"""Run with torchrun --standalone --nproc_per_node=2 tests/ddp_gradient_check.py."""
import os
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from avid_benchmark.models.mindeye2.adapter import MindEyeModule, RidgeRegression
from avid_benchmark.models.training_utils import gather_contrastive_batch


class Backbone(nn.Module):
    def __init__(self):
        super().__init__(); self.linear=nn.Linear(3,3)
    def forward(self,x):
        y=self.linear(x)
        return y,y,None


class Prior(nn.Module):
    def __init__(self):
        super().__init__(); self.linear=nn.Linear(3,3)
    def forward(self,text_embed,image_embed):
        y=self.linear(text_embed)
        return (y-image_embed).square().mean(),y


def build():
    model=MindEyeModule(); model.ridge=RidgeRegression([2,2],3)
    model.backbone=Backbone(); model.diffusion_prior=Prior()
    return model


dist.init_process_group('gloo')
rank=dist.get_rank(); torch.manual_seed(123)
model=build(); reference=build(); reference.load_state_dict(model.state_dict())
ddp=DistributedDataParallel(model,find_unused_parameters=True)
inputs=[torch.arange(8,dtype=torch.float32).reshape(4,1,2)/8,torch.arange(8,dtype=torch.float32).reshape(4,1,2)/7]
targets=torch.arange(24,dtype=torch.float32).reshape(8,1,3)/24
local=[batch[rank*2:(rank+1)*2] for batch in inputs]
local_targets=torch.cat([targets[rank*2:(rank+1)*2],targets[4+rank*2:4+(rank+1)*2]])
output=ddp(local,[0,1],local_targets)
loss=output[1].square().mean()+output[3]; loss.backward()
full=reference(inputs,[0,1],targets)
(full[1].square().mean()+full[3]).backward()
for (name,param),(_,expected) in zip(model.named_parameters(),reference.named_parameters()):
    assert param.grad is not None, name
    torch.testing.assert_close(param.grad,expected.grad,rtol=1e-5,atol=1e-7)
    param.data.add_(param.grad,alpha=-.1)
    other=param.detach().clone(); dist.broadcast(other,src=0)
    torch.testing.assert_close(param,other,rtol=0,atol=0)
if rank==0:
    print('PASS: two-rank gradients equal global-batch reference; parameters identical after step.')

# Compare the complete symmetric contrastive objectives, including cross-rank negatives.
for mode in ('mixco', 'soft'):
    torch.manual_seed(321)
    linear=nn.Linear(3,3)
    expected=nn.Linear(3,3); expected.load_state_dict(linear.state_dict())
    parallel=DistributedDataParallel(linear)
    examples=torch.arange(12,dtype=torch.float32).reshape(4,3)/12
    target=torch.nn.functional.normalize(torch.tensor([[1.,0.,1.],[0.,1.,1.],[1.,1.,0.],[1.,-1.,1.]]),dim=-1)
    local_output=torch.nn.functional.normalize(parallel(examples[rank*2:rank*2+2]),dim=-1)
    perm=torch.tensor([1,0]); betas=torch.tensor([.2,.7]); selected=torch.ones(2,dtype=torch.bool)
    prediction, gathered_target, gathered_perm, gathered_betas, _=gather_contrastive_batch(
        local_output,target[rank*2:rank*2+2],perm,betas,selected)
    assert gathered_perm.tolist()==[1,0,3,2]
    def objective(pred,targs,permutation,weights):
        logits=(pred@targs.T)/.125
        if mode=='mixco':
            probabilities=torch.diag(weights)
            probabilities[torch.arange(len(weights)),permutation]+=1-weights
            return -((logits.log_softmax(-1)*probabilities).sum(-1).mean()+(logits.T.log_softmax(-1)*probabilities.T).sum(-1).mean())/2
        probabilities=((targs@targs.T)/.125).softmax(-1)
        return -((logits.log_softmax(-1)*probabilities).sum(-1).mean()+(logits.T.log_softmax(-1)*probabilities).sum(-1).mean())/2
    actual_loss=objective(prediction,gathered_target,gathered_perm,gathered_betas)
    reference_loss=objective(torch.nn.functional.normalize(expected(examples),dim=-1),target,torch.tensor([1,0,3,2]),betas.repeat(2))
    actual_loss.backward(); reference_loss.backward()
    torch.testing.assert_close(actual_loss,reference_loss)
    for parameter,reference_parameter in zip(linear.parameters(),expected.parameters()):
        torch.testing.assert_close(parameter.grad,reference_parameter.grad,rtol=1e-5,atol=1e-6)
    if rank==0:
        print(f'PASS: global {mode} candidates and gradients equal single-process reference.')
dist.destroy_process_group()
