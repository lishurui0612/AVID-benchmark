import json
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
from avid_benchmark.cli import resolve_checkpoint


def test_mixco_targets_stay_in_subject_block():
    from avid_benchmark.models.training_utils import offset_mixco_permutation
    assert offset_mixco_permutation(torch.tensor([1,0]),[torch.zeros(2,1,3)]).tolist()==[3,2]
    assert offset_mixco_permutation(torch.tensor([1,0]),[]).tolist()==[1,0]


@pytest.mark.parametrize('model,filename',[('mindeye1','utils_1.py'),('mindeye2','utils.py')])
def test_mixco_self_permutation_retains_full_target_probability(model,filename):
    import ast
    source=Path(__file__).parents[1]/f'avid_benchmark/vendor/{model}/src/{filename}'
    function=next(node for node in ast.parse(source.read_text(encoding='utf-8')).body if isinstance(node,ast.FunctionDef) and node.name=='mixco_nce')
    namespace={'torch':torch,'F':torch.nn.functional}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source),'exec'),namespace)
    predictions=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
    targets=torch.tensor([[.8,.2],[.3,.7]])
    plain=namespace['mixco_nce'](predictions,targets)
    mixed=namespace['mixco_nce'](predictions,targets,perm=torch.arange(2),betas=torch.tensor([1.,.2]),select=torch.tensor([False,True]))
    torch.testing.assert_close(mixed,plain)
    mixed.backward()
    assert predictions.grad.abs().sum()>0


def test_loo_checkpoint_uses_finetune_not_pretrain(tmp_path):
    from avid_benchmark.types import PathConfig
    paths=PathConfig(tmp_path,tmp_path,tmp_path,tmp_path,tmp_path)
    cfg={'paths':paths,'model_name':'mindeye2','train':{'mode':'multi_subject_leave_one_out','run_name':'test'}}
    root=tmp_path/'mindeye2/test'
    (root/'logs').mkdir(parents=True)
    pretrained=root/'loo_pretrain_best.pth'; pretrained.touch()
    adapted=root/'loo_finetune_last.pth'; adapted.touch()
    (root/'logs/train_result.json').write_text(json.dumps({'best_checkpoint':str(pretrained),'finetunes':{'pretrain_best':{'last_checkpoint':str(adapted)}}}))
    assert resolve_checkpoint(cfg)==adapted.resolve()


def test_bridge_training_receives_requested_dataset_root():
    source=(Path(__file__).parents[1]/'avid_benchmark/models/mindbridge/adapter.py').read_text()
    assert '"--data_path",\n            str(self.paths.dataset_root)' in source


def test_bridge_lowlevel_uses_raw_vertices_before_highlevel_pooling(monkeypatch):
    import ast
    from types import SimpleNamespace
    source=Path(__file__).parents[1]/'avid_benchmark/vendor/mindbridge/src/recon_1.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    functions=[node for node in tree.body if isinstance(node,ast.FunctionDef)
               and node.name in ('prepare_data','prepare_voxel2sd')]
    calls={}
    class LowLevel(torch.nn.Module):
        def __init__(self,in_dim):
            super().__init__()
            self.lin0=torch.nn.Sequential(torch.nn.Linear(in_dim,2,bias=False))
    checkpoint={'model_state_dict':LowLevel(17).state_dict()}
    monkeypatch.setattr(torch,'load',lambda *a,**kw:checkpoint)
    data=SimpleNamespace(resolve_img_root=lambda args:'/images',
                         get_tar_dataloader=lambda **kw:calls.update(kw) or 'loader')
    namespace=dict(torch=torch,Voxel2StableDiffusionModel=LowLevel,data=data,
                   load_lowlevel_model=lambda cls,state:cls(in_dim=state['lin0.0.weight'].shape[1]))
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
    args=SimpleNamespace(pool_num=8,pool_type='max',img2img_strength=.85,
                         val_tar_path='/data.tar',subj_test=1,data='complex',seed=42)
    assert namespace['prepare_data'](args)=='loader'
    assert calls['pool_type'] is None
    model=namespace['prepare_voxel2sd'](args,'/checkpoint.pth','cpu')
    assert model.lin0[0].in_features==17
    assert args.pool_type=='max'
    args.img2img_strength=1
    namespace['prepare_data'](args)
    assert calls['pool_type']=='max'


def test_lowlevel_restores_projector_architecture_and_rejects_roi_mismatch():
    from avid_benchmark.models.lowlevel import load_lowlevel_model
    seen={}
    class Restored:
        def __init__(self,**kwargs):
            seen.update(kwargs)
        def load_state_dict(self,state,strict):
            assert strict
        def state_dict(self):
            return state
    state={'lin0.0.weight':torch.zeros(4,17),'norm.weight':torch.ones(64),
           'mlp.0.0.weight':torch.zeros(4,4),'maps_projector.0.weight':torch.zeros(2,2)}
    load_lowlevel_model(Restored,state,input_dim=17)
    assert seen==dict(in_dim=17,h=4,n_blocks=1,ups_mode='4x',use_cont=True)
    with pytest.raises(ValueError,match='expects 17 vertices'):
        load_lowlevel_model(Restored,state,input_dim=19)


def test_lowlevel_legacy_attention_rename_preserves_tensor_and_strict_loading():
    from avid_benchmark.models.lowlevel import load_lowlevel_model
    old='upsampler.mid_block.attentions.0.query.weight'
    new='upsampler.mid_block.attentions.0.to_q.weight'
    state={'lin0.0.weight':torch.zeros(4,17),'norm.weight':torch.ones(64),
           old:torch.arange(16).reshape(4,4)}
    class Decoder:
        def __init__(self,**kwargs):
            pass
        def state_dict(self):
            return {new:None,'lin0.0.weight':None,'norm.weight':None}
        def load_state_dict(self,restored,strict):
            assert strict and set(restored)==set(self.state_dict())
            assert restored[new] is state[old]
    model=load_lowlevel_model(Decoder,state,input_dim=17)
    assert model._avid_lowlevel_key_migrations=={old:new}
    with pytest.raises(ValueError,match='Conflicting legacy/current'):
        load_lowlevel_model(Decoder,dict(state,**{new:torch.ones(4,4)}))


def test_metric_import_does_not_import_torch_in_fresh_interpreter():
    import subprocess,sys
    subprocess.run([sys.executable,'-c',"import avid_benchmark.evaluation; import sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],check=True)


def test_reconstruction_range_is_explicit():
    import pytest
    from avid_benchmark.models.vendor_runtime import recon_tensor_to_uint8
    with pytest.raises(ValueError,match='range guessing'):
        recon_tensor_to_uint8(torch.full((1,3,2,2),1.001))
    assert recon_tensor_to_uint8(torch.ones((1,3,2,2))).max()==255


def test_mindeye2_exports_single_generated_image():
    from avid_benchmark.models.mindeye2.adapter import MindEye2Adapter
    image=torch.tensor([0.,.5,1.]).reshape(3,1,1)
    output=MindEye2Adapter._image_tensor_to_uint8(image)
    assert output.dtype==torch.uint8
    assert output.flatten().tolist()==[0,128,255]
