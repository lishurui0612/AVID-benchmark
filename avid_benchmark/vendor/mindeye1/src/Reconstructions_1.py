#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # Code to convert this notebook to .py if you want to run it via command line or with Slurm
# from subprocess import call
# command = "jupyter nbconvert Reconstructions.ipynb --to python"
# call(command,shell=True)


# run
# python Reconstructions_1.py --autoencoder_name model_low/autoencoder --model_name model_15000 --sub S5_WD --data main

# In[2]:


import os
import sys
import json
import inspect
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime
import webdataset as wds
import PIL
import argparse
from pathlib import Path
from PIL import Image
# gpu = 5
# device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(gpu)
# local_rank = 0
# print("device:",device)

import utils_1
from models import Clipper, OpenClipper, BrainNetwork, BrainDiffusionPrior, BrainDiffusionPriorOld, Voxel2StableDiffusionModel, VersatileDiffusionPriorNetwork

if utils_1.is_interactive():
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')

seed=42
utils_1.seed_everything(seed=seed)


# # Configurations

# In[2]:


# if running this interactively, can specify jupyter_args here for argparser to use
if utils_1.is_interactive():
    # Example use
    jupyter_args = "--data_path=/path/to/AVID/AVID_surfaceNoSpeechLanguage \
                    --subj=1 \
                    --model_name=prior_257_final_subj01_bimixco_softclip_byol"
    
    jupyter_args = jupyter_args.split()
    print(jupyter_args)


# In[3]:
parser = argparse.ArgumentParser(description="Model Training Configuration")
parser.add_argument(
    "--gpu",type=str, default='5',
)

parser.add_argument(
    "--model_name", type=str, default="testing",
    help="name of trained model",
)
parser.add_argument(
    "--autoencoder_name", type=str, default="None",
    help="name of trained autoencoder model",
)
parser.add_argument(
    "--data_path", type=str, default="/path/to/AVID/AVID_surfaceNoSpeechLanguage/",
    help="Root directory containing AVID tar shards when an explicit validation shard path is not provided.",
)
parser.add_argument(
    "--data", type=str, default="complex",
    help="complex or simple",
)
parser.add_argument(
    "--data_cro", type=str, default="None",
    help="Cross validation(e.g.,complex or simple)",
)
parser.add_argument(
    "--sub", type=str, default="None",
    help="name of subject (e.g., S1_YPP)",
)
parser.add_argument(
    "--pth", type=str, default="None",
    help="best or last",
)
parser.add_argument(
    "--subj",type=int, default=1,
)
parser.add_argument(
    "--val_tar_path", type=str, default=None,
    help="Direct path override for the validation tar shard.",
)
parser.add_argument(
    "--img2img_strength",type=float, default=.85,
    help="How much img2img (1=no img2img; 0=outputting the low-level image itself)",
)
parser.add_argument(
    "--recons_per_sample", type=int, default=5,
    help="How many recons to output, to then automatically pick the best one (MindEye uses 16)",
)
parser.add_argument(
    "--vd_cache_dir", type=str, default='',
    help="Where is cached Versatile Diffusion model; if not cached will download to this path",
)
parser.add_argument(
    "--img_root", type=str, default=None,
    help="Image root used to resolve img_name.npy values.",
)
parser.add_argument(
    "--train_outdir", type=str, default=None,
    help="Directory containing the trained checkpoints.",
)
parser.add_argument(
    "--recon_outdir", type=str, default=None,
    help="Directory where reconstruction outputs are written.",
)
parser.add_argument(
    "--checkpoint_path", type=str, default=None,
    help="Direct checkpoint path override for the main model.",
)
parser.add_argument(
    "--autoencoder_ckpt_path", type=str, default=None,
    help="Direct checkpoint path override for the low-level autoencoder.",
)
parser.add_argument(
    "--save_recon_tensor_path", type=str, default=None,
    help="Direct path for the saved reconstruction tensor.",
)
parser.add_argument(
    "--save_all_images_path", type=str, default=None,
    help="Direct path for the saved ground-truth tensor.",
)
parser.add_argument(
    "--num_inference_steps", type=int, default=20,
    help="Number of inference steps for diffusion reconstruction.",
)
parser.add_argument(
    "--guidance_scale", type=float, default=3.5,
    help="Guidance scale for diffusion reconstruction.",
)
parser.add_argument(
    "--text_image_ratio", type=float, default=0.0,
    help="Dual-guidance text/image blend ratio.",
)
parser.add_argument("--max_samples", type=int, default=0, help="0 predicts the full validation set.")
parser.add_argument("--seed", type=int, default=42)
if utils_1.is_interactive():
    args = parser.parse_args(jupyter_args)
else:
    args = parser.parse_args()
utils_1.seed_everything(seed=args.seed)
gpu  = int(args.gpu) 
device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(gpu)
local_rank = 0
print("device:",device)



# create global variables without the args prefix
for attribute_name in vars(args).keys():
    globals()[attribute_name] = getattr(args, attribute_name)
    
if autoencoder_name=="None":
    autoencoder_name = None


# In[4]:

val_url = args.val_tar_path or f"{data_path}/sub-{sub}/sub-{sub}_{data}_val.tar"
# val_url = f"{data_path}/{sub}_main_val.tar"

meta_url = None
if data == 'complex':
    num_train = 640
    num_val = 160
    img_root = args.img_root or "/path/to/AVID/coco_images"
elif data == 'simple':
    num_train = 320
    num_val = 80
    img_root = args.img_root or "/path/to/AVID/simple_stimuli_corresponding_image"
else:
    print(f'Error: {data}')
    


def load_image(sample, img_root):
    stim = str(sample["stim"].item())
    img_path = f"{img_root}/{stim}"

    image = Image.open(img_path).convert("RGB")
    
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    sample["images"] = img_transform(image)

    return sample
batch_size = val_batch_size = 1
voxels_key = 'responses.npy' # 1d inputs

val_data = wds.WebDataset(val_url, resampled=False)\
    .decode("torch")\
    .rename(voxels="responses.npy", stim="img_name.npy")\
    .map(lambda sample: load_image(sample, img_root))\
    .to_tuple("voxels", "stim", "images")\
    .batched(val_batch_size, partial=False)

# import pdb;pdb.set_trace()

val_dl = torch.utils.data.DataLoader(val_data, batch_size=None, shuffle=False)

# In[5]:
# check that your data loader is working
for val_i, (voxel, img_input, stim) in enumerate(val_dl):
    print("idx",val_i)
    print("voxel.shape",voxel.shape) # [1, 3, 15724]
    print("img_input.shape",img_input.shape) # [1, 3, 256, 256]
    num_voxels = voxel.shape[2]
    break
print("sub",sub,"num_voxels",num_voxels)
# import pdb;pdb.set_trace()
# ## Load autoencoder

# In[6]:


from models import Voxel2StableDiffusionModel
from avid_benchmark.models.lowlevel import load_lowlevel_model

# outdir = f'../train_logs/{autoencoder_name}'
outdir = f'./outputs/mindeye1/{autoencoder_name}/{data}/vertex_{sub}_non_spe_lan'
ckpt_path = args.autoencoder_ckpt_path or os.path.join(outdir, f'{pth}.pth')

if img2img_strength < 1:
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError('img2img requires the matching AVID low-level checkpoint')
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    voxel2sd = load_lowlevel_model(Voxel2StableDiffusionModel, state_dict, input_dim=num_voxels)
    del checkpoint, state_dict
    voxel2sd.eval()
    voxel2sd.to(device)
    print("Loaded low-level model!")
else:
    voxel2sd = None


# # Load VD pipe

# In[12]:


print('Creating versatile diffusion reconstruction pipeline...')
from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
from diffusers.models import DualTransformer2DModel
from diffusers.pipelines.versatile_diffusion import modeling_text_unet


def resolve_vd_repo_dir(vd_cache_dir: str) -> str:
    repo_dir = Path(vd_cache_dir)
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.is_dir():
        snapshot_paths = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])
        if not snapshot_paths:
            raise FileNotFoundError(f"No snapshot directories found under {snapshots_dir}")
        return str(snapshot_paths[-1])
    nested_snapshots = repo_dir / "models--shi-labs--versatile-diffusion" / "snapshots"
    if nested_snapshots.is_dir():
        snapshot_paths = sorted([path for path in nested_snapshots.iterdir() if path.is_dir()])
        if not snapshot_paths:
            raise FileNotFoundError(f"No snapshot directories found under {nested_snapshots}")
        return str(snapshot_paths[-1])
    return str(repo_dir)

# repo_dir = "../../../dataset/fsx/proj-medar/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7"
repo_dir = resolve_vd_repo_dir(vd_cache_dir)

if not getattr(modeling_text_unet, "_avid_transformer_layers_patch", False):
    original_get_down_block = modeling_text_unet.get_down_block
    original_get_up_block = modeling_text_unet.get_up_block
    down_signature = inspect.signature(original_get_down_block)
    up_signature = inspect.signature(original_get_up_block)

    def _compat_get_down_block(*compat_args, **compat_kwargs):
        filtered_kwargs = {
            key: value
            for key, value in compat_kwargs.items()
            if key in down_signature.parameters
        }
        return original_get_down_block(*compat_args, **filtered_kwargs)

    def _compat_get_up_block(*compat_args, **compat_kwargs):
        filtered_kwargs = {
            key: value
            for key, value in compat_kwargs.items()
            if key in up_signature.parameters
        }
        return original_get_up_block(*compat_args, **filtered_kwargs)

    modeling_text_unet.get_down_block = _compat_get_down_block
    modeling_text_unet.get_up_block = _compat_get_up_block
    modeling_text_unet._avid_transformer_layers_patch = True

if not getattr(DualTransformer2DModel, "_avid_forward_signature_patch", False):
    original_dual_transformer_forward = DualTransformer2DModel.forward
    dual_transformer_signature = inspect.signature(original_dual_transformer_forward)

    def _compat_dual_transformer_forward(self, *compat_args, **compat_kwargs):
        filtered_kwargs = {
            key: value
            for key, value in compat_kwargs.items()
            if key in dual_transformer_signature.parameters
        }
        return original_dual_transformer_forward(self, *compat_args, **filtered_kwargs)

    DualTransformer2DModel.forward = _compat_dual_transformer_forward
    DualTransformer2DModel._avid_forward_signature_patch = True

# scheduler = 'models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7/scheduler'
# try:
# import pdb;pdb.set_trace()
vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(repo_dir,local_files_only=True).to(device).to(torch.float16)
# import pdb;pdb.set_trace()
# except:
# print("Downloading Versatile Diffusion to", vd_cache_dir)
# vd_pipe =  VersatileDiffusionDualGuidedPipeline.from_pretrained(
#         "shi-labs/versatile-diffusion",
#         torch_dtype=torch.float16,
#         # cache_dir = vd_cache_dir).to(device).to(torch.float16)
#         cache_dir = vd_cache_dir).to(device)

vd_pipe.image_unet.eval()
vd_pipe.vae.eval()
vd_pipe.image_unet.requires_grad_(False)
vd_pipe.vae.requires_grad_(False)

vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(repo_dir, subfolder="scheduler")
# import pdb;pdb.set_trace()
# vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(vd_cache_dir)
num_inference_steps = args.num_inference_steps

# Set weighting of Dual-Guidance 
text_image_ratio = args.text_image_ratio # .5 means equally weight text and image, 0 means use only image
for name, module in vd_pipe.image_unet.named_modules():
    if isinstance(module, DualTransformer2DModel):
        module.mix_ratio = text_image_ratio
        for i, type in enumerate(("text", "image")):
            # import pdb;pdb.set_trace()
            if type == "text":
                module.condition_lengths[i] = 77
                module.transformer_index_for_condition[i] = 1  # use the second (text) transformer
            else:
                module.condition_lengths[i] = 257
                module.transformer_index_for_condition[i] = 0  # use the first (image) transformer

unet = vd_pipe.image_unet
vae = vd_pipe.vae
noise_scheduler = vd_pipe.scheduler


# ## Load Versatile Diffusion model

# In[8]:


img_variations = False

out_dim = 257 * 768
clip_extractor = Clipper("ViT-L/14", hidden_state=True, norm_embs=True, device=device)
voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim)
voxel2clip = BrainNetwork(**voxel2clip_kwargs)
voxel2clip.requires_grad_(False)
voxel2clip.eval()

out_dim = 768
depth = 6
dim_head = 64
heads = 12 # heads * dim_head = 12 * 64 = 768
timesteps = 100 #100
# Train_MindEye.py
prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        learned_query_mode="pos_emb"
    )

diffusion_prior = BrainDiffusionPrior(
    net=prior_network,
    image_embed_dim=out_dim,
    condition_on_text_encodings=False,
    timesteps=timesteps,
    cond_drop_prob=0.2,
    image_embed_scale=None,
    voxel2clip=voxel2clip,
)

outdir = args.train_outdir or f'./outputs/mindeye1/{data}/{model_name}_{sub}_non_spe_lan'
# import pdb;pdb.set_trace()
ckpt_path = args.checkpoint_path or os.path.join(outdir, f'{pth}.pth')
# ckpt_path = os.path.join(outdir, f'best.pth')

print("ckpt_path",ckpt_path)
checkpoint = torch.load(ckpt_path, map_location=device)
state_dict = checkpoint['model_state_dict']
print("EPOCH: ",checkpoint['epoch'])
diffusion_prior.load_state_dict(state_dict,strict=True)
diffusion_prior.eval().to(device)
diffusion_priors = [diffusion_prior]
pass


# ## Load Image Variations model

# In[9]:


# img_variations = True

# # CLS model
# out_dim = 768
# clip_extractor = Clipper("ViT-L/14", hidden_state=False, norm_embs=False, device=device)
# voxel2clip_kwargs = dict(in_dim=num_voxels,out_dim=out_dim)
# voxel2clip = BrainNetwork(**voxel2clip_kwargs)
# voxel2clip.requires_grad_(False)
# voxel2clip.eval()

# diffusion_prior = BrainDiffusionPriorOld.from_pretrained(
#     # kwargs for DiffusionPriorNetwork
#     dict(),
#     # kwargs for DiffusionNetwork
#     dict(
#         condition_on_text_encodings=False,
#         timesteps=1000,
#         voxel2clip=voxel2clip,
#     ),
#     voxel2clip_path=None,
# )

# outdir = f'../train_logs/{model_name}'
# ckpt_path = os.path.join(outdir, f'last.pth')

# print("ckpt_path",ckpt_path)
# checkpoint = torch.load(ckpt_path, map_location=device)
# state_dict = checkpoint['model_state_dict']
# print("EPOCH: ",checkpoint['epoch'])
# diffusion_prior.load_state_dict(state_dict,strict=False)
# diffusion_prior.eval().to(device)
# diffusion_priors = [diffusion_prior]
# pass


# In[10]:


# from diffusers import AutoencoderKL, UNet2DConditionModel, UniPCMultistepScheduler

# sd_cache_dir = '/fsx/home-paulscotti/.cache/huggingface/diffusers/models--lambdalabs--sd-image-variations-diffusers/snapshots/a2a13984e57db80adcc9e3f85d568dcccb9b29fc'
# unet = UNet2DConditionModel.from_pretrained(sd_cache_dir,subfolder="unet").to(device)

# unet.eval() # dont want to train model
# unet.requires_grad_(False) # dont need to calculate gradients

# vae = AutoencoderKL.from_pretrained(sd_cache_dir,subfolder="vae").to(device)
# vae.eval()
# vae.requires_grad_(False)

# noise_scheduler = UniPCMultistepScheduler.from_pretrained(sd_cache_dir, subfolder="scheduler")
# num_inference_steps = 20


# # Reconstruct one-at-a-time

# In[13]:


print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

retrieve = False
plotting = True
saving = True
verbose = False
imsize = 512

if img_variations:
    guidance_scale = 7.5
else:
    guidance_scale = args.guidance_scale
    
ind_include = np.arange(min(num_val, args.max_samples) if args.max_samples else num_val)
all_brain_recons = None
    
only_lowlevel = False
if img2img_strength == 1:
    img2img = False
elif img2img_strength == 0:
    img2img = True
    only_lowlevel = True
else:
    img2img = True
    
for val_i, (voxel, stim, img) in enumerate(tqdm(val_dl,total=len(ind_include))):
    # import pdb;pdb.set_trace()
    if val_i<np.min(ind_include):
        continue
    voxel = torch.mean(voxel,axis=1).to(device)
    # voxel = voxel[:,0].to(device)
    
    with torch.no_grad():
        if img2img:
            ae_preds = voxel2sd(voxel.float())
            blurry_recons = vd_pipe.vae.decode(ae_preds.to(device).half()/0.18215).sample / 2 + 0.5 # / 2 + 0.5是干嘛？
            

            if val_i==0:
                plt.imshow(utils_1.torch_to_Image(blurry_recons))
                plt.show()
        else:
            blurry_recons = None

        if only_lowlevel:
            brain_recons = blurry_recons
        else:
            grid, brain_recons, laion_best_picks, recon_img = utils_1.reconstruction(
                img, voxel,
                clip_extractor, unet, vae, noise_scheduler,
                voxel2clip_cls = None, #diffusion_prior_cls.voxel2clip,
                diffusion_priors = diffusion_priors,
                text_token = None,
                img_lowlevel = blurry_recons,
                num_inference_steps = num_inference_steps,
                n_samples_save = batch_size,
                recons_per_sample = recons_per_sample,
                guidance_scale = guidance_scale,
                img2img_strength = img2img_strength, # 0=fully rely on img_lowlevel, 1=not doing img2img
                timesteps_prior = 100,
                seed = seed,
                retrieve = retrieve,
                plotting = plotting,
                img_variations = img_variations,
                verbose = verbose,
            )

            if plotting:
                # plt.show()
                recon_dir = args.recon_outdir or f'./predictions/mindeye1/{pth}/{data}/{sub}/vertex_strength_{img2img_strength}_non_spe_lan'
                os.makedirs(recon_dir, exist_ok=True)
                grid.savefig(os.path.join(recon_dir, str(stim.item())))
                plt.close(grid)
            # import pdb;pdb.set_trace()
            brain_recons = brain_recons[:,laion_best_picks.astype(np.int8)]

        if all_brain_recons is None:
            all_brain_recons = brain_recons
            all_images = img
        else:
            all_brain_recons = torch.vstack((all_brain_recons,brain_recons))
            all_images = torch.vstack((all_images,img))

    if val_i>=np.max(ind_include):
        break

all_brain_recons = all_brain_recons.view(-1,3,imsize,imsize)
print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



if saving:
    recon_dir = args.recon_outdir or f'./predictions/mindeye1/{pth}/{data}/{sub}/vertex_strength_{img2img_strength}_non_spe_lan'
    os.makedirs(recon_dir, exist_ok=True)
    all_images_path = args.save_all_images_path or os.path.join(recon_dir, f'{model_name}_all_images.pt')
    recon_tensor_path = args.save_recon_tensor_path or os.path.join(recon_dir, f'{model_name}_recons_img2img{img2img_strength}_{recons_per_sample}samples.pt')
    torch.save(all_images, all_images_path)
    torch.save(all_brain_recons, recon_tensor_path)
print(args.save_recon_tensor_path or f'{args.recon_outdir or f"./predictions/mindeye1/{pth}/{data}/{sub}/vertex_strength_{img2img_strength}_non_spe_lan"}/{model_name}_recons_img2img{img2img_strength}_{recons_per_sample}samples')

if not utils_1.is_interactive():
    sys.exit(0)
