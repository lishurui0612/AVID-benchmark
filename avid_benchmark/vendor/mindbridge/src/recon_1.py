import os
import inspect
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
from datetime import datetime
from pathlib import Path

import utils
from models_1 import Clipper, MindBridge, MindSingle, Voxel2StableDiffusionModel
import data_1 as data
from options_1 import args
from avid_benchmark.models.lowlevel import load_lowlevel_model


# ===== AVID adaptation for tar-based multi-subject data: START =====
def prepare_subject_metadata(args):
    if args.subj_load is None:
        args.subj_load = [args.subj_test]


def prepare_data(args):
    args.batch_size = 1
    args.num_workers = 0
    val_path = args.val_tar_path or data.resolve_avid_shard_path(args.data_path, args.subj_test, args.data, "val")
    img_root = data.resolve_img_root(args)
    print(f"Using val tar: {val_path}", flush=True)
    print(f"Using image root: {img_root}", flush=True)

    val_dl = data.get_tar_dataloader(
        url=val_path,
        batch_size=args.batch_size,
        subject=args.subj_test,
        img_root=img_root,
        pool_num=args.pool_num,
        # The AVID low-level model consumes raw vertices. Pool only the
        # high-level MindBridge input after constructing the blurry image.
        pool_type=None if args.img2img_strength < 1 else args.pool_type,
        is_shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    print(f"Prepared recon dataloader | data={args.data} | subject={args.subj_test} | split=val | batch_size={args.batch_size}", flush=True)
    return val_dl


def prepare_train_log_dir(args):
    if args.train_outdir:
        return args.train_outdir
    return os.path.join('./outputs/mindbridge', args.data, args.model_name)


def prepare_recon_dir(args):
    if args.results_path:
        return args.results_path
    return os.path.join('./predictions/mindbridge', args.ckpt_from, args.data, f'{args.subj_test:02d}', args.model_name)
    # return os.path.join('./predictions/mindbridge', args.ckpt_from, args.data, f'S{args.subj_test}', args.model_name)
# ===== AVID adaptation for tar-based multi-subject data: END =====


## Load autoencoder
def resolve_vd_repo_dir(vd_cache_dir: str) -> str:
    # return vd_cache_dir
    repo_dir = Path(vd_cache_dir)
    snapshots_dir = repo_dir / "snapshots"
    if snapshots_dir.is_dir():
        snapshot_paths = sorted([path for path in snapshots_dir.iterdir() if path.is_dir()])
        if not snapshot_paths:
            raise FileNotFoundError(f"No snapshot directories found under {snapshots_dir}")
        return str(snapshot_paths[-1])
    return str(repo_dir)


def prepare_voxel2sd(args, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']

    voxel2sd = load_lowlevel_model(Voxel2StableDiffusionModel, state_dict)
    del checkpoint, state_dict
    voxel2sd.to(device)
    voxel2sd.eval()
    print("Loaded low-level model!")

    return voxel2sd


def prepare_VD(args, device):
    print('Creating versatile diffusion reconstruction pipeline...')
    from diffusers import VersatileDiffusionDualGuidedPipeline, UniPCMultistepScheduler
    from diffusers.models import DualTransformer2DModel
    from diffusers.pipelines.versatile_diffusion import modeling_text_unet

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

    # vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(args.vd_cache_dir)
    repo_dir = resolve_vd_repo_dir(args.vd_cache_dir)
    vd_pipe = VersatileDiffusionDualGuidedPipeline.from_pretrained(repo_dir, local_files_only=True)
    vd_pipe.image_unet.eval().to(device)
    vd_pipe.vae.eval().to(device)
    vd_pipe.image_unet.requires_grad_(False)
    vd_pipe.vae.requires_grad_(False)

    # vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(args.vd_cache_dir, subfolder="scheduler")
    vd_pipe.scheduler = UniPCMultistepScheduler.from_pretrained(repo_dir, subfolder="scheduler")

    for name, module in vd_pipe.image_unet.named_modules():
        if isinstance(module, DualTransformer2DModel):
            module.mix_ratio = args.text_image_ratio
            for i, type in enumerate(("text", "image")):
                if type == "text":
                    module.condition_lengths[i] = 77
                    module.transformer_index_for_condition[i] = 1
                else:
                    module.condition_lengths[i] = 257
                    module.transformer_index_for_condition[i] = 0

    return vd_pipe


def prepare_CLIP(args, device):
    clip_sizes = {"RN50": 1024, "ViT-L/14": 768, "ViT-B/32": 512, "ViT-H-14": 1024}
    clip_size = clip_sizes[args.clip_variant]
    out_dim_image = 257 * clip_size
    out_dim_text = 77 * clip_size
    clip_extractor = Clipper(args.clip_variant, hidden_state=True, norm_embs=args.norm_embs, device=device)

    return clip_extractor, out_dim_image, out_dim_text


def prepare_voxel2clip(args, out_dim_image, out_dim_text, device):
    voxel2clip_kwargs = dict(
        in_dim=args.pool_num, out_dim_image=out_dim_image, out_dim_text=out_dim_text,
        h=args.h_size, n_blocks=args.n_blocks, subj_list=args.subj_load
    )

    voxel2clip = MindSingle(**voxel2clip_kwargs)

    outdir = prepare_train_log_dir(args)
    primary_ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path else Path(outdir) / f"{args.ckpt_from}.pth"
    if args.checkpoint_path:
        fallback_ckpt_path = primary_ckpt_path.with_name("best.pth")
    else:
        fallback_ckpt_path = Path(outdir) / "best.pth"
    if primary_ckpt_path.exists():
        ckpt_path = primary_ckpt_path
    elif fallback_ckpt_path.exists():
        print(f"Checkpoint not found: {primary_ckpt_path}. Falling back to {fallback_ckpt_path}.", flush=True)
        ckpt_path = fallback_ckpt_path
    else:
        raise FileNotFoundError(
            f"Neither requested checkpoint nor best checkpoint exists: {primary_ckpt_path} | {fallback_ckpt_path}"
        )
    print("ckpt_path", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    print("EPOCH:", checkpoint['epoch'])
    state_dict = checkpoint['model_state_dict']

    missing, unexpected = voxel2clip.load_state_dict(state_dict, strict=False)
    # Joint checkpoints also contain other subjects' embedders and builders.
    if missing or any(not key.startswith(('embedder.', 'builder.')) for key in unexpected):
        raise ValueError(f'Incomplete MindBridge prediction weights: missing={missing}, unexpected={unexpected}')
    voxel2clip.requires_grad_(False)
    voxel2clip.eval().to(device)

    return voxel2clip


def main(device):
    prepare_subject_metadata(args)

    test_dl = prepare_data(args)

    if args.autoencoder_ckpt_path:
        ckpt_path = args.autoencoder_ckpt_path
    else:
        outdir_ae = os.path.join('./outputs/mindbridge', args.data, args.autoencoder_name) if args.autoencoder_name else None
        ckpt_path = os.path.join(outdir_ae, 'last.pth') if outdir_ae else None
    if args.img2img_strength < 1:
        if ckpt_path is None or not os.path.exists(ckpt_path):
            raise FileNotFoundError('img2img requires the matching AVID low-level checkpoint')
        voxel2sd = prepare_voxel2sd(args, ckpt_path, device)
    else:
        voxel2sd = None

    vd_pipe = prepare_VD(args, device)
    unet = vd_pipe.image_unet
    vae = vd_pipe.vae
    noise_scheduler = vd_pipe.scheduler

    clip_extractor, out_dim_image, out_dim_text = prepare_CLIP(args, device)
    voxel2clip = prepare_voxel2clip(args, out_dim_image, out_dim_text, device)

    save_dir = prepare_recon_dir(args)
    os.makedirs(save_dir, exist_ok=True)
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    all_brain_recons = []

    for val_i, (voxel, img, coco, subj) in enumerate(tqdm(test_dl)):
        if val_i < args.test_start:
            continue
        if args.test_end is not None and val_i >= args.test_end:
            break
        if (args.samples is not None) and (val_i not in args.samples):
            continue

        stim_ids = data.batch_stim_ids(coco)
        output_name = stim_ids[0] if stim_ids else f"recon_{val_i}.png"

        voxel = torch.mean(voxel, axis=1).float().to(device)
        img = img.to(device)

        with torch.no_grad():
            if args.only_embeddings:
                embeddings = voxel2clip(data.pool_voxels(voxel, args.pool_num, args.pool_type))[:2]
                torch.save(embeddings, os.path.join(save_dir, f'embeddings_{val_i}.pt'))
                continue

            if args.img2img_strength < 1 and voxel2sd is not None:
                if voxel.shape[-1] != voxel2sd.lin0[0].in_features:
                    raise ValueError(f'Low-level checkpoint expects {voxel2sd.lin0[0].in_features} vertices, '
                                     f'but target data has {voxel.shape[-1]}')
                ae_preds = voxel2sd(voxel)
                blurry_recons = vd_pipe.vae.decode(ae_preds.to(device) / 0.18215).sample / 2 + 0.5
                voxel = data.pool_voxels(voxel, args.pool_num, args.pool_type)
            else:
                blurry_recons = None

            if args.img2img_strength == 0 and blurry_recons is not None:
                brain_recons = blurry_recons
            else:
                grid, brain_recons, best_picks, recon_img = utils.reconstruction(
                    img, voxel, voxel2clip,
                    clip_extractor, unet, vae, noise_scheduler,
                    img_lowlevel=blurry_recons,
                    num_inference_steps=args.num_inference_steps,
                    n_samples_save=args.batch_size,
                    recons_per_sample=args.recons_per_sample,
                    guidance_scale=args.guidance_scale,
                    img2img_strength=args.img2img_strength,
                    seed=args.seed,
                    plotting=args.plotting,
                    verbose=args.verbose,
                    device=device,
                    mem_efficient=False,
                )

                if args.plotting:
                    grid.savefig(os.path.join(save_dir, output_name))

                brain_recons = brain_recons[:, best_picks.astype(np.int8)]

            all_brain_recons.append(brain_recons.cpu())

    if args.only_embeddings:
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print("save path:", save_dir)
        return

    if not all_brain_recons:
        raise ValueError("No reconstructions were generated.")

    all_brain_recons = torch.vstack(all_brain_recons)
    if all_brain_recons.ndim == 5:
        all_brain_recons = all_brain_recons.view(-1, all_brain_recons.shape[-3], all_brain_recons.shape[-2], all_brain_recons.shape[-1])

    torch.save(all_brain_recons, os.path.join(save_dir, 'recon_images.pt'))

    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("save path:", save_dir)
    print("saved recon tensor:", os.path.join(save_dir, 'recon_images.pt'))


if __name__ == "__main__":
    utils.seed_everything(seed=args.seed)
    device = torch.device('cuda:{}'.format(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    main(device)
