"""Independent AVID fMRI-to-VAE training, using the original low-level losses.

The supplied VAE/ConvNeXt are frozen. Only Voxel2StableDiffusionModel is trained.
There is no image generation or benchmark evaluation in this stage.
"""
import importlib
import math
import os
from pathlib import Path
import sys
import time
from ..lowlevel import validate_response_dimension, load_legacy_attention_state


def lowlevel_loss(predicted, target, student_maps, teacher_maps, augmented_maps,
                  contrastive_weight=.1, temperature=.075, chunk_size=256):
    """Exact symmetric soft-target loss in chunks; keep the full token candidate set."""
    import torch
    from torch.nn import functional as F
    latent_l1 = F.l1_loss(predicted, target) / .18215
    student = F.normalize(student_maps.flatten(0, 1), dim=-1)
    teacher = F.normalize(teacher_maps.flatten(0, 1), dim=-1)
    augmented = F.normalize(augmented_maps.flatten(0, 1), dim=-1)
    if student.shape != teacher.shape or teacher.shape != augmented.shape:
        raise ValueError('Student and teacher token shapes differ')
    # Checkpointing avoids saving an N_tokens squared matrix for backward.
    from torch.utils.checkpoint import checkpoint
    def chunk_loss(indices, student, teacher, augmented):
        first = -(F.log_softmax(student[indices] @ augmented.T / temperature, dim=-1)
                  * F.softmax(teacher[indices] @ augmented.T / temperature, dim=-1)).sum()
        second = -(F.log_softmax(augmented[indices] @ student.T / temperature, dim=-1)
                   * F.softmax(augmented[indices] @ teacher.T / temperature, dim=-1)).sum()
        return (first + second) / (2 * len(student))
    contrastive = latent_l1.new_zeros(())
    for start in range(0, len(student), chunk_size):
        indices = torch.arange(start, min(start + chunk_size, len(student)), device=student.device)
        contrastive = contrastive + checkpoint(chunk_loss, indices, student, teacher, augmented,
                                               use_reentrant=False)
    return latent_l1 + contrastive_weight * contrastive, latent_l1, contrastive


def train_lowlevel(adapter):
    import numpy as np
    import torch
    from torch.nn import functional as F
    from PIL import Image
    from diffusers import AutoencoderKL
    import kornia.augmentation as K
    from ...assets import atomic_torch_save, local_asset
    from ...data.dataset import build_loader, count_samples_in_tar
    from ...runtime import save_json

    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        raise ValueError('MindEye1 low-level training currently uses one GPU; launch with python train.py.')
    if not torch.cuda.is_available():
        raise RuntimeError('Low-level model training requires a GPU.')
    options = adapter.experiment['lowlevel']
    epochs = int(options['num_epochs'])
    batch_size = int(options['batch_size'])
    if epochs < 1 or batch_size < 1:
        raise ValueError('epochs and batch size must be positive')
    if options['ups_mode'] != '4x':
        raise ValueError('The AVID low-level contrastive recipe requires the 4x decoder.')
    sys.path.insert(0, str(adapter.vendor_root))
    utils = importlib.import_module('utils_1')
    model_class = importlib.import_module('models').Voxel2StableDiffusionModel
    teacher_class = importlib.import_module('convnext').ConvnextXL
    seed = int(adapter.train_cfg.get('seed', 0))
    utils.seed_everything(seed, cudnn_deterministic=False)
    torch.backends.cuda.matmul.allow_tf32 = True
    device = torch.device('cuda', adapter.model_cfg.gpu_id)
    train_tar, val_tar, image_root = adapter._common_paths()
    train_count, val_count = count_samples_in_tar(train_tar), count_samples_in_tar(val_tar)
    from ...data.surface_nospeech import _stimulus_names_in_tar
    train_names, val_names = _stimulus_names_in_tar(train_tar), _stimulus_names_in_tar(val_tar)
    if set(train_names) & set(val_names):
        raise ValueError('Training and validation stimulus IDs overlap')
    def loader(path):
        return build_loader(path, batch_size, int(adapter.train_cfg.get('num_workers', 1)),
                            resampled=False, seed=seed, drop_last=False)
    train_loader, val_loader = loader(train_tar), loader(val_tar)
    probe = next(iter(train_loader))[1]
    input_dim = validate_response_dimension(probe, options.get('input_dim'))
    del probe
    steps = math.ceil(train_count / batch_size)
    if int(options.get('steps_per_epoch', 0)):
        steps = min(steps, int(options['steps_per_epoch']))

    model = model_class(in_dim=input_dim, h=int(options['hidden_dim']),
                        n_blocks=int(options['n_blocks']), use_cont=True,
                        ups_mode=options['ups_mode']).to(device)
    # This is the same VAE architecture and checkpoint used by the AVID script.
    vae_path, teacher_path = local_asset('mindeye1_vae'), local_asset('mindeye1_convnext')
    vae = AutoencoderKL(down_block_types=['DownEncoderBlock2D'] * 4,
                        up_block_types=['UpDecoderBlock2D'] * 4,
                        block_out_channels=[128, 256, 512, 512],
                        layers_per_block=2, sample_size=256)
    print('Loading frozen VAE:', vae_path, flush=True)
    vae_migrations = load_legacy_attention_state(vae, torch.load(vae_path, map_location='cpu'))
    vae.requires_grad_(False).eval().to(device)
    print('Loading frozen ConvNeXt teacher:', teacher_path, flush=True)
    teacher = teacher_class(teacher_path).requires_grad_(False).eval().to(device)
    augmentation = K.AugmentationSequential(
        K.ColorJitter(brightness=.4, contrast=.4, saturation=.2, hue=.1, p=.8),
        K.RandomGrayscale(p=.2), K.RandomSolarize(p=.2),
        K.RandomGaussianBlur(kernel_size=(7, 7), sigma=(.1, 2.), p=.1),
        K.RandomResizedCrop((512, 512), scale=(.5, 1.)), data_keys=['input']).to(device)
    mean = torch.tensor([.485, .456, .406], device=device).reshape(1, 3, 1, 1)
    std = torch.tensor([.228, .224, .225], device=device).reshape(1, 3, 1, 1)
    def images(names):
        rows = []
        for name in names:
            with Image.open(image_root / name) as image:
                rows.append(utils.img_transform(image.convert('RGB')))
        return F.interpolate(torch.stack(rows).to(device), (512, 512), mode='bilinear',
                             align_corners=False, antialias=True)

    # Cache only deterministic frozen targets; augmentation remains fresh per batch.
    targets, maps = {}, {}
    cache_start = time.monotonic()
    for split_loader, is_train in ((train_loader, True), (val_loader, False)):
        for batch_index, (names, responses) in enumerate(split_loader):
            validate_response_dimension(responses, input_dim)
            image = images(names)
            with torch.no_grad():
                latent = vae.encode(2 * image - 1).latent_dist.mode() * .18215
                features = teacher((image - mean) / std)[1] if is_train else None
            for index, name in enumerate(names):
                targets[name] = latent[index].cpu()
                if is_train:
                    maps[name] = features[index].cpu()
            if batch_index % 10 == 0:
                print(f'Caching frozen targets: train={is_train}, batch={batch_index}, cached={len(targets)}', flush=True)
    assert len(targets) == train_count + val_count and len(maps) == train_count
    del vae
    torch.cuda.empty_cache()
    grouped = [{'params': [p for n, p in model.named_parameters() if not any(t in n for t in ('bias', 'LayerNorm.bias', 'LayerNorm.weight'))], 'weight_decay': .01},
               {'params': [p for n, p in model.named_parameters() if any(t in n for t in ('bias', 'LayerNorm.bias', 'LayerNorm.weight'))], 'weight_decay': 0.}]
    optimizer = torch.optim.AdamW(grouped, lr=float(options['initial_lr']))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=float(options['max_lr']), total_steps=epochs * steps,
        final_div_factor=1000, pct_start=min(2 / epochs, .5)) if epochs * steps > 2 else None
    provenance = dict(component='lowlevel', initialized_from='random', input_dim=input_dim,
                      subject_id=adapter.subject_id, stimulus_type=adapter.stimulus_type,
                      train_tar=str(train_tar), val_tar=str(val_tar),
                      train_samples=train_count, val_samples=val_count,
                      train_sample_ids=train_names, val_sample_ids=val_names,
                      steps_per_epoch=steps, epochs=epochs, seed=seed,
                      architecture={key: options[key] for key in ('hidden_dim', 'n_blocks', 'ups_mode')},
                      frozen_vae=vae_path, frozen_convnext=teacher_path,
                      vae_attention_key_migrations=vae_migrations,
                      teacher_checkpoint_report=teacher._avid_checkpoint_report,
                      optimizer='AdamW', target_cache_seconds=time.monotonic() - cache_start,
                      loss='latent L1 / 0.18215 + 0.1 * symmetric token soft contrastive',
                      checkpoint_selection='minimum validation latent MSE (historical AVID protocol)')
    save_json(adapter.run_dirs.logs / 'lowlevel_training_protocol.json', provenance)
    print('Low-level training protocol:', provenance, flush=True)
    history, best = [], float('inf')
    def save_checkpoint(tag, epoch, val_loss):
        payload = dict(epoch=epoch, model_state_dict=model.state_dict(),
                       input_dim=input_dim, provenance=provenance, history=history,
                       val_loss=val_loss)
        if tag == 'last':
            payload.update(optimizer_state_dict=optimizer.state_dict(),
                           lr_scheduler=scheduler.state_dict() if scheduler else None)
        atomic_torch_save(payload, adapter.run_dirs.checkpoints / (tag + '.pth'))
    log_path = adapter.run_dirs.logs / 'lowlevel_epochs.jsonl'
    for epoch in range(epochs):
        start = time.monotonic()
        model.train()
        totals = np.zeros(3)
        seen = []
        # Shuffle all finite batches once per epoch. No resampling can omit a stimulus.
        batches = list(train_loader)
        order = torch.randperm(len(batches)).tolist()
        for batch_index in order[:steps]:
            names, responses = batches[batch_index]
            validate_response_dimension(responses, input_dim)
            voxel = utils.voxel_select(responses.to(device).float())
            target = torch.stack([targets[name] for name in names]).to(device)
            clean_maps = torch.stack([maps[name] for name in names]).to(device)
            with torch.no_grad():
                augmented = teacher((augmentation(images(names)) - mean) / std)[1]
            optimizer.zero_grad(set_to_none=True)
            predicted, student_maps = model(voxel, return_transformer_feats=True)
            loss, latent_l1, contrastive = lowlevel_loss(
                predicted, target, student_maps, clean_maps, augmented,
                float(options['contrastive_weight']), float(options['contrastive_temperature']))
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite low-level loss at epoch {epoch}, batch {batch_index}')
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            totals += [loss.item() * len(names), latent_l1.item() * len(names), contrastive.item() * len(names)]
            seen.extend(names)
        if steps == math.ceil(train_count / batch_size) and (len(seen) != train_count or set(seen) != set(train_names)):
            raise RuntimeError('Full epoch did not cover all unique training stimuli exactly once')
        model.eval()
        validation_sum, validation_n = 0., 0
        with torch.no_grad():
            for names, responses in val_loader:
                validate_response_dimension(responses, input_dim)
                voxel = responses.mean(1) if responses.ndim == 3 else responses
                prediction = model(voxel.to(device).float())
                target = torch.stack([targets[name] for name in names]).to(device)
                val_mse = F.mse_loss(prediction, target)
                if not torch.isfinite(val_mse):
                    raise RuntimeError('Nonfinite validation loss')
                validation_sum += val_mse.item() * len(names)
                validation_n += len(names)
        if validation_n != val_count:
            raise RuntimeError('Validation sample count mismatch')
        val_loss = validation_sum / validation_n
        row = dict(epoch=epoch + 1, train_loss=totals[0] / len(seen),
                   latent_l1=totals[1] / len(seen), contrastive=totals[2] / len(seen),
                   val_mse=val_loss, train_samples=len(seen), val_samples=validation_n,
                   seconds=time.monotonic() - start, lr=optimizer.param_groups[0]['lr'])
        history.append(row)
        import json
        with log_path.open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(f'Low-level epoch {epoch + 1}/{epochs}: {json.dumps(row)}', flush=True)
        if val_loss < best:
            best = val_loss
            save_checkpoint('best', epoch, val_loss)
    save_checkpoint('last', epochs - 1, val_loss)
    result = dict(model_name='mindeye1', component='lowlevel', run_name=adapter.run_name,
                  subject_id=adapter.subject_id, stimulus_type=adapter.stimulus_type,
                  input_dim=input_dim, epochs=epochs, best_val_mse=best,
                  best_checkpoint=str(adapter.run_dirs.checkpoints / 'best.pth'),
                  last_checkpoint=str(adapter.run_dirs.checkpoints / 'last.pth'),
                  cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                  cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    save_json(adapter.run_dirs.logs / 'train_result.json', result)
    return result
