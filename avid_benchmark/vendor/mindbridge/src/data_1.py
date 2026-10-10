import os
import math
import random
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import webdataset as wds
from PIL import Image
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import utils
import kornia
from kornia.augmentation.container import AugmentationSequential


img_augment = AugmentationSequential(
    kornia.augmentation.RandomResizedCrop((224,224), (0.8,1), p=0.3),
    kornia.augmentation.Resize((224, 224)),
    kornia.augmentation.RandomBrightness(brightness=(0.8, 1.2), clip_output=True, p=0.2),
    kornia.augmentation.RandomContrast(contrast=(0.8, 1.2), clip_output=True, p=0.2),
    kornia.augmentation.RandomGamma((0.8, 1.2), (1.0, 1.3), p=0.2),
    kornia.augmentation.RandomSaturation((0.8,1.2), p=0.2),
    kornia.augmentation.RandomHue((-0.1,0.1), p=0.2),
    kornia.augmentation.RandomSharpness((0.8, 1.2), p=0.2),
    kornia.augmentation.RandomGrayscale(p=0.2),
    data_keys=["input"],
)

STIM_NAME_MAX_BYTES = 128


def passthrough_nodesplitter(urls):
    return urls


def distributed_nodesplitter(src):
    return wds.split_by_node(src)


def is_distributed_runtime():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def resolve_avid_shard_path(dataset_root, subject, data_split, split):
    token = f"{int(subject):02d}"
    filename = f"sub-{token}_{data_split}_{split}.tar"
    root = Path(dataset_root)
    candidates = [
        root / filename,
        root / f"sub-{token}" / filename,
        root / f"sub-{int(subject)}" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())

    matches = sorted(root.rglob(filename))
    if matches:
        return str(matches[0].resolve())

    raise FileNotFoundError(f"Could not find {filename} under {root.resolve()}")

class NSDDataset(Dataset):
    def __init__(self, root_dir, extensions=None, pool_num=8192, pool_type="max", length=None):
        self.root_dir = root_dir
        self.extensions = extensions if extensions else []
        self.pool_num = pool_num
        self.pool_type = pool_type
        self.samples = self._load_samples()
        self.samples_keys = sorted(self.samples.keys())
        self.length = length
        if length is not None:
            if length > len(self.samples_keys):
                pass # enlarge the dataset
            elif length > 0:
                self.samples_keys = self.samples_keys[:length]
            elif length < 0:
                self.samples_keys = self.samples_keys[length:]
            elif length == 0:
                raise ValueError("length must be a non-zero value!")
        else:
            self.length = len(self.samples_keys)

    def _load_samples(self):
        files = os.listdir(self.root_dir)
        samples = {}
        for file in files:
            file_path = os.path.join(self.root_dir, file)
            sample_id, ext = file.split(".",maxsplit=1)
            if ext in self.extensions:
                if sample_id in samples.keys():
                    samples[sample_id][ext] = file_path
                else:
                    samples[sample_id]={"subj": file_path}
                    samples[sample_id][ext] = file_path
            # print(samples)
        return samples
    
    def _load_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))
        return image
    
    def _load_npy(self, npy_path):
        array = np.load(npy_path)
        array = torch.from_numpy(array)
        return array
    
    def vox_process(self, x):
        if self.pool_num is not None:
            x = pool_voxels(x, self.pool_num, self.pool_type)
        return x
    
    def subj_process(self, key):
        id = int(key.split("/")[-2].split("subj")[-1])
        return id
    
    def aug_process(self, brain3d):
        return brain3d

    def __len__(self):
        # return len(self.samples_keys)
        return self.length

    def __getitem__(self, idx):
        idx = idx % len(self.samples_keys)
        sample_key = self.samples_keys[idx]
        sample = self.samples[sample_key]
        items = []
        for ext in self.extensions:
            if ext == "jpg":
                items.append(self._load_image(sample[ext]))
            elif ext == "nsdgeneral.npy":
                voxel = self._load_npy(sample[ext])
                items.append(self.vox_process(voxel))
            elif ext == "coco73k.npy":
                items.append(self._load_npy(sample[ext]))
            elif ext == "subj":
                items.append(self.subj_process(sample[ext]))
            elif ext == "wholebrain_3d.npy":
                brain3d = self._load_npy(sample[ext])
                items.append(self.aug_process(brain3d, ))

        return items

def pool_voxels(voxels, pool_num, pool_type):
    # voxels = voxels.float()
    voxels = torch.as_tensor(voxels).float()
    if pool_type == 'avg':
        voxels = nn.AdaptiveAvgPool1d(pool_num)(voxels)
    elif pool_type == 'max':
        voxels = nn.AdaptiveMaxPool1d(pool_num)(voxels)
    elif pool_type == "resize":
        voxels = voxels.unsqueeze(1) # Add a dimension to make it (B, 1, L)
        voxels = F.interpolate(voxels, size=pool_num, mode='linear', align_corners=False)
        voxels = voxels.squeeze(1)

    return voxels

def get_dataloader(
        root_dir,
        batch_size,
        num_workers=1,
        seed=42,
        is_shuffle=True,
        extensions=['nsdgeneral.npy', "jpg", 'coco73k.npy', "subj"],
        pool_type=None,
        pool_num=None,
        length=None,
    ):
    utils.seed_everything(seed)
    dataset = NSDDataset(root_dir=root_dir, extensions=extensions, pool_num=pool_num, pool_type=pool_type, length=length)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=is_shuffle)

    return dataloader

# ===== AVID adaptation for tar-based multi-subject data: START =====
def normalize_image_name(value):
    if isinstance(value, bytes):
        value = value.decode()
    elif isinstance(value, np.ndarray):
        if value.dtype == np.uint8 and value.ndim >= 1:
            flat = value.reshape(-1).tolist()
            value = bytes(int(item) for item in flat if int(item) != 0).decode("utf-8")
        elif value.ndim == 0:
            value = value.item()
        else:
            value = value.reshape(-1)[0]
            if hasattr(value, "item"):
                value = value.item()
    elif torch.is_tensor(value):
        if value.dtype == torch.uint8 and value.ndim >= 1:
            flat = value.reshape(-1).tolist()
            value = bytes(int(item) for item in flat if int(item) != 0).decode("utf-8")
        else:
            value = value.item()
    return os.path.basename(str(value).strip())


def batch_stim_ids(stim):
    if isinstance(stim, (list, tuple)):
        return [normalize_image_name(item) for item in stim]
    if isinstance(stim, np.ndarray):
        if stim.dtype == np.uint8 and stim.ndim >= 2:
            return [normalize_image_name(item) for item in stim]
        return [normalize_image_name(item) for item in stim.tolist()]
    if torch.is_tensor(stim):
        if stim.dtype == torch.uint8 and stim.ndim >= 2:
            return [normalize_image_name(item) for item in stim]
        if stim.ndim >= 1:
            return [normalize_image_name(item) for item in stim]
    return [normalize_image_name(stim)]


def concat_stim_batches(stim_batches):
    merged = []
    for stim in stim_batches:
        merged.extend(batch_stim_ids(stim))
    return merged


def load_caption_lookup(tsv_path):
    table = pd.read_csv(tsv_path, sep="\t", usecols=["image", "translated_caption"])
    table = table.dropna(subset=["image", "translated_caption"])
    return {
        normalize_image_name(image): str(caption).strip()
        for image, caption in zip(table["image"], table["translated_caption"])
    }


def encode_stim_name(stim_name: str) -> torch.Tensor:
    encoded = stim_name.encode("utf-8")
    if len(encoded) >= STIM_NAME_MAX_BYTES:
        raise ValueError(f"Stimulus name {stim_name!r} exceeds {STIM_NAME_MAX_BYTES - 1} bytes.")
    payload = torch.zeros(STIM_NAME_MAX_BYTES, dtype=torch.uint8)
    payload[: len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)
    return payload


def resolve_img_root(args):
    if args.img_root is not None:
        return args.img_root
    if args.data == 'complex':
        return "/path/to/AVID/coco_images"
    if args.data == 'simple':
        return "/path/to/AVID/simple_stimuli_corresponding_image"
    raise ValueError(f"Unsupported data split: {args.data}")


# def resolve_split_lengths(args):
#     if args.num_train is not None and args.num_val is not None:
#         return args.num_train, args.num_val
#     if args.data == 'complex':
#         return 640, 160
#     if args.data == 'simple':
#         return 320, 80
#     raise ValueError(f"Unsupported data split: {args.data}")


def load_sample(sample, img_root, subject, pool_num, pool_type):
    stim_name = normalize_image_name(sample["stim"])
    image = Image.open(os.path.join(img_root, stim_name)).convert("RGB")
    image = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])(image)

    sample["stim"] = encode_stim_name(stim_name)
    sample["images"] = image
    sample["voxels"] = pool_voxels(sample["voxels"], pool_num, pool_type)
    sample["subj"] = torch.tensor(subject, dtype=torch.long)
    return sample


def get_tar_dataloader(url, batch_size, subject, img_root, pool_num, pool_type, is_shuffle, seed, num_workers=1, epoch_batches=None):
    dataset_kwargs = {
        "resampled": is_shuffle,
        "nodesplitter": distributed_nodesplitter if is_distributed_runtime() else passthrough_nodesplitter,
    }
    try:
        dataset = wds.WebDataset(url, empty_check=False, **dataset_kwargs)
    except TypeError:
        try:
            dataset = wds.WebDataset(url, **dataset_kwargs)
        except TypeError:
            dataset = wds.WebDataset(url, resampled=is_shuffle)
    if is_shuffle:
        dataset = dataset.shuffle(500, initial=500, rng=random.Random(seed))

    dataset = (
        dataset
        .decode("torch")
        .rename(voxels="responses.npy", stim="img_name.npy")
        .map(lambda sample: load_sample(sample, img_root, subject, pool_num, pool_type))
        .to_tuple("voxels", "images", "stim", "subj")
    )
    if epoch_batches is not None:
        dataset = dataset.with_epoch(max(epoch_batches * batch_size, batch_size))

    persistent_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        timeout=120 if num_workers > 0 else 0,
        persistent_workers=persistent_workers,
    )


# def get_dls(subject, data_path, batch_size, val_batch_size, num_workers, pool_type, pool_num, length, seed):
#     train_path = "{}/webdataset_avg_split/train/subj0{}".format(data_path, subject)
#     val_path = "{}/webdataset_avg_split/val/subj0{}".format(data_path, subject)
#     extensions = ['nsdgeneral.npy', "jpg", 'coco73k.npy', "subj"]
#
#     train_dl = get_dataloader(
#         train_path,
#         batch_size=batch_size,
#         num_workers=num_workers,
#         seed=seed,
#         extensions=extensions,
#         pool_type=pool_type,
#         pool_num=pool_num,
#         is_shuffle=True,
#         length=length,
#     )
#
#     val_dl = get_dataloader(
#         val_path,
#         batch_size=val_batch_size,
#         num_workers=num_workers,
#         seed=seed,
#         extensions=extensions,
#         pool_type=pool_type,
#         pool_num=pool_num,
#         is_shuffle=False,
#     )
#
#     num_train=len(train_dl.dataset)
#     num_val=len(val_dl.dataset)
#     print(train_path,"\n",val_path)
#     print("number of train data:", num_train)
#     print("batch_size", batch_size)
#     print("number of val data:", num_val)
#     print("val_batch_size", val_batch_size)
#
#     return train_dl, val_dl
# ===== AVID adaptation for tar-based multi-subject data: END =====


def get_dls(subject, args):
    # subject_name = args.subj_name_map[subject]
    if args.train_tar_path and len(args.subj_list or []) == 1 and subject == args.subj_list[0]:
        train_path = args.train_tar_path
    else:
        train_path = resolve_avid_shard_path(args.data_path, subject, args.data, "train")
    if args.val_tar_path and len(args.subj_list or []) == 1 and subject == args.subj_list[0]:
        val_path = args.val_tar_path
    else:
        val_path = resolve_avid_shard_path(args.data_path, subject, args.data, "val")
    # train_path = f"{args.data_path}/S{subject}_main_train.tar"
    # val_path = f"{args.data_path}/S{subject}_main_val.tar"
    img_root = resolve_img_root(args)
    # num_train, num_val = resolve_split_lengths(args)
    if args.data == 'complex':
        num_train, num_val = 640, 160
    elif args.data == 'simple':
        num_train, num_val = 320, 80
    else:
        raise ValueError(f"Unsupported data split: {args.data}")

    global_batch_size = args.batch_size * max(int(os.environ.get("WORLD_SIZE", "1")), 1)
    num_batches = max(math.floor(num_train / global_batch_size), 1)
    # num_worker_batches = max(math.floor(num_batches / max(args.num_workers, 1)), 1)
    num_worker_batches = num_batches

    train_dl = get_tar_dataloader(
        url=train_path,
        batch_size=args.batch_size,
        subject=subject,
        img_root=img_root,
        pool_num=args.pool_num,
        pool_type=args.pool_type,
        is_shuffle=True,
        seed=args.seed,
        num_workers=args.num_workers,
        epoch_batches=num_worker_batches,
    )

    val_dl = get_tar_dataloader(
        url=val_path,
        batch_size=args.val_batch_size,
        subject=subject,
        img_root=img_root,
        pool_num=args.pool_num,
        pool_type=args.pool_type,
        is_shuffle=False,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    # print(train_path, "\n", val_path)
    # print("number of train data:", num_train)
    # print("batch_size", args.batch_size)
    # print("number of train batches:", num_worker_batches)
    # print("number of val data:", num_val)
    # print("val_batch_size", args.val_batch_size)

    return train_dl, val_dl, num_worker_batches
