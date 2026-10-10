"""Resolve existing pretrained assets without triggering downloads."""
import os
from pathlib import Path


def configure_assets(assets):
    for name,value in assets.items():
        os.environ['AVID_ASSET_'+name.upper()]=str(Path(value).expanduser().resolve())


def local_asset(name):
    value=os.environ.get('AVID_ASSET_'+name.upper())
    if not value or not Path(value).exists():
        raise FileNotFoundError(f'Set assets.{name} in config to the existing pretrained model.')
    return value


def atomic_torch_save(payload,destination):
    import torch
    import uuid
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name('.'+destination.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        torch.save(payload,temporary)
        os.replace(temporary,destination)
    finally:
        if temporary.exists():
            temporary.unlink()
