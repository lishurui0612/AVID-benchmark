"""Content keys for in-memory target caches scoped to a fixed encoder instance."""
import hashlib
from pathlib import Path


def image_content_key(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()
