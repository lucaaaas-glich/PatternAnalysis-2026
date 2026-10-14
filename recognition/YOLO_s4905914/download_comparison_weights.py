"""Obtain pinned official YOLOv8s weights; no training or source-data changes."""

import hashlib
import os
import tempfile
import urllib.request
from pathlib import Path

from train import COMPARISON_SHA256

URL = 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt'


def download(destination=Path('weights/yolov8s.pt')):
    destination = Path(destination)
    if destination.exists():
        with destination.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != COMPARISON_SHA256:
            raise ValueError('Existing weights differ; preserve them, do not overwrite')
        print('Existing official YOLOv8s weights verified')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.part', delete=False) as handle:
            temporary = Path(handle.name)
            with urllib.request.urlopen(URL, timeout=60) as response:
                total = 0
                while chunk := response.read(1024*1024):
                    total += len(chunk)
                    if total > 30*1024*1024:
                        raise ValueError('Weight download exceeds expected size')
                    handle.write(chunk)
        with temporary.open('rb') as handle:
            digest = hashlib.file_digest(handle, 'sha256').hexdigest()
        if digest != COMPARISON_SHA256:
            raise ValueError('Official checkpoint snapshot checksum mismatch')
        os.link(temporary, destination)  # Atomic publication without replacing another file.
        print('Official YOLOv8s checkpoint verified:', destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()  # Only this call's generated temporary file.


if __name__ == '__main__':
    download()
