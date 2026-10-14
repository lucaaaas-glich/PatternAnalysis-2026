"""Model adapter: uses Ultralytics' YOLO implementation, not a custom YOLO."""

from pathlib import Path


def load_detector(weights):
    weights = Path(weights)
    if not weights.is_file():
        raise ValueError('Download official weights before submitting the GPU job')
    from ultralytics import YOLO
    return YOLO(str(weights), task='detect')

