"""
TuSimple lane dataset -> binary lane segmentation masks.

TuSimple JSON: up to four lane center polylines per frame (x at fixed h_samples).
All valid lane pixels are class 1 (lane); everything else is background (0).
"""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import (
    CLASS_NAMES,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    LANE_LINE_WIDTH,
    NUM_CLASSES,
    TRAIN_IMAGE_ROOT,
)


def _load_label_records(label_paths: Iterable[Path]) -> list[dict]:
    records: list[dict] = []
    for path in label_paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def discover_train_labels(data_root: Path) -> list[Path]:
    paths: list[Path] = []
    train_dir = data_root / "train_set"
    for pattern in ("label_data_*.json",):
        paths.extend(Path(p) for p in glob(str(train_dir / pattern)))
    train_val = data_root / "train_set" / "seg_label" / "train_val.json"
    if train_val.is_file():
        paths.append(train_val)
    return sorted(set(paths))


def _rasterize_lanes(
    lanes: list[list[int]],
    h_samples: list[int],
    height: int,
    width: int,
    line_width: int = LANE_LINE_WIDTH,
) -> np.ndarray:
    """Return uint8 mask: 0=background, 1=lane (all TuSimple polylines)."""
    mask = np.zeros((height, width), dtype=np.uint8)

    for xs in lanes:
        points = []
        for x, y in zip(xs, h_samples):
            if x >= 0:
                points.append((int(x), int(y)))
        if len(points) < 2:
            continue
        pts = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            mask,
            [pts],
            isClosed=False,
            color=1,
            thickness=line_width,
            lineType=cv2.LINE_8,
        )
    return np.clip(mask, 0, NUM_CLASSES - 1).astype(np.uint8)


def _resize_mask_nearest(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


class TuSimpleLaneDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        label_paths: list[Path] | None = None,
        image_root: Path | None = None,
        height: int = INPUT_HEIGHT,
        width: int = INPUT_WIDTH,
        augment: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        # raw_file entries are like "clips/0601/<id>/20.jpg" relative to train_set/
        self.image_root = Path(image_root or (self.data_root / "train_set"))
        self.height = height
        self.width = width
        self.augment = augment

        if label_paths is None:
            label_paths = discover_train_labels(self.data_root)
        self.records = _load_label_records(label_paths)
        self.records = [r for r in self.records if self._image_exists(r)]

    def _image_exists(self, record: dict) -> bool:
        return (self.image_root / record["raw_file"]).is_file()

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        record = self.records[index]
        img_path = self.image_root / record["raw_file"]
        image = self._load_image(img_path)
        orig_h, orig_w = image.shape[:2]

        mask = _rasterize_lanes(
            record["lanes"],
            record["h_samples"],
            orig_h,
            orig_w,
            line_width=LANE_LINE_WIDTH,
        )

        image = cv2.resize(image, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        mask = _resize_mask_nearest(mask, self.height, self.width)
        mask = np.clip(mask, 0, NUM_CLASSES - 1)

        if self.augment and np.random.rand() > 0.5:
            image = np.ascontiguousarray(image[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = (image - mean) / std

        mask = torch.from_numpy(mask.astype(np.int64))
        return image, mask


def estimate_class_weights(
    dataset,
    max_samples: int = 256,
    min_pixels: int = 100,
    max_weight: float = 15.0,
) -> torch.Tensor:
    """
    Inverse-frequency weights from a random subset.

    Classes with fewer than ``min_pixels`` in the sample get weight 1.0
    (never use max(count, 1) on zero-count classes).
    """
    import random

    n = len(dataset)
    k = min(max_samples, n)
    indices = random.sample(range(n), k) if k < n else list(range(n))
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for i in indices:
        _, mask = dataset[i]
        for c in range(NUM_CLASSES):
            counts[c] += (mask.numpy() == c).sum()

    total = counts.sum()
    weights = np.ones(NUM_CLASSES, dtype=np.float64)
    for c in range(NUM_CLASSES):
        if counts[c] >= min_pixels:
            weights[c] = total / (NUM_CLASSES * counts[c])
    weights = np.clip(weights, 0.05, max_weight)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
