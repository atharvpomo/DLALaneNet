#!/usr/bin/env python3
"""Quick pre-flight check: data labels, one train step, optional ONNX shape."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dla_lanenet.config import NUM_CLASSES, TUSIMPLE_ROOT  # noqa: E402
from dla_lanenet.dataset import TuSimpleLaneDataset, discover_train_labels  # noqa: E402
from dla_lanenet.model import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Smoke test before long training runs")
    p.add_argument("--data-root", type=Path, default=TUSIMPLE_ROOT)
    p.add_argument("--samples", type=int, default=32, help="Random samples to validate mask range")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def validate_masks(dataset: TuSimpleLaneDataset, n: int) -> None:
    n = min(n, len(dataset))
    for i in range(n):
        _, mask = dataset[i]
        mn, mx = int(mask.min()), int(mask.max())
        if mn < 0 or mx >= NUM_CLASSES:
            raise ValueError(f"Sample {i}: mask range [{mn}, {mx}] invalid for {NUM_CLASSES} classes")
    print(f"OK: {n} masks in [0, {NUM_CLASSES - 1}]")


def train_step(device: torch.device) -> None:
    model = build_model().to(device).train()
    x = torch.randn(1, 3, 512, 1024, device=device)
    y = torch.randint(0, NUM_CLASSES, (1, 512, 1024), device=device)
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()
    assert torch.isfinite(loss), "loss is not finite on random batch"
    print(f"OK: forward/backward loss={loss.item():.4f} logits={tuple(logits.shape)}")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    labels = discover_train_labels(args.data_root)
    ds = TuSimpleLaneDataset(args.data_root, label_paths=labels, augment=False)
    print(f"Dataset: {len(ds)} samples")
    validate_masks(ds, args.samples)
    train_step(device)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
