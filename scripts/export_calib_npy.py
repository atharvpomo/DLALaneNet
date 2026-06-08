#!/usr/bin/env python3
"""
Export calibration / validation tensors for DriveWorks tensorRT_optimization.

Two layouts:

1) calib_npy/  — one file per sample for INT8 entropy calibration (flat .npy list)
   Used OFFLINE or with tools that read many inputs; NOT for --npyFileDir validation.

2) validate_npy/  — DriveWorks --npyFileDir layout (blob names exactly):
      input.npy
      logits.npy
   Single reference pair for I/O validation only.

DriveWorks --npyFileDir expects DNN blob names, NOT input_0000.npy (can crash with stoi).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dla_lanenet.config import INPUT_HEIGHT, INPUT_WIDTH, TUSIMPLE_ROOT  # noqa: E402
from dla_lanenet.dataset import discover_train_labels  # noqa: E402
from dla_lanenet.model import build_model  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(img_bgr: np.ndarray) -> np.ndarray:
    resized = cv2.resize(img_bgr, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return np.transpose(rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)


def collect_image_paths(data_root: Path, num_samples: int) -> list[Path]:
    root = data_root / "train_set"
    paths: list[Path] = []
    for label in discover_train_labels(data_root):
        with open(label, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                p = root / rec["raw_file"]
                if p.is_file():
                    paths.append(p)
                if len(paths) >= num_samples:
                    return paths
    return paths


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export calibration / validation npy for Orin build")
    p.add_argument("--data-root", type=Path, default=TUSIMPLE_ROOT)
    p.add_argument("--calib-dir", type=Path, default=ROOT / "artifacts" / "calib_npy")
    p.add_argument("--validate-dir", type=Path, default=ROOT / "artifacts" / "validate_npy")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints" / "dla_lanenet_best.pt")
    p.add_argument("--num-samples", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    paths = collect_image_paths(args.data_root, args.num_samples)
    if not paths:
        raise RuntimeError("No images found")

    # Flat calib list (float32 NCHW) — copy whole folder to tegra for custom calib scripts
    args.calib_dir.mkdir(parents=True, exist_ok=True)
    for old in args.calib_dir.glob("*.npy"):
        old.unlink()
    for i, path in enumerate(paths):
        img = cv2.imread(str(path))
        if img is None:
            continue
        np.save(args.calib_dir / f"sample_{i:04d}.npy", preprocess_bgr(img))
    print(f"Calib list: {len(list(args.calib_dir.glob('*.npy')))} files in {args.calib_dir}")

    # DriveWorks --npyFileDir validation pair (exact blob names)
    args.validate_dir.mkdir(parents=True, exist_ok=True)
    img = cv2.imread(str(paths[0]))
    tensor = preprocess_bgr(img)
    np.save(args.validate_dir / "input.npy", tensor)

    if args.checkpoint.is_file():
        model = build_model()
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            logits = model(torch.from_numpy(tensor)).numpy().astype(np.float32)
        np.save(args.validate_dir / "logits.npy", logits)
        print(f"Validate pair: {args.validate_dir}/input.npy + logits.npy")
    else:
        print(f"Validate input only (no checkpoint): {args.validate_dir}/input.npy")

    print("Copy to tegra:")
    print(f"  scp -r {args.calib_dir} benchdev2@tegra:/home/benchdev2/calib_npy")
    print(f"  scp -r {args.validate_dir} benchdev2@tegra:/home/benchdev2/validate_npy")


if __name__ == "__main__":
    main()
