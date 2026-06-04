#!/usr/bin/env python3
"""Run ONNX lane model on a TuSimple (or any) image and save an overlay."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dla_lanenet.config import INPUT_HEIGHT, INPUT_WIDTH  # noqa: E402

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_bgr(img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (tensor NCHW float32, resized BGR for visualization)."""
    resized = cv2.resize(img_bgr, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    tensor = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    return tensor, resized


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX inference + lane overlay")
    p.add_argument(
        "--onnx",
        type=Path,
        default=ROOT / "artifacts" / "dla_lanenet_best.onnx",
    )
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image path (default: first TuSimple train frame)",
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=Path("/home/atharvsh/tusimple/TUSimple"),
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "onnx_preview.png",
    )
    p.add_argument("--provider", choices=("cpu", "cuda"), default="cpu")
    return p.parse_args()


def default_image(data_root: Path) -> Path:
    import json

    label = data_root / "train_set" / "seg_label" / "train_val.json"
    with open(label, "r", encoding="utf-8") as f:
        rec = json.loads(f.readline())
    return data_root / "train_set" / rec["raw_file"]


def main() -> None:
    args = parse_args()
    import onnxruntime as ort

    image_path = args.image or default_image(args.data_root)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not args.onnx.is_file():
        raise FileNotFoundError(args.onnx)

    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if args.provider == "cuda"
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(str(args.onnx), providers=providers)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read {image_path}")

    tensor, vis = preprocess_bgr(img)
    logits = sess.run(None, {"input": tensor})[0]
    mask = logits.argmax(axis=1)[0].astype(np.uint8)

    lane_px = int((mask == 1).sum())
    total = mask.size
    print(f"image: {image_path}")
    print(f"ONNX:  {args.onnx}")
    print(f"mask:  {mask.shape}  lane_pixels={lane_px} ({100 * lane_px / total:.2f}%)")

    overlay = vis.copy()
    overlay[mask == 1] = (0, 255, 0)  # green lane pixels
    blend = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), blend)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
