#!/usr/bin/env python3
"""
Export DLALaneSegNet to ONNX for TensorRT 8.6.13 + DLA 2.0 (DRIVE OS 6.0.10).

Static shape only: [1, 3, 512, 1024] -> [1, num_classes, 512, 1024]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dla_lanenet.config import (  # noqa: E402
    BATCH_SIZE,
    INPUT_CHANNELS,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NUM_CLASSES,
)
from dla_lanenet.model import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export DLA-optimized lane net to ONNX")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints" / "dla_lanenet_best.pt",
        help="Optional trained weights (.pt state_dict or full checkpoint)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "dla_lanenet_int8_ready.onnx",
    )
    p.add_argument("--opset", type=int, default=13, help="ONNX opset (13 recommended for TRT 8.6)")
    p.add_argument(
        "--fp16-io",
        action="store_true",
        help="Cast model + I/O to FP16 so the ONNX input/output are FP16. "
        "Aims to remove the DLA input Reformat CopyNode (pure-DLA build).",
    )
    return p.parse_args()


def load_weights(model: torch.nn.Module, checkpoint: Path) -> None:
    if not checkpoint.is_file():
        print(f"No checkpoint at {checkpoint}; exporting random-init weights for graph validation.")
        return
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    print(f"Loaded weights from {checkpoint}")


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(num_classes=NUM_CLASSES)
    model.eval()
    load_weights(model, args.checkpoint)

    dummy = torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH)

    if args.fp16_io:
        model = model.half()
        dummy = dummy.half()
        print("FP16 I/O mode: model + input cast to float16 (input/output tensors will be FP16).")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            str(args.output),
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=None,
            training=torch.onnx.TrainingMode.EVAL,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX,
        )

    dtype = "float16" if args.fp16_io else "float32"
    print(f"Exported ONNX: {args.output}")
    print(f"  input : [{BATCH_SIZE}, {INPUT_CHANNELS}, {INPUT_HEIGHT}, {INPUT_WIDTH}] ({dtype})")
    print(f"  output: [{BATCH_SIZE}, {NUM_CLASSES}, {INPUT_HEIGHT}, {INPUT_WIDTH}] ({dtype})")
    print()
    print("Pure-DLA build on Orin (no GPU fallback):")
    print("  # Path A - Safe DLA (all layers on DLA, app feeds FP16/NCHWx):")
    print("    tensorRT_optimization --modelType=onnx --onnxFile=<onnx> --out=<engine> --useSafeDLA=1")
    print("  # Path B - normal DLA with FP16-I/O ONNX (this file, if --fp16-io was used):")
    print("    tensorRT_optimization --modelType=onnx --onnxFile=<onnx> --out=<engine> --useDLA=1")


if __name__ == "__main__":
    main()
