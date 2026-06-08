"""Static configuration for DRIVE Orin DLA deployment."""

from pathlib import Path

# Network I/O (strict static shapes for TensorRT / DLA)
BATCH_SIZE = 1
INPUT_CHANNELS = 3
INPUT_HEIGHT = 512
INPUT_WIDTH = 1024

# Segmentation classes: TuSimple provides lane polylines only (no marking types)
NUM_CLASSES = 2
CLASS_NAMES = ("background", "lane")

# TuSimple default layout (override via CLI)
TUSIMPLE_ROOT = Path("/home/atharvsh/tusimple/TUSimple")
TRAIN_IMAGE_ROOT = TUSIMPLE_ROOT / "train_set"
TRAIN_LABEL_GLOBS = (
    "label_data_*.json",
    str(TUSIMPLE_ROOT / "train_set" / "seg_label" / "train_val.json"),
)

# Decoder width — 256 overflows DLA CBUF on final 1x1 conv at 1024 width (FP16).
# 128 fits Orin DLA (Conv_71 data banks 8 + weight 1 <= 16).
DECODER_CHANNELS = 128

# Lane rasterization
LANE_LINE_WIDTH = 5
