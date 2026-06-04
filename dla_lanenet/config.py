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

# Decoder width (kept constant for clean INT8 calibration)
DECODER_CHANNELS = 256

# Lane rasterization
LANE_LINE_WIDTH = 5
