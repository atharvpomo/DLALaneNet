# DLALaneNet — DLA-Optimized Lane Segmentation

PyTorch lane segmentation + ONNX export for **NVIDIA DRIVE Orin DLA 2.0** (DRIVE OS 6.0.10, TensorRT 8.6.13, INT8).

## Dataset

TuSimple root (default):

```
/home/atharvsh/tusimple/TUSimple/
  train_set/clips/...
  train_set/label_data_*.json
  train_set/seg_label/train_val.json
```

## Setup

```bash
cd /home/atharvsh/DLALaneNet
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py --data-root /home/atharvsh/tusimple/TUSimple --epochs 30
```

## Export ONNX (static 1×3×512×1024)

```bash
python scripts/export_onnx.py \
  --checkpoint checkpoints/dla_lanenet_best.pt \
  --output artifacts/dla_lanenet_int8_ready.onnx
```

## TensorRT / DLA build (on target Orin)

```bash
trtexec --onnx=artifacts/dla_lanenet_int8_ready.onnx \
  --saveEngine=artifacts/dla_lanenet.dla.engine \
  --int8 --fp16 \
  --useDLACore=0 \
  --allowGPUFallback=false \
  --shapes=input:1x3x512x1024
```

Provide a calibration cache (`--calib`) built from representative front-camera frames.

## DLA design notes

| Constraint | Implementation |
|------------|----------------|
| Backbone | Vanilla ResNet-18 (`Conv2d`, `BatchNorm`, `ReLU`, `MaxPool`, residual **Add**) |
| Upsampling | Five `ConvTranspose2d(k=4, s=2, p=1)` stages — no `Resize`/bilinear |
| Skip fusion | `1×1` channel projection + **element-wise Add** (no `Concat`) |
| Single output | `forward()` returns final `[B, 4, H, W]` logits only |
| Static shapes | Fixed `512×1024`, batch `1` in export |

### Class labels

| ID | Name | TuSimple proxy |
|----|------|----------------|
| 0 | background | non-lane |
| 1 | solid | lanes 0–1 rasterized |
| 2 | dotted | lanes 2–3 rasterized |
| 3 | other | reserved |

TuSimple JSON has no true solid/dotted tags; lane index is used as a training proxy until type-annotated data is available.

## Project layout

```
dla_lanenet/     model, dataset, config
scripts/         train.py, export_onnx.py
checkpoints/     created by training
artifacts/       ONNX output
```
