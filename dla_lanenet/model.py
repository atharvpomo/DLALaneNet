"""
DLA 2.0 / TensorRT 8.6.13–optimized lane segmentation network.

Architecture: vanilla ResNet-18 backbone + shallow FCN decoder.
- Upsampling: ConvTranspose2d (stride=2), no bilinear/bicubic Resize.
- Skip fusion: element-wise Add after 1x1 channel projection (no Concat).
- Single output: final class logits [B, C, H, W] only.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import DECODER_CHANNELS, INPUT_CHANNELS, NUM_CLASSES


def _conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)


def _deconv2x(in_ch: int, out_ch: int) -> nn.ConvTranspose2d:
    """
    2x upsample for Orin DLA: ConvTranspose must use padding=0 only.

    kernel=4, padding=1 (standard FCN) fails DLA with:
      "DLA only supports padding in the range of [0-0]"
    kernel=2, stride=2, padding=0 yields the same spatial sizes (16->32, etc.).
    """
    return nn.ConvTranspose2d(
        in_ch,
        out_ch,
        kernel_size=2,
        stride=2,
        padding=0,
        output_padding=0,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = _conv3x3(in_ch, out_ch, stride)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(out_ch, out_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class ResNet18Backbone(nn.Module):
    """Standard ResNet-18 stem and stages (DLA-safe ops only)."""

    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            INPUT_CHANNELS, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, blocks=2, stride=1)
        self.layer2 = self._make_layer(128, blocks=2, stride=2)
        self.layer3 = self._make_layer(256, blocks=2, stride=2)
        self.layer4 = self._make_layer(512, blocks=2, stride=2)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        c0 = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(c0)
        c1 = self.layer1(x)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return c0, c1, c2, c3, c4


class DLALaneSegNet(nn.Module):
    """
    ResNet-18 + shallow FCN decoder for multi-class lane masks.

    Spatial flow at 512x1024 input:
      c0: 256x512 (64 ch)  stem conv
      c1: 128x256 (64 ch)  layer1
      c2:  64x128 (128 ch) layer2
      c3:  32x64  (256 ch) layer3
      c4:  16x32  (512 ch) layer4
    Five stride-2 transposed convolutions restore full resolution.
    Decoder width default 128 (DLA CBUF limit at 512x1024 head; see config.py).
    """

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.backbone = ResNet18Backbone()
        d = DECODER_CHANNELS

        self.reduce_c4 = nn.Sequential(
            nn.Conv2d(512, d, kernel_size=1, bias=False),
            nn.BatchNorm2d(d),
            nn.ReLU(inplace=True),
        )
        self.proj_c3 = nn.Conv2d(256, d, kernel_size=1, bias=False)
        self.proj_c2 = nn.Conv2d(128, d, kernel_size=1, bias=False)
        self.proj_c1 = nn.Conv2d(64, d, kernel_size=1, bias=False)
        self.proj_c0 = nn.Conv2d(64, d, kernel_size=1, bias=False)

        self.up1 = _deconv2x(d, d)
        self.up2 = _deconv2x(d, d)
        self.up3 = _deconv2x(d, d)
        self.up4 = _deconv2x(d, d)
        self.up5 = _deconv2x(d, d)

        self.fuse_bn = nn.ModuleList([nn.BatchNorm2d(d) for _ in range(5)])
        self.fuse_relu = nn.ReLU(inplace=True)

        self.head = nn.Conv2d(d, num_classes, kernel_size=1, bias=True)

    def _fuse(self, x: torch.Tensor, skip: torch.Tensor, stage: int) -> torch.Tensor:
        x = x + skip
        x = self.fuse_bn[stage](x)
        return self.fuse_relu(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c0, c1, c2, c3, c4 = self.backbone(x)

        x = self.reduce_c4(c4)

        x = self.up1(x)
        x = self._fuse(x, self.proj_c3(c3), 0)

        x = self.up2(x)
        x = self._fuse(x, self.proj_c2(c2), 1)

        x = self.up3(x)
        x = self._fuse(x, self.proj_c1(c1), 2)

        x = self.up4(x)
        x = self._fuse(x, self.proj_c0(c0), 3)

        x = self.up5(x)
        x = self.fuse_bn[4](x)
        x = self.fuse_relu(x)

        return self.head(x)  # [B, num_classes, H, W]; default num_classes=2 (bg, lane)


def build_model(num_classes: int = NUM_CLASSES) -> DLALaneSegNet:
    return DLALaneSegNet(num_classes=num_classes)
