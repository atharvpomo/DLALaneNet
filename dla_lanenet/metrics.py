"""Segmentation metrics for training / validation logs."""

from __future__ import annotations

import torch

from .config import CLASS_NAMES, NUM_CLASSES


@torch.no_grad()
def confusion_update(
    pred: torch.Tensor,
    target: torch.Tensor,
    conf: torch.Tensor,
) -> None:
    """Update confusion matrix from integer pred/target maps [N,H,W]."""
    pred = pred.view(-1)
    target = target.view(-1)
    k = NUM_CLASSES
    idx = target * k + pred
    bincount = torch.bincount(idx, minlength=k * k)
    conf += bincount.reshape(k, k).to(conf.dtype)


def metrics_from_confusion(conf: torch.Tensor) -> dict[str, float]:
    """Per-class IoU, mean foreground IoU, lane pixel accuracy."""
    conf = conf.double()
    diag = torch.diag(conf)
    sum_pred = conf.sum(dim=0)
    sum_tgt = conf.sum(dim=1)
    union = sum_pred + sum_tgt - diag
    iou = torch.where(union > 0, diag / union, torch.zeros_like(diag))

    per_class = {CLASS_NAMES[c]: float(iou[c].item()) for c in range(NUM_CLASSES)}
    miou = iou.mean().item()
    fg = iou[1:].mean().item() if NUM_CLASSES > 1 else 0.0

    lane_correct = diag[1:].sum()
    lane_total = sum_tgt[1:].sum()
    lane_acc = float((lane_correct / lane_total.clamp(min=1)).item())

    lane_iou = float(iou[1].item()) if NUM_CLASSES > 1 else 0.0

    return {
        "miou": miou,
        "lane_iou": lane_iou,
        "mean_fg_iou": fg,
        "lane_pixel_acc": lane_acc,
        "per_class_iou": per_class,
    }
