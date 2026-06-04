#!/usr/bin/env python3
"""Train DLALaneSegNet on TuSimple lane labels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dla_lanenet.config import INPUT_HEIGHT, INPUT_WIDTH, NUM_CLASSES, TUSIMPLE_ROOT  # noqa: E402
from dla_lanenet.dataset import TuSimpleLaneDataset, discover_train_labels  # noqa: E402
from dla_lanenet.model import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DLALaneSegNet (512x1024 is VRAM-heavy; use batch 1-2 on 8GB GPUs)",
    )
    p.add_argument("--data-root", type=Path, default=TUSIMPLE_ROOT)
    p.add_argument("--epochs", type=int, default=30)
    # 512x1024 full-res decoder: batch 4 needs ~10+ GB; batch 1-2 fits 8GB with AMP
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument(
        "--accum-steps",
        type=int,
        default=2,
        help="Gradient accumulation steps (effective batch = batch_size * accum_steps)",
    )
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mixed precision (recommended on CUDA, roughly halves activation memory)",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint (.pt with model + epoch)",
    )
    p.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print training loss every N optimizer steps (0 = epoch end only)",
    )
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == masks).sum().item()
        total += masks.numel()
    return total_loss / max(len(loader.dataset), 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.empty_cache()

    effective_batch = args.batch_size * args.accum_steps
    print(
        f"batch_size={args.batch_size} accum_steps={args.accum_steps} "
        f"effective_batch={effective_batch} amp={use_amp}"
    )
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {name} ({total_gb:.1f} GiB)")
        if args.batch_size >= 4 and not use_amp:
            print(
                "Warning: batch_size>=4 at 512x1024 usually exceeds 8GB VRAM. "
                "Use --batch-size 2 --accum-steps 2 --amp or --batch-size 1 --accum-steps 4."
            )

    print("Loading TuSimple labels and indexing images...")
    label_paths = discover_train_labels(args.data_root)
    dataset = TuSimpleLaneDataset(args.data_root, label_paths=label_paths, augment=True)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found under {args.data_root}")
    print(f"Dataset ready: {len(dataset)} samples from {len(label_paths)} label files")

    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(
        dataset,
        [train_len, val_len],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_miou_proxy = 0.0
    best_path = args.checkpoint_dir / "dla_lanenet_best.pt"
    start_epoch = 1
    if args.resume and args.resume.is_file():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        best_miou_proxy = float(ckpt.get("val_pixel_acc", 0.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from {args.resume} (epoch {start_epoch - 1}, val_pixel_acc={best_miou_proxy:.4f})")
    steps_per_epoch = len(train_loader)
    optim_steps_per_epoch = steps_per_epoch // args.accum_steps
    print(
        f"Train batches/epoch: {steps_per_epoch} "
        f"(~{optim_steps_per_epoch} optimizer steps). First epoch log may take several minutes."
    )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = 0.0
        optim_step = 0
        optimizer.zero_grad(set_to_none=True)
        print(f"Epoch {epoch}/{args.epochs} started...", flush=True)
        for step, (images, masks) in enumerate(train_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks) / args.accum_steps
            if not torch.isfinite(loss):
                print(f"  skip non-finite loss at epoch {epoch} step {step}", flush=True)
                optimizer.zero_grad(set_to_none=True)
                continue
            scaler.scale(loss).backward()

            if step % args.accum_steps == 0:
                optim_step += 1
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if args.log_interval > 0 and optim_step % args.log_interval == 0:
                    print(
                        f"  epoch {epoch} step {optim_step}/{optim_steps_per_epoch} "
                        f"loss={running / optim_step:.4f}",
                        flush=True,
                    )

            running += loss.item() * args.accum_steps

        if step % args.accum_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        val_loss, pixel_acc = evaluate(model, val_loader, device, use_amp)
        print(
            f"epoch {epoch:03d}/{args.epochs} "
            f"train_loss={running / len(train_loader):.4f} "
            f"val_loss={val_loss:.4f} val_pixel_acc={pixel_acc:.4f}"
        )

        if pixel_acc > best_miou_proxy:
            best_miou_proxy = pixel_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "val_pixel_acc": pixel_acc,
                    "input_shape": [1, 3, INPUT_HEIGHT, INPUT_WIDTH],
                    "num_classes": NUM_CLASSES,
                },
                best_path,
            )

    print(f"Best checkpoint: {best_path} (val_pixel_acc={best_miou_proxy:.4f})")


if __name__ == "__main__":
    main()
