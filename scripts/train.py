#!/usr/bin/env python3
"""Train DLALaneSegNet on TuSimple lane labels."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dla_lanenet.config import (  # noqa: E402
    INPUT_HEIGHT,
    INPUT_WIDTH,
    NUM_CLASSES,
    TUSIMPLE_ROOT,
)
from dla_lanenet.dataset import (  # noqa: E402
    TuSimpleLaneDataset,
    discover_train_labels,
    estimate_class_weights,
)
from dla_lanenet.metrics import confusion_update, metrics_from_confusion  # noqa: E402
from dla_lanenet.model import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DLALaneSegNet")
    p.add_argument("--data-root", type=Path, default=TUSIMPLE_ROOT)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--accum-steps", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--val-batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--no-class-weights", action="store_true", help="Use unweighted CE")
    p.add_argument("--weight-samples", type=int, default=256, help="Samples for CE weight estimate")
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--log-interval", type=int, default=100)
    p.add_argument("--tensorboard", action="store_true")
    p.add_argument("--skip-smoke", action="store_true")
    p.add_argument("--patience", type=int, default=0, help="Early stop if mean_fg_iou stalls (0=off)")
    p.add_argument("--save-every", type=int, default=5, help="Save epoch_N.pt every N epochs (0=off)")
    p.add_argument("--no-tqdm", action="store_true", help="Disable progress bar")
    return p.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_batch_masks(masks: torch.Tensor, tag: str) -> None:
    mn, mx = int(masks.min()), int(masks.max())
    if mn < 0 or mx >= NUM_CLASSES:
        raise ValueError(f"{tag}: mask values [{mn}, {mx}] outside [0, {NUM_CLASSES - 1}]")


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def build_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    best_score: float,
    val: dict,
    args: argparse.Namespace,
) -> dict:
    return {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "best_score": best_score,
        "val_pixel_acc": val["val_pixel_acc"],
        "val_lane_acc": val["lane_pixel_acc"],
        "val_miou": val["miou"],
        "val_lane_iou": val["lane_iou"],
        "val_mean_fg_iou": val["mean_fg_iou"],
        "per_class_iou": val["per_class_iou"],
        "input_shape": [1, 3, INPUT_HEIGHT, INPUT_WIDTH],
        "num_classes": NUM_CLASSES,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float, bool]:
    """Returns (start_epoch, best_score, scheduler_restored)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_score = float(
        ckpt.get(
            "best_score",
            ckpt.get("val_lane_iou", ckpt.get("val_mean_fg_iou", ckpt.get("val_pixel_acc", 0.0))),
        )
    )
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("  restored optimizer")
    sched_ok = False
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
        sched_ok = True
        print("  restored scheduler")
    if "scaler" in ckpt and scaler.is_enabled():
        scaler.load_state_dict(ckpt["scaler"])
        print("  restored AMP scaler")
    return start_epoch, best_score, sched_ok


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    use_amp: bool,
) -> dict:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    correct = 0
    total = 0
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.int64)

    for images, masks in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        bs = images.size(0)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item() * bs
        n_samples += bs
        pred = logits.argmax(dim=1)
        correct += (pred == masks).sum().item()
        total += masks.numel()
        confusion_update(pred.cpu(), masks.cpu(), conf)

    metrics = metrics_from_confusion(conf)
    metrics["val_loss"] = total_loss / max(n_samples, 1)
    metrics["val_pixel_acc"] = correct / max(total, 1)
    return metrics


def _optimizer_step(
    scaler: torch.cuda.amp.GradScaler,
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    use_amp: bool,
) -> None:
    if use_amp:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.log_dir / "train_metrics.jsonl"

    device = torch.device(args.device)
    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.cuda.empty_cache()

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(args.log_dir / "tensorboard"))
            print(f"TensorBoard: {args.log_dir / 'tensorboard'}")
        except ImportError as e:
            print(f"Warning: TensorBoard disabled ({e}). Try: pip install tensorboard six")

    print(
        f"batch_size={args.batch_size} accum_steps={args.accum_steps} "
        f"effective_batch={args.batch_size * args.accum_steps} amp={use_amp}"
    )
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {torch.cuda.get_device_name(0)} ({props.total_memory / 2**30:.1f} GiB)")

    label_paths = discover_train_labels(args.data_root)
    dataset = TuSimpleLaneDataset(args.data_root, label_paths=label_paths, augment=True)
    print(f"Dataset: {len(dataset)} samples")

    val_len = max(1, int(len(dataset) * args.val_ratio))
    train_set, val_set = random_split(
        dataset,
        [len(dataset) - val_len, val_len],
        generator=torch.Generator().manual_seed(args.seed),
    )

    loader_kw = dict(
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    if args.num_workers > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kw,
    )
    val_loader = DataLoader(val_set, batch_size=args.val_batch_size, shuffle=False, **loader_kw)

    model = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    if args.no_class_weights:
        class_weights = None
        print("Loss: unweighted CrossEntropy")
    else:
        class_weights = estimate_class_weights(train_set, max_samples=args.weight_samples).to(device)
        print(f"Loss: weighted CE weights={[round(w, 3) for w in class_weights.tolist()]}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    start_epoch = 1
    best_score = 0.0
    best_path = args.checkpoint_dir / "dla_lanenet_best.pt"
    last_path = args.checkpoint_dir / "dla_lanenet_last.pt"
    stale_epochs = 0

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    if args.resume:
        if not args.resume.is_file():
            raise FileNotFoundError(f"--resume not found: {args.resume}")
        start_epoch, best_score, sched_ok = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )
        if not sched_ok:
            remaining = max(1, args.epochs - start_epoch + 1)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining)
            print(f"  new cosine schedule over {remaining} remaining epochs")
        print(f"Resume: continue from epoch {start_epoch}, best_lane_iou={best_score:.4f}")
        if start_epoch == 1 and "optimizer" not in torch.load(args.resume, map_location="cpu"):
            print(
                "  Note: legacy checkpoint (weights only). Use dla_lanenet_best.pt from "
                "epoch 9+ or finish one full epoch to build a full last.pt."
            )

    optim_steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
    print(f"~{optim_steps_per_epoch} optimizer steps/epoch | log: {metrics_path}")

    if not args.skip_smoke:
        img0, mask0 = next(iter(train_loader))
        validate_batch_masks(mask0, "smoke")
        model.train()
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss0 = criterion(model(img0.to(device)), mask0.to(device))
        if not torch.isfinite(loss0):
            raise RuntimeError("Smoke test: non-finite loss")
        print(f"Smoke OK (loss={loss0.item():.4f})")

    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = 0.0
        optim_step = 0
        skipped = 0
        optimizer.zero_grad(set_to_none=True)

        iterator = train_loader
        if not args.no_tqdm and tqdm is not None:
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")

        for step, (images, masks) in enumerate(iterator, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            if step == 1:
                validate_batch_masks(masks, f"e{epoch}")

            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(images), masks) / args.accum_steps

            if not torch.isfinite(loss):
                skipped += 1
                optimizer.zero_grad(set_to_none=True)
                continue

            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % args.accum_steps == 0:
                optim_step += 1
                _optimizer_step(scaler, optimizer, model, use_amp)
                optimizer.zero_grad(set_to_none=True)
                if args.log_interval > 0 and optim_step % args.log_interval == 0:
                    avg = running / max(optim_step, 1)
                    msg = f"  step {optim_step}/{optim_steps_per_epoch} loss={avg:.4f} lr={optimizer.param_groups[0]['lr']:.2e}"
                    if tqdm is not None and hasattr(iterator, "set_postfix"):
                        iterator.set_postfix(loss=f"{avg:.4f}", refresh=False)
                    else:
                        print(msg, flush=True)

            running += loss.item() * args.accum_steps

        if step % args.accum_steps != 0:
            _optimizer_step(scaler, optimizer, model, use_amp)
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        train_loss = running / max(len(train_loader) - skipped, 1)
        val = evaluate(model, val_loader, device, criterion, use_amp)
        elapsed = time.time() - t0
        samples_per_sec = (len(train_loader) * args.batch_size) / max(elapsed, 1e-6)

        iou_str = " ".join(f"{k}={v:.3f}" for k, v in val["per_class_iou"].items())
        print(
            f"epoch {epoch:03d}/{args.epochs} ({elapsed / 60:.1f}m, {samples_per_sec:.1f} img/s) "
            f"lr={optimizer.param_groups[0]['lr']:.2e} "
            f"train_loss={train_loss:.4f} val_loss={val['val_loss']:.4f} "
            f"miou={val['miou']:.4f} lane_iou={val['lane_iou']:.4f} "
            f"lane_acc={val['lane_pixel_acc']:.4f} pixel_acc={val['val_pixel_acc']:.4f} | {iou_str}"
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_sec": round(elapsed, 1),
            "samples_per_sec": round(samples_per_sec, 2),
            **{k: v for k, v in val.items() if k != "per_class_iou"},
            "per_class_iou": val["per_class_iou"],
        }
        append_jsonl(metrics_path, record)

        if writer:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val["val_loss"], epoch)
            writer.add_scalar("iou/miou", val["miou"], epoch)
            writer.add_scalar("iou/lane", val["lane_iou"], epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        ckpt = build_checkpoint(epoch, model, optimizer, scheduler, scaler, best_score, val, args)
        torch.save(ckpt, last_path)

        if args.save_every > 0 and epoch % args.save_every == 0:
            ep_path = args.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            torch.save(ckpt, ep_path)

        score = val["lane_iou"]
        if score > best_score:
            best_score = score
            stale_epochs = 0
            ckpt["best_score"] = best_score
            torch.save(ckpt, best_path)
            print(f"  -> best checkpoint (lane_iou={score:.4f})")
        else:
            stale_epochs += 1
            if args.patience > 0 and stale_epochs >= args.patience:
                print(f"Early stop: no lane_iou improvement for {args.patience} epochs")
                break

    if writer:
        writer.close()
    print(f"Done. best={best_path} (lane_iou={best_score:.4f}) last={last_path}")


if __name__ == "__main__":
    main()
