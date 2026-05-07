"""
U-Net training driver.

Run:
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python -m train.train_unet
    HSA_OVERRIDE_GFX_VERSION=10.3.0 python -m train.train_unet --epochs 30 --batch-size 8
    python -m train.train_unet --resume results/unet/checkpoints/best.pth

Resume and continue for 100 more epochs with early stopping:
    python -m train.train_unet --resume results/unet/checkpoints/best.pth \
        --epochs 100 --patience 20 --batch-size 8
"""

from __future__ import annotations

# The HSA override MUST be applied before torch is imported on ROCm/RX 6700S.
from utils.device import apply_hsa_override  # noqa: E402

apply_hsa_override()

import argparse
import signal
from contextlib import nullcontext
from pathlib import Path
import csv
import time

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from data.potsdam_dataset import get_dataloaders
from models.unet import UNet
from utils.cfg import load_config, resolve_path
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.device import get_device
from utils.seed import set_seed

# ---------------------------------------------------------------------------
# Graceful interruption: Ctrl+C saves ``last.pth`` before exiting.
# ---------------------------------------------------------------------------
_INTERRUPTED = False


def _handle_sigint(signum, frame):
    global _INTERRUPTED
    if _INTERRUPTED:
        # Second Ctrl+C → hard exit
        raise SystemExit(1)
    _INTERRUPTED = True
    print("\n[train] Ctrl+C received – finishing current epoch and saving …")


def _wrap_tqdm(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if tqdm is not None else iterable


def _amp_context(device: torch.device, enabled: bool):
    """Return a torch.amp autocast context, or nullcontext if disabled."""
    if not enabled or device.type not in ("cuda", "mps"):
        return nullcontext()
    try:
        return torch.amp.autocast(device_type=device.type, enabled=True)
    except (AttributeError, RuntimeError):
        if device.type == "cuda":
            return torch.cuda.amp.autocast()
        return nullcontext()


def _build_scaler(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return None
    # Prefer the non-deprecated API (PyTorch 2.4+).
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler=None,
    use_amp: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    ctx = _amp_context(device, use_amp)

    for images, masks in _wrap_tqdm(loader, desc="  train", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with ctx:
            logits = model(images)
            loss = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
) -> float:
    model.eval()
    total_loss = 0.0
    ctx = _amp_context(device, use_amp)

    for images, masks in _wrap_tqdm(loader, desc="  val  ", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with ctx:
            logits = model(images)
            loss = criterion(logits, masks)
        total_loss += loss.item() * images.size(0)

    return total_loss / len(loader.dataset)


def main(args: argparse.Namespace) -> None:
    global _INTERRUPTED

    cfg = load_config(args.config)
    set_seed(cfg.device.seed)
    device = get_device()

    epochs = args.epochs or cfg.unet.epochs
    batch_size = args.batch_size or cfg.unet.batch_size
    lr = args.lr or cfg.unet.lr
    patience = args.patience  # None = disabled
    num_workers = args.num_workers if args.num_workers is not None else cfg.device.num_workers

    if args.image_size is not None:
        h, w = args.image_size
        cfg.unet.image_size = [int(h), int(w)]

    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        cfg=cfg,
    )

    model = UNet(in_channels=3, num_classes=cfg.unet.num_classes).to(device)
    if args.compile:
        try:
            model = torch.compile(model, mode="reduce-overhead")
        except (RuntimeError, AttributeError) as e:
            print(f"[train] torch.compile disabled: {e}")
            args.compile = False

    criterion = nn.CrossEntropyLoss()

    use_amp = (args.amp or (args.amp is None and cfg.device.amp)) and device.type in ("cuda", "mps")
    scaler = _build_scaler(device, use_amp)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=cfg.unet.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    start_epoch = 1
    best_val = float("inf")
    epochs_no_improve = 0

    # ------------------------------------------------------------------
    # Resume: restore model, optimizer, scheduler, scaler, and counters
    # ------------------------------------------------------------------
    if args.resume:
        ckpt = load_checkpoint(Path(args.resume), model, optimizer, device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val = float(ckpt.get("best_val", ckpt.get("val_loss", best_val)))
        epochs_no_improve = int(ckpt.get("epochs_no_improve", 0))

        # Restore LR scheduler state (if saved).
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])

        # Restore GradScaler state (if saved and we have a scaler).
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

        # When resuming, --epochs means the *target* epoch, not additional.
        # If --epochs was not explicitly given, default to resuming for the
        # same total as the config (so the user must pass --epochs to extend).
        print(f"[train] resumed at epoch {start_epoch}  "
              f"(best_val={best_val:.4f}, no_improve={epochs_no_improve})")

    ckpt_dir = resolve_path(cfg.paths.unet_ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = ckpt_dir.parent / "results.csv"
    csv_exists = csv_path.exists()
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["epoch", "time", "train_loss", "val_loss", "lr/pg0"])

    tags = []
    if use_amp:
        tags.append("AMP")
    if args.compile:
        tags.append("compile")
    if patience is not None:
        tags.append(f"patience={patience}")
    tag_str = f"  [{', '.join(tags)}]" if tags else ""

    print(f"\n{'=' * 60}")
    print(f"  U-Net | epochs {start_epoch}→{epochs} | device: {device}{tag_str}")
    print(f"{'=' * 60}\n")

    # Install Ctrl+C handler *after* setup so data loading can be interrupted
    # normally during init.
    prev_handler = signal.signal(signal.SIGINT, _handle_sigint)

    def _save_last(epoch, val_loss):
        """Save last.pth with full training state for seamless resume."""
        extra = {
            "best_val": best_val,
            "epochs_no_improve": epochs_no_improve,
            "scheduler_state_dict": scheduler.state_dict(),
        }
        if scaler is not None:
            extra["scaler_state_dict"] = scaler.state_dict()
        save_checkpoint(model, optimizer, epoch, val_loss, ckpt_dir / "last.pth",
                        extra=extra)

    val_loss = float("inf")
    stopped_reason = None
    start_time = time.time()

    for epoch in range(start_epoch, epochs + 1):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, use_amp=use_amp,
        )
        val_loss = validate(model, val_loader, criterion, device, use_amp=use_amp)
        scheduler.step(val_loss)

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_no_improve = 0
            extra = {
                "best_val": best_val,
                "epochs_no_improve": epochs_no_improve,
                "scheduler_state_dict": scheduler.state_dict(),
            }
            if scaler is not None:
                extra["scaler_state_dict"] = scaler.state_dict()
            save_checkpoint(model, optimizer, epoch, val_loss, ckpt_dir / "best.pth",
                            extra=extra)
        else:
            epochs_no_improve += 1

        lr_now = optimizer.param_groups[0]["lr"]
        flag = "✓ best" if improved else f"(no improve {epochs_no_improve})"
        print(
            f"Epoch [{epoch:03d}/{epochs}]  "
            f"train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}  "
            f"lr: {lr_now:.2e}  {flag}"
        )
        
        epoch_time = time.time() - epoch_start_time
        csv_writer.writerow([epoch, epoch_time, train_loss, val_loss, lr_now])
        csv_file.flush()

        # ----- early stopping check -----
        if patience is not None and epochs_no_improve >= patience:
            stopped_reason = "early_stop"
            print(f"\n[train] Early stopping triggered after {patience} epochs "
                  f"without improvement.")
            break

        # ----- Ctrl+C check -----
        if _INTERRUPTED:
            stopped_reason = "interrupted"
            print(f"\n[train] Interrupted at epoch {epoch}.")
            break

    # Always save last.pth so training can be resumed from wherever we stopped.
    _save_last(epoch, val_loss)

    # Restore original signal handler.
    signal.signal(signal.SIGINT, prev_handler)
    csv_file.close()

    if stopped_reason == "interrupted":
        print(f"[train] State saved. Resume with:")
        print(f"  python -m train.train_unet --resume {ckpt_dir / 'last.pth'} "
              f"--epochs {epochs}")
    elif stopped_reason == "early_stop":
        print(f"[train] Best val loss: {best_val:.4f}")
    else:
        print(f"\n[train] Done. Best val loss: {best_val:.4f}")
        
    print("\n[train] Running final evaluation...")
    try:
        from train.eval_unet import evaluate_unet
        evaluate_unet(cfg, ckpt_dir / "best.pth")
    except Exception as e:
        print(f"[train] Could not run final evaluation: {e}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net for aerial segmentation")
    p.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early stopping patience: stop after N epochs without val_loss "
        "improvement. Default: disabled.",
    )
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="Enable mixed precision (default: follow config.device.amp)")
    p.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Override U-Net train/val resize (default: config unet.image_size). "
        "Use smaller values on mobile AMD GPUs if training hangs (e.g. 384 384).",
    )
    p.add_argument("--compile", action="store_true", help="Use torch.compile (may fail on 3.12)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
