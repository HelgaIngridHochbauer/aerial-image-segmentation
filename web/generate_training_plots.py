"""
Generate comprehensive training analytics plots for the web dashboard.

Reads:
  - UNet checkpoints (best.pth / last.pth) for model metadata
  - YOLO results.csv for training curves
  - YOLO confusion matrices and PR curves (existing PNGs from Ultralytics)

Writes matplotlib figures to  figures/  for serving by the web app.
Optionally writes LaTeX-ready (white background, PDF-safe) versions to
figures/latex/.

Usage:
    python -m web.generate_training_plots
    python -m web.generate_training_plots --latex
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Two palettes: dark (web) and light (LaTeX)
# ---------------------------------------------------------------------------
_DARK = dict(
    BG="#0f1e2b", SURFACE="#162a3a", ACCENT="#27b17b", ACCENT2="#8fe4ff",
    TEXT="#d5e8f0", GRID="#1e3a4d",
    COLORS=["#27b17b", "#8fe4ff", "#e8a838", "#e05858", "#9b6fd9", "#4fc1e8"],
)
_LIGHT = dict(
    BG="#ffffff", SURFACE="#ffffff", ACCENT="#1a7a52", ACCENT2="#1565a0",
    TEXT="#111111", GRID="#cccccc",
    COLORS=["#1a7a52", "#1565a0", "#c47800", "#c03030", "#7b4db8", "#2896b8"],
)

# Active palette — set by generate_all()
P = _DARK


def _style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(P["SURFACE"])
    ax.set_title(title, color=P["TEXT"], fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, color=P["TEXT"], fontsize=9)
    ax.set_ylabel(ylabel, color=P["TEXT"], fontsize=9)
    ax.tick_params(colors=P["TEXT"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(P["GRID"])
    ax.grid(True, color=P["GRID"], alpha=0.5, linewidth=0.5)


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[plots] saved → {path}")


# ===================================================================
# UNet plots
# ===================================================================

def _load_unet_checkpoint(ckpt_path: Path) -> dict | None:
    try:
        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        return ckpt
    except Exception as e:
        print(f"[plots] could not load {ckpt_path}: {e}")
        return None


def plot_unet_architecture(out_dir: Path):
    """Visual summary of U-Net architecture (layer sizes)."""
    encoder_channels = [3, 64, 128, 256, 512, 1024]
    decoder_channels = [1024, 512, 256, 128, 64]
    labels_enc = ["Input\n3ch", "Enc1\n64", "Enc2\n128", "Enc3\n256",
                  "Enc4\n512", "Bottleneck\n1024"]
    labels_dec = ["Up1\n512", "Up2\n256", "Up3\n128", "Up4\n64"]

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.set_facecolor(P["BG"])
    ax.set_facecolor(P["BG"])

    lbl_color = P["TEXT"]
    enc_color = P["ACCENT"]
    dec_color = P["ACCENT2"]
    btn_color = P["COLORS"][3]

    for i, (ch, lbl) in enumerate(zip(encoder_channels, labels_enc)):
        height = ch / 1024 * 3.0
        c = btn_color if i == len(encoder_channels) - 1 else enc_color
        ax.barh(0, 0.8, left=i, height=height, color=c, alpha=0.8,
                edgecolor=P["GRID"], linewidth=0.5, align="center")
        ax.text(i + 0.4, 0, lbl, ha="center", va="center", color=lbl_color,
                fontsize=7, fontweight="bold")

    for i, (ch, lbl) in enumerate(zip(decoder_channels[1:], labels_dec)):
        height = ch / 1024 * 3.0
        ax.barh(0, 0.8, left=len(encoder_channels) + i, height=height,
                color=dec_color, alpha=0.7, edgecolor=P["GRID"], linewidth=0.5,
                align="center")
        txt_c = P["BG"] if P is _DARK else P["TEXT"]
        ax.text(len(encoder_channels) + i + 0.4, 0, lbl, ha="center",
                va="center", color=txt_c, fontsize=7, fontweight="bold")

    ox = len(encoder_channels) + len(labels_dec)
    ax.barh(0, 0.8, left=ox, height=0.02, color=P["COLORS"][2], alpha=0.9,
            edgecolor=P["GRID"], linewidth=0.5, align="center")
    ax.text(ox + 0.4, 0, "Out\n6cls", ha="center", va="center",
            color=lbl_color, fontsize=7, fontweight="bold")

    for i in range(4):
        enc_x = i + 1 + 0.4
        dec_x = len(encoder_channels) + (3 - i) + 0.4
        ax.annotate("", xy=(dec_x, 0.25), xytext=(enc_x, 0.25),
                    arrowprops=dict(arrowstyle="->", color=P["COLORS"][2] + "80",
                                    lw=1.2, connectionstyle="arc3,rad=0.3"))

    ax.set_xlim(-0.2, ox + 1.2)
    ax.set_ylim(-2, 2)
    ax.axis("off")
    ax.set_title("U-Net Architecture — Channel Progression & Skip Connections",
                 color=P["TEXT"], fontsize=12, fontweight="bold", pad=12)
    _save(fig, out_dir / "unet_architecture.png")


def plot_unet_model_summary(out_dir: Path, ckpt_dir: Path):
    """Checkpoint metadata summary card."""
    best = _load_unet_checkpoint(ckpt_dir / "best.pth")
    last = _load_unet_checkpoint(ckpt_dir / "last.pth")
    if not best and not last:
        print("[plots] no UNet checkpoints found, skipping model summary")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.set_facecolor(P["BG"])
    ax.set_facecolor(P["SURFACE"])
    ax.axis("off")

    lines = []
    if best:
        n_params = sum(v.numel() for v in best.get("model_state_dict", {}).values())
        lines.append(("Model", "U-Net (Conv-BN-ReLU)"))
        lines.append(("Parameters", f"{n_params:,}"))
        lines.append(("Best epoch", str(best.get("epoch", "?"))))
        lines.append(("Best val loss", f"{best.get('val_loss', 0):.4f}"))
    if last:
        lines.append(("Last epoch", str(last.get("epoch", "?"))))
        lines.append(("Last val loss", f"{last.get('val_loss', 0):.4f}"))
        bv = last.get("best_val")
        if bv is not None:
            lines.append(("Tracked best_val", f"{bv:.4f}"))
        eni = last.get("epochs_no_improve")
        if eni is not None:
            lines.append(("Epochs no improve", str(eni)))

    y = 0.92
    ax.text(0.5, 0.98, "U-Net Checkpoint Summary", ha="center", va="top",
            color=P["ACCENT"], fontsize=13, fontweight="bold",
            transform=ax.transAxes)
    for label, value in lines:
        ax.text(0.15, y, label, ha="left", va="top", color=P["TEXT"],
                fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(0.65, y, value, ha="left", va="top", color=P["ACCENT2"],
                fontsize=10, family="monospace", transform=ax.transAxes)
        y -= 0.1
    _save(fig, out_dir / "unet_model_summary.png")


def plot_unet_param_distribution(out_dir: Path, ckpt_dir: Path):
    """Histogram of weight magnitudes per layer group."""
    best = _load_unet_checkpoint(ckpt_dir / "best.pth")
    if not best:
        return

    sd = best.get("model_state_dict", {})
    groups = {"Encoder": [], "Decoder": [], "Bottleneck": [], "Output": []}
    for k, v in sd.items():
        flat = v.float().cpu().numpy().ravel()
        if "down" in k or "inc" in k:
            groups["Encoder"].append(flat)
        elif "up" in k:
            groups["Decoder"].append(flat)
        elif "outc" in k:
            groups["Output"].append(flat)
        else:
            groups["Bottleneck"].append(flat)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    fig.set_facecolor(P["BG"])

    for ax, (name, arrays), color in zip(axes, groups.items(), P["COLORS"]):
        _style_ax(ax, title=name, xlabel="Weight value", ylabel="Density")
        if arrays:
            all_vals = np.concatenate(arrays)
            ax.hist(all_vals, bins=80, color=color, alpha=0.8, density=True,
                    edgecolor="none")
            ax.axvline(0, color=P["GRID"], linewidth=0.8, linestyle="--")
            ax.text(0.95, 0.95,
                    f"μ={all_vals.mean():.4f}\nσ={all_vals.std():.4f}",
                    ha="right", va="top", transform=ax.transAxes,
                    color=P["TEXT"], fontsize=7, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=P["BG"],
                              alpha=0.8))

    fig.suptitle("U-Net Weight Distribution by Layer Group", color=P["TEXT"],
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "unet_weight_distribution.png")

def plot_unet_losses(out_dir: Path, csv_path: Path):
    """UNet training vs validation loss curves."""
    data = _read_yolo_csv(csv_path) # Uses the same CSV reader function
    if data is None or "train_loss" not in data:
        return

    epochs = data.get("epoch", np.arange(1, len(data["train_loss"]) + 1))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.set_facecolor(P["BG"])
    _style_ax(ax, "U-Net Training vs Validation Loss", "Epoch", "Cross-Entropy Loss")

    ax.plot(epochs, data["train_loss"], color=P["COLORS"][0], linewidth=1.8, label="Train Loss", alpha=0.9)
    if "val_loss" in data:
        ax.plot(epochs, data["val_loss"], color=P["COLORS"][1], linewidth=1.8, label="Val Loss", alpha=0.9)

    ax.legend(fontsize=8, facecolor=P["SURFACE"], edgecolor=P["GRID"], labelcolor=P["TEXT"])
    fig.tight_layout()
    _save(fig, out_dir / "unet_loss_curves.png")

def plot_unet_lr(out_dir: Path, csv_path: Path):
    """UNet learning rate schedule."""
    data = _read_yolo_csv(csv_path)
    if data is None or "lr/pg0" not in data:
        return

    epochs = data.get("epoch", np.arange(1, len(data["lr/pg0"]) + 1))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.set_facecolor(P["BG"])
    _style_ax(ax, "U-Net Learning Rate Schedule", "Epoch", "LR")

    ax.plot(epochs, data["lr/pg0"], color=P["COLORS"][2], linewidth=1.5, label="LR (pg0)", alpha=0.9)

    ax.legend(fontsize=7, facecolor=P["SURFACE"], edgecolor=P["GRID"], labelcolor=P["TEXT"])
    fig.tight_layout()
    _save(fig, out_dir / "unet_lr_schedule.png")

def copy_unet_figures(out_dir: Path, unet_run_dir: Path):
    """Copy UNet evaluation plots."""
    copies = {
        "confusion_matrix.png": "unet_confusion_matrix.png",
        "confusion_matrix_normalized.png": "unet_confusion_matrix_norm.png",
        "unet_pr_curve.png": "unet_pr_curve.png",
        "unet_f1_curve.png": "unet_f1_curve.png",
    }
    for src_name, dst_name in copies.items():
        src = unet_run_dir / src_name
        dst = out_dir / dst_name
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"[plots] copied → {dst}")



# ===================================================================
# YOLO plots
# ===================================================================

def _read_yolo_csv(csv_path: Path) -> dict[str, np.ndarray] | None:
    try:
        import csv
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return None
        data: dict[str, list[float]] = {}
        for key in rows[0]:
            clean = key.strip()
            vals = []
            for r in rows:
                try:
                    vals.append(float(r[key]))
                except (ValueError, TypeError):
                    vals.append(float("nan"))
            data[clean] = np.array(vals)
        return data
    except Exception as e:
        print(f"[plots] could not read {csv_path}: {e}")
        return None


def plot_yolo_losses(out_dir: Path, csv_path: Path):
    """YOLO training vs validation loss curves."""
    data = _read_yolo_csv(csv_path)
    if data is None:
        return

    epochs = data.get("epoch", np.arange(1, len(next(iter(data.values()))) + 1))
    train_keys = [k for k in data if k.startswith("train/")]
    val_keys = [k for k in data if k.startswith("val/")]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.set_facecolor(P["BG"])

    _style_ax(axes[0], "Training Losses", "Epoch", "Loss")
    for i, k in enumerate(train_keys):
        label = k.replace("train/", "")
        axes[0].plot(epochs, data[k], color=P["COLORS"][i % len(P["COLORS"])],
                     linewidth=1.5, label=label, alpha=0.9)
    axes[0].legend(fontsize=7, facecolor=P["SURFACE"], edgecolor=P["GRID"],
                   labelcolor=P["TEXT"])

    _style_ax(axes[1], "Validation Losses", "Epoch", "Loss")
    for i, k in enumerate(val_keys):
        label = k.replace("val/", "")
        axes[1].plot(epochs, data[k], color=P["COLORS"][i % len(P["COLORS"])],
                     linewidth=1.5, label=label, alpha=0.9)
    axes[1].legend(fontsize=7, facecolor=P["SURFACE"], edgecolor=P["GRID"],
                   labelcolor=P["TEXT"])

    fig.suptitle("YOLOv8 OBB — Training & Validation Losses",
                 color=P["TEXT"], fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "yolo_loss_curves.png")


def plot_yolo_metrics(out_dir: Path, csv_path: Path):
    """YOLO precision, recall, mAP curves."""
    data = _read_yolo_csv(csv_path)
    if data is None:
        return

    epochs = data.get("epoch", np.arange(1, len(next(iter(data.values()))) + 1))
    metric_keys = {
        "metrics/precision(B)": ("Precision", P["COLORS"][0]),
        "metrics/recall(B)": ("Recall", P["COLORS"][1]),
        "metrics/mAP50(B)": ("mAP@50", P["COLORS"][2]),
        "metrics/mAP50-95(B)": ("mAP@50-95", P["COLORS"][3]),
    }

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.set_facecolor(P["BG"])
    _style_ax(ax, "YOLOv8 OBB — Detection Metrics Over Training",
              "Epoch", "Score")

    for key, (label, color) in metric_keys.items():
        if key in data:
            vals = data[key]
            ax.plot(epochs, vals, color=color, linewidth=1.8, label=label,
                    alpha=0.9)
            ax.annotate(f"{vals[-1]:.3f}", xy=(epochs[-1], vals[-1]),
                        xytext=(5, 0), textcoords="offset points",
                        color=color, fontsize=7.5, fontweight="bold")

    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, facecolor=P["SURFACE"], edgecolor=P["GRID"],
              labelcolor=P["TEXT"], loc="lower right")
    fig.tight_layout()
    _save(fig, out_dir / "yolo_metrics_curves.png")


def plot_yolo_lr(out_dir: Path, csv_path: Path):
    """YOLO learning rate schedule."""
    data = _read_yolo_csv(csv_path)
    if data is None:
        return

    lr_keys = [k for k in data if k.startswith("lr/")]
    if not lr_keys:
        return

    epochs = data.get("epoch", np.arange(1, len(next(iter(data.values()))) + 1))
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fig.set_facecolor(P["BG"])
    _style_ax(ax, "YOLOv8 Learning Rate Schedule", "Epoch", "LR")

    for i, k in enumerate(lr_keys):
        label = k.replace("lr/", "")
        ax.plot(epochs, data[k], color=P["COLORS"][i % len(P["COLORS"])],
                linewidth=1.5, label=label, alpha=0.9)

    ax.legend(fontsize=7, facecolor=P["SURFACE"], edgecolor=P["GRID"],
              labelcolor=P["TEXT"])
    fig.tight_layout()
    _save(fig, out_dir / "yolo_lr_schedule.png")


def copy_yolo_figures(out_dir: Path, yolo_run_dir: Path):
    """Copy Ultralytics-generated plots (confusion matrices, PR/F1 curves)."""
    copies = {
        "confusion_matrix.png": "yolo_confusion_matrix.png",
        "confusion_matrix_normalized.png": "yolo_confusion_matrix_norm.png",
        "BoxPR_curve.png": "yolo_pr_curve.png",
        "BoxF1_curve.png": "yolo_f1_curve.png",
        "results.png": "yolo_results_ultralytics.png",
    }
    for src_name, dst_name in copies.items():
        src = yolo_run_dir / src_name
        dst = out_dir / dst_name
        if src.is_file():
            shutil.copy2(src, dst)
            print(f"[plots] copied → {dst}")

# ===================================================================
# Mixed plots
# ===================================================================

def plot_mixed_losses(out_dir: Path, yolo_csv: Path, unet_csv: Path):
    """Overlay YOLO and UNet losses to compare convergence."""
    yolo_data = _read_yolo_csv(yolo_csv)
    unet_data = _read_yolo_csv(unet_csv)
    
    if yolo_data is None or unet_data is None:
        return
        
    y_epochs = yolo_data.get("epoch", np.arange(1, len(yolo_data.get("train/box_loss", [])) + 1))
    u_epochs = unet_data.get("epoch", np.arange(1, len(unet_data.get("train_loss", [])) + 1))
    
    # We'll normalize losses to their max value for better overlay comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.set_facecolor(P["BG"])
    _style_ax(ax, "Convergence Comparison (Normalized Training Loss)", "Epoch", "Normalized Loss")
    
    if "train/box_loss" in yolo_data and "train/cls_loss" in yolo_data:
        # Sum of key YOLO losses
        y_loss = yolo_data["train/box_loss"] + yolo_data["train/cls_loss"]
        if len(y_loss) > 0 and np.max(y_loss) > 0:
            y_loss_norm = y_loss / np.nanmax(y_loss)
            ax.plot(y_epochs, y_loss_norm, color=P["COLORS"][0], linewidth=2, label="YOLO Train Loss (Normalized)", alpha=0.85)

    if "train_loss" in unet_data:
        u_loss = unet_data["train_loss"]
        if len(u_loss) > 0 and np.max(u_loss) > 0:
            u_loss_norm = u_loss / np.nanmax(u_loss)
            ax.plot(u_epochs, u_loss_norm, color=P["COLORS"][1], linewidth=2, label="U-Net Train Loss (Normalized)", alpha=0.85)
            
    ax.legend(fontsize=9, facecolor=P["SURFACE"], edgecolor=P["GRID"], labelcolor=P["TEXT"])
    fig.tight_layout()
    _save(fig, out_dir / "mixed_loss_curves.png")

def plot_mixed_metrics(out_dir: Path, yolo_csv: Path, unet_csv: Path):
    """Overlay YOLO mAP and UNet F1/IoU (if available) or just provide a side-by-side."""
    yolo_data = _read_yolo_csv(yolo_csv)
    unet_data = _read_yolo_csv(unet_csv)
    
    if yolo_data is None or unet_data is None:
        return
        
    y_epochs = yolo_data.get("epoch", np.arange(1, len(yolo_data.get("metrics/mAP50(B)", [])) + 1))
    
    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.set_facecolor(P["BG"])
    _style_ax(ax1, "Model Performance Over Time", "Epoch", "YOLO mAP@50")
    
    if "metrics/mAP50(B)" in yolo_data:
        ax1.plot(y_epochs, yolo_data["metrics/mAP50(B)"], color=P["COLORS"][0], linewidth=2, label="YOLO mAP@50", alpha=0.9)
    
    # UNet metrics might not be in CSV since we run eval_unet at the end.
    # If they are not in CSV, we can just plot the YOLO metrics and maybe a final UNet point if we parse it,
    # but for now we just plot YOLO and if UNet has metrics we plot them on a secondary Y-axis.
    ax1.legend(loc="upper left", fontsize=9, facecolor=P["SURFACE"], edgecolor=P["GRID"], labelcolor=P["TEXT"])
    
    if "val_loss" in unet_data:
        ax2 = ax1.twinx()
        u_epochs = unet_data.get("epoch", np.arange(1, len(unet_data["val_loss"]) + 1))
        # As a proxy for performance over time, plot validation loss inverted or just plot val loss on right axis
        ax2.plot(u_epochs, unet_data["val_loss"], color=P["COLORS"][1], linewidth=2, label="U-Net Val Loss", alpha=0.9, linestyle="--")
        ax2.set_ylabel("U-Net Validation Loss", color=P["TEXT"], fontsize=9)
        ax2.tick_params(axis="y", colors=P["TEXT"], labelsize=8)
        for spine in ax2.spines.values():
            spine.set_color(P["GRID"])
        ax2.legend(loc="upper right", fontsize=9, facecolor=P["SURFACE"], edgecolor=P["GRID"], labelcolor=P["TEXT"])

    fig.tight_layout()
    _save(fig, out_dir / "mixed_metrics_curves.png")


# ===================================================================
# Pipeline overview
# ===================================================================

def plot_pipeline_overview(out_dir: Path):
    """Visual summary of the dual-model pipeline."""
    fig, ax = plt.subplots(figsize=(10, 3))
    fig.set_facecolor(P["BG"])
    ax.set_facecolor(P["BG"])
    ax.axis("off")

    boxes = [
        (0.05, "Aerial\nImage", P["COLORS"][1]),
        (0.22, "U-Net\nSegmentation", P["ACCENT"]),
        (0.22, "YOLOv8\nOBB Detection", P["COLORS"][2]),
        (0.55, "Morphological\nPost-Processing", P["COLORS"][4]),
        (0.75, "Composite\nOutput", P["COLORS"][3]),
    ]
    y_positions = [0.5, 0.7, 0.3, 0.5, 0.5]

    for (x, label, color), y in zip(boxes, y_positions):
        ax.add_patch(plt.Rectangle(
            (x - 0.07, y - 0.15), 0.14, 0.3,
            facecolor=color, alpha=0.85, edgecolor=P["GRID"], linewidth=1.5,
            transform=ax.transAxes, zorder=2, clip_on=False))
        txt_c = "#fff" if P is _DARK else "#fff"
        ax.text(x, y, label, ha="center", va="center", color=txt_c,
                fontsize=8, fontweight="bold", transform=ax.transAxes, zorder=3)

    arrow_pairs = [(0.12, 0.5, 0.15, 0.7), (0.12, 0.5, 0.15, 0.3),
                   (0.29, 0.7, 0.48, 0.55), (0.29, 0.3, 0.48, 0.45),
                   (0.62, 0.5, 0.68, 0.5)]
    arrow_c = "#ffffff60" if P is _DARK else "#33333380"
    for x1, y1, x2, y2 in arrow_pairs:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=arrow_c, lw=1.5),
                    xycoords="axes fraction", textcoords="axes fraction")

    ax.set_title("URBAN-SYNAPSE — Dual-Model Inference Pipeline",
                 color=P["TEXT"], fontsize=12, fontweight="bold", pad=10)
    _save(fig, out_dir / "pipeline_overview.png")


# ===================================================================
# Main
# ===================================================================

def generate_all(out_dir: Path, project_root: Path, latex: bool = False):
    global P
    P = _LIGHT if latex else _DARK
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = project_root / "results" / "unet" / "checkpoints"
    yolo_csv = project_root / "results" / "yolo" / "safe" / "results.csv"
    yolo_run = project_root / "results" / "yolo" / "safe"

    mode = "LaTeX (white)" if latex else "web (dark)"
    print(f"[plots] Generating training analytics ({mode}) → {out_dir}")

    plot_pipeline_overview(out_dir)
    plot_unet_architecture(out_dir)
    plot_unet_model_summary(out_dir, ckpt_dir)
    plot_unet_param_distribution(out_dir, ckpt_dir)
    
    unet_csv = project_root / "results" / "unet" / "results.csv"
    if unet_csv.is_file():
        plot_unet_losses(out_dir, unet_csv)
        plot_unet_lr(out_dir, unet_csv)
    else:
        print(f"[plots] UNet CSV not found at {unet_csv}, skipping UNet curve plots")
        
    unet_run_dir = project_root / "results" / "unet"
    if unet_run_dir.is_dir():
        copy_unet_figures(out_dir, unet_run_dir)

    if yolo_csv.is_file():
        plot_yolo_losses(out_dir, yolo_csv)
        plot_yolo_metrics(out_dir, yolo_csv)
        plot_yolo_lr(out_dir, yolo_csv)
    else:
        print(f"[plots] YOLO CSV not found at {yolo_csv}, skipping YOLO curve plots")

    if yolo_run.is_dir():
        copy_yolo_figures(out_dir, yolo_run)
    else:
        print(f"[plots] YOLO run dir not found at {yolo_run}, skipping copies")

    if yolo_csv.is_file() and unet_csv.is_file():
        plot_mixed_losses(out_dir, yolo_csv, unet_csv)
        plot_mixed_metrics(out_dir, yolo_csv, unet_csv)

    print("[plots] Done.")


def main():
    p = argparse.ArgumentParser(description="Generate training analytics plots")
    p.add_argument("--out-dir", type=str, default="figures",
                   help="Output directory for plots (default: figures)")
    p.add_argument("--project-root", type=str, default=None)
    p.add_argument("--latex", action="store_true",
                   help="Also export white-background LaTeX-ready versions "
                   "to <out-dir>/latex/")
    args = p.parse_args()

    root = Path(args.project_root) if args.project_root else \
        Path(__file__).resolve().parent.parent
    out = Path(args.out_dir)

    # Dark mode for website
    generate_all(out, root, latex=False)

    # LaTeX mode
    if args.latex:
        generate_all(out / "latex", root, latex=True)


if __name__ == "__main__":
    main()
