"""
Evaluation script for U-Net. Computes Confusion Matrix, PR Curve, and F1 Curve.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.device import apply_hsa_override
apply_hsa_override()

import torch
import torch.nn.functional as F
from pathlib import Path
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from data.potsdam_dataset import get_dataloaders
from models.unet import UNet
from utils.cfg import load_config, resolve_path
from utils.device import get_device

def _wrap_tqdm(iterable, **kwargs):
    return tqdm(iterable, **kwargs) if tqdm is not None else iterable

def evaluate_unet(cfg, ckpt_path):
    device = get_device()
    _, val_loader = get_dataloaders(cfg=cfg)
    
    num_classes = cfg.unet.num_classes
    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    
    model.eval()
    
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    # For PR curves
    thresholds = np.linspace(0.0, 1.0, 51)
    # tp, fp, fn per class per threshold
    tp = np.zeros((num_classes, len(thresholds)), dtype=np.int64)
    fp = np.zeros((num_classes, len(thresholds)), dtype=np.int64)
    fn = np.zeros((num_classes, len(thresholds)), dtype=np.int64)

    print("[eval] Running evaluation on validation set...")
    with torch.no_grad():
        for images, masks in _wrap_tqdm(val_loader, desc="eval"):
            images = images.to(device)
            masks = masks.to(device) # (B, H, W)
            
            logits = model(images)
            probs = F.softmax(logits, dim=1) # (B, C, H, W)
            preds = torch.argmax(probs, dim=1)
            
            # Update confusion matrix
            mask_flat = masks.flatten()
            pred_flat = preds.flatten()
            
            # Bincount for confusion matrix
            idx = mask_flat * num_classes + pred_flat
            counts = torch.bincount(idx, minlength=num_classes**2).cpu().numpy()
            confusion_matrix += counts.reshape((num_classes, num_classes))
            
            # Update PR curves per threshold
            # Since masks are mutually exclusive, a pixel belongs to exactly one class.
            for c in range(num_classes):
                true_c = (masks == c)
                prob_c = probs[:, c, :, :]
                
                # To save memory, we can do it on GPU then move to CPU
                for t_idx, t in enumerate(thresholds):
                    pred_c = (prob_c >= t)
                    
                    tp[c, t_idx] += (pred_c & true_c).sum().item()
                    fp[c, t_idx] += (pred_c & ~true_c).sum().item()
                    fn[c, t_idx] += (~pred_c & true_c).sum().item()

    # Save directly to figures/ since results/unet is root-owned
    out_dir = resolve_path(cfg.paths.unet_ckpt_dir).parent.parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    class_names = [info[1] for info in cfg.unet.class_info]
    
    # 1. Plot Confusion Matrix
    _plot_confusion_matrix(confusion_matrix, class_names, out_dir / "unet_confusion_matrix.png", normalize=False)
    _plot_confusion_matrix(confusion_matrix, class_names, out_dir / "unet_confusion_matrix_norm.png", normalize=True)
    
    # 2. Plot PR Curve
    _plot_pr_curve(tp, fp, fn, thresholds, class_names, out_dir)
    
    print(f"[eval] Evaluation complete. Plots saved to {out_dir}")

def _plot_confusion_matrix(cm, class_names, out_path, normalize=False):
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        fmt = ".2f"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"
        
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt_str = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt_str),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
                    
    fig.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def _plot_pr_curve(tp, fp, fn, thresholds, class_names, out_dir):
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F1 = 2 * (P * R) / (P + R)
    
    epsilon = 1e-6
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    
    # Exclude background/clutter if it's the last class or just plot all.
    # YOLO usually plots macro-average as a thick line.
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(len(class_names)):
        ax.plot(recall[c, :], precision[c, :], label=class_names[c], alpha=0.7)
        
    # Macro average
    mean_p = np.mean(precision, axis=0)
    mean_r = np.mean(recall, axis=0)
    ax.plot(mean_r, mean_p, label="all classes", linewidth=3, color="black")
    
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "unet_pr_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # F1 Curve
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in range(len(class_names)):
        ax.plot(thresholds, f1[c, :], label=class_names[c], alpha=0.7)
        
    mean_f1 = np.mean(f1, axis=0)
    ax.plot(thresholds, mean_f1, label="all classes", linewidth=3, color="black")
    
    ax.set_title("F1-Confidence Curve")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("F1")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "unet_f1_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--config", type=str, default=None)
    args = p.parse_args()
    
    cfg = load_config(args.config)
    evaluate_unet(cfg, args.ckpt)
