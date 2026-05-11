# Aerial Image Segmentation + Detection

![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-ROCm-ee4c2c.svg)


A production-ready dual-model pipeline for **semantic segmentation** and **object detection** on aerial/drone imagery using a combination of **U-Net** and **YOLOv8 OBB** models. Detect pedestrians, buildings, roads, vehicles, and vegetation in high-resolution aerial images with a browser-based Flask UI.

**Key Features:**
- **Dual-Model Pipeline**: U-Net for 6-class semantic segmentation + YOLOv8 OBB for oriented vehicle detection
- **Morphological Post-Processing**: Automated cleanup of segmentation masks using shape filtering and overlap heuristics
- **ROCm Support**: Full compatibility with AMD GPUs (tested on RX 6700S gfx1032)
- **Docker Ready**: Isolated environment with GPU passthrough
- **Web UI**: Interactive Flask application for upload, inference, and result visualization
- **Production Inference**: Batch processing with uncertainty quantification
- **Comprehensive Tests**: Unit tests, integration tests, and hardware validation

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Web UI](#web-ui-flask)
- [Results & Outputs](#results--outputs)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Hardware & ROCm](#hardware--rocm)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Citations](#citations)
- [License](#license)

---

## Quick Start

### Option 1: Docker (Recommended)

```bash
# Interactive shell
./docker-run.sh

# Single inference command
./docker-run.sh python infer.py --image sample.jpg

# Web UI (single GPU worker)
./docker-run.sh gunicorn -w 1 -b 0.0.0.0:5000 'web.app:create_app()'
```

### Option 2: Local Installation (Python 3.12)

```bash
# Install ROCm PyTorch first (skip if using NVIDIA CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Install dependencies
pip install -r requirements.txt

# Optional: Web UI dependencies
pip install -r requirements-web.txt

# Download datasets
python -m data.download_potsdam
python -m data.download_vsai

# Run inference
python infer.py --image path/to/aerial.jpg
```

---

## Architecture Overview

### Model Pipeline

```
Input Image (RGB)
    ↓
    ├─→ [U-Net Encoder] → [Bottleneck] → [U-Net Decoder] → Semantic Mask (6-class)
    │       (512×512)                                           + Uncertainty Map
    │
    └─→ [YOLOv8 OBB] → Oriented Bounding Boxes (vehicles)
            (1024×1024)
    ↓
[Morphological Post-Processing]
    ├─ Shape Filtering (buildings, vegetation, cars)
    ├─ Aspect Ratio Analysis
    └─ Car-Road Overlap Heuristic
    ↓
[Composite Visualization]
    └─ Overlay Mask + Draw OBB Polygons → result.png
```

### Models

| Model | Dataset | Task | Architecture | Input Size |
|-------|---------|------|--------------|-----------|
| **U-Net** | ISPRS Potsdam | Semantic Segmentation | Pure PyTorch (5-layer encoder-decoder) | 512×512 |
| **YOLOv8 OBB** | VSAI | Oriented Bounding Box Detection | Ultralytics (nano variant) | 1024×1024 |

**Key Design Decision**: Models are trained **independently** on **separate datasets** and combined **only at inference time** — no instance segmentation overhead, clean separation of concerns.

### Output Classes (U-Net)

| ID | Class | RGB Color | Kernel Size |
|----|-------|-----------|------------|
| 0 | Roads/Pavement | (255, 255, 255) | 7×7 Rect |
| 1 | Building | (0, 0, 255) | 7×7 Rect |
| 2 | Low Vegetation | (0, 255, 255) | 5×5 Ellipse |
| 3 | Tree | (0, 255, 0) | 5×5 Ellipse |
| 4 | Car | (255, 255, 0) | 3×3 Rect |
| 5 | Clutter | (255, 0, 0) | 3×3 Ellipse |

---

## Installation

### Prerequisites

- **Python 3.12** (or 3.11 fallback for torch.compile issues)
- **GPU Memory**: ≥4 GB (recommended ≥6 GB for comfortable training)
- **Disk Space**: ≥50 GB (for datasets + checkpoints)
- **ROCm 6.2+** (for AMD GPUs) or **CUDA 11.8+** (for NVIDIA)

### Environment Setup

#### Option A: Docker (Preferred)

```bash
./docker-run.sh                          # Interactive shell
./docker-run.sh python infer.py --image sample.jpg
```

The script builds on `rocm/pytorch:latest` and automatically:
- Sets `HSA_OVERRIDE_GFX_VERSION` for AMD compatibility
- Passes GPU device nodes (`/dev/kfd`, `/dev/dri`)
- Allocates 8GB shared memory
- Mounts the repository directory

#### Option B: Local Install (Python 3.12)

```bash
# Create virtual environment
python3.12 -m venv venv
source venv/bin/activate

# Install PyTorch from official index (ROCm 6.2)
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2

# Install project dependencies
pip install -r requirements.txt

# Verify setup
python -c "import torch; print(f'PyTorch {torch.__version__} - CUDA Available: {torch.cuda.is_available()}')"
```

#### Option C: Python 3.11 Fallback

If torch.compile fails on 3.12:

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r requirements.txt
```

#### Optional: Web UI

```bash
pip install -r requirements-web.txt
# or
pip install flask gunicorn werkzeug
```

---

## Dataset Preparation

Both datasets are downloaded lazily (cached via `kagglehub`):

### U-Net Training Data (ISPRS Potsdam)

```bash
python -m data.download_potsdam
```

- **Dataset**: ISPRS Potsdam patches (semantic labels)
- **Size**: ~1 GB
- **Format**: 512×512 RGB patches + 512×512 RGB masks
- **Auto-cached**: `~/.cache/kagglehub/datasets/deasadiqbal/private-data-1/`
- **Classes**: 6 (roads, buildings, vegetation, trees, cars, clutter)

### YOLO Training Data (VSAI)

```bash
python -m data.download_vsai
```

- **Dataset**: VSAI Vehicle Detection Dataset (OBB format)
- **Size**: ~2 GB
- **Generates**: `data/vsai_dataset.yaml` (auto-updated with absolute paths)
- **Auto-cached**: `~/.cache/kagglehub/datasets/redzapdos123/vsai-dataset-yolo11-obb-format/`
- **Classes**: Vehicle (oriented bounding boxes)

### Lazy Loading

Datasets are downloaded on first use and cached locally:

```python
from data.potsdam_dataset import get_dataloaders
train_loader, val_loader = get_dataloaders(cfg=cfg)
# Downloads + caches automatically if not already present
```

---

## Training

### U-Net Training

**Overview**: Pure PyTorch implementation with learning rate scheduling and early stopping.

#### Basic Training

```bash
python -m train.train_unet
```

#### With Custom Hyperparameters

```bash
python -m train.train_unet \
    --epochs 50 \
    --batch-size 8 \
    --lr 1e-4 \
    --patience 20  # Early stopping after 20 epochs without improvement
```

#### Resume from Checkpoint

```bash
python -m train.train_unet --resume results/unet/checkpoints/best.pth
```

Resumes:
- Model weights + optimizer state
- Learning rate scheduler state
- Epoch counter and best validation loss

#### Advanced Options

```bash
python -m train.train_unet \
    --epochs 100 \
    --batch-size 4 \
    --image-size 384 384 \      # Smaller size for limited VRAM
    --amp                        # Enable mixed precision (default: config)
    --compile                    # Use torch.compile (requires PyTorch 2.0+)
    --num-workers 8              # Parallel data loading
```

**Output**: Checkpoints saved to `results/unet/checkpoints/`:
- `best.pth` — Best validation loss
- `last.pth` — Latest checkpoint (for resuming interrupted training)
- `results.csv` — Training history (epoch, loss, LR per epoch)

#### Training Features

- **Mixed Precision (AMP)**: Reduces memory usage; enabled by default for CUDA/ROCm
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early Stopping**: Optional patience-based stopping (e.g., --patience 20)
- **Graceful Interrupt**: Ctrl+C saves state → resume seamlessly
- **CSV Logging**: Per-epoch metrics for external analysis

### YOLO Training

**Overview**: Ultralytics YOLOv8 OBB training via wrapper.

#### Basic Training

```bash
python -m train.train_yolo
```

#### Custom Parameters

```bash
python -m train.train_yolo \
    --epochs 100 \
    --batch 8 \
    --imgsz 1024 \              # Image size (1024 recommended for OBB)
    --patience 20 \             # Early stopping patience
    --device 0                  # GPU device (auto-detected if 'auto')
```

#### With Custom Project Name

```bash
python -m train.train_yolo \
    --name "v2-augmented" \
    --project "results/yolo"
```

Output: `results/yolo/<name>/weights/best.pt`

#### Skip Download (Use Cached Dataset)

```bash
python -m train.train_yolo --skip-download
```

**Output**: Results saved to `results/yolo/<run_name>/`:
- `weights/best.pt` — Best model
- `weights/last.pt` — Latest checkpoint
- `results.png` — Training curves (loss, mAP)

---

## Inference

### Command-Line Inference

**Basic Usage**:

```bash
python infer.py --image path/to/aerial.jpg
```

**With Custom Weights**:

```bash
python infer.py \
    --image path/to/aerial.jpg \
    --unet-weights results/unet/checkpoints/best.pth \
    --yolo-weights results/yolo/custom/weights/best.pt \
    --output /custom/output/dir
```


---

## Morphological Post-Processing

The raw U-Net segmentation masks are automatically refined at inference time using targeted OpenCV operations:

### Processing Pipeline

1. **Building Cleanup (Class 1)**
   - Morphological open/close with 7×7 rectangular kernel
   - Extract contours and analyze shape properties
   - **Filter Criteria**:
     - Aspect ratio > 7.0 → likely road (elongated)
     - Circularity < 0.07 → likely road (non-compact)
     - Overlaps with car pixels → likely road (heuristic)
   - Isolated small blobs (area < 50 px) → remove

2. **Vegetation Cleanup (Classes 2 & 3)**
   - Morphological open/close with 5×5 elliptical kernel
   - Preserves organic shapes (ellipse better than rectangle)

3. **Car Cleanup (Class 4)**
   - Morphological open/close with 3×3 rectangular kernel
   - Tight kernel preserves individual vehicle boundaries

4. **Clutter Cleanup (Class 5)**
   - Morphological open/close with 3×3 elliptical kernel

### Statistics Logged

```json
{
  "shape_filter_blobs": 42,           // Contours removed by aspect ratio/circularity
  "shape_filter_pixels": 15234,       // Total pixels reclassified
  "car_overlap_blobs": 5,             // Building contours overlapping cars
  "car_overlap_pixels": 3421          // Pixels reclassified due to car overlap
}
```

---

## Web UI (Flask)

Browser-based inference with interactive visualization.

### Setup

```bash
pip install -r requirements.txt -r requirements-web.txt
```

### Launch

```bash
# Development server
flask --app web.app:create_app run --host 0.0.0.0 --port 5000

# Production server (single GPU worker)
gunicorn -w 1 -b 0.0.0.0:5000 'web.app:create_app()'
```

Open http://127.0.0.1:5000/

### Features

#### Upload & Parameters
- Drag-and-drop image upload
- Optional threshold overrides:
  - **U-Net Mask Alpha** (0.0–1.0): Mask blend opacity
  - **YOLO Confidence** (0.0–1.0): Detection confidence threshold
  - **YOLO IoU** (0.0–1.0): NMS IoU threshold
- One-click **Run Models** button

#### Results View
- **Input image**: Original uploaded image
- **Composite**: Mask overlay + OBB polygons with class labels
- **Segmentation mask**: Colorized 6-class output
- **Uncertainty heatmap**: Per-pixel entropy visualization
- **Analytics**:
  - Class mix bar chart (pixel count per class)
  - YOLO detections per class
  - Detection confidence distribution
  - Detection center scatter plot

#### Detection Crops
- Grid of 256×256 crops centered on each YOLO detection
- Individual vehicle/object inspection

#### Raw Outputs
- Download `detections.json`, `mask.png`, `uncertainty.png`, etc.


---

## Results & Outputs

### Training Artifacts

**U-Net**:
- `results/unet/checkpoints/best.pth` — Best weights
- `results/unet/checkpoints/last.pth` — Latest weights
- `results/unet/results.csv` — Per-epoch metrics

**YOLO**:
- `results/yolo/<name>/weights/best.pt` — Best weights
- `results/yolo/<name>/results.png` — Training curves
- `results/yolo/<name>/confusion_matrix.png` — Class confusion

### Inference Artifacts

Per-image results in `results/inference/`:

```
results/inference/
├── result.png              # Composite visualization
├── mask.png                # Semantic segmentation (colorized)
├── mask_raw.png            # Pre-post-processing mask
├── mask_diff.png           # Pixels changed by post-processing (heatmap)
├── uncertainty.png         # U-Net entropy per-pixel (colormap)
├── detections.json         # YOLO OBB data
└── morph_stats.json        # Post-processing statistics
```

### Example Detections JSON

```json
[
  {
    "class_id": 0,
    "class_name": "vehicle",
    "conf": 0.92,
    "corners": [
      [100.5, 150.2],
      [250.3, 145.8],
      [248.9, 290.1],
      [98.7, 295.4]
    ]
  },
  ...
]
```

---

## API Reference

### `inference.pipeline.run_inference()`

```python
def run_inference(
    image_path: str | Path,
    cfg: Optional[Config] = None,
    unet_weights: Optional[str | Path] = None,
    yolo_weights: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    unet_model: Optional[UNet] = None,
) -> dict[str, str]
```

**Args**:
- `image_path` (str | Path): Path to input image
- `cfg` (Config, optional): Loaded config; defaults to `config.yaml`
- `unet_weights` (Path, optional): Override U-Net weights path
- `yolo_weights` (Path, optional): Override YOLO weights path
- `output_dir` (Path, optional): Output directory; defaults to `cfg.paths.inference_out_dir`
- `unet_model` (UNet, optional): Pre-loaded model (for batch inference)

**Returns**: Dictionary mapping artifact names to absolute paths:
```python
{
    'mask_raw': '/abs/path/to/mask_raw.png',
    'mask': '/abs/path/to/mask.png',
    'mask_diff': '/abs/path/to/mask_diff.png',
    'result': '/abs/path/to/result.png',
    'detections': '/abs/path/to/detections.json',
    'uncertainty': '/abs/path/to/uncertainty.png',
    'morph_stats': '/abs/path/to/morph_stats.json',
}
```

### `models.unet.UNet`

```python
class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,      # RGB
        num_classes: int = 6,      # Potsdam classes
        base_filters: int = 64,    # Channel multiplier
    )
```

**Features**:
- 5-layer encoder-decoder with skip connections
- BatchNorm after every Conv2d
- Bilinear upsampling (no transposed conv)
- Kaiming weight initialization
- ~7.8M trainable parameters

### `utils.cfg.Config`

```python
from utils.cfg import load_config

cfg = load_config("config.yaml")

# Attribute access
cfg.unet.lr  # 1e-4
cfg.inference.mask_alpha  # 0.45

# Dict access
cfg["unet"]["epochs"]  # 50

# Nested structures
for cls in cfg.unet.class_info:
    print(cls)  # [0, "roads/pavement", 255, 255, 255]
```

---

## Testing

### Unit & Integration Tests

```bash
# Run all tests (CPU-friendly)
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_unet_smoke.py

# Show print statements
pytest -s
```

### Long-Running Tests

```bash
# 1-epoch overfit test on CPU (validates training loop)
pytest -m slow

# Time limit: ~5 minutes
```

### GPU/Hardware Validation

```bash
# Check ROCm + GPU availability
./tests/rocm_check.sh

# Run 1-epoch U-Net + YOLO training on GPU
./tests/rocm_check.sh --with-training
```

### Test Coverage

| Test | Purpose |
|------|---------|
| `test_unet_smoke.py` | Forward/backward pass, parameter count |
| `test_dataset_potsdam.py` | Dataset loading, color-to-class mapping |
| `test_augmentations.py` | Transform pipelines (train/val/infer) |
| `test_combine.py` | Mask overlay + OBB drawing |
| `test_infer_synthetic.py` | Full pipeline on synthetic image |
| `test_cfg.py` | Config loading and nested access |
| `test_device.py` | Device detection + HSA override |
| `test_vsai_yaml.py` | YOLO dataset YAML generation |
| `test_overfit_batch.py` | 1-epoch training (slow) |

---

## Hardware & ROCm

### AMD GPU Compatibility

This project targets AMD RX 6000-series GPUs with ROCm. The RX 6700S (`gfx1032`) is **not officially supported** by ROCm, so we apply a workaround:

```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

This environment variable is automatically set by:
- All training/inference entry points (before `import torch`)
- Docker image and `docker-run.sh`
- All test scripts

No manual export needed in most cases.

### Device Detection

```python
from utils.device import get_device

device = get_device()  # Returns torch.device('cuda') or 'cpu'
# HSA override already applied before import
```

### Supported Configurations

| GPU | Driver | ROCm | Status |
|-----|--------|------|--------|
| RX 6700S (gfx1032) | amdgpu | 6.2 | Tested |
| RX 6800 (gfx1030) | amdgpu | 6.2 | Expected |
| NVIDIA A100 (CUDA) | nvidia-driver | 12.x | Supported |
| CPU (no GPU) | — | — | Fallback |

### Memory Requirements

| Task | Min VRAM | Recommended |
|------|----------|------------|
| Inference | 2 GB | 4 GB |
| U-Net Training (batch=4) | 3 GB | 6 GB |
| YOLO Training (batch=8) | 4 GB | 8 GB |
| Both simultaneously | N/A | 16 GB |

**Note**: ROCm's managed memory can use host RAM as overflow; behavior varies by driver version.

---

## Project Structure

```
aerial-image-segmentation/
├── README.md                    ← You are here
├── config.yaml                  Unified config (YOLO + U-Net + paths)
├── requirements.txt             Core ML dependencies
├── requirements-web.txt         Flask web UI dependencies
├── Dockerfile                   ROCm + project setup
├── docker-run.sh                GPU passthrough wrapper
├── infer.py                     Combined inference CLI
├── conftest.py                  Pytest configuration
├── pytest.ini                   Pytest settings
│
├── models/                      Model implementations
│   ├── unet.py                  Pure PyTorch U-Net (5-layer)
│   └── yolo.py                  Ultralytics YOLO wrapper
│
├── data/                        Dataset handling
│   ├── download_potsdam.py      ISPRS Potsdam download (kagglehub)
│   ├── download_vsai.py         VSAI OBB download + YAML generation
│   ├── vsai_dataset.yaml        Auto-generated YOLO dataset YAML
│   ├── potsdam_dataset.py       PyTorch Dataset class + color mapping
│   └── augmentations.py         Albumentations transform pipelines
│
├── train/                       Training scripts
│   ├── train_unet.py            U-Net training loop (PyTorch)
│   ├── train_yolo.py            YOLO training wrapper
│   └── eval_unet.py             [Optional] Post-training evaluation
│
├── inference/                   Inference pipeline
│   ├── pipeline.py              Combined YOLO + U-Net orchestration
│   ├── combine.py               Overlay mask + draw OBB polygons
│   └── visualization.py         Color palette helpers
│
├── utils/                       Utilities
│   ├── cfg.py                   YAML config loader + nested Config class
│   ├── device.py                ROCm device detection + HSA override
│   ├── checkpoint.py            torch.save/load wrappers
│   └── seed.py                  Deterministic seeding
│
├── web/                         Flask web UI
│   ├── app.py                   Application factory (HSA setup)
│   ├── routes.py                Upload, inference, result views
│   ├── templates/               HTML templates
│   ├── static/                  CSS, JS, assets
│   └── uploads/                 [Generated] Temporary uploads
│
├── tests/                       Test suite
│   ├── test_unet_smoke.py       Model forward/backward
│   ├── test_dataset_potsdam.py  Dataset loading
│   ├── test_augmentations.py    Transform pipelines
│   ├── test_combine.py          Visualization
│   ├── test_infer_synthetic.py  Full pipeline
│   ├── test_cfg.py              Config loading
│   ├── test_device.py           Device detection
│   ├── test_vsai_yaml.py        YOLO YAML generation
│   ├── test_overfit_batch.py    Training validation (slow)
│   └── rocm_check.sh            Hardware smoke test
│
├── figures/                     [Optional] README images
│   ├── training_yolo.png        YOLO training curves
│   ├── training_unet.png        U-Net loss curves
│   ├── inference_composite.png  Example output
│   └── ...
│
├── results/                     [Generated during training/inference]
│   ├── unet/
│   │   ├── checkpoints/         U-Net weights
│   │   └── figures/             Training artifacts
│   ├── yolo/                    YOLO runs (Ultralytics format)
│   └── inference/               Inference outputs
│
└── data/                        [Generated during data prep]
    ├── <kagglehub-cache>/       Downloaded datasets (auto-managed)
    └── vsai_dataset.yaml        Auto-generated YOLO YAML
```

---

## Citations

### Academic References

- **U-Net**: Ronneberger, O., Fischer, P., & Brox, T. (2015). "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI*, 234–241. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597)

- **YOLOv8 OBB**: Jocher, G., et al. (2023). "Ultralytics YOLOv8." GitHub. [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

### Datasets

- **ISPRS Potsdam**: Crall, A., et al. "ISPRS 2D Semantic Labeling Contest." [isprs-wg-iii-4](http://www2.isprs.org/commissions/comm3/wg4/semantic-labeling.html)

- **VSAI**: Vehicle detection dataset in OBB format. [Kaggle: VSAI Dataset](https://www.kaggle.com/datasets/redzapdos123/vsai-dataset-yolo11-obb-format)

### Libraries

- [PyTorch](https://pytorch.org/) — Deep learning framework
- [Ultralytics YOLOv8](https://docs.ultralytics.com/) — Object detection
- [Albumentations](https://albumentations.ai/) — Image augmentation
- [Flask](https://flask.palletsprojects.com/) — Web framework
- [ROCm](https://rocmdocs.amd.com/) — AMD GPU support

---
