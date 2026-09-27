# RF-DETR: Mammography Lesion Detection

A modular, production-ready implementation of **RF-DETR** tailored for lesion detection in 16-bit mammography images, powered by **DINOv2** and **detrex**.

---

## Architecture Overview

![RF-DETR Architecture](assets/architecture.jpg)

```
16-bit Mammogram (DICOM/PNG uint16)
         │
         ▼
[Mammo16BitMapper]
  - Foreground percentile intensity windowing (1% - 99%, excluding black air)
  - Continuous float32 [0.0, 1.0] output preserving full 16-bit dynamic range (RGB 3 channels)
  - LesionAwareCrop augmentation (Detectron2 canonical, crop_prob=0.0 default)
  - Geometric & multi-scale augmentations (Detectron2, multiples of 14: 644..812)
         │
         ▼
[DINOv2MultiScaleBackbone] (ViT-S/14 pretrained on mammography)
  - Intermediate blocks extracted: (2, 5, 8, 11) -> blocks 3, 6, 9, 12
  - Output: 4 feature maps of uniform stride 14 and 384 channels: (B, 384, H/14, W/14)
         │
         ▼
[MultiScaleProjector] (RF-DETR C2f feature pyramid)
  - Dynamic scaling: 2.0x (ConvTranspose2d), 1.0x (identity), 0.5x (stride-2 conv), 0.25x (extra pooling)
  - Output pyramid: p2 (stride 7), p3 (stride 14), p4 (stride 28), p5 (stride 56) with 256 channels
         │
         ▼
[detrex ChannelMapper (Neck)]
  - Maps (p2, p3, p4, p5) with GroupNorm into the DINO transformer interface
         │
         ▼
[detrex DINO Transformer & Head]
  - Deformable attention multi-scale encoder & decoder (6 layers each)
  - 2-stage query generation & Hungarian Matcher
  - Focal Loss + L1 Bounding Box Loss + GIoU Loss + Contrastive DeNoising (CDN)
```

### Key Principles
- **Minimal custom footprint**: Only custom modules not available in standard libraries are implemented (`backbone.py`, `projector.py`, `mapper.py`). Everything else (transformer, neck, matcher, criterion, optimizer, trainers, evaluators) is imported directly from `detrex` and `detectron2`.
- **Automatic dynamic class detection**: Class names (`thing_classes`) and count (`num_classes`) are extracted on-the-fly from the COCO JSON `categories` field — no hardcoding required.
- **Robust medical imaging support**: Dedicated 16-bit percentile windowing preserves micro-calcifications and subtle mass margins without dynamic range loss.

---

## Repository Structure

```
RF-DETR/
├── configs/
│   └── mammo_dinov2_dino.py       # Detrex LazyConfig (extends dino_r50, swaps backbone)
├── rfdetr/
│   ├── models/
│   │   ├── backbone.py            # DINOv2 multi-scale intermediate feature extractor
│   │   ├── projector.py           # RF-DETR multi-scale C2f projector & wrapper
│   │   └── dino.py                # Detrex DINO model import shim
│   ├── data/
│   │   ├── mapper.py              # Mammo16BitMapper (16-bit uint16 reading & normalization)
│   │   └── registration.py        # COCO dataset registration with auto-detected classes
│   └── utils/
│       └── visualize.py           # Matplotlib dataset & ground-truth visualizer
├── scripts/
│   └── setup_colab.sh             # Universal environment installer & Detrex compiler
├── notebooks/
│   └── colab_quickstart.ipynb     # Jupyter/Colab quickstart guide
├── tests/
│   ├── test_architecture.py       # Full pipeline shape & interface coordination tests
│   ├── test_backbone.py           # Unit tests for DINOv2 feature extractor & freezing
│   └── test_projector.py          # Unit tests for RF-DETR projector & pyramid shapes
├── train.py                       # CLI entrypoint for training & resume
├── eval.py                        # Standalone COCO evaluation script
├── visualize.py                   # CLI tool to visualize dataset annotations
├── pyproject.toml                 # Package definition & dependencies
├── requirements.txt               # Pinned dependencies
└── README.md
```

---

## Installation & Setup

### Option A: Google Colab / Remote GPU (Recommended for Training)

Run the automated setup script to install dependencies and compile the Detrex CUDA extensions:

```bash
git clone https://github.com/matardoky/MammoDINOv2.git
cd MammoDINOv2
bash scripts/setup_colab.sh
```

### Option B: Local Machine / CPU Testing

To run unit tests and verify model architecture without CUDA:

```bash
git clone https://github.com/matardoky/MammoDINOv2.git
cd MammoDINOv2
pip install -e ".[dev]"
```

---

## Command Line Interface (CLI)

### 1. Visualizing the Dataset

Check that 16-bit images and bounding box annotations are correctly parsed before launching training:

```bash
python visualize.py \
    --train-json /path/to/mass_train.json \
    --val-json /path/to/mass_val.json \
    --images-dir /path/to/images \
    --split train \
    --num-images 4 \
    --save-dir ./viz_output
```

Options:
- `--split`: Choose `train` or `val`.
- `--num-images`: Number of random samples to display (default: `3`).
- `--low-pct`, `--high-pct`: Intensity clipping percentiles (default: `1.0` and `99.0`).
- `--save-dir`: Save figures to disk as PNG (recommended for headless servers / Colab).

---

### 2. Training

Launch distributed or single-GPU training with automatic dataset class detection:

```bash
python train.py \
    --config-file configs/mammo_dinov2_dino.py \
    --train-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_train.json \
    --val-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_val.json \
    --images-dir /content/mammo_data/images \
    --dinov2-weights /content/drive/MyDrive/EMBED_Dataset/checkpoints/dinov2_latest_checkpoint.pth \
    --output-dir /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/RF_DETR \
    --accum-steps 8 \
    --amp-dtype auto \
    --num-gpus 1
```

> **Key Runtime Optimizations:**
> - **Effective Batch Size = 16**: With physical `batch_size = 2` (to fit on 15 GB Colab GPUs) and `accum_steps = 8`, the model trains with an effective batch size of $2 \times 8 = 16$, ensuring stable Hungarian matching and contrastive denoising.
> - **Mixed Precision (`--amp-dtype auto`)**: Automatically uses `bfloat16` on modern GPUs (A100, L4) without loss scaler overhead, and safely falls back to `float16` with `GradScaler` on T4 / V100.
> - **Gradient Norm Logging**: Logs `grad_norm` at each step to TensorBoard to monitor transformer stability.

#### Testing Lesion-Aware Crop after First Training Run
By default, `crop_prob = 0.0` (first training runs on full uncropped images with multi-scale lengths $644 \dots 812$).
To activate lesion-aware cropping (guaranteeing 100% containment of target lesions in $518 \times 518$ windows):

```bash
python train.py \
    --config-file configs/mammo_dinov2_dino.py \
    --train-json ... --val-json ... --images-dir ... \
    --output-dir ./output \
    --opts dataloader.train.mapper.crop_prob=0.2
```

#### Overriding Hyperparameters via `--opts`
Detrex LazyConfig allows overriding any parameter directly from the command line:

```bash
python train.py \
    --config-file configs/mammo_dinov2_dino.py \
    --train-json ... --val-json ... --images-dir ... \
    --output-dir ./output \
    --opts train.max_iter=7200 \
           train.eval_period=600 \
           train.log_period=20 \
           model.num_queries=50 \
           model.backbone.backbone.freeze_blocks=8
```

#### Resuming Training
To resume from the latest checkpoint in the output directory:

```bash
python train.py \
    --config-file configs/mammo_dinov2_dino.py \
    --train-json ... --val-json ... --images-dir ... \
    --output-dir /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/RF_DETR \
    --num-gpus 1 \
    --resume
```

---

---

### 3. Evaluation

Evaluate a trained model checkpoint on the validation set to obtain standard COCO metrics ($AP, AP_{50}, AP_{75}, AP_s, AP_m, AP_l$):

```bash
python eval.py \
    --config-file configs/mammo_dinov2_dino.py \
    --weights /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/RF_DETR/model_best.pth \
    --val-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_val.json \
    --images-dir /content/mammo_data/images \
    --output-dir ./eval_output \
    --test-size 812 \
    --max-size 1624
```

---

### 4. Side-by-Side Visualization (Ground Truth vs Predictions)

Inspect model detections alongside ground-truth radiologist annotations on validation mammograms:

```bash
python visualize.py \
    --weights /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/RF_DETR/model_best.pth \
    --config-file configs/mammo_dinov2_dino.py \
    --val-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_val.json \
    --images-dir /content/mammo_data/images \
    --conf-thresh 0.30 \
    --num-images 4 \
    --save-dir ./viz_predictions
```

- **Left panel**: Ground-truth bounding boxes and lesion classes.
- **Right panel**: Model predicted bounding boxes with confidence scores (e.g. `Mass 87%`, `ArchDistortion 65%`).
- Automatically prioritizes mammograms with active lesions for informative inspection.

---

## Python Notebook API

You can also use the modules directly in Python or Jupyter:

```python
from rfdetr.data.registration import register_mammo_dataset
from rfdetr.utils.visualize import visualize_dataset

# 1. Register COCO datasets (classes are auto-detected)
thing_classes = register_mammo_dataset(
    train_json="path/to/mass_train.json",
    val_json="path/to/mass_val.json",
    images_dir="path/to/images"
)
print("Detected classes:", thing_classes)

# 2. Visualize samples inline
visualize_dataset("mammo_train", num_images=3, seed=42)
```

---

## Automated Verification & Tests

Run the comprehensive automated test suite (85 tests covering multi-scale feature shapes, pyramid strides, partial freezing logic, gradient backpropagation, 16-bit float32 normalization, lesion-aware cropping containment, gradient accumulation equivalence, CLI parsing, and end-to-end integration contracts):

```bash
python -m pytest tests/
```

All unit and integration tests run entirely on synthetic data and execute on CPU without requiring CUDA or GPU hardware.
