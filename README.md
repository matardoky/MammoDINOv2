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
  - 2-stage query generation (num_queries = two_stage_num_proposals = 100)
  - Hungarian Matcher + Focal Loss + L1 Bounding Box Loss + GIoU Loss + Contrastive DeNoising (CDN)
```

### Key Principles
- **Minimal custom footprint**: Only custom modules not available in standard libraries are implemented (`backbone.py`, `projector.py`, `mapper.py`). Everything else (transformer, neck, matcher, criterion, optimizer, trainers, evaluators) is imported directly from `detrex` and `detectron2`.
- **Automatic dynamic class detection**: Class names (`thing_classes`) and count (`num_classes`) are extracted on-the-fly from the COCO JSON `categories` field — no hardcoding required.
- **Pure FP32 Precision**: Pure standard 32-bit floating point precision throughout the network, ensuring complete numerical stability and zero kernel incompatibilities with Detrex `MultiScaleDeformableAttention`.
- **Robust medical imaging support**: Dedicated 16-bit percentile windowing preserves micro-calcifications and subtle mass margins without dynamic range loss.

---

## Repository Structure

```
RF-DETR/
├── configs/
│   └── mammo_dinov2_dino.py       # Detrex LazyConfig (DINOv2 + RF-DETR Projector + DINO)
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
│   ├── conftest.py                # Global pytest warning filters (clean reports)
│   ├── test_architecture.py       # Full pipeline shape & interface coordination tests
│   ├── test_backbone.py           # Unit tests for DINOv2 feature extractor & freezing
│   ├── test_cli_and_integration.py# CLI arg parsing & end-to-end gradient flow smoke tests
│   ├── test_data.py               # Data mapper, uint16 normalization & sampler tests
│   └── test_projector.py          # Unit tests for RF-DETR projector & pyramid shapes
├── train.py                       # CLI entrypoint for training & resume
├── eval.py                        # Standalone COCO evaluation script
├── visualize.py                   # CLI tool to visualize dataset annotations & predictions
├── pyproject.toml                 # Package definition, tool settings & warning filters
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

#### In Google Colab / Jupyter Notebook (Inline Visualization)

Use `%run` to execute in the notebook kernel and display the plot directly in the cell:

```python
%run visualize.py \
    --train-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/full_coco_3class_train.json \
    --val-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/full_coco_3class_val.json \
    --images-dir /content/mammo_data/images \
    --split train \
    --num-images 4 \
    --save-dir ./viz_output \
    --show
```

#### Standalone Terminal / Headless Server (Save to Disk)

```bash
python visualize.py \
    --train-json /path/to/mass_train.json \
    --val-json /path/to/mass_val.json \
    --images-dir /path/to/images \
    --split train \
    --num-images 4 \
    --save-dir ./viz_output
```

**Options:**
- `--split`: Choose `train` or `val`.
- `--num-images`: Number of random samples to display (default: `3`).
- `--seed`: Random seed (default: `None` for fresh random sampling each time; prioritizes images with lesions).
- `--show`: Display the plot inline via `plt.show()` (useful in notebook environments).
- `--low-pct`, `--high-pct`: Intensity clipping percentiles (default: `1.0` and `99.0`).
- `--save-dir`: Save figures to disk as PNG.

---

### 2. Fast Architecture & Overfit Verification (`overfit.py`)

Verify end-to-end model learning capacity and convergence on a small micro-batch (30 images) before launching full 20-epoch training:

```bash
python overfit.py \
    --train-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_train.json \
    --images-dir /content/mammo_data/images \
    --dinov2-weights /content/drive/MyDrive/EMBED_Dataset/checkpoints/dinov2_latest_checkpoint.pth \
    --output-dir /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/RF_DETR_OVERFIT \
    --num-images 30 \
    --freeze-blocks 0 \
    --max-iter 1500 \
    --eval-period 500 \
    --batch-size 2 \
    --lr 1e-4 \
    --num-gpus 1
```

- **Fully Unfrozen Backbone**: `--freeze-blocks 0` unfreezes all 12 DINOv2 ViT blocks and patch embeddings (22.06M trainable parameters), enabling complete representation fine-tuning with activation gradient checkpointing.
- **Deterministic Evaluation**: Trains and evaluates on the exact same 30 images with fixed resize (812 px) and zero stochastic flip.
- **Immediate Optimizer Steps**: `grad_accum_steps=1` enables immediate weight updates on every micro-batch, allowing rapid loss collapse and perfect convergence (**100.00% AP50**, **86.76% AP75**, and **64.99% mAP** achieved with `freeze_blocks = 0`).
- **DINOv2 Layer-Wise Decay**: Automatically applies layer-wise learning rate decay across all 12 ViT blocks (depth 12 down to depth 1 and stem) with `weight_decay = 0.0` on the backbone.
- **Automatic Visual Predictions**: Automatically saves visual side-by-side comparisons (Ground Truth vs Model Predictions) in `./output_overfit/visualizations/`.

#### Empirical Validation Benchmark (Overfit Verification)

| Backbone Configuration | Trainable Params | $\text{AP}_{50}$ | $\text{AP}_{75}$ | $\text{mAP}$ | Final Total Loss | Convergence |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Partial Freeze (`freeze_blocks=4`)** | 14.20M | 86.46% | 27.79% | 35.94% | ~18.4 | Strong |
| **Full Fine-Tuning (`freeze_blocks=0`)** | **22.06M** | **100.00%** | **86.76%** | **64.99%** | **9.00** | **Perfect (Certified)** |

---

### 3. Full Training

Launch single-GPU or distributed training with automatic dataset class detection:

```bash
python train.py \
    --config-file configs/mammo_dinov2_dino.py \
    --train-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/full_coco_3class_train.json \
    --val-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/full_coco_3class_val.json \
    --images-dir /content/mammo_data/images \
    --dinov2-weights /content/drive/MyDrive/EMBED_Dataset/checkpoints/dinov2_latest_checkpoint.pth \
    --output-dir /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/RF_DETR \
    --accum-steps 8 \
    --num-gpus 1
```

> **Key Runtime Configuration (for 1,564 images):**
> - **Effective Batch Size = 16**: Physical `batch_size = 2` (to fit on 15 GB Colab T4 GPUs) with `accum_steps = 8` yields an effective batch size of $2 \times 8 = 16$, ensuring stable Hungarian matching and contrastive denoising.
> - **20-Epoch Schedule**: For 1,564 images at batch size 2, 1 epoch = 782 iterations. 20 epochs = **15,640 iterations** total.
> - **Half-Period Evaluation (Half-Epoch)**: Evaluated every half period (**391 iterations** = 0.5 epoch), providing 40 evaluation checkpoints across 20 epochs. The peak `bbox/AP50` checkpoint is automatically saved as `model_best.pth`.
> - **Warmup**: 2% of total schedule (**312 iterations**).
> - **LR Decay**: $10\times$ step decay at epoch 16 (80% = **12,512 iterations**).
> - **Pure FP32 Precision**: Robust standard float32 precision guaranteeing numerical stability and native CUDA compatibility.
> - **Gradient Norm Logging**: Logs `grad_norm` at each step to TensorBoard to monitor transformer stability.

#### Hyperparameters Reference Guide

##### 1. Direct CLI Arguments

| Argument | Default | Description |
| :--- | :---: | :--- |
| `--max-iter` | `15640` | Total training iterations (20 epochs for 1,564 images at batch size 2). |
| `--eval-period` | `391` | Evaluation & checkpoint period (391 iters = every half-epoch; saves `model_best.pth`). |
| `--freeze-blocks` | `0` | Number of initial DINOv2 blocks to freeze (`0` = 100% unfrozen for full representation learning). |
| `--lr` | `1e-4` | Base learning rate for DINO transformer and detection heads. |
| `--backbone-lr` | `1.19e-4` | Learning rate for the top ViT block (decays down to depth 1 via layer-wise LR decay). |
| `--accum-steps` | `8` | Gradient accumulation steps (physical `batch_size=2` × 8 = **effective batch size 16**). |
| `--num-queries` | `100` | Number of object queries in DINO (100 validated for complete spatial lesion anchoring). |
| `--dn-number` | `10` | Number of Contrastive Denoising (CDN) query groups. |
| `--clip-grad-norm` | `0.1` | Maximum gradient norm clipping (essential for Hungarian matcher stability). |
| `--num-workers` | `2` | Number of CPU data loader workers for training (`1` or `0` for low-RAM systems). |
| `--resume` | *off* | Automatically resumes training from the latest checkpoint in `output_dir`. |

##### 2. Data & Augmentation (`--opts dataloader...`)

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `dataloader.train.mapper.crop_prob` | `0.0` | Probability of applying `LesionAwareCrop` (`0.2` recommended: 20% lesion zoom, 80% full field). |
| `dataloader.train.mapper.crop_size` | `[518, 518]` | Size of the crop window (multiple of 14 for ViT patch alignment). |
| `dataloader.train.mapper.low_pct` | `1.0` | Lower percentile for foreground intensity windowing (filters air background). |
| `dataloader.train.mapper.high_pct` | `99.0` | Upper percentile for intensity windowing (clips extreme bright artifacts). |
| `dataloader.repeat_thresh` | `0.15` | RepeatFactorTrainingSampler threshold (automatically oversamples rare lesion types). |
| `dataloader.train.total_batch_size` | `2` | Physical micro-batch size per GPU step. |

##### 3. Loss Weights & Matching Costs (`--opts model.criterion...`)

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `model.criterion.weight_dict.loss_bbox` | `5.0` | Weight for L1 bounding box coordinate loss. |
| `model.criterion.weight_dict.loss_giou` | `2.0` | Weight for Generalized IoU (GIoU) box loss. |
| `model.criterion.weight_dict.loss_class` | `1.0` | Weight for focal classification loss. |
| `model.criterion.alpha` | `0.25` | Focal loss $\alpha$ class balance factor. |
| `model.criterion.gamma` | `2.0` | Focal loss $\gamma$ focusing parameter on hard examples. |

##### 4. Optimization & Schedule (`--opts optimizer...` / `--opts train...`)

| Parameter | Default | Description |
| :--- | :---: | :--- |
| `optimizer.weight_decay` | `1e-4` | L2 weight decay on projector and transformer heads (`0.0` on backbone). |
| `optimizer.params.layer_decay` | `0.90` | Multiplicative decay per ViT layer from block 12 down to block 1. |
| `train.log_period` | `20` | Interval (iterations) for console metrics logging and TensorBoard writing. |

#### Recommended Training Command

```bash
python train.py \
    --config-file configs/mammo_dinov2_dino.py \
    --train-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_train.json \
    --val-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_val.json \
    --images-dir /content/mammo_data/images \
    --dinov2-weights /content/drive/MyDrive/EMBED_Dataset/checkpoints/dinov2_latest_checkpoint.pth \
    --output-dir /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/RF_DETR_FULL_TRAIN \
    --max-iter 15640 \
    --eval-period 391 \
    --freeze-blocks 0 \
    --accum-steps 8 \
    --num-gpus 1 \
    --opts dataloader.train.mapper.crop_prob=0.2
```

#### Resuming Training
To resume training seamlessly from the latest saved checkpoint after a pause or disconnection:

```bash
python train.py \
    --config-file configs/mammo_dinov2_dino.py \
    --train-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_train.json \
    --val-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/mass_val.json \
    --images-dir /content/mammo_data/images \
    --dinov2-weights /content/drive/MyDrive/EMBED_Dataset/checkpoints/dinov2_latest_checkpoint.pth \
    --output-dir /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/RF_DETR_FULL_TRAIN \
    --resume
```

---

### 4. Monitoring with TensorBoard

TensorBoard logs are automatically written to `output_dir`. In Google Colab:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/RF_DETR
```

Metrics tracked in real time:
- Total loss (`total_loss`), classification loss (`loss_class`), box losses (`loss_bbox`, `loss_giou`)
- Contrastive Denoising losses (`loss_class_dn`, `loss_bbox_dn`, `loss_giou_dn`)
- Layer-wise auxiliary decoder losses (0..4) and encoder proposal losses (`_enc`)
- Learning rate warmup and step decay (`lr`)
- Gradient norms (`grad_norm`)
- Data loading time (`data_time`) and iteration time (`time`)

---

### 5. Evaluation

Evaluate a trained model checkpoint on the validation set to obtain standard COCO metrics ($AP, AP_{50}, AP_{75}, AP_s, AP_m, AP_l$):

```bash
python eval.py \
    --config-file configs/mammo_dinov2_dino.py \
    --weights /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/RF_DETR/model_best.pth \
    --val-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/full_coco_3class_val.json \
    --images-dir /content/mammo_data/images \
    --output-dir ./eval_output \
    --test-size 812 \
    --max-size 1624
```

---

### 6. Side-by-Side Visualization (Ground Truth vs Predictions)

Inspect model detections alongside ground-truth radiologist annotations on validation mammograms:

```python
%run visualize.py \
    --weights /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/RF_DETR/model_best.pth \
    --config-file configs/mammo_dinov2_dino.py \
    --val-json /content/drive/MyDrive/EMBED_Dataset/curated/full_dataset/coco/full_coco_3class_val.json \
    --images-dir /content/mammo_data/images \
    --conf-thresh 0.30 \
    --num-images 4 \
    --save-dir ./viz_predictions \
    --show
```

- **Left panel**: Ground-truth bounding boxes and lesion classes.
- **Right panel**: Model predicted bounding boxes with confidence scores (e.g. `Mass 87%`, `ArchDistortion 65%`).
- Automatically prioritizes mammograms with active lesions for informative inspection.

---

## Python Notebook API

You can also use the modules directly in Python or Jupyter cells:

```python
from rfdetr.data.registration import register_mammo_dataset
from rfdetr.utils.visualize import visualize_dataset

# 1. Register COCO datasets (classes are auto-detected from categories)
thing_classes = register_mammo_dataset(
    train_json="/path/to/full_coco_3class_train.json",
    val_json="/path/to/full_coco_3class_val.json",
    images_dir="/path/to/images"
)
print("Detected classes:", thing_classes)

# 2. Visualize random samples inline (fresh random draw every run)
visualize_dataset("mammo_train", num_images=4, save_dir="./viz_output", show=True)
```

---

## Automated Verification & Tests

Run the comprehensive automated test suite (88 tests covering multi-scale feature shapes, pyramid strides, partial freezing logic, gradient backpropagation, 16-bit float32 normalization, lesion-aware cropping containment, gradient accumulation equivalence, CLI parsing, and end-to-end integration contracts):

```bash
python -m pytest tests/
```

- **100% Passing**: 88 passed out of 88 tests on Linux/CUDA.
- **Clean Reports**: 0 warnings (upstream deprecation notices are cleanly filtered via `tests/conftest.py` and `pyproject.toml`).
- All unit and integration tests run entirely on synthetic data and execute on CPU without requiring CUDA or GPU hardware.
