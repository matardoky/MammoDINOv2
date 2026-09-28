"""configs/mammo_dinov2_dino.py

LazyConfig for mammography lesion detection with DINOv2 + RF-DETR Projector + Detrex DINO.

Architecture & Inheritance:
  - Follows the official Detrex design pattern demonstrated in dino_vitdet.py
    (https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/models/dino_vitdet.py):
    inherits the canonical model definition directly from dino_r50.py
    (https://github.com/IDEA-Research/detrex/blob/main/projects/dino/configs/models/dino_r50.py).
  - Replaces model.backbone with DINOv2MultiScaleBackbone + MultiScaleProjector (Roboflow RF-DETR).
  - Projector uses layer_norm=True (ConvNeXt-style LayerNorm2d) matching official Roboflow RF-DETR,
    completely eliminating BatchNorm2d instability on small batch sizes (batch_size=1 or 2).
  - Adapts model.neck (ChannelMapper) input_shapes to match the 4-level 256-channel projector output.
  - Dynamically synchronizes two_stage_num_proposals with num_queries and criterion.num_classes
    with num_classes via OmegaConf references defined in dino_r50.py.
  - Pure FP32 training with activation checkpointing (use_checkpoint=True) and gradient checkpointing.
"""

from __future__ import annotations

import copy
import importlib.util
import os
import sys
import torch
import torch.nn as nn
from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2.config import LazyCall as L
from detectron2.data import (
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.evaluation import COCOEvaluator
from detectron2.layers import ShapeSpec
from detectron2.solver import WarmupParamScheduler
from detectron2.solver.build import get_default_optimizer_params
from fvcore.common.param_scheduler import MultiStepParamScheduler

from rfdetr.data.mapper import Mammo16BitMapper
from rfdetr.models.backbone import DINOv2MultiScaleBackbone
from rfdetr.models.projector import BackboneProjectorWrapper, MultiScaleProjector


# ─── num_classes placeholder ──────────────────────────────────────────────────
# Sentinel value — auto-detected and injected at runtime by train.py / overfit.py / eval.py
# from the dataset's COCO JSON categories.
_NUM_CLASSES = -1


# ─── Base Model Loading (dino_vitdet.py pattern) ──────────────────────────────
def _load_dino_r50_model():
    """Load canonical Detrex DINO R50 model definition.

    Priority:
      1. Detrex repository clone on Colab: /content/detrex/projects/dino/configs/models/dino_r50.py
      2. Environment variable DETREX_DIR if set
      3. Local mirror in configs/models/dino_r50.py
    """
    detrex_env = os.environ.get("DETREX_DIR", "")
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        "/content/detrex/projects/dino/configs/models/dino_r50.py",
        os.path.join(detrex_env, "projects", "dino", "configs", "models", "dino_r50.py") if detrex_env else "",
        os.path.join(cur_dir, "models", "dino_r50.py"),
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("_dino_r50_base", path)
            if spec is not None and spec.loader is not None:
                mod = importlib.util.module_from_spec(spec)
                sys.modules["_dino_r50_base"] = mod
                spec.loader.exec_module(mod)
                if hasattr(mod, "model"):
                    return copy.deepcopy(mod.model)

    try:
        from .models.dino_r50 import model as _m
        return copy.deepcopy(_m)
    except Exception:
        from configs.models.dino_r50 import model as _m
        return copy.deepcopy(_m)


# Inherit canonical Detrex DINO model
model = _load_dino_r50_model()


# ─── 1. Override Backbone (DINOv2 + Roboflow RF-DETR Projector) ──────────────
model.backbone = L(BackboneProjectorWrapper)(
    backbone=L(DINOv2MultiScaleBackbone)(
        checkpoint_path=None,          # set via train.py --dinov2-weights
        model_name="vit_small_patch14_dinov2.lvd142m",
        pretrained=False,
        freeze_blocks=2,               # freeze first 2 blocks for memory efficiency; 10 learnable
        grad_checkpointing=True,       # memory optimization for ViT
        out_features=["block3", "block6", "block9", "block12"],
    ),
    projector=L(MultiScaleProjector)(
        in_channels=[384, 384, 384, 384],
        out_channels=256,
        scale_factors=(2.0, 1.0, 0.5, 0.25),
        num_blocks=3,
        layer_norm=True,               # Roboflow RF-DETR standard: LayerNorm2d eliminates BatchNorm2d issues
        survival_prob=1.0,
        force_drop_last_n_features=0,
    ),
)


# ─── 2. Adapt Neck (ChannelMapper) for 4-Level Projector Output ──────────────
model.neck.input_shapes = {
    "p2": ShapeSpec(channels=256, stride=7),
    "p3": ShapeSpec(channels=256, stride=14),
    "p4": ShapeSpec(channels=256, stride=28),
    "p5": ShapeSpec(channels=256, stride=56),
}
model.neck.in_features = ["p2", "p3", "p4", "p5"]
model.neck.num_outs = 4
model.neck.norm_layer = L(nn.GroupNorm)(num_groups=32, num_channels=256)


# ─── 3. Transformer Memory & Feature Levels ──────────────────────────────────
model.transformer.num_feature_levels = 4
model.transformer.encoder.use_checkpoint = True
model.transformer.decoder.use_checkpoint = True


# ─── 4. Mammography Lesion Detection Hyperparameters ────────────────────────
model.embed_dim = 256                  # embed_dim=256 latent dimension
model.num_classes = _NUM_CLASSES       # auto-patched at runtime from dataset JSON
model.num_queries = 50                 # dynamically propagates to two_stage_num_proposals via dino_r50.py
model.dn_number = 6
model.pixel_mean = [0.3192, 0.3192, 0.3192]
model.pixel_std = [0.2603, 0.2603, 0.2603]
model.select_box_nums_for_evaluation = model.num_queries

# Ensure aux_loss weight dictionary is fully populated
if getattr(model, "aux_loss", True):
    base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items() if not k.endswith("_enc") and not any(k.endswith(f"_{i}") for i in range(10))})
    dec_layers = getattr(getattr(getattr(model, "transformer", None), "decoder", None), "num_layers", 6)
    if isinstance(dec_layers, int):
        for i in range(dec_layers - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items() if not k.endswith("_enc") and not any(k.endswith(f"_{j}") for j in range(10))})
    base_weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = base_weight_dict


# ─── Data Loaders ─────────────────────────────────────────────────────────────

dataloader = OmegaConf.create()

# Threshold for RepeatFactorTrainingSampler (RFS)
# Automatically over-samples categories with frequency < 15% (e.g. ArchDistortion ~7.2%)
dataloader.repeat_thresh = 0.15

dataloader.train = L(build_detection_train_loader)(
    dataset=L(get_detection_dataset_dicts)(names="mammo_train"),
    sampler=L(RepeatFactorTrainingSampler)(
        repeat_factors=L(RepeatFactorTrainingSampler.repeat_factors_from_category_frequency)(
            dataset_dicts="${dataloader.train.dataset}",
            repeat_thresh="${dataloader.repeat_thresh}",
        ),
    ),
    mapper=L(Mammo16BitMapper)(
        augmentation=[
            L(T.RandomFlip)(prob=0.5, horizontal=True, vertical=False),
            L(T.ResizeShortestEdge)(
                short_edge_length=(644, 672, 700, 728, 756, 784, 812),
                max_size=1624,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=[
            L(T.RandomFlip)(prob=0.5, horizontal=True, vertical=False),
            L(T.ResizeShortestEdge)(
                short_edge_length=(644, 672, 700, 728, 756, 784, 812),
                max_size=1624,
                sample_style="choice",
            ),
            L(T.RandomCrop)(
                crop_type="absolute",
                crop_size=(518, 518),
            ),
        ],
        crop_prob=0.0,
        crop_size=(518, 518),
        is_train=True,
        mask_on=False,
    ),
    total_batch_size=2,
    num_workers=2,
)

dataloader.test = L(build_detection_test_loader)(
    dataset=L(get_detection_dataset_dicts)(names="mammo_val", filter_empty=False),
    mapper=L(Mammo16BitMapper)(
        augmentation=[
            L(T.ResizeShortestEdge)(
                short_edge_length=(812,),
                max_size=1624,
                sample_style="choice",
            ),
        ],
        augmentation_with_crop=None,
        is_train=False,
        mask_on=False,
    ),
    num_workers=2,
)

dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="mammo_val",
    output_dir="${train.output_dir}/eval",
    use_fast_impl=False,
)


# ─── Optimizer (AdamW with backbone LR multiplier) ────────────────────────────

optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        base_lr="${..lr}",
        weight_decay_norm=0.0,
        lr_factor_func=lambda module_name: 1.0 if "projector" in module_name else (0.1 if ("backbone.backbone" in module_name or "backbone.vit" in module_name) else 1.0),
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
)


# ─── LR Scheduler ─────────────────────────────────────────────────────────────

# 20 epochs × 2835 iters/epoch (5669 images / batch_size 2) = 56 700 iters total
_ITERS_PER_EPOCH = 2835    # ceil(5669 / 2)
_MAX_ITER        = 56_700  # 20 epochs
_WARMUP_ITERS    = int(_MAX_ITER * 0.02)   # 2 % → 1 134 iters
_LR_DECAY_ITER   = 16 * _ITERS_PER_EPOCH  # epoch 16 → 45 360

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[_LR_DECAY_ITER, _MAX_ITER],
    ),
    warmup_length=_WARMUP_ITERS / _MAX_ITER,  # 0.02  (2 %)
    warmup_method="linear",
    warmup_factor=0.001,
)


# ─── Training Runtime Configuration (FP32 Pur, No AMP) ───────────────────────

train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter=_MAX_ITER,
    eval_period=_ITERS_PER_EPOCH,
    log_period=20,
    device="cuda",
    grad_accum_steps=8,  # Effective batch size = total_batch_size (2) × 8 = 16
    checkpointer=dict(period=_ITERS_PER_EPOCH, max_to_keep=5),
    clip_grad=dict(
        enabled=True,
        params=dict(
            max_norm=1.0,
            norm_type=2,
        ),
    ),
    ddp=dict(
        broadcast_buffers=False,
        find_unused_parameters=False,
        fp16_compression=False,
    ),
    model_ema=dict(
        enabled=False,
    ),
)
