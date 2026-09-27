"""configs/mammo_dinov2_dino.py

LazyConfig for mammography lesion detection with DINOv2 + RF-DETR Projector + detrex DINO.

Structure:
  - Starts from detrex DINO architecture (HungarianMatcher, DINOCriterion, Transformer, CDN
    are all imported from detrex — NOT reimplemented).
  - Replaces model.backbone with DINOv2MultiScaleBackbone + BackboneProjectorWrapper.
  - Adjusts ChannelMapper input_shapes to match the projector's 4-level 256-channel output.
  - Full Detrex configuration specification:
      * model: DINO detection transformer
      * dataloader: train and test loaders with Mammo16BitMapper (16-bit uint16)
      * optimizer: AdamW with weight decay
      * lr_multiplier: WarmupParamScheduler with MultiStep decay
      * train: training hyperparameters (iterations, eval_period, AMP, clip_grad)

num_classes:
  NOT hardcoded here. Set to -1 as a sentinel value.
  train.py and eval.py automatically read the real value from the COCO JSON
  'categories' field and patch cfg.model.num_classes and cfg.model.criterion.num_classes
  before the model is instantiated.
"""

from __future__ import annotations

import copy
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

from detrex.layers import PositionEmbeddingSine
from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper

from projects.dino.modeling import (
    DINO,
    DINOCriterion,
    DINOTransformer,
    DINOTransformerDecoder,
    DINOTransformerEncoder,
)

from rfdetr.data.mapper import Mammo16BitMapper
from rfdetr.models.backbone import DINOv2MultiScaleBackbone
from rfdetr.models.projector import BackboneProjectorWrapper, MultiScaleProjector


# ─── num_classes placeholder ──────────────────────────────────────────────────
# Sentinel value — DO NOT change this manually.
# train.py / eval.py automatically detect and inject the correct value
# from the COCO JSON 'categories' field before the model is instantiated.
_NUM_CLASSES = -1  # auto-set at runtime from dataset JSON


# ─── Model ───────────────────────────────────────────────────────────────────

model = L(DINO)(
    # ← Only this block is custom; everything else is detrex-native
    backbone=L(BackboneProjectorWrapper)(
        backbone=L(DINOv2MultiScaleBackbone)(
            checkpoint_path=None,          # set via train.py --dinov2-weights
            model_name="vit_small_patch14_dinov2.lvd142m",
            pretrained=False,
            freeze_blocks=8,
            out_features=["block3", "block6", "block9", "block12"],
        ),
        projector=L(MultiScaleProjector)(
            in_channels=[384, 384, 384, 384],
            out_channels=256,
            scale_factors=(2.0, 1.0, 0.5, 0.25),
            num_blocks=3,
            survival_prob=1.0,
            force_drop_last_n_features=0,
        ),
    ),

    # ── Positional encoding (detrex-native) ──────────────────────────────────
    position_embedding=L(PositionEmbeddingSine)(
        num_pos_feats=128,
        temperature=10000,
        normalize=True,
        offset=-0.5,
    ),

    # ── Neck: maps projector output (all 256ch) to DINO latent dim ───────────
    # The projector already outputs 256ch at 4 strides (7, 14, 28, 56).
    # ChannelMapper here provides GroupNorm and the FPN interface expected by DINO.
    neck=L(ChannelMapper)(
        input_shapes={
            "p2": ShapeSpec(channels=256, stride=7),
            "p3": ShapeSpec(channels=256, stride=14),
            "p4": ShapeSpec(channels=256, stride=28),
            "p5": ShapeSpec(channels=256, stride=56),
        },
        in_features=["p2", "p3", "p4", "p5"],
        out_channels=256,
        num_outs=4,
        norm_layer=L(nn.GroupNorm)(num_groups=32, num_channels=256),
    ),

    # ── Transformer (detrex-native) ──────────────────────────────────────────
    transformer=L(DINOTransformer)(
        encoder=L(DINOTransformerEncoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            post_norm=False,
            num_feature_levels=4,
        ),
        decoder=L(DINOTransformerDecoder)(
            embed_dim=256,
            num_heads=8,
            feedforward_dim=2048,
            attn_dropout=0.0,
            ffn_dropout=0.0,
            num_layers=6,
            return_intermediate=True,
            num_feature_levels=4,
        ),
    ),

    # ── Criterion (detrex-native) ─────────────────────────────────────────────
    criterion=L(DINOCriterion)(
        num_classes=_NUM_CLASSES,   # auto-set at runtime from dataset JSON
        matcher=L(HungarianMatcher)(
            cost_class=2.0,
            cost_bbox=5.0,
            cost_giou=2.0,
            cost_class_type="focal_loss_cost",
            alpha=0.25,
            gamma=2.5,  # Focus matching cost on minority / harder lesions (ArchDistortion)
        ),
        weight_dict={
            "loss_class": 1.0,
            "loss_bbox": 5.0,
            "loss_giou": 2.0,
            "loss_class_dn": 1.0,
            "loss_bbox_dn": 5.0,
            "loss_giou_dn": 2.0,
        },
        losses=["class", "boxes"],
        eos_coef=0.1,
        loss_class_type="focal_loss",
        alpha=0.25,
        gamma=2.5,      # Focus gradients on minority / harder lesions (ArchDistortion ~7%)
    ),

    # ── Misc ─────────────────────────────────────────────────────────────────
    embed_dim=256,
    num_classes=_NUM_CLASSES,       # auto-set at runtime from dataset JSON
    num_queries=50,
    dn_number=6,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    # Mammography-specific normalization for float32 [0.0, 1.0] image tensors
    pixel_mean=[0.3192, 0.3192, 0.3192],
    pixel_std=[0.2603, 0.2603, 0.2603],
    device="cuda",
    select_box_nums_for_evaluation=50,
)



# ─── Auxiliary Loss Weights (Detrex DINO standard) ───────────────────────────
# Deep supervision for the 6 decoder layers and the encoder proposal head
base_weight_dict = copy.deepcopy(model.criterion.weight_dict)
if model.get("aux_loss", True):
    weight_dict = copy.deepcopy(base_weight_dict)
    aux_weight_dict = {}
    aux_weight_dict.update({k + "_enc": v for k, v in base_weight_dict.items()})
    for i in range(model.transformer.decoder.num_layers - 1):
        aux_weight_dict.update({k + f"_{i}": v for k, v in base_weight_dict.items()})
    weight_dict.update(aux_weight_dict)
    model.criterion.weight_dict = weight_dict


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
)


# ─── Optimizer (Detrex standard with backbone LR multiplier) ──────────────────

optimizer = L(torch.optim.AdamW)(
    params=L(get_default_optimizer_params)(
        base_lr="${..lr}",
        weight_decay_norm=0.0,
        lr_factor_func=lambda module_name: 0.1 if "backbone" in module_name else 1.0,
    ),
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=1e-4,
)


# ─── LR Scheduler ─────────────────────────────────────────────────────────────

# 7200 total iterations, warmup 500 steps, decay by 10x at 5760 steps (80%)
lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1],
        milestones=[5760, 7200],
    ),
    warmup_length=500 / 7200,
    warmup_method="linear",
    warmup_factor=0.001,
)


# ─── Training Runtime Configuration ──────────────────────────────────────────

train = dict(
    output_dir="./output",
    init_checkpoint="",
    max_iter=7200,
    eval_period=600,
    log_period=20,
    device="cuda",
    amp=dict(
        enabled=False,  # Baseline standard: FP32 (AMP disabled for maximum simplicity and stability)
        dtype="float16",  # Used if AMP is explicitly enabled via --amp
    ),
    grad_accum_steps=8,  # Effective batch size = total_batch_size (2) * 8 = 16
    checkpointer=dict(period=600, max_to_keep=5),
    clip_grad=dict(
        enabled=True,
        params=dict(
            max_norm=0.1,
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
