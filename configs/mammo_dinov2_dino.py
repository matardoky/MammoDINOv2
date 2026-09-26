"""configs/mammo_dinov2_dino.py

LazyConfig for mammography lesion detection with DINOv2 + RF-DETR Projector + detrex DINO.

Structure:
  - Starts from detrex dino_r50.py (HungarianMatcher, DINOCriterion, Transformer, CDN
    are all imported from detrex — NOT reimplemented).
  - Replaces only model.backbone with DINOv2MultiScaleBackbone + BackboneProjectorWrapper.
  - Adjusts ChannelMapper input_shapes to match the projector's 4-level 256-channel output.

num_classes:
  NOT hardcoded here. Set to -1 as a sentinel value.
  train.py and eval.py automatically read the real value from the COCO JSON
  'categories' field and patch cfg.model.num_classes and cfg.model.criterion.num_classes
  before the model is instantiated.
"""

import copy
import torch.nn as nn

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec

from detrex.modeling.matcher import HungarianMatcher
from detrex.modeling.neck import ChannelMapper
from detrex.layers import PositionEmbeddingSine

from projects.dino.modeling import (
    DINO,
    DINOTransformerEncoder,
    DINOTransformerDecoder,
    DINOTransformer,
    DINOCriterion,
)

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
    # ChannelMapper here is identity (in=out=256) but provides the FPN interface
    # expected by the DINO transformer.
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
            gamma=2.0,
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
        gamma=2.0,
    ),

    # ── Misc ─────────────────────────────────────────────────────────────────
    num_classes=_NUM_CLASSES,       # auto-set at runtime from dataset JSON
    num_queries=50,
    dn_number=6,
    label_noise_ratio=0.5,
    box_noise_scale=1.0,
    # Mammography-specific normalization (grayscale replicated to 3 channels)
    pixel_mean=[77.76, 77.76, 77.76],   # 0.3051 * 255
    pixel_std=[67.76, 67.76, 67.76],    # 0.2658 * 255
    device="cuda",
    select_box_nums_for_evaluation=50,
)
