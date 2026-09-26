"""rfdetr.models

Custom components for mammography lesion detection with DINOv2 + RF-DETR + detrex DINO.

Only what detrex does NOT provide is implemented here:
  - DINOv2MultiScaleBackbone: ViT-S/14 with 4-level intermediate feature extraction
  - RF-DETR projector: fuses 4 uniform-stride DINOv2 features into a feature pyramid

Everything else (DINO transformer, matcher, criterion, CDN, neck, positional encoding,
training loop, LR scheduling, checkpointing, evaluation) comes directly from detrex
and detectron2 — imported, not reimplemented.
"""

from rfdetr.models.backbone import DINOv2MultiScaleBackbone
from rfdetr.models.projector import (
    BackboneProjectorWrapper,
    Bottleneck,
    C2f,
    Conv2d,
    ConvX,
    LayerNorm,
    LayerNorm2d,
    MultiScaleProjector,
)

__all__ = [
    "DINOv2MultiScaleBackbone",
    "LayerNorm2d",
    "LayerNorm",
    "ConvX",
    "Conv2d",
    "Bottleneck",
    "C2f",
    "MultiScaleProjector",
    "BackboneProjectorWrapper",
]
