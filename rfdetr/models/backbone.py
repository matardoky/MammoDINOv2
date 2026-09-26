"""rfdetr.models.backbone

Multi-scale DINOv2 vision transformer backbone extracting 4 intermediate
feature levels at uniform spatial stride 14.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import timm
import torch
import torch.nn as nn

try:
    from detectron2.modeling.backbone import Backbone
    from detectron2.layers import ShapeSpec
except ImportError:
    # Lightweight fallback for environments without detectron2 (local testing, CI)
    from dataclasses import dataclass
    from typing import Optional as _Optional

    class Backbone(nn.Module):  # type: ignore[no-redef]
        """Minimal Backbone stub when detectron2 is not installed."""
        def output_shape(self):
            return {}
        @property
        def size_divisibility(self) -> int:
            return 0

    @dataclass
    class ShapeSpec:  # type: ignore[no-redef]
        channels: _Optional[int] = None
        height: _Optional[int] = None
        width: _Optional[int] = None
        stride: _Optional[int] = None

logger = logging.getLogger(__name__)


class DINOv2MultiScaleBackbone(Backbone):
    """DINOv2 ViT-S/14 backbone with multi-layer intermediate feature extraction.

    Extracts features from 4 transformer blocks at indices (2, 5, 8, 11),
    corresponding to blocks 3, 6, 9, and 12. Each extracted level has 384
    channels and spatial stride 14.

    Args:
        checkpoint_path: Optional path to a pretrained teacher checkpoint (.pth).
        model_name: timm model architecture name (default: "vit_small_patch14_dinov2.lvd142m").
        pretrained: Whether to load timm's default pretrained weights (default: False).
        freeze_blocks: Number of initial transformer blocks to freeze (0 to 12, default: 2).
        out_features: Names of intermediate output feature keys.
    """

    BLOCK_INDICES: Tuple[int, ...] = (2, 5, 8, 11)

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_name: str = "vit_small_patch14_dinov2.lvd142m",
        pretrained: bool = False,
        freeze_blocks: int = 2,
        out_features: Sequence[str] = ("block3", "block6", "block9", "block12"),
    ) -> None:
        super().__init__()
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            dynamic_img_size=True,
            dynamic_img_pad=True,
            num_classes=0,
        )
        self._p: int = int(self.vit.patch_embed.patch_size[0])
        self._dim: int = int(self.vit.embed_dim)
        self.n_blocks: int = len(self.vit.blocks)
        self.out_features: List[str] = list(out_features)
        self._out_channels: Dict[str, int] = {n: self._dim for n in self.out_features}
        self._out_strides: Dict[str, int] = {n: self._p for n in self.out_features}

        if checkpoint_path is not None:
            self._load_teacher(checkpoint_path)

        if not (0 <= freeze_blocks <= self.n_blocks):
            raise ValueError(f"freeze_blocks must be in range [0, {self.n_blocks}], got {freeze_blocks}")

        self.freeze_blocks: int = int(freeze_blocks)
        self._apply_partial_freeze()

    def _load_teacher(self, path: str) -> None:
        """Load pretrained teacher weights with prefix cleanup and validation.

        Args:
            path: Path to checkpoint .pth file.

        Raises:
            FileNotFoundError: If checkpoint file does not exist.
            RuntimeError: If fewer than 100 tensors successfully match.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")

        if "teacher_backbone" in ckpt:
            sd = ckpt["teacher_backbone"]
        elif "model_state_dict" in ckpt:
            sd = {
                k: v for k, v in ckpt["model_state_dict"].items()
                if k.startswith("teacher_backbone.")
            }
        else:
            sd = {k: v for k, v in ckpt.items() if "student" not in k.lower()}

        prefixes = ("teacher_backbone.vit.", "teacher_backbone.", "vit.", "backbone.")
        clean = {}
        for k, v in sd.items():
            for pre in prefixes:
                if k.startswith(pre):
                    k = k[len(pre):]
                    break
            if not k.startswith(("head", "dino_head", "ibot_head")):
                clean[k] = v
        clean = {k: v for k, v in clean.items() if k != "mask_token"}

        missing, unexpected = self.vit.load_state_dict(clean, strict=False)
        logger.info(
            f"[DINOv2] Loaded {len(clean)} tensors from {path} "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )
        if len(clean) < 100:
            raise RuntimeError(
                f"Too few tensors loaded ({len(clean)}). Checkpoint structure may be incompatible."
            )

    def _apply_partial_freeze(self) -> None:
        """Freeze the first N blocks while keeping upper blocks and LayerNorm trainable."""
        for p in self.vit.parameters():
            p.requires_grad = False
        for i, blk in enumerate(self.vit.blocks):
            if i >= self.freeze_blocks:
                for p in blk.parameters():
                    p.requires_grad = True
        for p in self.vit.norm.parameters():
            p.requires_grad = True

        n_trainable = sum(p.numel() for p in self.vit.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in self.vit.parameters())
        logger.info(
            f"[DINOv2] Frozen blocks: 0..{self.freeze_blocks - 1} / {self.n_blocks} | "
            f"Trainable params: {n_trainable / 1e6:.2f}M / {n_total / 1e6:.2f}M"
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract intermediate feature representations.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            Dictionary mapping out_features keys to tensors of shape (B, 384, H/14, W/14).
        """
        intermediates = self.vit.forward_intermediates(
            x,
            indices=self.BLOCK_INDICES,
            output_fmt="NCHW",
            intermediates_only=True,
        )
        return {
            name: feat.contiguous()
            for name, feat in zip(self.out_features, intermediates)
        }

    def output_shape(self) -> Dict[str, ShapeSpec]:
        """Return ShapeSpec metadata for each output feature level."""
        return {
            name: ShapeSpec(channels=self._out_channels[name], stride=self._out_strides[name])
            for name in self.out_features
        }

    @property
    def size_divisibility(self) -> int:
        """Required input divisibility (14 * 4 = 56)."""
        return self._p * 4
