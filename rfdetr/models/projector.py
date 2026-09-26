"""rfdetr.models.projector

RF-DETR multi-scale projector adapted for DINOv2 uniform-stride feature maps.
Fuses 4 backbone levels (channels=384, stride=14) into 4 feature pyramid levels
(channels=256, strides=(7, 14, 28, 56)).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from detectron2.layers import ShapeSpec
except ImportError:
    from dataclasses import dataclass
    from typing import Optional as _Optional

    @dataclass
    class ShapeSpec:  # type: ignore[no-redef]
        channels: _Optional[int] = None
        height: _Optional[int] = None
        width: _Optional[int] = None
        stride: _Optional[int] = None


class LayerNorm2d(nn.Module):
    """Channel-first 2D LayerNorm for (B, C, H, W) tensors (ConvNeXt style).

    Normalizes along the channel dimension (dim=1) and applies learnable
    scale and shift parameters.
    """

    def __init__(self, normalized_shape: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


# Backwards compatibility alias
LayerNorm = LayerNorm2d


def get_norm(norm: Optional[Union[str, Callable[[int], nn.Module]]], out_channels: int) -> Optional[nn.Module]:
    """Factory helper to build normalization layers."""
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm_map = {"LN": lambda ch: LayerNorm2d(ch)}
        if norm not in norm_map:
            raise KeyError(f"Unsupported norm string: {norm}")
        return norm_map[norm](out_channels)
    return norm(out_channels)


def get_activation(name: Optional[str], inplace: bool = False) -> nn.Module:
    """Factory helper to build activation layers."""
    if name is None:
        return nn.Identity()
    name_lower = name.lower()
    if name_lower == "silu":
        return nn.SiLU(inplace=inplace)
    if name_lower == "relu":
        return nn.ReLU(inplace=inplace)
    if name_lower in ["leakyrelu", "lrelu"]:
        return nn.LeakyReLU(0.1, inplace=inplace)
    if name_lower == "identity":
        return nn.Identity()
    raise AttributeError(f"Unsupported activation type: {name}")


class ConvX(nn.Module):
    """Standard Convolution + Normalization + Activation block (YOLO/RF-DETR style)."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel: Union[int, Tuple[int, int]] = 3,
        stride: int = 1,
        groups: int = 1,
        dilation: int = 1,
        act: str = "relu",
        layer_norm: bool = False,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(kernel, tuple):
            kernel = (kernel, kernel)
        padding = (kernel[0] // 2, kernel[1] // 2)
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=False,
        )
        if rms_norm:
            self.bn = nn.RMSNorm(out_planes)
        else:
            self.bn = get_norm("LN", out_planes) if layer_norm else nn.BatchNorm2d(out_planes)
        self.act = get_activation(act, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


# Backwards compatibility alias
Conv2d = ConvX


class Bottleneck(nn.Module):
    """Standard 2-convolution residual bottleneck block."""

    def __init__(
        self,
        c1: int,
        c2: int,
        shortcut: bool = True,
        g: int = 1,
        k: Tuple[int, int] = (3, 3),
        e: float = 0.5,
        act: str = "silu",
        layer_norm: bool = False,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = ConvX(c1, c_, k[0], 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX(c_, c2, k[1], 1, groups=g, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    """CSP Bottleneck with 2 convolutions (RF-DETR)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
        act: str = "silu",
        layer_norm: bool = False,
        rms_norm: bool = False,
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = ConvX(c1, 2 * self.c, 1, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.cv2 = ConvX((2 + n) * self.c, c2, 1, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, act=act, layer_norm=layer_norm, rms_norm=rms_norm)
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class MultiScaleProjector(nn.Module):
    """RF-DETR official multi-scale projector adapted for DINOv2 feature maps.

    Fuses 4 backbone levels into 4 multi-scale detection levels.

    Args:
        in_channels: List of input feature channels (default: [384, 384, 384, 384]).
        out_channels: Target latent channel dimension (default: 256).
        scale_factors: Scaling factors for pyramid levels (default: (2.0, 1.0, 0.5, 0.25)).
        num_blocks: Number of Bottleneck blocks in each C2f module (default: 3).
        layer_norm: Whether to use LayerNorm2d in convolutions (default: False).
        rms_norm: Whether to use RMSNorm (default: False).
        survival_prob: Feature survival probability during training (1.0 = disable dropout).
        force_drop_last_n_features: Number of trailing feature maps to force drop for ablation.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (384, 384, 384, 384),
        out_channels: int = 256,
        scale_factors: Sequence[float] = (2.0, 1.0, 0.5, 0.25),
        num_blocks: int = 3,
        layer_norm: bool = False,
        rms_norm: bool = False,
        survival_prob: float = 1.0,
        force_drop_last_n_features: int = 0,
    ) -> None:
        super().__init__()
        self.scale_factors = list(scale_factors)
        self.survival_prob = float(survival_prob)
        self.force_drop_last_n_features = int(force_drop_last_n_features)
        self.use_extra_pool = False

        stages_sampling: List[List[nn.Module]] = []
        stages: List[nn.Module] = []

        for scale in self.scale_factors:
            if scale == 0.25:
                self.use_extra_pool = True
                continue

            current_stage_sampling: List[nn.Module] = []
            for in_dim in in_channels:
                layers: List[nn.Module] = []
                if scale == 4.0:
                    layers.extend([
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                        get_norm("LN", in_dim // 2),
                        nn.GELU(),
                        nn.ConvTranspose2d(in_dim // 2, in_dim // 4, kernel_size=2, stride=2),
                    ])
                elif scale == 2.0:
                    layers.extend([
                        nn.ConvTranspose2d(in_dim, in_dim // 2, kernel_size=2, stride=2),
                    ])
                elif scale == 1.0:
                    pass
                elif scale == 0.5:
                    layers.extend([
                        ConvX(in_dim, in_dim, kernel=3, stride=2, layer_norm=layer_norm),
                    ])
                else:
                    raise NotImplementedError(f"Unsupported scale factor: {scale}")

                current_stage_sampling.append(nn.Sequential(*layers))

            stages_sampling.append(nn.ModuleList(current_stage_sampling))

            fused_in_dim = int(sum(ic // max(1, scale) for ic in in_channels))
            stage_layers = [
                C2f(fused_in_dim, out_channels, num_blocks, layer_norm=layer_norm, rms_norm=rms_norm),
                get_norm("LN", out_channels),
            ]
            stages.append(nn.Sequential(*stage_layers))

        self.stages_sampling = nn.ModuleList(stages_sampling)
        self.stages = nn.ModuleList(stages)

    def forward(self, x: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        """Project multi-scale feature maps.

        Args:
            x: List of 4 feature tensors from backbone, each of shape (B, 384, Hp, Wp).

        Returns:
            List of 4 projected feature tensors:
            - p2: shape (B, 256, 2Hp, 2Wp)
            - p3: shape (B, 256, Hp, Wp)
            - p4: shape (B, 256, Hp/2, Wp/2)
            - p5: shape (B, 256, Hp/4, Wp/4)
        """
        feats = list(x)
        num_features = len(feats)

        if self.survival_prob < 1.0 and self.training:
            final_drop_prob = 1.0 - self.survival_prob
            drop_p = float(torch.rand(1).item())
            for i in range(1, num_features):
                critical_drop_prob = i * (final_drop_prob / (num_features - 1))
                if drop_p < critical_drop_prob:
                    feats[i] = torch.zeros_like(feats[i])
        elif self.force_drop_last_n_features > 0:
            for i in range(self.force_drop_last_n_features):
                idx = -(i + 1)
                feats[idx] = torch.zeros_like(feats[idx])

        results: List[torch.Tensor] = []
        for i, stage in enumerate(self.stages):
            fused_list = [sampling(feats[j]) for j, sampling in enumerate(self.stages_sampling[i])]
            fused = torch.cat(fused_list, dim=1) if len(fused_list) > 1 else fused_list[0]
            results.append(stage(fused))

        if self.use_extra_pool:
            results.append(F.max_pool2d(results[-1], kernel_size=1, stride=2, padding=0))

        return results


class BackboneProjectorWrapper(nn.Module):
    """Composite module encapsulating DINOv2MultiScaleBackbone and MultiScaleProjector.

    Implements output_shape() and size_divisibility interfaces for full Detectron2 compatibility.
    """

    def __init__(self, backbone: nn.Module, projector: MultiScaleProjector) -> None:
        super().__init__()
        self.backbone = backbone
        self.projector = projector

        # Base stride dynamically calculated from backbone patch size
        self._base_stride: int = getattr(backbone, "_p", 14) // 2  # e.g., 14 // 2 = 7

        n_stages = len(projector.stages)
        self.out_names: List[str] = [f"p{i + 2}" for i in range(n_stages)]
        self.out_strides: Dict[str, int] = {
            name: self._base_stride * (2 ** i)
            for i, name in enumerate(self.out_names)
        }
        if projector.use_extra_pool:
            extra_name = f"p{n_stages + 2}"
            self.out_names.append(extra_name)
            self.out_strides[extra_name] = self.out_strides[self.out_names[-2]] * 2

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.backbone(x)
        feat_list = [features[n] for n in self.backbone.out_features]
        pyramid = self.projector(feat_list)
        return {name: feat for name, feat in zip(self.out_names, pyramid)}

    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(channels=256, stride=stride)
            for name, stride in self.out_strides.items()
        }

    @property
    def size_divisibility(self) -> int:
        return self.backbone.size_divisibility
