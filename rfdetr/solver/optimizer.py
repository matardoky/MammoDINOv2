"""rfdetr.solver.optimizer

Layer-wise LR decay optimizer parameter group builder for DINOv2 + Detrex DINO.
Reproduces the exact optimizer configuration from the reference implementation:
- Layer-wise learning rate decay for DINOv2 ViT blocks (depth 12 down to depth 1)
- 0.0 weight decay on backbone and 1D/bias parameters
- Standard weight decay on head/projector 2D weights
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_dinov2_optimizer_params(
    model: nn.Module,
    base_lr: float = 1e-4,
    backbone_lr: float = 1.19e-4,
    layer_decay: float = 0.90,
    weight_decay: float = 1e-4,
    num_layers: int = 12,
    backbone_prefix: str = "backbone.backbone.vit.",
) -> List[Dict[str, Any]]:
    """Build layer-wise parameter groups matching the reference DINOv2 setup.

    Args:
        model: PyTorch model containing the backbone and detection heads.
        base_lr: Learning rate for detection head, projector, and transformer.
        backbone_lr: Maximum learning rate for the top ViT block (depth 12).
        layer_decay: Multiplicative decay factor per ViT layer from top to bottom.
        weight_decay: L2 penalty for 2D weights in head/projector.
        num_layers: Total number of Transformer blocks in ViT (12 for ViT-S).
        backbone_prefix: Name prefix for the ViT backbone module.

    Returns:
        List of parameter group dictionaries compatible with torch.optim.AdamW.
    """
    layer_groups: Dict[int, Dict[str, Any]] = {}
    other_decay: List[nn.Parameter] = []
    other_no_decay: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(backbone_prefix):
            m = re.search(r"blocks\.(\d+)\.", name)
            if m:
                depth = int(m.group(1)) + 1
                lr = backbone_lr * (layer_decay ** (num_layers - depth))
                layer_groups.setdefault(depth, {"params": [], "lr": lr})
                layer_groups[depth]["params"].append(param)
            else:
                # ViT stem (patch_embed, pos_embed) or head norm
                depth = 0
                lr = backbone_lr * (layer_decay ** 7)  # ~5.64e-05
                layer_groups.setdefault(0, {"params": [], "lr": lr})
                layer_groups[0]["params"].append(param)
        else:
            if param.ndim == 1 or name.endswith(".bias"):
                other_no_decay.append(param)
            else:
                other_decay.append(param)

    groups: List[Dict[str, Any]] = []
    for depth in sorted(layer_groups.keys()):
        g = layer_groups[depth]
        groups.append({
            "params": g["params"],
            "lr": g["lr"],
            "weight_decay": 0.0,
            "name": f"backbone_depth_{depth}",
        })
    if other_decay:
        groups.append({
            "params": other_decay,
            "lr": base_lr,
            "weight_decay": weight_decay,
            "name": "head_decay",
        })
    if other_no_decay:
        groups.append({
            "params": other_no_decay,
            "lr": base_lr,
            "weight_decay": 0.0,
            "name": "head_no_decay",
        })

    logger.info("Optimizer Parameter Groups:")
    for g in groups:
        n = sum(p.numel() for p in g["params"])
        logger.info(
            f"   {g['name']:>22s} | lr={g['lr']:.2e} | "
            f"wd={g['weight_decay']:.1e} | {n / 1e6:.2f}M"
        )

    return groups
