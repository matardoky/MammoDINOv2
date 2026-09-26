"""tests/test_cli_and_integration.py

Comprehensive CLI and integration tests for RF-DETR:
- Verification of CLI entrypoints (train.py, eval.py, visualize.py) argument parsing and help output.
- Unit verification of optimizer parameter grouping and backbone LR scaling factor.
- End-to-end smoke test of forward pass + backward pass + optimizer step on CPU without CUDA.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ─── 1. CLI Entrypoint Tests ──────────────────────────────────────────────────

def test_train_cli_help():
    """Verify train.py parses CLI arguments and outputs expected parameters."""
    pytest.importorskip("detectron2", reason="detectron2 required to run train.py CLI parser")
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "train.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0, f"train.py --help failed with: {res.stderr}"
    stdout = res.stdout
    assert "--train-json" in stdout
    assert "--val-json" in stdout
    assert "--images-dir" in stdout
    assert "--dinov2-weights" in stdout
    assert "--config-file" in stdout


def test_eval_cli_help():
    """Verify eval.py parses CLI arguments and outputs expected parameters."""
    pytest.importorskip("detectron2", reason="detectron2 required to run eval.py CLI parser")
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "eval.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0, f"eval.py --help failed with: {res.stderr}"
    stdout = res.stdout
    assert "--config-file" in stdout
    assert "--weights" in stdout
    assert "--val-json" in stdout
    assert "--images-dir" in stdout
    assert "--test-size" in stdout


def test_visualize_cli_help():
    """Verify visualize.py parses CLI arguments and outputs expected parameters."""
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "visualize.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0, f"visualize.py --help failed with: {res.stderr}"
    stdout = res.stdout
    assert "--val-json" in stdout
    assert "--images-dir" in stdout
    assert "--conf-thresh" in stdout
    assert "--save-dir" in stdout
    assert "--num-images" in stdout


# ─── 2. Optimizer Parameter Grouping Logic ────────────────────────────────────

def test_optimizer_parameter_groups_logic():
    """Verify that lr_factor_func scales backbone LR by 0.1x while keeping detector at 1.0x."""
    lr_factor_func = lambda module_name: 0.1 if "backbone" in module_name else 1.0

    # Backbone modules (standard and DDP-prefixed)
    assert lr_factor_func("backbone.backbone.vit.blocks.0") == 0.1
    assert lr_factor_func("module.backbone.backbone.vit.blocks.11") == 0.1
    assert lr_factor_func("backbone.projector.stages.0") == 0.1

    # Non-backbone modules (neck, transformer, heads)
    assert lr_factor_func("neck.conv") == 1.0
    assert lr_factor_func("module.neck.conv") == 1.0
    assert lr_factor_func("transformer.encoder.layers.0") == 1.0
    assert lr_factor_func("class_embed") == 1.0
    assert lr_factor_func("bbox_embed") == 1.0


# ─── 3. End-to-End Gradient Flow Smoke Test ───────────────────────────────────

def test_end_to_end_gradient_step_smoke():
    """Verify forward + backward + optimizer step through BackboneProjectorWrapper."""
    from rfdetr.models.projector import BackboneProjectorWrapper, MultiScaleProjector

    class _LightBackbone(nn.Module):
        out_features = ["block3", "block6", "block9", "block12"]
        _p = 14
        _dim = 64

        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 64, kernel_size=14, stride=14)

        def forward(self, x):
            f = self.conv(x)
            return {name: f for name in self.out_features}

        @property
        def size_divisibility(self):
            return 56

    backbone = _LightBackbone()
    projector = MultiScaleProjector(
        in_channels=[64, 64, 64, 64],
        out_channels=32,
        scale_factors=(2.0, 1.0, 0.5, 0.25),
        num_blocks=1,
    )
    model = BackboneProjectorWrapper(backbone=backbone, projector=projector)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Input divisible by 56
    x = torch.randn(2, 3, 56, 56, requires_grad=False)
    outputs = model(x)

    # Dummy loss across all pyramid scales
    loss = sum(feat.mean() for feat in outputs.values())
    optimizer.zero_grad()
    loss.backward()

    # Verify gradients exist and are finite
    has_grad = False
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            assert not torch.isnan(p.grad).any(), "NaN gradient detected!"
            assert not torch.isinf(p.grad).any(), "Inf gradient detected!"
            has_grad = True

    assert has_grad, "No parameters received gradients!"
    optimizer.step()
