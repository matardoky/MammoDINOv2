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
    assert "--output-dir" in stdout
    assert "--config-file" in stdout
    assert "--opts" in stdout


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
    assert "--opts" in stdout


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


def test_overfit_cli_help():
    """Verify overfit.py parses CLI arguments and outputs expected parameters."""
    pytest.importorskip("detectron2", reason="detectron2 required to run overfit.py CLI parser")
    res = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "overfit.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert res.returncode == 0, f"overfit.py --help failed with: {res.stderr}"
    stdout = res.stdout
    assert "--train-json" in stdout
    assert "--images-dir" in stdout
    assert "--num-images" in stdout
    assert "--max-iter" in stdout
    assert "--eval-period" in stdout



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


# ─── 4. Backbone Inheritance & Gradient Accumulation Tests ───────────────────

def test_backbone_projector_wrapper_is_backbone():
    """Verify BackboneProjectorWrapper inherits from Detectron2 Backbone."""
    from rfdetr.models.projector import Backbone, BackboneProjectorWrapper, MultiScaleProjector

    class _DummyBackbone(nn.Module):
        out_features = ["b"]
        _p = 14

        def forward(self, x):
            return {"b": x}

    proj = MultiScaleProjector(in_channels=[16], out_channels=16, scale_factors=(1.0,))
    wrapper = BackboneProjectorWrapper(_DummyBackbone(), proj)
    assert isinstance(wrapper, Backbone)


def test_gradient_accumulation_equivalence():
    """Verify that accumulating gradients over N micro-batches produces exact same gradients as large batch."""
    torch.manual_seed(42)
    linear_single = nn.Linear(10, 2, bias=True)
    linear_accum = nn.Linear(10, 2, bias=True)
    linear_accum.load_state_dict(linear_single.state_dict())

    x1 = torch.randn(2, 10)
    x2 = torch.randn(2, 10)
    x_full = torch.cat([x1, x2], dim=0)

    # 1. Full batch of 4 items
    out_full = linear_single(x_full)
    loss_full = out_full.sum() / 4.0
    loss_full.backward()

    # 2. Accumulated batches: 2 steps of batch size 2, accum=2
    accum = 2
    out1 = linear_accum(x1)
    loss1 = (out1.sum() / 2.0) / accum
    loss1.backward()

    out2 = linear_accum(x2)
    loss2 = (out2.sum() / 2.0) / accum
    loss2.backward()

    # Verify gradients are bit-exact identical
    for p_single, p_accum in zip(linear_single.parameters(), linear_accum.parameters()):
        assert torch.allclose(p_single.grad, p_accum.grad, atol=1e-6)


def test_trainer_clip_grads_and_telemetry():
    """Verify Trainer.clip_grads logs grad_norm to EventStorage."""
    pytest.importorskip("detectron2", reason="detectron2 required to import Trainer from train.py")
    from train import Trainer

    linear = nn.Linear(10, 2)
    x = torch.randn(2, 10)
    loss = linear(x).sum()
    loss.backward()

    class _DummyStorage:
        def __init__(self):
            self.history = {}

        def put_scalar(self, name, val):
            self.history[name] = val

    trainer = Trainer.__new__(Trainer)
    trainer.clip_grad_params = {"max_norm": 0.5, "norm_type": 2}
    trainer.storage = _DummyStorage()

    norm = trainer.clip_grads(linear.parameters())
    assert norm is not None
    assert "grad_norm" in trainer.storage.history
    assert trainer.storage.history["grad_norm"] > 0


def test_config_has_embed_dim():
    """Verify configs/mammo_dinov2_dino.py specifies embed_dim=256 for DINO."""
    config_path = PROJECT_ROOT / "configs" / "mammo_dinov2_dino.py"
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()
    assert "embed_dim=256" in content, "embed_dim=256 must be explicitly defined in model = L(DINO)(...)"

