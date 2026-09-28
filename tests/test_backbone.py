"""Automated unit and empirical stress tests for DINOv2MultiScaleBackbone.
"""

import tempfile
import pytest
import torch

from rfdetr.models.backbone import DINOv2MultiScaleBackbone


@pytest.fixture(scope="module")
def default_backbone():
    return DINOv2MultiScaleBackbone(pretrained=False, freeze_blocks=2)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("h, w", [
    (196, 196),
    (224, 224),
    (252, 252),
    (224, 280),
    (140, 280),
    (336, 336),
])
def test_backbone_variable_resolutions_and_batches(default_backbone, batch_size, h, w):
    """Verify DINOv2 multi-scale feature maps across diverse resolutions and batch sizes."""
    x = torch.randn(batch_size, 3, h, w)
    out = default_backbone(x)

    expected_keys = ["block3", "block6", "block9", "block12"]
    assert list(out.keys()) == expected_keys

    expected_h = h // 14
    expected_w = w // 14
    for key, feat in out.items():
        assert feat.shape == (batch_size, 384, expected_h, expected_w), (
            f"Mismatch for {key}: expected {(batch_size, 384, expected_h, expected_w)}, got {feat.shape}"
        )
        assert feat.dtype == torch.float32
        assert torch.isfinite(feat).all(), f"Non-finite values detected in {key}"


def test_backbone_output_shape_metadata(default_backbone):
    """Verify output_shape ShapeSpec metadata and size_divisibility."""
    shapes = default_backbone.output_shape()
    for name in ["block3", "block6", "block9", "block12"]:
        assert name in shapes
        spec = shapes[name]
        assert spec.channels == 384
        assert spec.stride == 14
    assert default_backbone.size_divisibility == 56


@pytest.mark.parametrize("freeze_blocks, expected_frozen, expected_unfrozen", [
    (0, 0, 12),
    (2, 2, 10),
    (6, 6, 6),
    (12, 12, 0),
])
def test_backbone_freezing_logic(freeze_blocks, expected_frozen, expected_unfrozen):
    """Verify parameter freeze states for different freeze_blocks configurations."""
    backbone = DINOv2MultiScaleBackbone(pretrained=False, freeze_blocks=freeze_blocks)
    
    frozen_count = 0
    unfrozen_count = 0
    for i, blk in enumerate(backbone.vit.blocks):
        if i < freeze_blocks:
            assert all(not p.requires_grad for p in blk.parameters()), f"Block {i} should be frozen"
            frozen_count += 1
        else:
            assert all(p.requires_grad for p in blk.parameters()), f"Block {i} should be unfrozen"
            unfrozen_count += 1

    assert frozen_count == expected_frozen
    assert unfrozen_count == expected_unfrozen


def test_backbone_invalid_freeze_blocks():
    """Verify ValueError is raised on out-of-range freeze_blocks."""
    with pytest.raises(ValueError):
        DINOv2MultiScaleBackbone(pretrained=False, freeze_blocks=-1)
    with pytest.raises(ValueError):
        DINOv2MultiScaleBackbone(pretrained=False, freeze_blocks=13)


@pytest.mark.parametrize("freeze_blocks", [0, 2, 4, 12])
def test_backbone_gradient_backpropagation(freeze_blocks):
    """Verify that gradients flow to unfrozen blocks while frozen blocks receive none."""
    backbone = DINOv2MultiScaleBackbone(pretrained=False, freeze_blocks=freeze_blocks)
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    out = backbone(x)
    loss = sum(v.sum() for v in out.values())
    loss.backward()

    for i, blk in enumerate(backbone.vit.blocks):
        params = list(blk.parameters())
        if i < freeze_blocks:
            assert all(p.grad is None for p in params), f"Frozen block {i} unexpectedly received gradient"
        else:
            assert all(p.grad is not None and (p.grad.abs().sum() > 0) for p in params), (
                f"Unfrozen block {i} failed to receive non-zero gradient"
            )


def test_backbone_teacher_loading_validation():
    """Verify error handling and state loading of teacher checkpoint loader."""
    # 1. Non-existent file
    with pytest.raises(FileNotFoundError):
        DINOv2MultiScaleBackbone(pretrained=False, checkpoint_path="non_existent_model.pth")

    # 2. Too few tensors in checkpoint (< 100)
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        ckpt_path = f.name
        torch.save({"dummy_weight": torch.randn(10, 10)}, ckpt_path)

    with pytest.raises(RuntimeError) as exc_info:
        DINOv2MultiScaleBackbone(pretrained=False, checkpoint_path=ckpt_path)
    assert "Too few tensors loaded" in str(exc_info.value)

    # 3. Compatible checkpoint with >= 100 tensors
    base_model = DINOv2MultiScaleBackbone(pretrained=False)
    sd = {"teacher_backbone." + k: v for k, v in base_model.vit.state_dict().items()}
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        valid_ckpt_path = f.name
        torch.save({"model_state_dict": sd}, valid_ckpt_path)

    loaded_model = DINOv2MultiScaleBackbone(pretrained=False, checkpoint_path=valid_ckpt_path)
    assert loaded_model is not None


def test_backbone_layernorm_intermediate_features():
    """Verify that forward_intermediates extracts LayerNorm-normalized features (norm=True)."""
    backbone = DINOv2MultiScaleBackbone(pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        feats = backbone(x)

    for name, feat in feats.items():
        # Normalized LayerNorm features across channel dimension have mean ~0 and std ~1
        # Mean across channel dimension (dim=1) should be very close to 0
        ch_mean = feat.mean(dim=1)
        assert torch.allclose(ch_mean, torch.zeros_like(ch_mean), atol=1e-3), (
            f"{name}: feature map is not LayerNorm normalized along channel axis"
        )

