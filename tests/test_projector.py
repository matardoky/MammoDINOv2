"""Automated unit and empirical stress tests for MultiScaleProjector and BackboneProjectorWrapper.
"""

import pytest
import torch

from rfdetr.models.backbone import DINOv2MultiScaleBackbone
from rfdetr.models.projector import BackboneProjectorWrapper, MultiScaleProjector


@pytest.fixture(scope="module")
def default_backbone():
    return DINOv2MultiScaleBackbone(pretrained=False, freeze_blocks=2)


@pytest.fixture(scope="module")
def default_projector():
    return MultiScaleProjector()


@pytest.fixture(scope="module")
def default_wrapper(default_backbone, default_projector):
    return BackboneProjectorWrapper(default_backbone, default_projector)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("hp, wp", [(14, 14), (16, 16), (18, 18), (16, 20)])
def test_multiscale_projector_shapes(default_projector, batch_size, hp, wp):
    """Verify MultiScaleProjector output shapes for various patch grid sizes."""
    dummy_feats = [torch.randn(batch_size, 384, hp, wp) for _ in range(4)]
    out = default_projector(dummy_feats)

    assert len(out) == 4
    # p2: stride 7 (2x upsampled relative to stride 14) -> 2*hp, 2*wp
    assert out[0].shape == (batch_size, 256, hp * 2, wp * 2)
    # p3: stride 14 (1x identical to backbone feature grid) -> hp, wp
    assert out[1].shape == (batch_size, 256, hp, wp)
    # p4: stride 28 (downsampled by 2) -> hp//2, wp//2
    assert out[2].shape == (batch_size, 256, hp // 2, wp // 2)
    # p5: stride 56 (max-pooled by 2 from p4) -> (hp//2)//2, (wp//2)//2
    assert out[3].shape == (batch_size, 256, (hp // 2 + 1) // 2 if (hp // 2) % 2 != 0 else (hp // 2) // 2,
                                            (wp // 2 + 1) // 2 if (wp // 2) % 2 != 0 else (wp // 2) // 2)

    for i, level_feat in enumerate(out):
        assert level_feat.dtype == torch.float32
        assert torch.isfinite(level_feat).all(), f"Level {i} contains non-finite values"


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("h, w", [
    (196, 196),
    (224, 224),
    (252, 252),
    (224, 280),
])
def test_backbone_projector_wrapper_end_to_end(default_wrapper, batch_size, h, w):
    """Verify composite BackboneProjectorWrapper output keys, shapes, and channels."""
    x = torch.randn(batch_size, 3, h, w)
    out = default_wrapper(x)

    expected_keys = ["p2", "p3", "p4", "p5"]
    assert list(out.keys()) == expected_keys

    # Channel dimensions must be 256 across all levels
    for key, feat in out.items():
        assert feat.shape[0] == batch_size
        assert feat.shape[1] == 256
        assert feat.dtype == torch.float32
        assert torch.isfinite(feat).all(), f"Non-finite values in wrapper level {key}"

    # For multiples of 56, exact stride matching must hold
    if h % 56 == 0 and w % 56 == 0:
        assert out["p2"].shape == (batch_size, 256, h // 7, w // 7)
        assert out["p3"].shape == (batch_size, 256, h // 14, w // 14)
        assert out["p4"].shape == (batch_size, 256, h // 28, w // 28)
        assert out["p5"].shape == (batch_size, 256, h // 56, w // 56)


def test_wrapper_output_shape_metadata(default_wrapper):
    """Verify ShapeSpec dictionary produced by wrapper.output_shape()."""
    shapes = default_wrapper.output_shape()
    expected = {
        "p2": (256, 7),
        "p3": (256, 14),
        "p4": (256, 28),
        "p5": (256, 56),
    }
    for key, (exp_ch, exp_stride) in expected.items():
        assert key in shapes, f"Missing key {key} in output_shape"
        spec = shapes[key]
        assert spec.channels == exp_ch, f"Channel mismatch for {key}: {spec.channels} vs {exp_ch}"
        assert spec.stride == exp_stride, f"Stride mismatch for {key}: {spec.stride} vs {exp_stride}"
    assert default_wrapper.size_divisibility == 56


def test_wrapper_gradient_backpropagation():
    """Verify gradient flow through BackboneProjectorWrapper into backbone and projector."""
    backbone = DINOv2MultiScaleBackbone(pretrained=False, freeze_blocks=2)
    projector = MultiScaleProjector()
    wrapper = BackboneProjectorWrapper(backbone, projector)

    x = torch.randn(2, 3, 224, 224)
    out = wrapper(x)
    loss = sum(v.sum() for v in out.values())
    loss.backward()

    # 1. Projector gradients
    proj_trainable = [p for p in projector.parameters() if p.requires_grad]
    assert len(proj_trainable) > 0
    for p in proj_trainable:
        assert p.grad is not None, "Projector parameter has None gradient"
        assert torch.isfinite(p.grad).all(), "Projector gradient has NaN/Inf"
        assert (p.grad.abs().sum() > 0), "Projector parameter has zero gradient"

    # 2. Backbone frozen blocks (0, 1)
    for i in range(2):
        blk = backbone.vit.blocks[i]
        assert all(p.grad is None for p in blk.parameters()), f"Frozen block {i} unexpectedly received gradients"

    # 3. Backbone unfrozen blocks (2..11)
    for i in range(2, 12):
        blk = backbone.vit.blocks[i]
        for p in blk.parameters():
            assert p.grad is not None, f"Unfrozen block {i} parameter has None gradient"
            assert (p.grad.abs().sum() > 0), f"Unfrozen block {i} parameter has zero gradient"


def test_projector_layernorm_variant():
    """Verify that MultiScaleProjector with layer_norm=True runs forward and backward."""
    p_ln = MultiScaleProjector(layer_norm=True)
    dummy_feats = [torch.randn(1, 384, 16, 16, requires_grad=True) for _ in range(4)]
    out = p_ln(dummy_feats)
    assert len(out) == 4
    loss = sum(v.sum() for v in out)
    loss.backward()
    for feat in dummy_feats:
        assert feat.grad is not None and (feat.grad.abs().sum() > 0)


def test_projector_rmsnorm_vulnerability():
    """Document and empirically verify the known RMSNorm dimension bug in ConvX.

    ConvX uses `nn.RMSNorm(out_planes)` which expects trailing channel format [*C],
    but receives 4D NCHW tensors, resulting in RuntimeError:
    'Given normalized_shape=[256], expected input with shape [*256], but got input of size [1, 256, 32, 32]'
    """
    p_rms = MultiScaleProjector(rms_norm=True)
    dummy_feats = [torch.randn(1, 384, 16, 16) for _ in range(4)]
    with pytest.raises(RuntimeError) as exc_info:
        p_rms(dummy_feats)
    assert "Given normalized_shape=[256]" in str(exc_info.value)


def test_projector_dropout_and_forced_drop():
    """Verify feature survival and forced ablation drop logic."""
    # Forced drop of 2 trailing features
    p_drop = MultiScaleProjector(force_drop_last_n_features=2)
    dummy_feats = [torch.ones(1, 384, 16, 16) for _ in range(4)]
    # In forward, feats[-1] and feats[-2] get zeroed out
    out = p_drop(dummy_feats)
    assert len(out) == 4
    for feat in out:
        assert torch.isfinite(feat).all()

    # Survival prob in eval mode should not alter outputs
    p_eval = MultiScaleProjector(survival_prob=0.5)
    p_eval.eval()
    out1 = p_eval(dummy_feats)
    out2 = p_eval(dummy_feats)
    for f1, f2 in zip(out1, out2):
        assert torch.equal(f1, f2), "Eval mode must be deterministic despite survival_prob < 1.0"
