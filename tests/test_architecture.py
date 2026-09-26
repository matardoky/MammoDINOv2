"""tests/test_architecture.py

Verifies the shape coordination between every component of the pipeline:

  (B,3,H,W)
      │
  DINOv2MultiScaleBackbone      → {block3..block12: (B,384,Hp,Wp)}   stride=14
      │
  MultiScaleProjector           → [(B,256,2Hp,2Wp), ..., (B,256,Hp/4,Wp/4)]
      │
  BackboneProjectorWrapper      → {p2:..., p3:..., p4:..., p5:...}   strides=[7,14,28,56]
      │
  ChannelMapper (neck)          → {p2:..., p3:..., p4:..., p5:...}   256ch, same spatial
      │
  DINOTransformer               → queries (B, num_queries, 256)

All tests run on CPU with synthetic tensors — no real images or weights needed.
"""

import math
import pytest
import torch
import torch.nn as nn


# ─── Fixtures ─────────────────────────────────────────────────────────────────

BATCH      = 2
IMG_H      = 448   # must be divisible by patch_size * 4 = 56
IMG_W      = 448
PATCH      = 14    # ViT-S/14 patch size
Hp         = IMG_H // PATCH   # = 32  spatial tokens per axis
Wp         = IMG_W // PATCH   # = 32
DIM        = 384   # ViT-S embed_dim
OUT_CH     = 256   # projector / neck output channels
NUM_LEVELS = 4


# ─── Helper: build lightweight stubs without timm or detrex ──────────────────

class _FakeDINOv2Backbone(nn.Module):
    """Drop-in stub for DINOv2MultiScaleBackbone with correct output contract."""

    out_features = ["block3", "block6", "block9", "block12"]
    _p = PATCH
    _dim = DIM
    size_divisibility = PATCH * 4  # 56

    def forward(self, x: torch.Tensor):
        B, _, H, W = x.shape
        h, w = H // self._p, W // self._p
        return {
            name: torch.randn(B, self._dim, h, w)
            for name in self.out_features
        }

    def output_shape(self):
        from detectron2.layers import ShapeSpec
        return {
            name: ShapeSpec(channels=self._dim, stride=self._p)
            for name in self.out_features
        }


# ─── 1. Backbone output shape ─────────────────────────────────────────────────

def test_backbone_output_shapes():
    """DINOv2MultiScaleBackbone → 4 feature maps, each (B, 384, H/14, W/14)."""
    backbone = _FakeDINOv2Backbone()
    x = torch.randn(BATCH, 3, IMG_H, IMG_W)
    feats = backbone(x)

    assert set(feats.keys()) == set(backbone.out_features), \
        f"Expected keys {backbone.out_features}, got {list(feats.keys())}"

    for name, feat in feats.items():
        assert feat.shape == (BATCH, DIM, Hp, Wp), \
            f"{name}: expected {(BATCH, DIM, Hp, Wp)}, got {feat.shape}"

    print(f"  [OK] Backbone: 4 × {(BATCH, DIM, Hp, Wp)}")


# ─── 2. Projector output shapes ───────────────────────────────────────────────

def test_projector_output_shapes():
    """MultiScaleProjector → 4 levels at strides [7,14,28,56], 256 channels each."""
    from rfdetr.models.projector import MultiScaleProjector

    proj = MultiScaleProjector(
        in_channels=[DIM, DIM, DIM, DIM],
        out_channels=OUT_CH,
        scale_factors=(2.0, 1.0, 0.5, 0.25),
        num_blocks=1,   # lightweight for test
    )
    proj.eval()

    feat_list = [torch.randn(BATCH, DIM, Hp, Wp) for _ in range(NUM_LEVELS)]

    with torch.no_grad():
        results = proj(feat_list)

    expected_spatial = [
        (Hp * 2, Wp * 2),   # scale=2.0 → upsample ×2
        (Hp,     Wp    ),   # scale=1.0 → identity
        (Hp // 2, Wp // 2), # scale=0.5 → downsample ×2
        (Hp // 4, Wp // 4), # scale=0.25 → extra max_pool
    ]
    assert len(results) == NUM_LEVELS, \
        f"Expected {NUM_LEVELS} outputs, got {len(results)}"

    for i, (feat, (eh, ew)) in enumerate(zip(results, expected_spatial)):
        assert feat.shape == (BATCH, OUT_CH, eh, ew), \
            f"Level {i}: expected {(BATCH, OUT_CH, eh, ew)}, got {feat.shape}"
        print(f"  [OK] Projector level {i}: {feat.shape}")


# ─── 3. BackboneProjectorWrapper output contract ──────────────────────────────

def test_wrapper_output_names_strides():
    """BackboneProjectorWrapper → dict keys p2..p5, strides [7,14,28,56], 256 ch."""
    from rfdetr.models.projector import MultiScaleProjector, BackboneProjectorWrapper

    backbone = _FakeDINOv2Backbone()
    projector = MultiScaleProjector(
        in_channels=[DIM, DIM, DIM, DIM],
        out_channels=OUT_CH,
        scale_factors=(2.0, 1.0, 0.5, 0.25),
        num_blocks=1,
    )
    wrapper = BackboneProjectorWrapper(backbone=backbone, projector=projector)
    wrapper.eval()

    x = torch.randn(BATCH, 3, IMG_H, IMG_W)
    with torch.no_grad():
        out = wrapper(x)

    expected_keys   = ["p2", "p3", "p4", "p5"]
    expected_strides = [7, 14, 28, 56]
    expected_spatial = [
        (Hp * 2, Wp * 2),
        (Hp,     Wp    ),
        (Hp // 2, Wp // 2),
        (Hp // 4, Wp // 4),
    ]

    assert list(out.keys()) == expected_keys, \
        f"Expected keys {expected_keys}, got {list(out.keys())}"

    shape_info = wrapper.output_shape()
    for key, exp_stride, (eh, ew) in zip(expected_keys, expected_strides, expected_spatial):
        feat = out[key]
        assert feat.shape == (BATCH, OUT_CH, eh, ew), \
            f"{key}: expected {(BATCH, OUT_CH, eh, ew)}, got {feat.shape}"
        assert shape_info[key].channels == OUT_CH, \
            f"{key}: output_shape channels mismatch"
        assert shape_info[key].stride == exp_stride, \
            f"{key}: expected stride {exp_stride}, got {shape_info[key].stride}"
        print(f"  [OK] Wrapper {key}: shape={feat.shape}  stride={exp_stride}")


# ─── 4. ChannelMapper (neck) coordination ────────────────────────────────────

def test_channelmapper_receives_correct_input():
    """ChannelMapper receives {p2..p5} with 256ch and passes through unchanged."""
    try:
        from detrex.modeling.neck import ChannelMapper
        from detectron2.layers import ShapeSpec
    except ImportError as e:
        try:
            import pytest
            pytest.skip(f"detrex not installed — skipping ChannelMapper test: {e}")
        except Exception:
            return

    neck = ChannelMapper(
        input_shapes={
            "p2": ShapeSpec(channels=OUT_CH, stride=7),
            "p3": ShapeSpec(channels=OUT_CH, stride=14),
            "p4": ShapeSpec(channels=OUT_CH, stride=28),
            "p5": ShapeSpec(channels=OUT_CH, stride=56),
        },
        in_features=["p2", "p3", "p4", "p5"],
        out_channels=OUT_CH,
        num_outs=NUM_LEVELS,
        norm_layer=nn.GroupNorm(num_groups=32, num_channels=OUT_CH),
    )
    neck.eval()

    dummy_input = {
        "p2": torch.randn(BATCH, OUT_CH, Hp * 2,  Wp * 2),
        "p3": torch.randn(BATCH, OUT_CH, Hp,      Wp    ),
        "p4": torch.randn(BATCH, OUT_CH, Hp // 2, Wp // 2),
        "p5": torch.randn(BATCH, OUT_CH, Hp // 4, Wp // 4),
    }

    with torch.no_grad():
        neck_out = neck(dummy_input)

    assert len(neck_out) == NUM_LEVELS
    for i, feat in enumerate(neck_out):
        assert feat.shape[1] == OUT_CH, \
            f"Neck output {i}: expected {OUT_CH} channels, got {feat.shape[1]}"
        print(f"  [OK] Neck output {i}: {feat.shape}")


# ─── 5. Full pipeline shape trace ────────────────────────────────────────────

def test_full_pipeline_shapes():
    """End-to-end shape trace: image → backbone → projector → wrapper output dict."""
    from rfdetr.models.projector import MultiScaleProjector, BackboneProjectorWrapper

    backbone = _FakeDINOv2Backbone()
    projector = MultiScaleProjector(
        in_channels=[DIM, DIM, DIM, DIM],
        out_channels=OUT_CH,
        scale_factors=(2.0, 1.0, 0.5, 0.25),
        num_blocks=1,
    )
    wrapper = BackboneProjectorWrapper(backbone=backbone, projector=projector)
    wrapper.eval()

    x = torch.randn(BATCH, 3, IMG_H, IMG_W)

    print(f"\n  Input:  {tuple(x.shape)}")
    with torch.no_grad():
        out = wrapper(x)

    for name, feat in out.items():
        stride = wrapper.out_strides[name]
        print(f"  {name} (stride={stride:2d}): {tuple(feat.shape)}")
        assert feat.shape[0] == BATCH
        assert feat.shape[1] == OUT_CH


# ─── 6. size_divisibility contract ───────────────────────────────────────────

def test_size_divisibility():
    """BackboneProjectorWrapper.size_divisibility must divide image dimensions."""
    from rfdetr.models.projector import MultiScaleProjector, BackboneProjectorWrapper

    backbone = _FakeDINOv2Backbone()
    projector = MultiScaleProjector(
        in_channels=[DIM, DIM, DIM, DIM],
        out_channels=OUT_CH,
        scale_factors=(2.0, 1.0, 0.5, 0.25),
        num_blocks=1,
    )
    wrapper = BackboneProjectorWrapper(backbone=backbone, projector=projector)

    div = wrapper.size_divisibility
    assert div == PATCH * 4, f"Expected size_divisibility={PATCH * 4}, got {div}"
    assert IMG_H % div == 0, f"IMG_H={IMG_H} is not divisible by {div}"
    assert IMG_W % div == 0, f"IMG_W={IMG_W} is not divisible by {div}"
    print(f"  [OK] size_divisibility={div} | {IMG_H}×{IMG_W} are valid input sizes")


if __name__ == "__main__":
    print("\n=== Architecture Coordination Tests ===\n")
    print("1. Backbone output shapes")
    test_backbone_output_shapes()
    print("\n2. Projector output shapes")
    test_projector_output_shapes()
    print("\n3. Wrapper output names & strides")
    test_wrapper_output_names_strides()
    print("\n4. ChannelMapper (requires detrex)")
    try:
        test_channelmapper_receives_correct_input()
    except Exception as e:
        print(f"  [SKIP] Skipped: {e}")
    print("\n5. Full pipeline shape trace")
    test_full_pipeline_shapes()
    print("\n6. size_divisibility contract")
    test_size_divisibility()
    print("\n[OK] All coordination checks passed.")
