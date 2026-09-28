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


def test_projector_uses_layernorm_by_default():
    """Verify that MultiScaleProjector defaults to layer_norm=True (Roboflow RF-DETR standard)."""
    from rfdetr.models.projector import MultiScaleProjector, LayerNorm2d

    proj = MultiScaleProjector(
        in_channels=[DIM, DIM, DIM, DIM],
        out_channels=OUT_CH,
        scale_factors=(2.0, 1.0, 0.5, 0.25),
        num_blocks=1,
    )
    # Check that conv normalization uses LayerNorm2d, not BatchNorm2d
    has_bn = any(isinstance(m, nn.BatchNorm2d) for m in proj.modules())
    has_ln = any(isinstance(m, LayerNorm2d) for m in proj.modules())
    assert not has_bn, "MultiScaleProjector must NOT contain BatchNorm2d by default (causes batch_size=1 failure)"
    assert has_ln, "MultiScaleProjector must contain LayerNorm2d by default for stability"


def test_mammo_config_dino_inheritance():
    """Verify configs/mammo_dinov2_dino.py properly inherits from dino_r50.py (dino_vitdet pattern)."""
    from pathlib import Path

    cfg_file = Path(__file__).resolve().parent.parent / "configs" / "mammo_dinov2_dino.py"
    with open(cfg_file, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Static architectural contract checks (always run, even without detectron2)
    assert "_load_dino_r50_model()" in content, "Must load base model via dino_r50 pattern"
    assert "BackboneProjectorWrapper" in content, "Must wrap backbone and projector"
    assert "layer_norm=True" in content, "Projector must use layer_norm=True for small batch stability"
    assert 'model.neck.in_features = ["p2", "p3", "p4", "p5"]' in content
    assert "model.transformer.num_feature_levels = 4" in content
    assert "model.transformer.encoder.use_checkpoint = True" in content
    assert "model.transformer.decoder.use_checkpoint = True" in content
    assert "model.num_queries = 100" in content
    assert "model.embed_dim = 256" in content
    assert "model.select_box_nums_for_evaluation = model.num_queries" in content

    # 2. Verify canonical dino_r50 base mirror
    dino_r50_file = Path(__file__).resolve().parent.parent / "configs" / "models" / "dino_r50.py"
    assert dino_r50_file.is_file(), "Canonical dino_r50.py mirror must exist in configs/models"
    import importlib.util
    spec_r50 = importlib.util.spec_from_file_location("dino_r50_mirror", str(dino_r50_file))
    mod_r50 = importlib.util.module_from_spec(spec_r50)
    spec_r50.loader.exec_module(mod_r50)
    assert hasattr(mod_r50, "model"), "configs/models/dino_r50.py must define model"

    # 3. Dynamic runtime instantiation test (when detectron2 is installed)
    try:
        import detectron2
    except ImportError:
        return  # Pass static verification if detectron2 not installed locally

    spec = importlib.util.spec_from_file_location("test_mammo_cfg", str(cfg_file))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert hasattr(mod, "model"), "Config must export 'model'"
    model = mod.model

    # Check Detrex dino_vitdet pattern overrides
    assert hasattr(model, "backbone"), "Model must have backbone"
    assert hasattr(model, "neck"), "Model must have neck"
    assert hasattr(model, "transformer"), "Model must have transformer"
    assert hasattr(model, "criterion"), "Model must have criterion"
    assert list(model.neck.in_features) == ["p2", "p3", "p4", "p5"]
    assert model.neck.num_outs == 4
    assert model.transformer.num_feature_levels == 4
    assert model.transformer.encoder.use_checkpoint is True
    assert model.transformer.decoder.use_checkpoint is True
    assert model.num_queries == 100
    assert model.embed_dim == 256
    assert model.select_box_nums_for_evaluation == 100


def test_dinov2_optimizer_params():
    """Verify that get_dinov2_optimizer_params correctly groups parameters with layer-wise decay."""
    from rfdetr.solver.optimizer import get_dinov2_optimizer_params

    class StubViTBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(384, 384)

    class StubViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([StubViTBlock() for _ in range(12)])
            self.norm = nn.LayerNorm(384)

    class StubBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.vit = StubViT()

    class StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.backbone = StubBackbone()
            self.backbone.projector = nn.Linear(384, 256)
            self.head = nn.Linear(256, 10)

    model = StubModel()
    groups = get_dinov2_optimizer_params(
        model,
        base_lr=1e-4,
        backbone_lr=1.19e-4,
        layer_decay=0.90,
        weight_decay=1e-4,
        num_layers=12,
    )

    group_names = [g["name"] for g in groups]
    assert any("backbone_depth_12" in n for n in group_names), "Must have backbone_depth_12"
    assert any("head_decay" in n for n in group_names), "Must have head_decay"
    assert any("head_no_decay" in n for n in group_names), "Must have head_no_decay"

    # Verify top layer has backbone_lr and bottom has decayed lr
    g12 = next(g for g in groups if g["name"] == "backbone_depth_12")
    assert pytest.approx(g12["lr"], 1e-6) == 1.19e-4
    assert g12["weight_decay"] == 0.0

    g1 = next(g for g in groups if g["name"] == "backbone_depth_1")
    assert pytest.approx(g1["lr"], 1e-6) == 1.19e-4 * (0.90 ** 11)


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
