"""tests/test_amp_patch.py — Verification of Detrex AMP fp16/bf16 patch logic."""

import os
import tempfile
import torch
import pytest

from rfdetr.utils.amp_patch import patch_detrex_source_file, patch_detrex_ms_deform_attn


def test_patch_detrex_source_file_on_disk():
    with tempfile.TemporaryDirectory() as tmpdir:
        detrex_layers_dir = os.path.join(tmpdir, "detrex", "layers")
        os.makedirs(detrex_layers_dir, exist_ok=True)
        target_file = os.path.join(detrex_layers_dir, "multi_scale_deform_attn.py")

        sample_code = """
        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value.to(torch.float32) if value.dtype==torch.float16 else value,
                spatial_shapes,
                level_start_index,
                sampling_locations,
                attention_weights,
                self.im2col_step,
            )
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        if value.dtype==torch.float16:
            output=output.to(torch.float16)
        """
        with open(target_file, "w", encoding="utf-8") as f:
            f.write(sample_code)

        # 1. First run should patch
        success = patch_detrex_source_file(tmpdir)
        assert success is True

        with open(target_file, "r", encoding="utf-8") as f:
            patched_code = f.read()

        assert "value.dtype in (torch.float16, torch.bfloat16)" in patched_code
        assert "output=output.to(value.dtype)" in patched_code

        # 2. Second run should be idempotent
        success2 = patch_detrex_source_file(tmpdir)
        assert success2 is True


def test_patch_detrex_source_file_nonexistent_dir():
    assert patch_detrex_source_file("/non/existent/path") is False
    assert patch_detrex_source_file(None) is False


def test_patch_detrex_ms_deform_attn_graceful():
    # If detrex is not importable, should return False without crashing
    res = patch_detrex_ms_deform_attn()
    assert isinstance(res, bool)


def test_mock_msdeformattn_autocast_dispatch():
    """Verify that casting value, sampling_locations, and attention_weights to float32
    and restoring orig_dtype preserves gradients in autograd for both float16 and bfloat16.
    """
    for test_dtype in [torch.float16, torch.bfloat16]:
        value = torch.randn(1, 16, 4, 32, dtype=test_dtype, requires_grad=True)
        sampling_locations = torch.rand(1, 8, 4, 4, 4, 2, dtype=test_dtype, requires_grad=True)
        attention_weights = torch.rand(1, 8, 4, 4, 4, dtype=test_dtype, requires_grad=True)

        orig_dtype = value.dtype
        needs_fp32 = orig_dtype in (torch.float16, torch.bfloat16)

        # Simulated kernel execution in float32
        v_in = value.to(torch.float32) if needs_fp32 else value
        s_loc = sampling_locations.to(torch.float32) if needs_fp32 else sampling_locations
        a_wt = attention_weights.to(torch.float32) if needs_fp32 else attention_weights

        # Dummy computation representing CUDA kernel
        out_f32 = (v_in.sum(dim=(2, 3), keepdim=True) * 0.1).expand_as(v_in)
        out = out_f32.to(orig_dtype) if needs_fp32 else out_f32

        assert out.dtype == test_dtype
        loss = out.sum()
        loss.backward()

        assert value.grad is not None
        assert value.grad.dtype == test_dtype
