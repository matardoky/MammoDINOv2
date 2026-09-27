"""rfdetr.utils.amp_patch — CUDA Mixed Precision (FP16/BF16) Compatibility Patch for Detrex.

Detrex's native MultiScaleDeformableAttn only converts `torch.float16` to `torch.float32`
prior to calling the C++ CUDA operator `_C.ms_deform_attn_forward`.
When executing under `torch.bfloat16` autocast, Detrex passes raw `bfloat16` into C++,
which immediately throws:
    NotImplementedError: "ms_deform_attn_forward_cuda" not implemented for 'BFloat16'

Furthermore, under certain AMP configurations, `sampling_locations` or `attention_weights`
can have mismatched precisions with `value`, triggering CUDA kernel dispatch errors.

This module provides two layers of safety:
1. `patch_detrex_source_file(detrex_root)`: Modifies the on-disk python file in the cloned
   detrex repo if present.
2. `patch_detrex_ms_deform_attn()`: Monkey-patches `MultiScaleDeformableAttn.forward`
   in-memory so any instantiated layer executes safely in both float16 and bfloat16.
"""

from __future__ import annotations

import logging
import os
import torch

logger = logging.getLogger("rfdetr.amp_patch")


def patch_detrex_source_file(detrex_root: str | None = None) -> bool:
    """Optionally patch detrex's multi_scale_deform_attn.py file on disk."""
    if not detrex_root or not os.path.isdir(detrex_root):
        return False

    candidates = [
        os.path.join(detrex_root, "detrex", "layers", "multi_scale_deform_attn.py"),
        os.path.join(detrex_root, "layers", "multi_scale_deform_attn.py"),
    ]

    for path in candidates:
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                if "value.dtype in (torch.float16, torch.bfloat16)" in content:
                    logger.debug(f"{path} is already patched on disk.")
                    return True

                old_str1 = "value.to(torch.float32) if value.dtype==torch.float16 else value"
                new_str1 = "value.to(torch.float32) if value.dtype in (torch.float16, torch.bfloat16) else value"

                old_str2 = "if value.dtype==torch.float16:\n            output=output.to(torch.float16)"
                new_str2 = "if value.dtype in (torch.float16, torch.bfloat16):\n            output=output.to(value.dtype)"

                modified = False
                if old_str1 in content:
                    content = content.replace(old_str1, new_str1)
                    modified = True
                if old_str2 in content:
                    content = content.replace(old_str2, new_str2)
                    modified = True

                if modified:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(content)
                    logger.info(f"Successfully patched Detrex on disk: {path}")
                    return True
            except Exception as e:
                logger.warning(f"Could not patch Detrex source file {path}: {e}")

    return False


def patch_detrex_ms_deform_attn() -> bool:
    """In-memory monkey-patch for Detrex MultiScaleDeformableAttention.forward.

    Guarantees that `value`, `sampling_locations`, and `attention_weights`
    are in float32 when calling C++ `_C.ms_deform_attn_forward`, and that
    the returned output is restored to the input tensor's original dtype
    (float16 or bfloat16).
    """
    try:
        import detrex.layers.multi_scale_deform_attn as msda
        cls = getattr(msda, "MultiScaleDeformableAttention", getattr(msda, "MultiScaleDeformableAttn", None))
        if cls is None:
            logger.debug("MultiScaleDeformableAttention not found in detrex.")
            return False

        if getattr(cls, "_is_mammo_patched", False):
            return True
    except Exception as e:
        logger.debug(f"Detrex not importable or could not inspect: {e}")
        return False

    def safe_forward(
        self,
        query,
        reference_points,
        value=None,
        identity=None,
        query_pos=None,
        key_padding_mask=None,
        spatial_shapes=None,
        level_start_index=None,
        **kwargs,
    ):
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query, embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        # value projection
        value = self.value_proj(value)
        # fill "0" for the padding part
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        # [bs, all hw, 256] -> [bs, all hw, 8, 32]
        value = value.view(bs, num_value, self.num_heads, -1)
        # [bs, all hw, 8, 4, 4, 2]: 8 heads, 4 level features, 4 sampling points, 2 offsets
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2
        )
        # [bs, all hw, 8, 16]: 4 level 4 sampling points: 16 features total
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points
        )
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.view(
            bs,
            num_query,
            self.num_heads,
            self.num_levels,
            self.num_points,
        )

        # bs, num_query, num_heads, num_levels, num_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = (
                reference_points[:, :, None, :, None, :]
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            )
        elif reference_points.shape[-1] == 4:
            sampling_locations = (
                reference_points[:, :, None, :, None, :2]
                + sampling_offsets
                / self.num_points
                * reference_points[:, :, None, :, None, 2:]
                * 0.5
            )
        else:
            raise ValueError(
                f"Last dim of reference_points must be 2 or 4, but got {reference_points.shape[-1]}."
            )

        orig_dtype = value.dtype
        if torch.cuda.is_available() and value.is_cuda:
            needs_fp32 = orig_dtype in (torch.float16, torch.bfloat16)
            output = msda.MultiScaleDeformableAttnFunction.apply(
                value.to(torch.float32) if needs_fp32 else value,
                spatial_shapes,
                level_start_index,
                sampling_locations.to(torch.float32) if needs_fp32 else sampling_locations,
                attention_weights.to(torch.float32) if needs_fp32 else attention_weights,
                self.im2col_step,
            )
            if needs_fp32:
                output = output.to(orig_dtype)
        else:
            output = msda.multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights
            )

        output = self.output_proj(output)

        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity
    try:
        cls.forward = safe_forward
        cls._is_mammo_patched = True
        logger.info("Detrex MultiScaleDeformableAttention successfully patched for fp16/bf16 CUDA execution.")
        return True
    except Exception as e:
        logger.warning(f"Could not monkey-patch MultiScaleDeformableAttention: {e}")
        return False
