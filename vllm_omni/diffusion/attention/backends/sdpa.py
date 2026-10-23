# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from typing import Literal

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)
from vllm_omni.diffusion.attention.backends.utils.attn_runtime_selector import can_sdpa_use_fused_gqa

logger = init_logger(__name__)


SDPAMaskMode = Literal["broadcast_k", "full_qk"]


def _maybe_reshape_attn_mask(
    query: torch.Tensor,
    key: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    mask_mode: SDPAMaskMode = "broadcast_k",
):
    """
    Reshape Attention Mask
    2D [batch_size, seq_len_k] ->
      - broadcast_k: [batch_size, 1, 1, seq_len_k]
      - full_qk: [batch_size, 1, seq_len_q, seq_len_k]
    """
    # Reshape Attention Mask
    # 2D [batch_size, seq_len_k] mask only.
    if (
        attn_mask is not None
        and attn_mask.ndim == 2
        and attn_mask.shape[0] == query.shape[0]
        and attn_mask.shape[1] == key.shape[1]
    ):
        B, Sq, Skv = attn_mask.shape[0], query.shape[1], key.shape[1]
        attn_mask = attn_mask.to(torch.bool)
        if mask_mode == "full_qk":
            # NPU path requires explicit [B, 1, Q, K] mask layout.
            attn_mask = attn_mask.unsqueeze(1).expand(B, Sq, Skv).unsqueeze(1).contiguous()
        elif mask_mode == "broadcast_k":
            # CUDA-like backends prefer [B, 1, 1, K] and rely on SDPA broadcast.
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
        else:
            raise ValueError(f"Unsupported SDPA mask mode: {mask_mode}")
    return attn_mask


def _h3_sparse_override(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_metadata: AttentionMetadata,
    softmax_scale: float,
) -> torch.Tensor | None:
    """MiniMax-H3 opt-in block-sparse attention for the packed SDPA layout.

    Confined to metadata that publishes a ``VideoTokenLayout`` carrying both
    ``used_len`` and ``prefix_len`` -- i.e. MiniMax-H3 -- and gated by
    ``H3_BSA_ENABLE=1``. Returns ``None`` for every other model and whenever
    the sparse kernel is unusable, so the dense call below is unchanged.
    """

    layout = getattr(attn_metadata, "video_layout", None)
    used_len = None if layout is None else getattr(layout, "used_len", None)
    prefix_len = None if layout is None else getattr(layout, "prefix_len", None)
    if prefix_len is None:
        # Ref2VA publishes used_len + video_spans instead of a single
        # prefix_len; the non-video head is everything before the target
        # video rows, and those rows must stay visible to every query.
        spans = () if layout is None else (getattr(layout, "video_spans", ()) or ())
        starts = [int(span.start) for span in spans if getattr(span, "role", "") == "target"]
        if not starts:
            starts = [int(span.start) for span in spans]
        prefix_len = min(starts) if starts else None
    if used_len is None or prefix_len is None:
        return None
    if query.shape[2] != key.shape[2]:
        # The kernel call assumes one head group per KV head.
        return None

    from vllm_omni.diffusion.attention.backends.h3_bsa import (
        h3_bsa_enabled,
        h3_block_sparse_attention_single,
    )

    if not h3_bsa_enabled():
        return None
    out = h3_block_sparse_attention_single(
        query[0],
        key[0],
        value[0],
        used_len=int(used_len),
        prefix_len=int(prefix_len),
        softmax_scale=softmax_scale,
    )
    return None if out is None else out.unsqueeze(0)


class SDPABackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls, attention_spec: object | None = None) -> bool:
        del attention_spec
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [x for x in range(1024)]  # todo

    @staticmethod
    def get_name() -> str:
        return "SDPA"

    @staticmethod
    def get_impl_cls() -> type["SDPAImpl"]:
        return SDPAImpl


class SDPAImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        if backend_kwargs:
            logger.warning("SDPAImpl ignoring backend_kwargs: %s", list(backend_kwargs.keys()))

    def _forward_impl(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
        mask_mode: SDPAMaskMode = "broadcast_k",
    ) -> torch.Tensor:
        # Normalize mask before permuting q/k/v.
        # _maybe_reshape_attn_mask expects sequence length on dim=1.
        attention_mask = None
        if attn_metadata:
            attention_mask = _maybe_reshape_attn_mask(query, key, attn_metadata.attn_mask, mask_mode=mask_mode)

        if attn_metadata is not None and not self.causal and query.shape[0] == 1:
            sparse_out = _h3_sparse_override(
                query,
                key,
                value,
                attn_metadata,
                self.softmax_scale,
            )
            if sparse_out is not None:
                return sparse_out

        enable_gqa = query.shape[2] != key.shape[2]
        query, key, value = (x.permute(0, 2, 1, 3) for x in (query, key, value))
        # Only the PyTorch SDPA backend needs this dispatch check. If SDPA
        # cannot select a fused GQA kernel for the runtime shape/mask, expand
        # K/V locally so it can use the better-supported equal-head path.
        if enable_gqa and not can_sdpa_use_fused_gqa(query, key, value, attention_mask, self.causal):
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError(
                    "GQA requires query heads to be a multiple of KV heads, "
                    f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}."
                )
            repeat_num = query.shape[1] // key.shape[1]
            key = key.repeat_interleave(repeat_num, dim=1)
            value = value.repeat_interleave(repeat_num, dim=1)
            enable_gqa = False
            logger.debug(
                "CUDA SDPA cannot use a fused native-GQA kernel for this shape; expanding K/V heads before SDPA."
            )
        output = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=self.causal,
            scale=self.softmax_scale,
            enable_gqa=enable_gqa,
        )
        out = output.permute(0, 2, 1, 3)
        return out

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="broadcast_k")

    def forward_xpu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="broadcast_k")

    def forward_hip(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="broadcast_k")

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        return self._forward_impl(query, key, value, attn_metadata, mask_mode="full_qk")
