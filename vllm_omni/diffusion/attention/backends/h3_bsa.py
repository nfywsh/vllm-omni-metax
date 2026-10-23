# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Opt-in block-sparse attention for MiniMax-H3 packed video rows.

Ported from the SGLang delivery build
(``sglang/multimodal_gen/runtime/models/dits/minimax_h3_bsa.py``).

H3 forwards every modality of a request as one packed varlen document
(``[text | imgvid_cond | audio | video | pad]`` with
``cu_seqlens = [0, used, seq_len]``), so the dense flash kernel behind
``_forward_varlen_packed`` dominates denoise time. This module routes the
longest segment of a packed call through the Wan2.2-style block-sparse kernel
(``block_sparse_attn``, MACA build) and keeps every other segment on the
existing dense backend.

Knobs -- the feature is inert unless ``H3_BSA_ENABLE=1``:
  H3_BSA_ENABLE=1        enable
  H3_BSA_SPARSITY=0.65   fraction of key blocks dropped per query block
  H3_BSA_WINDOW=1        forced diagonal (temporal-locality) window, blocks
  H3_BSA_PATH=<dir>      override the bundled kernel directory

The kernel is loaded from ``third_party/bsa`` at first use (never at import
time) and is not installed into the interpreter's site-packages.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

_BLOCK = 128  # CUDA kernel block size (query and key)
_DEFAULT_SPARSITY = 0.65
_DEFAULT_WINDOW = 1
# Segments shorter than this many blocks are not worth sparsifying.
_MIN_SPARSE_BLOCKS = 4
_DEFAULT_KERNEL_DIR = Path(__file__).resolve().parents[4] / "third_party" / "bsa"

_bsa = None
_bsa_error: str | None = None


def h3_bsa_enabled() -> bool:
    return os.environ.get("H3_BSA_ENABLE", "0") == "1"


def _sparsity() -> float:
    """Fraction of key blocks dropped per query block (default 0.65)."""
    raw = os.environ.get("H3_BSA_SPARSITY")
    if not raw:
        return _DEFAULT_SPARSITY
    try:
        sparsity = float(raw)
    except ValueError:
        logger.warning_once("Invalid H3_BSA_SPARSITY=%r; using %.2f", raw, _DEFAULT_SPARSITY)
        return _DEFAULT_SPARSITY
    if not 0.0 <= sparsity < 1.0:
        logger.warning_once("H3_BSA_SPARSITY=%r is outside [0,1); using %.2f", raw, _DEFAULT_SPARSITY)
        return _DEFAULT_SPARSITY
    return sparsity


def _window() -> int:
    """Forced diagonal (temporal-spatial locality) window in blocks.

    Video tokens are laid out (t, h, w) row-major, so block i's temporal
    neighbors are block i+-1. Dropping them for even one step causes visible
    flicker; keep them unconditionally. A negative value disables the window.
    """
    raw = os.environ.get("H3_BSA_WINDOW")
    if not raw:
        return _DEFAULT_WINDOW
    try:
        return int(raw)
    except ValueError:
        logger.warning_once("Invalid H3_BSA_WINDOW=%r; using %d", raw, _DEFAULT_WINDOW)
        return _DEFAULT_WINDOW


def _load_kernel():
    """Import the MACA ``block_sparse_attn`` extension once."""
    global _bsa, _bsa_error
    if _bsa is not None:
        return _bsa
    if _bsa_error is not None:
        raise ImportError(_bsa_error)
    kernel_dir = os.environ.get("H3_BSA_PATH") or str(_DEFAULT_KERNEL_DIR)
    if not (Path(kernel_dir) / "block_sparse_attn").is_dir():
        _bsa_error = f"block_sparse_attn kernel directory {kernel_dir!r} does not exist"
        raise ImportError(_bsa_error)
    if kernel_dir not in sys.path:
        sys.path.insert(0, kernel_dir)
    try:
        import block_sparse_attn as bsa
    except Exception as exc:  # noqa: BLE001
        _bsa_error = f"cannot import block_sparse_attn from {kernel_dir!r}: {exc}"
        raise ImportError(_bsa_error) from exc
    _bsa = bsa
    return bsa


@torch.no_grad()
def _topk_block_mask(
    q: torch.Tensor, k: torch.Tensor, sparsity: float, prefix_blocks: int = 0
) -> torch.Tensor:
    """Per-head top-k block mask + forced locality/keyframe anchors.

    q, k: [S, H, D] rows padded to a multiple of _BLOCK. Returns [1, H, nb, nb]
    bool mask where True = attend.

    Base strategy is the Wan2.2 reference (mean-pool per block, block score,
    top-k per query row). On top of that we force:
      - the diagonal window +- H3_BSA_WINDOW blocks (temporal locality)
      - the first and last key blocks (FL2VA keyframe anchors)
    Forced blocks are unioned with the top-k picks (slight over-keep).
    """
    seq_len, heads, dim = q.shape
    nb = seq_len // _BLOCK
    q_pooled = q.view(nb, _BLOCK, heads, dim).mean(dim=1).permute(1, 0, 2).float()
    k_pooled = k.view(nb, _BLOCK, heads, dim).mean(dim=1).permute(1, 0, 2).float()
    scores = torch.matmul(q_pooled, k_pooled.transpose(-2, -1)) / (dim**0.5)
    keep = max(10, int(nb * (1.0 - sparsity)))
    keep = min(keep, nb)
    top = torch.topk(scores, k=keep, dim=-1).indices
    mask = torch.zeros(heads, nb, nb, device=q.device, dtype=torch.bool)
    mask.scatter_(-1, top, True)

    forced = torch.zeros(nb, nb, device=q.device, dtype=torch.bool)
    window = _window()
    if window >= 0 and nb > 1:
        pos = torch.arange(nb, device=q.device)
        forced[torch.abs(pos[:, None] - pos[None, :]) <= window] = True
    if nb > 1:
        # FL2VA packs the keyframe latents as the first/last latent frames of
        # the video segment; they anchor identity and must stay visible to
        # every query block, which pure content top-k does not guarantee.
        forced[:, 0] = True
        forced[:, -1] = True
    if prefix_blocks > 0:
        # SDPA layouts keep text, visual conditions and audio in the sequence
        # prefix. Every query row must still see them, or the video loses
        # prompt adherence and keyframe identity.
        forced[:, : min(int(prefix_blocks), nb)] = True
    if forced.any():
        mask |= forced.unsqueeze(0)
    return mask.unsqueeze(0)


def _dense_varlen_segment(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    softmax_scale: float,
) -> torch.Tensor:
    """Run one segment through the model's existing dense varlen kernel."""
    from vllm_omni.diffusion.attention.backends.utils.fa import flash_attn_varlen_func

    if flash_attn_varlen_func is None:
        raise ImportError("Dense variable-length attention requires flash_attn_varlen_func")
    length = q.shape[0]
    cu_seqlens = torch.tensor([0, length], device=q.device, dtype=torch.int32)
    out = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=length,
        max_seqlen_k=length,
        causal=False,
        softmax_scale=softmax_scale,
    )
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def h3_block_sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
    cu_seqlens_host: tuple[int, ...] | None = None,
) -> torch.Tensor | None:
    """Drop-in replacement for one packed varlen call, or ``None``.

    q, k, v: packed [T, H, D] rows; the segments per cu_seqlens are
    independent varlen documents. The longest segment runs block-sparse, every
    other segment runs the model's existing dense backend unchanged.

    Returns ``None`` -- after one warning -- when the kernel cannot be imported
    or the shapes are unsupported, so the caller keeps its dense path instead
    of failing the request.
    """
    try:
        return _run(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            softmax_scale=softmax_scale,
            cu_seqlens_host=cu_seqlens_host,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning_once(
            "H3 block-sparse attention unavailable (%s: %s); using dense flash attention.",
            type(exc).__name__,
            exc,
        )
        return None


@torch.no_grad()
def _run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    softmax_scale: float,
    cu_seqlens_host: tuple[int, ...] | None,
) -> torch.Tensor:
    if not q.is_cuda:
        raise RuntimeError("block-sparse attention requires a CUDA/MACA device")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            f"packed self-attention expects matching q/k/v rows, got "
            f"{tuple(q.shape)}/{tuple(k.shape)}/{tuple(v.shape)}"
        )
    if q.shape[-1] != _BLOCK:
        raise ValueError(f"head dim {q.shape[-1]} is unsupported (the kernel expects {_BLOCK})")
    if not 0 < max_seqlen:
        raise ValueError(f"max_seqlen must be positive, got {max_seqlen}")

    bsa = _load_kernel()
    # TODO(h3-bsa): thread a host-side ``cu_seqlens_q_host`` down from
    # MiniMaxH3Attention so this .tolist() device sync disappears. The packing
    # layer already knows the bounds as Python ints.
    bounds = (
        tuple(cu_seqlens_host)
        if cu_seqlens_host is not None
        else tuple(int(item) for item in cu_seqlens.tolist())
    )
    if len(bounds) < 2 or bounds[-1] != q.shape[0]:
        raise ValueError(f"cu_seqlens {bounds!r} does not describe the packed rows {q.shape[0]}")
    lens = [stop - start for start, stop in zip(bounds[:-1], bounds[1:])]
    out = torch.empty_like(v)

    sparsity = _sparsity()
    heads = q.shape[1]
    longest = max(range(len(lens)), key=lambda index: lens[index])
    logger.info_once(
        "MiniMax-H3 block-sparse attention active: sparsity=%.2f, window=%d, "
        "longest segment %d rows of %d total, dense for %d other segment(s)",
        sparsity,
        _window(),
        lens[longest],
        sum(lens),
        len(lens) - 1,
    )
    for seg, (start, stop) in enumerate(zip(bounds[:-1], bounds[1:])):
        length = stop - start
        if length <= 0:
            continue
        if seg != longest or length < _MIN_SPARSE_BLOCKS * _BLOCK:
            out[start:stop] = _dense_varlen_segment(
                q[start:stop],
                k[start:stop],
                v[start:stop],
                softmax_scale=softmax_scale,
            )
            continue

        qs, ks, vs = q[start:stop], k[start:stop], v[start:stop]
        pad = (-length) % _BLOCK
        if pad:
            # K/V padding uses REAL rows sampled from the tail region with a
            # stride (97) coprime with the span instead of repeating the last
            # row. Copies of one token concentrate softmax mass on a single
            # key; the end-of-sequence queries reach the padded rows through
            # the forced diagonal window and last-block anchor, collapsing
            # their outputs toward one token's value -> dark smudge artifact.
            # Zero padding is also wrong (logit-0 attractor).
            span = min(length, 4096)
            src = length - 1 - (torch.arange(pad, device=q.device) * 97) % span
            qs = torch.nn.functional.pad(qs, (0, 0, 0, 0, 0, pad))
            ks = torch.cat([ks, ks[src]], dim=0)
            vs = torch.cat([vs, vs[src]], dim=0)
        if qs.dtype != torch.bfloat16:
            qs, ks, vs = (tensor.to(torch.bfloat16) for tensor in (qs, ks, vs))
        seq_padded = qs.shape[0]
        mask = _topk_block_mask(qs, ks, sparsity)
        cu = torch.tensor([0, seq_padded], device=q.device, dtype=torch.int32)
        head_mask_type = torch.ones(heads, device=q.device, dtype=torch.int32)
        result = bsa.block_sparse_attn_func(
            qs,
            ks,
            vs,
            cu,
            cu,
            head_mask_type,
            None,
            mask,
            seq_padded,
            seq_padded,
            0.0,
            deterministic=False,
            softmax_scale=softmax_scale,
            is_causal=False,
            exact_streaming=False,
            return_attn_probs=False,
        )
        out[start:stop] = result[:length].to(v.dtype)
    return out


_LAST_SINGLE_LOG: tuple[float, int, bool] | None = None


def _log_single_request(
    sparsity: float | None,
    length: int,
    prefix_rows: int,
    prefix_blocks: int,
    *,
    engaged: bool,
) -> None:
    """One line per request, not per layer/step.

    The attention module runs once per layer per denoise step (~200 calls for a
    four-step H3 request), so the log is deduplicated on the tuple that changes
    only when a new request starts. Workers serve one request at a time, which
    makes a module-level key safe here.
    """
    global _LAST_SINGLE_LOG
    resolved = _sparsity() if sparsity is None else float(sparsity)
    key = (resolved, int(length), bool(engaged))
    if _LAST_SINGLE_LOG == key:
        return
    _LAST_SINGLE_LOG = key
    if engaged:
        logger.info(
            "MiniMax-H3 block-sparse attention requested=%.2f engaged=true: "
            "rows %d, prefix rows %d (%d blocks forced), window=%d",
            resolved,
            int(length),
            int(prefix_rows),
            int(prefix_blocks),
            _window(),
        )
    else:
        logger.info(
            "MiniMax-H3 block-sparse attention requested=%.2f engaged=false: "
            "rows %d < %d minimum, using dense SDPA",
            resolved,
            int(length),
            _MIN_SPARSE_BLOCKS * _BLOCK,
        )


@torch.no_grad()
def h3_block_sparse_attention_single(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    used_len: int,
    prefix_len: int,
    softmax_scale: float,
    sparsity: float | None = None,
) -> torch.Tensor | None:
    """Whole-sequence block-sparse attention for the packed SDPA layout.

    The SDPA backend attends one packed row as a single document, so unlike
    :func:`h3_block_sparse_attention` there are no cu_seqlens segments to
    split on. ``prefix_len`` marks the leading rows that are not video (text,
    visual conditions, audio); their blocks stay visible to every query.

    Returns ``None`` -- after one warning -- when the kernel cannot be
    imported or the shapes are unsupported, so the caller keeps SDPA.
    """
    try:
        return _run_single(
            q,
            k,
            v,
            used_len=used_len,
            prefix_len=prefix_len,
            softmax_scale=softmax_scale,
            sparsity=sparsity,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning_once(
            "H3 block-sparse attention unavailable (%s: %s); using dense attention.",
            type(exc).__name__,
            exc,
        )
        return None


@torch.no_grad()
def _run_single(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    used_len: int,
    prefix_len: int,
    softmax_scale: float,
    sparsity: float | None = None,
) -> torch.Tensor | None:
    if not q.is_cuda:
        raise RuntimeError("block-sparse attention requires a CUDA/MACA device")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(f"expects matching q/k/v rows, got {tuple(q.shape)}/{tuple(k.shape)}/{tuple(v.shape)}")
    if q.shape[-1] != _BLOCK:
        raise ValueError(f"head dim {q.shape[-1]} is unsupported (the kernel expects {_BLOCK})")

    length = int(used_len)
    if length < _MIN_SPARSE_BLOCKS * _BLOCK:
        # Too short to be worth sparsifying: let the dense backend handle it.
        _log_single_request(sparsity, length, 0, 0, engaged=False)
        return None
    if length > q.shape[0]:
        raise ValueError(f"used_len {length} exceeds the packed rows {q.shape[0]}")

    bsa = _load_kernel()
    qs, ks, vs = q[:length], k[:length], v[:length]
    pad = (-length) % _BLOCK
    if pad:
        span = min(length, 4096)
        src = length - 1 - (torch.arange(pad, device=q.device) * 97) % span
        qs = torch.nn.functional.pad(qs, (0, 0, 0, 0, 0, pad))
        ks = torch.cat([ks, ks[src]], dim=0)
        vs = torch.cat([vs, vs[src]], dim=0)
    if qs.dtype != torch.bfloat16:
        qs, ks, vs = (tensor.to(torch.bfloat16) for tensor in (qs, ks, vs))

    if sparsity is None:
        sparsity = _sparsity()
    sparsity = float(sparsity)
    heads = qs.shape[1]
    seq_padded = qs.shape[0]
    prefix_blocks = (max(int(prefix_len), 0) + _BLOCK - 1) // _BLOCK
    _log_single_request(sparsity, length, int(prefix_len), prefix_blocks, engaged=True)
    mask = _topk_block_mask(qs, ks, sparsity, prefix_blocks=prefix_blocks)
    cu = torch.tensor([0, seq_padded], device=q.device, dtype=torch.int32)
    head_mask_type = torch.ones(heads, device=q.device, dtype=torch.int32)
    result = bsa.block_sparse_attn_func(
        qs,
        ks,
        vs,
        cu,
        cu,
        head_mask_type,
        None,
        mask,
        seq_padded,
        seq_padded,
        0.0,
        deterministic=False,
        softmax_scale=softmax_scale,
        is_causal=False,
        exact_streaming=False,
        return_attn_probs=False,
    )
    out = torch.zeros_like(v)
    out[:length] = result[:length].to(v.dtype)
    return out


__all__ = [
    "h3_block_sparse_attention",
    "h3_block_sparse_attention_single",
    "h3_bsa_enabled",
]
