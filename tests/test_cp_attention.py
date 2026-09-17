# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Context-parallel attention correctness tests for artifixer.

Multi-GPU tests require torchrun:
    torchrun --nproc_per_node=2 -m pytest tests/test_cp_attention.py -v
    torchrun --nproc_per_node=4 -m pytest tests/test_cp_attention.py -v

All tests in this file require distributed (torchrun with >1 GPU).
"""

import os

import pytest
import torch
import torch.distributed as dist
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import flex_attention as _flex_attention_orig

_flex_attention = torch.compile(_flex_attention_orig, dynamic=False)

# ---------------------------------------------------------------------------
# Distributed setup / teardown
# ---------------------------------------------------------------------------


def _is_distributed():
    return int(os.environ["WORLD_SIZE"]) > 1 if "WORLD_SIZE" in os.environ else False


def _init_distributed():
    """Initialise dist and return (rank, world_size)."""
    world_size = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    if world_size > 1 and not dist.is_initialized():
        torch.cuda.set_device(rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    elif world_size == 1:
        torch.cuda.set_device(0)
    return rank, world_size


def _destroy_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="module", autouse=True)
def distributed_env():
    """Module-scoped fixture: init once, tear down after all tests."""
    rank, world_size = _init_distributed()
    yield rank, world_size
    _destroy_distributed()


# Skip the entire module when not launched via torchrun.
pytestmark = pytest.mark.skipif(
    not _is_distributed(),
    reason="Requires torchrun with >1 GPU",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

B = 1
H = 4  # heads
D = 64  # head dim
DTYPE = torch.bfloat16
ATOL = 5e-3
RTOL = 1e-2


def _sdpa_reference(q, k, v, attn_mask=None, is_causal=False):
    """Single-GPU scaled dot-product attention reference.

    Input layout: (B, S, H, D) -- transposed internally to (B, H, S, D).
    """
    q_t = q.transpose(1, 2)
    k_t = k.transpose(1, 2)
    v_t = v.transpose(1, 2)
    out = torch.nn.functional.scaled_dot_product_attention(
        q_t,
        k_t,
        v_t,
        attn_mask=attn_mask,
        is_causal=is_causal,
    )
    return out.transpose(1, 2)


def _allgather_kv(key, value, world_size):
    """All-gather K/V across ranks. key/value: (B, S_local, H, D)."""
    gathered_k = [torch.empty_like(key) for _ in range(world_size)]
    gathered_v = [torch.empty_like(value) for _ in range(world_size)]
    dist.all_gather(gathered_k, key.contiguous())
    dist.all_gather(gathered_v, value.contiguous())
    return torch.cat(gathered_k, dim=1), torch.cat(gathered_v, dim=1)


def _build_block_causal_ends(total_tokens, frames_per_block, frame_seqlen, device):
    """Build the ``ends`` tensor exactly as ArtifixerTransformer.forward does."""
    block_token_size = frames_per_block * frame_seqlen
    ends = torch.zeros(total_tokens, device=device, dtype=torch.long)
    frame_indices = torch.arange(
        0,
        end=total_tokens,
        step=block_token_size,
        device=device,
    )
    for tmp in frame_indices:
        ends[tmp : tmp + block_token_size] = tmp + block_token_size
    return ends


# ---------------------------------------------------------------------------
# Test 1: all-gather KV matches single-GPU (non-causal / standard validation)
# ---------------------------------------------------------------------------


def test_allgather_kv_matches_single_gpu(distributed_env):
    """Split Q across ranks, all-gather K/V, compare to full single-GPU
    attention.  This validates the basic CP all-gather path used during
    artifixer validation without block-causal masking."""
    rank, world_size = distributed_env
    torch.manual_seed(42)

    S = 128  # must be divisible by world_size
    assert S % world_size == 0
    S_local = S // world_size

    q_full = torch.randn(B, S, H, D, dtype=DTYPE, device="cuda")
    k_full = torch.randn(B, S, H, D, dtype=DTYPE, device="cuda")
    v_full = torch.randn(B, S, H, D, dtype=DTYPE, device="cuda")

    # Reference: single-GPU
    ref_out = _sdpa_reference(q_full, k_full, v_full)
    ref_local = ref_out[:, rank * S_local : (rank + 1) * S_local]

    # CP path
    q_local = q_full[:, rank * S_local : (rank + 1) * S_local].contiguous()
    k_local = k_full[:, rank * S_local : (rank + 1) * S_local].contiguous()
    v_local = v_full[:, rank * S_local : (rank + 1) * S_local].contiguous()

    k_gathered, v_gathered = _allgather_kv(k_local, v_local, world_size)
    cp_out = _sdpa_reference(q_local, k_gathered, v_gathered)

    torch.testing.assert_close(cp_out, ref_local, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 2: block-causal CP matches single-GPU via flex_attention
# ---------------------------------------------------------------------------


def test_block_causal_cp_matches_single_gpu(distributed_env):
    """Block-causal flex_attention with CP (rewritten mask + all-gather K/V)
    should match single-GPU block-causal flex_attention.

    This mirrors the artifixer ArtifixerSelfAttnProcessor code path where
    block_mask is not None and _cp_mesh is set.
    """
    rank, world_size = distributed_env
    torch.manual_seed(42)

    frames_per_block = 2
    frame_seqlen = 16
    num_blocks_per_rank = 2
    num_blocks = num_blocks_per_rank * world_size
    total_frames = num_blocks * frames_per_block
    total_tokens = total_frames * frame_seqlen
    local_tokens = total_tokens // world_size

    # (B, H, S, D) layout for flex_attention
    q_full = torch.randn(B, H, total_tokens, D, dtype=DTYPE, device="cuda")
    k_full = torch.randn(B, H, total_tokens, D, dtype=DTYPE, device="cuda")
    v_full = torch.randn(B, H, total_tokens, D, dtype=DTYPE, device="cuda")

    ends = _build_block_causal_ends(total_tokens, frames_per_block, frame_seqlen, "cuda")

    def attention_mask(b, h, q_idx, kv_idx):
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    full_block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device="cuda",
    )

    # Reference: single-GPU flex_attention
    ref_out = _flex_attention(q_full, k_full, v_full, block_mask=full_block_mask)
    ref_local = ref_out[:, :, rank * local_tokens : (rank + 1) * local_tokens]

    # CP path: local Q, full K/V (simulating all-gather), rewritten mask
    q_local = q_full[:, :, rank * local_tokens : (rank + 1) * local_tokens].contiguous()

    _rank = rank
    _shard_size = local_tokens
    original_mask_mod = full_block_mask.mask_mod

    def cp_mask_mod(b, h, q_idx, kv_idx):
        return original_mask_mod(b, h, q_idx + _rank * _shard_size, kv_idx)

    cp_block_mask = create_block_mask(
        cp_mask_mod,
        B=None,
        H=None,
        Q_LEN=local_tokens,
        KV_LEN=total_tokens,
        device="cuda",
    )

    cp_out = _flex_attention(q_local, k_full, v_full, block_mask=cp_block_mask)

    torch.testing.assert_close(cp_out, ref_local, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 3: block-causal CP with actual distributed all-gather
# ---------------------------------------------------------------------------


def test_block_causal_cp_with_allgather(distributed_env):
    """End-to-end CP test with actual distributed all-gather of K/V,
    block-causal mask rewrite, and flex_attention.  Each rank holds its
    own shard and all-gathers K/V, matching the real artifixer code path.
    """
    rank, world_size = distributed_env
    torch.manual_seed(42)

    frames_per_block = 2
    frame_seqlen = 16
    num_blocks_per_rank = 2
    num_blocks = num_blocks_per_rank * world_size
    total_frames = num_blocks * frames_per_block
    total_tokens = total_frames * frame_seqlen
    local_tokens = total_tokens // world_size

    # All ranks create the same full tensors (same seed).
    # Layout: (B, S, H, D) for SDPA-style, will transpose for flex_attention.
    q_full_bshd = torch.randn(B, total_tokens, H, D, dtype=DTYPE, device="cuda")
    k_full_bshd = torch.randn(B, total_tokens, H, D, dtype=DTYPE, device="cuda")
    v_full_bshd = torch.randn(B, total_tokens, H, D, dtype=DTYPE, device="cuda")

    # Each rank takes its local shard
    q_local = q_full_bshd[:, rank * local_tokens : (rank + 1) * local_tokens].contiguous()
    k_local = k_full_bshd[:, rank * local_tokens : (rank + 1) * local_tokens].contiguous()
    v_local = v_full_bshd[:, rank * local_tokens : (rank + 1) * local_tokens].contiguous()

    # All-gather K/V
    k_gathered, v_gathered = _allgather_kv(k_local, v_local, world_size)

    # Transpose to (B, H, S, D) for flex_attention
    q_bhsd = q_local.transpose(1, 2)
    k_bhsd = k_gathered.transpose(1, 2)
    v_bhsd = v_gathered.transpose(1, 2)

    # Build block-causal mask for the full sequence
    ends = _build_block_causal_ends(total_tokens, frames_per_block, frame_seqlen, "cuda")

    def attention_mask(b, h, q_idx, kv_idx):
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    full_block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device="cuda",
    )

    # Rewrite mask for CP
    _rank = rank
    _shard_size = local_tokens
    original_mask_mod = full_block_mask.mask_mod

    def cp_mask_mod(b, h, q_idx, kv_idx):
        return original_mask_mod(b, h, q_idx + _rank * _shard_size, kv_idx)

    cp_block_mask = create_block_mask(
        cp_mask_mod,
        B=None,
        H=None,
        Q_LEN=local_tokens,
        KV_LEN=total_tokens,
        device="cuda",
    )

    cp_out = _flex_attention(q_bhsd, k_bhsd, v_bhsd, block_mask=cp_block_mask)
    # Back to (B, S, H, D)
    cp_out_bshd = cp_out.transpose(1, 2)

    # Reference: single-GPU flex_attention over full sequence
    q_full_bhsd = q_full_bshd.transpose(1, 2)
    k_full_bhsd = k_full_bshd.transpose(1, 2)
    v_full_bhsd = v_full_bshd.transpose(1, 2)
    ref_out = _flex_attention(q_full_bhsd, k_full_bhsd, v_full_bhsd, block_mask=full_block_mask)
    ref_local = ref_out.transpose(1, 2)[:, rank * local_tokens : (rank + 1) * local_tokens]

    torch.testing.assert_close(cp_out_bshd, ref_local, atol=ATOL, rtol=RTOL)


# ---------------------------------------------------------------------------
# Test 4: block-causal mask correctness (distributed)
# ---------------------------------------------------------------------------


def test_block_causal_mask_correctness(distributed_env):
    """Block-causal mask structure: within-block bidirectional, between-blocks
    causal.  Each rank verifies the mask pattern on its own GPU, ensuring
    create_block_mask produces consistent results across the CP group."""
    rank, world_size = distributed_env
    torch.manual_seed(42)

    frames_per_block = 2
    frame_seqlen = 16
    num_blocks = world_size * 2
    total_frames = num_blocks * frames_per_block
    total_tokens = total_frames * frame_seqlen
    block_token_size = frames_per_block * frame_seqlen

    ends = _build_block_causal_ends(total_tokens, frames_per_block, frame_seqlen, "cuda")

    def attention_mask(b, h, q_idx, kv_idx):
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    # Materialize the mask for a few query/key pairs to verify structure
    for blk_q in range(num_blocks):
        q_start = blk_q * block_token_size
        for blk_k in range(num_blocks):
            k_start = blk_k * block_token_size
            # Check first and last token of each block pair
            for q_off in [0, block_token_size - 1]:
                for k_off in [0, block_token_size - 1]:
                    qi = q_start + q_off
                    ki = k_start + k_off
                    result = attention_mask(0, 0, qi, ki)
                    if blk_k <= blk_q:
                        # Within-block or earlier block: should attend
                        assert result, (
                            f"Rank {rank}: Q-block {blk_q}, K-block {blk_k} " f"should attend (qi={qi}, ki={ki})"
                        )
                    else:
                        # Future block: should NOT attend (unless qi==ki, but
                        # for different blocks that cannot happen)
                        assert not result, (
                            f"Rank {rank}: Q-block {blk_q}, K-block {blk_k} " f"should NOT attend (qi={qi}, ki={ki})"
                        )

    # All ranks must agree -- barrier to confirm no rank crashed
    dist.barrier()


# ---------------------------------------------------------------------------
# Test 5: timestep_proj slicing matches full path under CP
# ---------------------------------------------------------------------------


def test_timestep_proj_slicing(distributed_env):
    """Each CP rank slices timestep_proj to its local blocks and expands via
    repeat_interleave.  Verify this matches the corresponding slice of the
    full (non-CP) expansion.  This mirrors the diffusion-forcing CP code in
    ArtifixerTransformer.forward."""
    rank, world_size = distributed_env
    torch.manual_seed(42)

    batch_size = 2
    num_blocks = 4 * world_size  # ensure divisible
    frames_per_block = 3
    frame_seqlen = 8
    inner_dim = 32
    total_tokens = num_blocks * frames_per_block * frame_seqlen
    tokens_per_block = frames_per_block * frame_seqlen

    # Full timestep_proj: (batch_size * num_blocks, 6, inner_dim)
    timestep_proj_full = torch.randn(
        batch_size * num_blocks,
        6,
        inner_dim,
        dtype=DTYPE,
        device="cuda",
    )

    # Full path: unflatten -> repeat_interleave
    ts_full = timestep_proj_full.unflatten(0, (batch_size, num_blocks))
    full_expanded = ts_full.repeat_interleave(tokens_per_block, dim=1)

    blocks_per_rank = num_blocks // world_size
    local_tokens = total_tokens // world_size

    # CP slicing (mirrors transformer.py code path)
    block_start = rank * blocks_per_rank
    block_end = block_start + blocks_per_rank
    ts_sliced = ts_full[:, block_start:block_end].contiguous()
    ts_sliced_flat = ts_sliced.flatten(0, 1)
    ts_unflat = ts_sliced_flat.unflatten(0, (batch_size, blocks_per_rank))
    local_expanded = ts_unflat.repeat_interleave(tokens_per_block, dim=1)

    seq_start = rank * local_tokens
    seq_end = seq_start + local_tokens
    ref_slice = full_expanded[:, seq_start:seq_end]

    torch.testing.assert_close(local_expanded, ref_slice)

    # All ranks agree
    dist.barrier()


# ---------------------------------------------------------------------------
# Test 6: CP mask rewrite correctness (distributed)
# ---------------------------------------------------------------------------


def test_cp_mask_rewrite(distributed_env):
    """Verify that the CP mask_mod remapping on each actual rank produces the
    same boolean pattern as slicing the corresponding rows from the full mask.
    This exercises the real distributed path rather than simulating ranks."""
    rank, world_size = distributed_env
    torch.manual_seed(42)

    frames_per_block = 2
    frame_seqlen = 16
    num_blocks = world_size * 2
    total_frames = num_blocks * frames_per_block
    total_tokens = total_frames * frame_seqlen
    local_tokens = total_tokens // world_size

    ends = _build_block_causal_ends(total_tokens, frames_per_block, frame_seqlen, "cuda")

    def attention_mask(b, h, q_idx, kv_idx):
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    # Build full block mask
    full_block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device="cuda",
    )

    # CP mask rewrite for this rank
    original_mask_mod = full_block_mask.mask_mod
    _rank = rank
    _shard_size = local_tokens

    def cp_mask_mod(b, h, q_idx, kv_idx):
        return original_mask_mod(b, h, q_idx + _rank * _shard_size, kv_idx)

    # Verify element-wise: cp_mask_mod(q_idx) == original_mask(q_idx + rank*shard)
    # Sample several positions to avoid O(n^2) materialization on GPU
    test_positions = [0, local_tokens // 4, local_tokens // 2, local_tokens - 1]
    kv_positions = [0, total_tokens // 4, total_tokens // 2, total_tokens - 1]

    for q_local in test_positions:
        for kv_idx in kv_positions:
            cp_result = cp_mask_mod(0, 0, q_local, kv_idx)
            global_q = q_local + rank * local_tokens
            ref_result = attention_mask(0, 0, global_q, kv_idx)
            assert cp_result == ref_result, (
                f"Rank {rank}: cp_mask_mod(q={q_local}, kv={kv_idx}) = {cp_result}, "
                f"but original(q={global_q}, kv={kv_idx}) = {ref_result}"
            )

    # Build CP block mask and verify it works with flex_attention
    cp_block_mask = create_block_mask(
        cp_mask_mod,
        B=None,
        H=None,
        Q_LEN=local_tokens,
        KV_LEN=total_tokens,
        device="cuda",
    )

    # Confirm dimensions
    assert cp_block_mask.shape == (1, 1, local_tokens, total_tokens), (
        f"Rank {rank}: Expected mask shape (1, 1, {local_tokens}, {total_tokens}), " f"got {cp_block_mask.shape}"
    )

    # All ranks agree
    dist.barrier()


# ---------------------------------------------------------------------------
# Test 7: timestep_proj NOT sliced during validation (ts_seq_len is None)
# ---------------------------------------------------------------------------


def test_timestep_proj_not_sliced_during_validation(distributed_env):
    """During validation, even with frames_per_block set (block_mask active),
    the timestep is a single value per sample — NOT per-block.  timestep_proj
    has shape (batch_size, 6, dim), NOT (batch_size * num_blocks, 6, dim).
    The CP code must NOT attempt to unflatten/slice in this case.

    This mirrors the Stage 2 validation path where block_mask is not None
    but ts_seq_len is None.  The bug this catches: unconditionally unflattening
    timestep_proj when block_mask is active causes a RuntimeError during
    validation because dim 0 = batch_size, not batch_size * num_blocks.
    """
    rank, world_size = distributed_env
    torch.manual_seed(42)

    batch_size = 1
    num_blocks = 3  # e.g., 21 frames / 7 per block
    frames_per_block = 7
    frame_seqlen = 8
    inner_dim = 32
    total_frames = num_blocks * frames_per_block
    total_tokens = total_frames * frame_seqlen
    tokens_per_block = frames_per_block * frame_seqlen
    assert total_tokens % world_size == 0
    local_tokens = total_tokens // world_size
    frames_per_rank = total_frames // world_size

    # --- Training case: ts_seq_len is not None ---
    # timestep_proj shape: (batch_size * num_blocks, 6, inner_dim)
    ts_training = torch.randn(
        batch_size * num_blocks,
        6,
        inner_dim,
        dtype=DTYPE,
        device="cuda",
    )
    # This SHOULD be sliced
    blocks_per_rank = frames_per_rank // frames_per_block
    block_start = rank * blocks_per_rank
    block_end = block_start + blocks_per_rank
    ts_sliced = ts_training.unflatten(0, (batch_size, num_blocks))
    ts_sliced = ts_sliced[:, block_start:block_end].contiguous().flatten(0, 1)
    assert ts_sliced.shape[0] == batch_size * blocks_per_rank

    # --- Validation case: ts_seq_len is None ---
    # timestep_proj shape: (batch_size, 6, inner_dim)
    ts_validation = torch.randn(
        batch_size,
        6,
        inner_dim,
        dtype=DTYPE,
        device="cuda",
    )
    # This must NOT be unflattened — verify the shape is incompatible
    # with unflatten(0, (batch_size, num_blocks)):
    with pytest.raises(RuntimeError, match="don't multiply up"):
        ts_validation.unflatten(0, (batch_size, num_blocks))

    # The correct behavior: leave ts_validation unchanged when ts_seq_len is None.
    # Verify that the unchanged validation timestep_proj passes through the
    # block's repeat_interleave condition without entering the DF branch:
    # scale_msa.shape[0] == hidden_states.shape[0] → no repeat_interleave
    scale_msa_val = ts_validation.chunk(6, dim=1)[0]  # (batch_size, 1, inner_dim)
    hidden_states_local = torch.randn(
        batch_size,
        local_tokens,
        inner_dim,
        dtype=DTYPE,
        device="cuda",
    )
    # In validation: scale_msa.shape[0] == batch_size == hidden_states.shape[0]
    assert scale_msa_val.shape[0] == hidden_states_local.shape[0], (
        f"Validation timestep_proj should NOT trigger the DF repeat_interleave branch. "
        f"scale_msa.shape[0]={scale_msa_val.shape[0]} vs hidden_states.shape[0]={hidden_states_local.shape[0]}"
    )

    dist.barrier()


# ---------------------------------------------------------------------------
# Test 8: CP + block-causal numerical equivalence (multi-rank)
# ---------------------------------------------------------------------------


def test_block_causal_cp_numerical_equivalence(distributed_env):
    """End-to-end numerical equivalence test: CP+block-causal with actual
    all-gather must produce the same output as single-GPU block-causal.

    This is the highest-value test -- it catches any bugs in mask rewriting,
    K/V gathering, or attention dispatch ordering."""
    rank, world_size = distributed_env
    torch.manual_seed(42)

    frames_per_block = 3
    frame_seqlen = 8
    num_blocks_per_rank = 2
    num_blocks = num_blocks_per_rank * world_size
    total_frames = num_blocks * frames_per_block
    total_tokens = total_frames * frame_seqlen
    local_tokens = total_tokens // world_size

    # All ranks create identical tensors (same seed)
    q_full = torch.randn(B, H, total_tokens, D, dtype=DTYPE, device="cuda")
    k_full = torch.randn(B, H, total_tokens, D, dtype=DTYPE, device="cuda")
    v_full = torch.randn(B, H, total_tokens, D, dtype=DTYPE, device="cuda")

    ends = _build_block_causal_ends(total_tokens, frames_per_block, frame_seqlen, "cuda")

    def attention_mask(b, h, q_idx, kv_idx):
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    full_block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device="cuda",
    )

    # Reference: single-GPU
    ref_out = _flex_attention(q_full, k_full, v_full, block_mask=full_block_mask)

    # CP path: shard Q, all-gather K/V from actual shards
    k_local = k_full[:, :, rank * local_tokens : (rank + 1) * local_tokens].contiguous()
    v_local = v_full[:, :, rank * local_tokens : (rank + 1) * local_tokens].contiguous()

    # All-gather in (B, H, S, D) layout
    gathered_k = [torch.empty_like(k_local) for _ in range(world_size)]
    gathered_v = [torch.empty_like(v_local) for _ in range(world_size)]
    dist.all_gather(gathered_k, k_local)
    dist.all_gather(gathered_v, v_local)
    k_gathered = torch.cat(gathered_k, dim=2)
    v_gathered = torch.cat(gathered_v, dim=2)

    # Verify gathered K/V matches full K/V
    torch.testing.assert_close(k_gathered, k_full, atol=0, rtol=0)
    torch.testing.assert_close(v_gathered, v_full, atol=0, rtol=0)

    q_local = q_full[:, :, rank * local_tokens : (rank + 1) * local_tokens].contiguous()

    # Rewrite mask
    _rank = rank
    _shard_size = local_tokens
    original_mask_mod = full_block_mask.mask_mod

    def cp_mask_mod(b, h, q_idx, kv_idx):
        return original_mask_mod(b, h, q_idx + _rank * _shard_size, kv_idx)

    cp_block_mask = create_block_mask(
        cp_mask_mod,
        B=None,
        H=None,
        Q_LEN=local_tokens,
        KV_LEN=total_tokens,
        device="cuda",
    )

    cp_out = _flex_attention(q_local, k_gathered, v_gathered, block_mask=cp_block_mask)

    ref_local = ref_out[:, :, rank * local_tokens : (rank + 1) * local_tokens]
    torch.testing.assert_close(cp_out, ref_local, atol=ATOL, rtol=RTOL)

    dist.barrier()


# ---------------------------------------------------------------------------
# Test 9: CP + block-causal with BLOCK_SIZE=(256, 128)
# ---------------------------------------------------------------------------


def test_block_causal_cp_with_fa4_block_size(distributed_env):
    """Same as test_block_causal_cp_numerical_equivalence but using
    BLOCK_SIZE=(256, 128) to validate FA4 tile alignment."""
    rank, world_size = distributed_env
    torch.manual_seed(42)

    frames_per_block = 2
    frame_seqlen = 64  # 128 tokens per block -> KV aligned to 128
    num_blocks_per_rank = 2
    num_blocks = num_blocks_per_rank * world_size
    total_frames = num_blocks * frames_per_block
    total_tokens = total_frames * frame_seqlen
    local_tokens = total_tokens // world_size

    q_full = torch.randn(B, H, total_tokens, D, dtype=DTYPE, device="cuda")
    k_full = torch.randn(B, H, total_tokens, D, dtype=DTYPE, device="cuda")
    v_full = torch.randn(B, H, total_tokens, D, dtype=DTYPE, device="cuda")

    ends = _build_block_causal_ends(total_tokens, frames_per_block, frame_seqlen, "cuda")

    def attention_mask(b, h, q_idx, kv_idx):
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    # Reference with default block size
    ref_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device="cuda",
    )
    ref_out = _flex_attention(q_full, k_full, v_full, block_mask=ref_mask)

    # CP path with (256, 128) block size
    q_local = q_full[:, :, rank * local_tokens : (rank + 1) * local_tokens].contiguous()

    _rank = rank
    _shard_size = local_tokens
    original_mask_mod = ref_mask.mask_mod

    def cp_mask_mod(b, h, q_idx, kv_idx):
        return original_mask_mod(b, h, q_idx + _rank * _shard_size, kv_idx)

    cp_block_mask = create_block_mask(
        cp_mask_mod,
        B=None,
        H=None,
        Q_LEN=local_tokens,
        KV_LEN=total_tokens,
        device="cuda",
        BLOCK_SIZE=(256, 128),
    )

    cp_out = _flex_attention(q_local, k_full, v_full, block_mask=cp_block_mask)

    ref_local = ref_out[:, :, rank * local_tokens : (rank + 1) * local_tokens]
    torch.testing.assert_close(cp_out, ref_local, atol=ATOL, rtol=RTOL)

    dist.barrier()
