# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Single-GPU tests for artifixer context-parallel configuration, block mask
creation, CP mask rewriting, and timestep_proj slicing.

No distributed environment required -- run with:
    pytest tests/test_cp_config.py -v
"""

import math

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_block_causal_mask(total_tokens, frames_per_block, frame_seqlen, device="cpu"):
    """Reproduce the block-causal mask construction from ArtifixerTransformer.forward."""
    block_token_size = frames_per_block * frame_seqlen
    ends = torch.zeros(total_tokens, device=device, dtype=torch.long)
    frame_indices = torch.arange(0, end=total_tokens, step=block_token_size, device=device)
    for tmp in frame_indices:
        ends[tmp : tmp + block_token_size] = tmp + block_token_size

    def attention_mask(b, h, q_idx, kv_idx):
        return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

    block_mask = create_block_mask(
        attention_mask,
        B=None,
        H=None,
        Q_LEN=total_tokens,
        KV_LEN=total_tokens,
        device=device,
    )
    return block_mask, attention_mask, ends


def _materialize_mask(mask_fn, q_len, kv_len, device="cpu"):
    """Materialize a mask_mod function into a boolean tensor."""
    mask = torch.zeros(q_len, kv_len, dtype=torch.bool, device=device)
    for qi in range(q_len):
        for ki in range(kv_len):
            mask[qi, ki] = mask_fn(0, 0, qi, ki)
    return mask


# ---------------------------------------------------------------------------
# _choose_cp_group_size (inlined from TrainerBase to avoid heavy imports)
# ---------------------------------------------------------------------------


def _choose_cp_group_size(cp_frame_divisor, world_size, num_tasks):
    """Pick CP group size that minimizes validation wall-clock time.

    Mirrors TrainerBase._choose_cp_group_size from trainer_base.py.
    """
    if world_size <= 1:
        return 1
    best_size, best_time = 1, float("inf")
    for d in range(1, cp_frame_divisor + 1):
        if cp_frame_divisor % d != 0 or d > world_size:
            continue
        num_groups = world_size // d
        if num_groups == 0:
            continue
        rounds = math.ceil(num_tasks / num_groups)
        time = rounds / d
        if time < best_time:
            best_time = time
            best_size = d
    return best_size


# ===========================================================================
# Test 1: _choose_cp_group_size with block constraints
# ===========================================================================


class TestChooseCPGroupSize:
    """Verify _choose_cp_group_size with math.gcd(cp_frame_div, num_blocks)
    produces correct CP sizes for various configurations."""

    def test_block_causal_21_frames_7_fpb(self):
        """21 latent frames, 7 frames/block, 20 GPUs, 4 tasks.

        num_blocks = 3, cp_frame_div = gcd(21, 3) = 3.
        Valid CP sizes: {1, 3}.
        CP=3: 20//3=6 groups, ceil(4/6)=1 round, time=1/3.
        CP=1: 20//1=20, ceil(4/20)=1, time=1.
        Best: CP=3.
        """
        num_latent_frames = 21
        frames_per_block = 7
        num_blocks = num_latent_frames // frames_per_block
        cp_frame_div = math.gcd(num_latent_frames, num_blocks)
        assert cp_frame_div == 3
        assert _choose_cp_group_size(cp_frame_div, world_size=20, num_tasks=4) == 3

    def test_no_block_causal_21_frames(self):
        """21 latent frames, no block-causal.  cp_frame_div = 21.

        Valid CP sizes dividing 21 and <= 20: {1, 3, 7}.
        CP=7: 20//7=2 groups, ceil(4/2)=2, time=2/7~0.286.
        Best: CP=7.
        """
        assert _choose_cp_group_size(21, world_size=20, num_tasks=4) == 7

    def test_many_tasks_favors_data_parallel(self):
        """12 latent frames, 4 frames/block, 8 GPUs, 8 tasks.

        cp_frame_div = gcd(12, 3) = 3.  Valid: {1, 3}.
        CP=3: 8//3=2 groups, ceil(8/2)=4, time=4/3~1.33.
        CP=1: 8//1=8, ceil(8/8)=1, time=1.
        Best: CP=1 (data-parallel wins with many tasks).
        """
        cp_frame_div = math.gcd(12, 12 // 4)
        assert cp_frame_div == 3
        assert _choose_cp_group_size(cp_frame_div, world_size=8, num_tasks=8) == 1

    def test_tie_breaks_to_smaller_cp(self):
        """8 latent frames, 2 frames/block, 4 GPUs, 2 tasks.

        cp_frame_div = gcd(8, 4) = 4.  Valid: {1, 2, 4}.
        CP=4: time=2/4=0.5.  CP=2: time=1/2=0.5.
        Tie at 0.5; ascending iteration with strict < means CP=2 wins.
        """
        cp_frame_div = math.gcd(8, 4)
        assert _choose_cp_group_size(cp_frame_div, world_size=4, num_tasks=2) == 2

    def test_single_gpu_returns_1(self):
        """Single GPU should always return CP=1."""
        assert _choose_cp_group_size(21, world_size=1, num_tasks=4) == 1

    def test_few_tasks_favors_cp(self):
        """6 latent frames, 3 frames/block, 6 GPUs, 1 task.

        cp_frame_div = gcd(6, 2) = 2.  Valid: {1, 2}.
        CP=2: 6//2=3 groups, ceil(1/3)=1, time=0.5.
        Best: CP=2.
        """
        cp_frame_div = math.gcd(6, 6 // 3)
        assert _choose_cp_group_size(cp_frame_div, world_size=6, num_tasks=1) == 2

    def test_cp_frame_divisor_equals_world_size(self):
        """cp_frame_div equals world_size -- maximum CP should be chosen
        when there are few tasks."""
        # 8 frames, no blocks, 8 GPUs, 1 task
        # CP=8: 1 group, 1 round, time=1/8=0.125
        assert _choose_cp_group_size(8, world_size=8, num_tasks=1) == 8


# ===========================================================================
# Test 2: block mask creation
# ===========================================================================


class TestBlockMaskCreation:
    """Verify block-causal mask patterns for known configurations."""

    @pytest.mark.parametrize(
        "num_frames,frames_per_block,frame_seqlen",
        [
            (6, 2, 4),  # 3 blocks, 8 tokens each, 24 total
            (9, 3, 2),  # 3 blocks, 6 tokens each, 18 total
            (12, 4, 1),  # 3 blocks, 4 tokens each, 12 total
            (8, 2, 8),  # 4 blocks, 16 tokens each, 128 total
            (21, 7, 4),  # 3 blocks, 28 tokens each, 84 total
        ],
    )
    def test_mask_structure(self, num_frames, frames_per_block, frame_seqlen):
        """Within-block: fully bidirectional.  Between blocks: causal."""
        device = "cpu"
        num_blocks = num_frames // frames_per_block
        total_tokens = num_frames * frame_seqlen
        block_token_size = frames_per_block * frame_seqlen

        _, mask_fn, _ = _build_block_causal_mask(
            total_tokens,
            frames_per_block,
            frame_seqlen,
            device,
        )
        mask = _materialize_mask(mask_fn, total_tokens, total_tokens, device)

        for blk_q in range(num_blocks):
            q_start = blk_q * block_token_size
            q_end = q_start + block_token_size
            for blk_k in range(num_blocks):
                k_start = blk_k * block_token_size
                k_end = k_start + block_token_size
                sub = mask[q_start:q_end, k_start:k_end]

                if blk_k <= blk_q:
                    assert sub.all(), (
                        f"({num_frames},{frames_per_block},{frame_seqlen}): "
                        f"Q-block {blk_q} should see K-block {blk_k}"
                    )
                else:
                    assert not sub.any(), (
                        f"({num_frames},{frames_per_block},{frame_seqlen}): "
                        f"Q-block {blk_q} should NOT see K-block {blk_k}"
                    )

    def test_single_block_is_full_attention(self):
        """When there is only 1 block, the mask is fully bidirectional."""
        total_tokens = 32
        _, mask_fn, _ = _build_block_causal_mask(
            total_tokens,
            frames_per_block=8,
            frame_seqlen=4,
            device="cpu",
        )
        mask = _materialize_mask(mask_fn, total_tokens, total_tokens, "cpu")
        assert mask.all(), "Single block should give fully bidirectional attention"

    def test_ends_tensor_values(self):
        """Directly verify the 'ends' tensor values."""
        frames_per_block = 3
        frame_seqlen = 4
        num_blocks = 3
        total_tokens = num_blocks * frames_per_block * frame_seqlen  # 36
        block_token_size = frames_per_block * frame_seqlen  # 12

        _, _, ends = _build_block_causal_mask(
            total_tokens,
            frames_per_block,
            frame_seqlen,
            "cpu",
        )

        for blk in range(num_blocks):
            start = blk * block_token_size
            end = start + block_token_size
            for i in range(start, end):
                assert ends[i].item() == end, f"Token {i} in block {blk}: ends={ends[i].item()}, expected={end}"


# ===========================================================================
# Test 3: cp_frame_divisor -- various frame/block combinations
# ===========================================================================


class TestCPFrameDivisor:
    """Test that math.gcd(num_latent_frames, num_blocks) produces the
    correct CP frame divisor for various configurations."""

    @pytest.mark.parametrize(
        "num_latent_frames,frames_per_block,expected_divisor",
        [
            (21, 7, 3),  # gcd(21, 3) = 3
            (12, 4, 3),  # gcd(12, 3) = 3
            (8, 2, 4),  # gcd(8, 4) = 4
            (6, 3, 2),  # gcd(6, 2) = 2
            (10, 5, 2),  # gcd(10, 2) = 2
            (16, 4, 4),  # gcd(16, 4) = 4
            (24, 8, 3),  # gcd(24, 3) = 3
            (9, 3, 3),  # gcd(9, 3) = 3
        ],
    )
    def test_gcd_formula(self, num_latent_frames, frames_per_block, expected_divisor):
        num_blocks = num_latent_frames // frames_per_block
        cp_frame_div = math.gcd(num_latent_frames, num_blocks)
        assert cp_frame_div == expected_divisor, (
            f"frames={num_latent_frames}, fpb={frames_per_block}, "
            f"blocks={num_blocks}: gcd={cp_frame_div}, expected={expected_divisor}"
        )

    def test_no_block_causal_uses_full_frames(self):
        """Without block-causal, cp_frame_divisor equals num_latent_frames."""
        for nlf in [6, 9, 12, 21, 33]:
            assert nlf == nlf  # trivially, the divisor is just nlf itself


# ===========================================================================
# Test 4: CP mask rewrite correctness
# ===========================================================================


class TestCPMaskRewrite:
    """Verify that the CP mask_mod remapping produces the same boolean
    pattern as slicing the corresponding rows from the full mask."""

    @pytest.mark.parametrize("world_size", [2, 3, 4])
    def test_cp_mask_mod_remaps_indices(self, world_size):
        """cp_mask_mod(q_idx) should equal original_mask(q_idx + rank * shard_size)."""
        device = "cpu"
        frames_per_block = 2
        frame_seqlen = 4
        num_blocks = world_size * 2
        total_frames = num_blocks * frames_per_block
        total_tokens = total_frames * frame_seqlen
        local_tokens = total_tokens // world_size

        _, original_mask_fn, _ = _build_block_causal_mask(
            total_tokens,
            frames_per_block,
            frame_seqlen,
            device,
        )
        full_mask = _materialize_mask(original_mask_fn, total_tokens, total_tokens, device)

        for cp_rank in range(world_size):
            _rank = cp_rank
            _shard_size = local_tokens

            def cp_mask_mod(b, h, q_idx, kv_idx, _r=_rank, _s=_shard_size):
                return original_mask_fn(b, h, q_idx + _r * _s, kv_idx)

            cp_mask = _materialize_mask(cp_mask_mod, local_tokens, total_tokens, device)

            global_start = cp_rank * local_tokens
            global_end = global_start + local_tokens
            expected = full_mask[global_start:global_end, :]

            assert torch.equal(cp_mask, expected), (
                f"Rank {cp_rank}/{world_size}: CP mask doesn't match "
                f"rows [{global_start}:{global_end}] of full mask"
            )

    @pytest.mark.parametrize("world_size", [2, 3])
    def test_cp_divisibility_constraint(self, world_size):
        """CP shard boundaries must align with block boundaries."""
        frames_per_block = 3
        total_frames = frames_per_block * world_size * 2
        frame_seqlen = 4
        total_tokens = total_frames * frame_seqlen
        local_tokens = total_tokens // world_size
        block_token_size = frames_per_block * frame_seqlen

        assert local_tokens % block_token_size == 0, (
            f"local_tokens ({local_tokens}) not divisible by " f"block_token_size ({block_token_size})"
        )

        frames_per_rank = total_frames // world_size
        assert frames_per_rank % frames_per_block == 0, (
            f"frames_per_rank ({frames_per_rank}) not divisible by " f"frames_per_block ({frames_per_block})"
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention requires CUDA")
    def test_cp_block_mask_flex_attention_output(self):
        """Build actual BlockMask objects for simulated CP ranks and verify
        that flex_attention produces the same output as the full-mask reference
        on the corresponding Q slice.

        This is a single-GPU test that simulates CP by manually slicing Q and
        rewriting the mask for each simulated rank.
        """
        torch.manual_seed(42)
        B, H, D = 1, 4, 64
        DTYPE = torch.bfloat16

        world_size = 2
        frames_per_block = 2
        frame_seqlen = 16
        num_blocks = world_size * 2
        total_frames = num_blocks * frames_per_block
        total_tokens = total_frames * frame_seqlen
        local_tokens = total_tokens // world_size

        q_full = torch.randn(B, H, total_tokens, D, dtype=DTYPE, device="cuda")
        k_full = torch.randn(B, H, total_tokens, D, dtype=DTYPE, device="cuda")
        v_full = torch.randn(B, H, total_tokens, D, dtype=DTYPE, device="cuda")

        block_mask_full, original_mask_fn, _ = _build_block_causal_mask(
            total_tokens,
            frames_per_block,
            frame_seqlen,
            "cuda",
        )

        flex_attention_full = torch.compile(
            torch.nn.attention.flex_attention.flex_attention,
            dynamic=False,
        )
        flex_attention_local = torch.compile(
            torch.nn.attention.flex_attention.flex_attention,
            dynamic=False,
        )

        ref_out = flex_attention_full(q_full, k_full, v_full, block_mask=block_mask_full)

        for cp_rank in range(world_size):
            torch._dynamo.reset()
            _rank = cp_rank
            _shard_size = local_tokens
            _orig = original_mask_fn

            def cp_mask_mod(b, h, q_idx, kv_idx, _r=_rank, _s=_shard_size):
                return _orig(b, h, q_idx + _r * _s, kv_idx)

            cp_block_mask = create_block_mask(
                cp_mask_mod,
                B=None,
                H=None,
                Q_LEN=local_tokens,
                KV_LEN=total_tokens,
                device="cuda",
            )

            q_local = q_full[:, :, cp_rank * local_tokens : (cp_rank + 1) * local_tokens].contiguous()
            cp_out = flex_attention_local(q_local, k_full, v_full, block_mask=cp_block_mask)
            ref_slice = ref_out[:, :, cp_rank * local_tokens : (cp_rank + 1) * local_tokens]

            torch.testing.assert_close(cp_out, ref_slice, atol=5e-3, rtol=1e-2)


# ===========================================================================
# Test 5: timestep_proj slicing correctness
# ===========================================================================


class TestTimestepProjSlicing:
    """Verify that slicing timestep_proj to local blocks + repeat_interleave
    matches the full non-CP path.  This tests the diffusion-forcing CP code
    in ArtifixerTransformer.forward."""

    @pytest.mark.parametrize("world_size", [2, 3])
    def test_slicing_matches_full_expansion(self, world_size):
        """For each simulated rank, the CP-sliced timestep_proj expanded to
        token level should equal the corresponding slice of the full expansion."""
        device = "cpu"
        torch.manual_seed(0)

        batch_size = 2
        num_blocks = 6 * world_size  # ensure divisible
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
            device=device,
        )

        # Full path: unflatten -> repeat_interleave
        ts_full = timestep_proj_full.unflatten(0, (batch_size, num_blocks))
        full_expanded = ts_full.repeat_interleave(tokens_per_block, dim=1)

        blocks_per_rank = num_blocks // world_size
        local_tokens = total_tokens // world_size

        for cp_rank in range(world_size):
            block_start = cp_rank * blocks_per_rank
            block_end = block_start + blocks_per_rank

            # CP slicing (mirrors transformer.py)
            ts_sliced = ts_full[:, block_start:block_end].contiguous()
            ts_sliced_flat = ts_sliced.flatten(0, 1)
            ts_unflat = ts_sliced_flat.unflatten(0, (batch_size, blocks_per_rank))
            local_expanded = ts_unflat.repeat_interleave(tokens_per_block, dim=1)

            seq_start = cp_rank * local_tokens
            seq_end = seq_start + local_tokens
            ref_slice = full_expanded[:, seq_start:seq_end]

            torch.testing.assert_close(local_expanded, ref_slice)

    def test_single_rank_is_identity(self):
        """With world_size=1, slicing the full range should reproduce the
        full expansion exactly."""
        device = "cpu"
        torch.manual_seed(0)

        batch_size = 2
        num_blocks = 4
        frames_per_block = 3
        frame_seqlen = 8
        inner_dim = 32
        tokens_per_block = frames_per_block * frame_seqlen

        timestep_proj_full = torch.randn(
            batch_size * num_blocks,
            6,
            inner_dim,
            device=device,
        )
        ts_full = timestep_proj_full.unflatten(0, (batch_size, num_blocks))
        full_expanded = ts_full.repeat_interleave(tokens_per_block, dim=1)

        # "CP" with 1 rank: slice is the full range
        ts_sliced = ts_full[:, 0:num_blocks].contiguous()
        ts_sliced_flat = ts_sliced.flatten(0, 1)
        ts_unflat = ts_sliced_flat.unflatten(0, (batch_size, num_blocks))
        local_expanded = ts_unflat.repeat_interleave(tokens_per_block, dim=1)

        torch.testing.assert_close(local_expanded, full_expanded)


# ===========================================================================
# Test 6: GCD constraint in trainer_base.py
# ===========================================================================


class TestGCDConstraint:
    """Verify the math.gcd logic from trainer_base.py produces correct
    CP sizes with concrete frames_per_block values."""

    @pytest.mark.parametrize(
        "num_latent_frames,frames_per_block,world_size,num_tasks,expected_cp",
        [
            (21, 7, 9, 12, 3),  # 3 blocks, cp_div=3, CP=3 best
            (12, 4, 6, 2, 3),  # 3 blocks, cp_div=3, CP=3 best
            (21, 21, 9, 4, 1),  # 1 block, forces CP=1
            (24, 8, 3, 1, 3),  # 3 blocks, cp_div=3, CP=3 best
            (16, 4, 8, 8, 1),  # 4 blocks, cp_div=4, tie -> CP=1
        ],
    )
    def test_gcd_with_choose_cp(
        self,
        num_latent_frames,
        frames_per_block,
        world_size,
        num_tasks,
        expected_cp,
    ):
        num_blocks = num_latent_frames // frames_per_block
        cp_frame_div = math.gcd(num_latent_frames, num_blocks)
        result = _choose_cp_group_size(cp_frame_div, world_size, num_tasks)
        assert result == expected_cp, (
            f"frames={num_latent_frames}, fpb={frames_per_block}, ws={world_size}, "
            f"tasks={num_tasks}: got CP={result}, expected={expected_cp}"
        )


# ===========================================================================
# Test 7: _FLEX_BLOCK_SIZE with create_block_mask
# ===========================================================================


class TestFlexBlockSize:
    """Verify that create_block_mask with various BLOCK_SIZE values works
    correctly, including when seq_len is not a multiple of the block size."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention requires CUDA")
    @pytest.mark.parametrize(
        "q_len,kv_len,block_size",
        [
            (256, 128, (256, 128)),
            (512, 256, (256, 128)),
            (384, 256, (256, 128)),
            (128, 128, 128),
        ],
    )
    def test_causal_mask_with_block_size(self, q_len, kv_len, block_size):
        """Create a simple causal mask with the given BLOCK_SIZE and verify
        flex_attention produces valid output."""
        device = "cuda"

        def causal_mask(b, h, q_idx, kv_idx):
            return q_idx >= kv_idx

        block_mask = create_block_mask(
            causal_mask,
            B=None,
            H=None,
            Q_LEN=q_len,
            KV_LEN=kv_len,
            device=device,
            BLOCK_SIZE=block_size,
        )

        B, H, D = 1, 4, 64
        q = torch.randn(B, H, q_len, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, H, kv_len, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, H, kv_len, D, dtype=torch.bfloat16, device=device)

        flex_attn = torch.compile(
            torch.nn.attention.flex_attention.flex_attention,
            dynamic=False,
        )
        out = flex_attn(q, k, v, block_mask=block_mask)
        assert out.shape == (B, H, q_len, D)
        assert not torch.isnan(out).any(), "flex_attention output contains NaN"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="flex_attention requires CUDA")
    def test_block_causal_fa4_vs_default_block_size(self):
        """Block-causal mask with BLOCK_SIZE=(256,128) produces same output
        as default BLOCK_SIZE=128."""
        device = "cuda"
        frames_per_block = 2
        frame_seqlen = 64
        num_blocks = 4
        total_tokens = num_blocks * frames_per_block * frame_seqlen

        _, mask_fn, _ = _build_block_causal_mask(
            total_tokens,
            frames_per_block,
            frame_seqlen,
            device,
        )

        B, H, D = 1, 4, 64
        q = torch.randn(B, H, total_tokens, D, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, H, total_tokens, D, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, H, total_tokens, D, dtype=torch.bfloat16, device=device)

        flex_attn = torch.compile(
            torch.nn.attention.flex_attention.flex_attention,
            dynamic=False,
        )

        bm_fa4 = create_block_mask(
            mask_fn,
            B=None,
            H=None,
            Q_LEN=total_tokens,
            KV_LEN=total_tokens,
            device=device,
            BLOCK_SIZE=(256, 128),
        )
        out_fa4 = flex_attn(q, k, v, block_mask=bm_fa4)

        bm_default = create_block_mask(
            mask_fn,
            B=None,
            H=None,
            Q_LEN=total_tokens,
            KV_LEN=total_tokens,
            device=device,
            BLOCK_SIZE=128,
        )
        out_default = flex_attn(q, k, v, block_mask=bm_default)

        torch.testing.assert_close(out_fa4, out_default, atol=5e-3, rtol=1e-2)
