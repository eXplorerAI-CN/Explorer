# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import torch
import torch.distributed as dist
from diffusers import WanTransformer3DModel
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanRotaryPosEmbed,
    WanTransformerBlock,
    _get_added_kv_projections,
    _get_qkv_projections,
)
from diffusers.utils import logging, scale_lora_layers, unscale_lora_layers
from diffusers.utils.constants import USE_PEFT_BACKEND
from einops import rearrange
from torch import nn
from torch.nn.attention import activate_flash_attention_impl, list_flash_attention_impls
from torch.nn.attention.flex_attention import BlockMask, create_block_mask
from torch.nn.attention.flex_attention import flex_attention as _flex_attention_orig

from model_training.net.prope import PropeDotProductAttention

# flex_attention kernel_options are set based on GPU arch after _select_attention_config runs.
_FLEX_KERNEL_OPTIONS: dict[str, str] = {}


def _flex_attention_with_options(query, key, value, block_mask=None):
    return _flex_attention_orig(query, key, value, block_mask=block_mask, kernel_options=_FLEX_KERNEL_OPTIONS)


flex_attention = torch.compile(_flex_attention_with_options, dynamic=False)

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


# ---------------------------------------------------------------------------
# Fix: diffusers >= 0.37 wraps flash_attn_3_func in a _custom_op without
# autograd backward, breaking training.  We replace the _flash_3 backend with
# a torch.autograd.Function that supports both torch.compile (via
# allow_in_graph) and training (proper backward).  This is the same pattern
# FA2 uses (FlashAttnFunc) and should be upstreamed to diffusers.
# See: https://github.com/huggingface/diffusers/issues/12022
# ---------------------------------------------------------------------------
def _patch_flash_3_backend() -> None:
    try:
        from diffusers.models.attention_dispatch import AttentionBackendName, _AttentionBackendRegistry
        from flash_attn_interface import _flash_attn_backward as _fa3_bwd
        from flash_attn_interface import flash_attn_func as _fa3_func

        @torch.compiler.allow_in_graph
        class _FlashAttn3Func(torch.autograd.Function):
            @staticmethod
            def forward(ctx, q, k, v, softmax_scale, causal):
                out, lse, *_ = _fa3_func(
                    q=q,
                    k=k,
                    v=v,
                    softmax_scale=softmax_scale,
                    causal=causal,
                    return_attn_probs=True,
                )
                ctx.save_for_backward(q, k, v, out, lse)
                ctx.softmax_scale = softmax_scale
                ctx.causal = causal
                return out, lse

            @staticmethod
            def backward(ctx, grad_out, grad_lse):
                q, k, v, out, lse = ctx.saved_tensors
                dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
                # Call positionally to match FA3's own backward and avoid
                # custom op kwarg dispatch issues.
                _fa3_bwd(
                    grad_out.contiguous(),
                    q,
                    k,
                    v,
                    out,
                    lse,
                    None,
                    None,  # cu_seqlens_q, cu_seqlens_k
                    None,
                    None,  # sequed_q, sequed_k
                    None,
                    None,  # max_seqlen_q, max_seqlen_k
                    dq,
                    dk,
                    dv,
                    ctx.softmax_scale,
                    ctx.causal,
                    -1,
                    -1,  # window_size_left, window_size_right
                    0.0,  # softcap
                    False,  # deterministic
                    0,  # sm_margin
                )
                dq = dq[..., : q.shape[-1]]
                dk = dk[..., : k.shape[-1]]
                dv = dv[..., : v.shape[-1]]
                return dq, dk, dv, None, None

        def _flash_attention_3(
            query,
            key,
            value,
            attn_mask=None,
            scale=None,
            is_causal=False,
            return_lse=False,
            _parallel_config=None,
            **kwargs,
        ):
            if attn_mask is not None:
                raise ValueError("`attn_mask` is not supported for flash-attn 3.")
            out, lse = _FlashAttn3Func.apply(query, key, value, scale, is_causal)
            return (out, lse) if return_lse else out

        _AttentionBackendRegistry._backends[AttentionBackendName._FLASH_3] = _flash_attention_3
        print("Patched _flash_3 backend with autograd.Function + allow_in_graph")
    except ImportError:
        pass  # FA3 not installed


_patch_flash_3_backend()


def _select_attention_config() -> str | None:
    """Select FA implementation and return the default diffusers attention backend
    based on the GPU architecture.

    Returns the recommended ``dispatch_attention_fn`` backend string, or *None*
    to let diffusers auto-detect (cuDNN SDPA — best for A100 / sm_80).

    Three mechanisms work together:

    * ``activate_flash_attention_impl("FA4")`` — registers FA4 kernels into
      ``aten::_scaled_dot_product_flash_attention`` (used by SDPA ``_native_flash``).
    * ``_FLEX_KERNEL_OPTIONS['BACKEND'] = 'FLASH'`` — tells ``flex_attention`` to
      use FA4's CuTeDSL kernels instead of Triton codegen.
    * ``dispatch_attention_fn(backend=…)`` — diffusers' own dispatch for cross-attn,
      non-block self-attn, and KV-cache attn.

    ======== ==================================== ==========================================
    GPU      ``flex_attention``                    ``dispatch_attention_fn`` backend
    ======== ==================================== ==========================================
    A100     Triton (AUTO)                        ``None`` → auto SDPA (cuDNN)
    H100     FA4 CuTeDSL (BACKEND=FLASH)          ``"_flash_3"`` → Dao-AILab FA3 Hopper
    GB200    FA4 CuTeDSL (BACKEND=FLASH)          ``None`` → auto SDPA (FA4 via aten)
    ======== ==================================== ==========================================

    FA3 is Hopper-only and does not run on Blackwell.
    """
    major, _ = torch.cuda.get_device_capability()
    impls = list_flash_attention_impls()

    if major >= 9 and "FA4" in impls:
        # Activate FA4 for aten SDPA flash override (used by _native_flash on GB200).
        try:
            activate_flash_attention_impl("FA4")
            print(f"Activated FA4 on sm_{major}0")
        except Exception as e:
            print(f"WARNING: FA4 activation failed on sm_{major}0: {e}")

        # Use FA4's CuTeDSL kernels for flex_attention (requires flash_attn.cute).
        _FLEX_KERNEL_OPTIONS["BACKEND"] = "FLASH"
        print("flex_attention will use FA4 CuTeDSL backend")

    if major >= 10:
        # Blackwell (sm_100+) — auto SDPA picks FA4-flash or cuDNN as appropriate.
        print(f"Attention config: sm_{major}0 (Blackwell) — auto SDPA, dispatch=auto")
        return None

    if major >= 9:
        # Hopper (sm_90) — FA3 (Dao-AILab Hopper kernels) for dispatch.
        print(f"Attention config: sm_{major}0 (Hopper) — FA4 aten + FA3 dispatch")
        return "_flash_3"

    # A100 (sm_80) or older — auto SDPA (picks cuDNN).
    print(f"Attention config: sm_{major}0 — auto SDPA (cuDNN), dispatch=auto")
    return None


# Resolved once at import time; used as the default when --attention_backend is
# not supplied (see train_utils.py).
_DEFAULT_ATTENTION_BACKEND = _select_attention_config()

# FA4's CUTE tile sizes require flex_attention BlockMask BLOCK_SIZE to align
# with kernel tile dimensions.  On Blackwell: tile_m (Q) = 256, tile_n (KV) = 128.
# The Q block size must be a multiple of tile_m; the KV block size must equal tile_n.
# When FA4 is not active the default 128 works for both dimensions.
_FLEX_BLOCK_SIZE: int | tuple[int, int] = (256, 128) if _FLEX_KERNEL_OPTIONS.get("BACKEND") == "FLASH" else 128


class ArtifixerTransformer(nn.Module):

    # Context-parallelism state — set via enable/disable_context_parallel.
    _cp_mesh: torch.distributed.DeviceMesh | None = None
    _cp_rank: int = 0
    _cp_world_size: int = 1

    def __init__(
        self,
        base_transformer: WanTransformer3DModel,
        frames_per_block: int | None,
        local_attn_size: int | None,
        sink_size: int,
        vae_scale_factor_spatial: int,
        vae_scale_factor_temporal: int,
        gradient_checkpointing: bool,
        checkpoint_every_n_blocks: int,
    ):
        super().__init__()

        self.rope = OffsetableWanRotaryPosEmbed(
            attention_head_dim=base_transformer.rope.attention_head_dim,
            patch_size=base_transformer.rope.patch_size,
            max_seq_len=base_transformer.rope.max_seq_len,
        )

        # TODO: Add clip image embeddings?
        self.patch_embedding = base_transformer.patch_embedding

        self.patch_size = base_transformer.config.patch_size
        self.condition_embedder = base_transformer.condition_embedder

        opacity_embedding_dim = (
            vae_scale_factor_temporal
            * vae_scale_factor_spatial
            * self.patch_size[1]
            * vae_scale_factor_spatial
            * self.patch_size[2]
        )

        camera_dims = vae_scale_factor_spatial * self.patch_size[1] * vae_scale_factor_spatial * self.patch_size[2]
        camera_embedding_dim = camera_dims * 6

        self.frames_per_block = frames_per_block
        self.block_masks = {}
        self.blocks = nn.ModuleList(
            [
                ArtifixerTransformerBlock(
                    block,
                    opacity_embedding_dim,
                    camera_embedding_dim,
                    local_attn_size,
                    sink_size,
                )
                for block in base_transformer.blocks
            ]
        )

        self.norm_out = base_transformer.norm_out
        self.proj_out = base_transformer.proj_out
        self.scale_shift_table = base_transformer.scale_shift_table
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_every_n_blocks = checkpoint_every_n_blocks

        self.prope_cross_attn_src = PropeDotProductAttention(head_dim=self.rope.attention_head_dim)
        self.prope_cross_attn_tgt = PropeDotProductAttention(head_dim=self.rope.attention_head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        neighbor_hidden_states: torch.Tensor | None,
        opacity: torch.Tensor,
        camera_rays: torch.Tensor,
        w2cs: torch.Tensor,
        neighbor_w2cs: torch.Tensor | None,
        Ks: torch.Tensor,
        neighbor_Ks: torch.Tensor | None,
        ignore_neighbors: bool = False,
        kv_cache: dict[str, torch.Tensor] | None = None,
        crossattn_cache: dict[str, torch.Tensor | bool] | None = None,
        neighbor_crossattn_cache: dict[str, torch.Tensor | bool] | None = None,
        current_start: int = 0,
        frame_offset: int = 0,
        return_dict: bool = False,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if attention_kwargs is not None:
            attention_kwargs = attention_kwargs.copy()
            lora_scale = attention_kwargs.pop("scale", 1.0)
        else:
            lora_scale = 1.0

        if USE_PEFT_BACKEND:
            # weight the lora layers by setting `lora_scale` for each PEFT layer
            scale_lora_layers(self, lora_scale)
        else:
            if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
                logger.warning("Passing `scale` via `attention_kwargs` when not using the PEFT backend is ineffective.")

        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states, frame_offset)

        hidden_states = self.patch_embedding(hidden_states)

        if frame_offset == 0:
            # Pad the first frame to account for the VAE "1 + 4" temporal downsampling
            opacity = torch.cat([opacity[:, :1].repeat_interleave(3, dim=1), opacity], dim=1)

        opacity_extra_patches = rearrange(
            opacity,
            "b (t t4) (h h8) (w w8) -> b (h8 w8 t4) t h w",
            h8=self.vae_scale_factor_spatial * self.patch_size[1],
            w8=self.vae_scale_factor_spatial * self.patch_size[2],
            t4=self.vae_scale_factor_temporal,
        )
        opacity_extra_patches = opacity_extra_patches.flatten(2).transpose(1, 2)

        if camera_rays.shape[1] == hidden_states.shape[2]:
            # Temporal downsampling is already applied
            camera_extra_patches = rearrange(
                camera_rays,
                "b t (h h8) (w w8) c -> b (c h8 w8) t h w",
                h8=self.vae_scale_factor_spatial * self.patch_size[1],
                w8=self.vae_scale_factor_spatial * self.patch_size[2],
            )
        else:
            if frame_offset == 0:
                # Pad the first frame to account for the VAE "1 + 4" temporal downsampling
                camera_rays = torch.cat([camera_rays[:, :1].repeat_interleave(3, dim=1), camera_rays], dim=1)
            camera_extra_patches = rearrange(
                camera_rays,
                "b (t t4) (h h8) (w w8) c -> b (c h8 w8 t4) t h w",
                h8=self.vae_scale_factor_spatial * self.patch_size[1],
                w8=self.vae_scale_factor_spatial * self.patch_size[2],
                t4=self.vae_scale_factor_temporal,
            )
        camera_extra_patches = camera_extra_patches.flatten(2).transpose(1, 2)

        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        if neighbor_hidden_states is not None:
            if (
                self.prope_cross_attn_src.patches_x != post_patch_width
                or self.prope_cross_attn_src.patches_y != post_patch_height
            ):
                self.prope_cross_attn_src.update_coeffs(post_patch_width, post_patch_height, hidden_states.device)

            if (
                self.prope_cross_attn_tgt.patches_x != post_patch_width
                or self.prope_cross_attn_tgt.patches_y != post_patch_height
            ):
                self.prope_cross_attn_tgt.update_coeffs(
                    post_patch_width, post_patch_height, neighbor_hidden_states.device
                )
            self.prope_cross_attn_tgt._precompute_and_cache_apply_fns(neighbor_w2cs, neighbor_Ks)

            neighbor_hidden_states = self.patch_embedding(neighbor_hidden_states)
            neighbor_hidden_states = neighbor_hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len (wan 2.2 ti2v)
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()  # batch_size * seq_len
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, _ = self.condition_embedder(
            timestep, encoder_hidden_states, None, timestep_seq_len=ts_seq_len
        )
        if ts_seq_len is not None:
            # batch_size, seq_len, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            # batch_size, 6, inner_dim
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        frame_seqlen = post_patch_height * post_patch_width
        if self.frames_per_block is not None:
            if hidden_states.shape[1] not in self.block_masks:
                ends = torch.zeros(hidden_states.shape[1], device=hidden_states.device, dtype=torch.long)

                # Block-wise causal mask will attend to all elements that are before the end of the current chunk
                frame_indices = torch.arange(
                    0,
                    end=hidden_states.shape[1],
                    step=frame_seqlen * self.frames_per_block,
                    device=hidden_states.device,
                )

                for tmp in frame_indices:
                    ends[tmp : tmp + frame_seqlen * self.frames_per_block] = tmp + frame_seqlen * self.frames_per_block

                def attention_mask(b, h, q_idx, kv_idx):
                    return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

                self.block_masks[hidden_states.shape[1]] = create_block_mask(
                    attention_mask,
                    B=None,
                    H=None,
                    Q_LEN=hidden_states.shape[1],
                    KV_LEN=hidden_states.shape[1],
                    device=hidden_states.device,
                    BLOCK_SIZE=_FLEX_BLOCK_SIZE,
                )
            block_mask = self.block_masks[hidden_states.shape[1]]
        else:
            block_mask = None

        # Context parallelism: shard sequence at frame boundaries before the block loop.
        # Each rank processes a subset of frames; self-attention all-gathers K/V.
        cp_active = self._cp_mesh is not None
        if cp_active:
            total_frames = post_patch_num_frames
            if total_frames % self._cp_world_size != 0:
                raise ValueError(
                    f"Cannot evenly split {total_frames} post-patch frames across {self._cp_world_size} CP ranks. "
                    f"Adjust num_frames or CP degree so that latent frames are divisible."
                )
            frames_per_rank = total_frames // self._cp_world_size
            seq_start = self._cp_rank * frames_per_rank * frame_seqlen
            seq_end = seq_start + frames_per_rank * frame_seqlen

            hidden_states = hidden_states[:, seq_start:seq_end].contiguous()
            rotary_emb = (
                rotary_emb[0][:, seq_start:seq_end].contiguous(),
                rotary_emb[1][:, seq_start:seq_end].contiguous(),
            )

            if block_mask is not None:
                # Block-causal CP: rewrite mask for local Q range with full KV range.
                # Following imaginaire4's flex_attention_cp pattern.
                if frames_per_rank % self.frames_per_block != 0:
                    raise ValueError(
                        f"CP shard ({frames_per_rank} frames/rank) must be divisible by "
                        f"frames_per_block ({self.frames_per_block}) for block-causal CP."
                    )
                shard_size = frames_per_rank * frame_seqlen
                full_seq_len = total_frames * frame_seqlen
                original_mask_mod = block_mask.mask_mod
                _cp_rank = self._cp_rank
                _shard_size = shard_size

                def cp_mask_mod(b, h, q_idx, kv_idx):
                    return original_mask_mod(b, h, q_idx + _cp_rank * _shard_size, kv_idx)

                cp_cache_key = (shard_size, full_seq_len, self._cp_rank)
                if cp_cache_key not in self.block_masks:
                    self.block_masks[cp_cache_key] = create_block_mask(
                        cp_mask_mod,
                        B=None,
                        H=None,
                        Q_LEN=shard_size,
                        KV_LEN=full_seq_len,
                        device=hidden_states.device,
                        BLOCK_SIZE=_FLEX_BLOCK_SIZE,
                    )
                block_mask = self.block_masks[cp_cache_key]

                # Slice per-block timestep_proj to local blocks — only when
                # timesteps are actually per-block (training with diffusion
                # forcing).  During validation, timestep is a single value per
                # sample so timestep_proj is (batch_size, 6, dim), not
                # (batch_size * num_blocks, 6, dim).
                if ts_seq_len is not None:
                    num_blocks = total_frames // self.frames_per_block
                    blocks_per_rank = frames_per_rank // self.frames_per_block
                    block_start = self._cp_rank * blocks_per_rank
                    block_end = block_start + blocks_per_rank
                    timestep_proj = timestep_proj.unflatten(0, (batch_size, num_blocks))
                    timestep_proj = timestep_proj[:, block_start:block_end].contiguous()
                    timestep_proj = timestep_proj.flatten(0, 1)
            opacity_extra_patches = opacity_extra_patches[:, seq_start:seq_end].contiguous()
            camera_extra_patches = camera_extra_patches[:, seq_start:seq_end].contiguous()

            # PRoPE src: precompute for LOCAL camera/frame range only (tgt already done above with full neighbors)
            if neighbor_hidden_states is not None:
                cam_start = self._cp_rank * frames_per_rank
                cam_end = cam_start + frames_per_rank
                self.prope_cross_attn_src._precompute_and_cache_apply_fns(
                    w2cs[:, cam_start:cam_end], Ks[:, cam_start:cam_end]
                )
        else:
            # Standard path: precompute PRoPE src for all cameras
            if neighbor_hidden_states is not None:
                self.prope_cross_attn_src._precompute_and_cache_apply_fns(w2cs, Ks)

        # 4. Transformer blocks
        use_checkpointing = torch.is_grad_enabled() and self.gradient_checkpointing

        if use_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs, **kwargs):
                    return module(*inputs, **kwargs)

                return custom_forward

        for i, block in enumerate(self.blocks):
            # NOTE (ruilong): in the case of training (gradient is enabled), we want to update
            # the learnable crossattn so do not use crossattn_cache.
            if torch.is_grad_enabled():
                block_crossattn_cache = None
                block_neighbor_crossattn_cache = None
            else:
                block_crossattn_cache = crossattn_cache[i] if crossattn_cache is not None else None
                block_neighbor_crossattn_cache = (
                    neighbor_crossattn_cache[i] if neighbor_crossattn_cache is not None else None
                )

            args = (
                hidden_states,
                encoder_hidden_states,
                neighbor_hidden_states,
                ignore_neighbors,
                timestep_proj,
                rotary_emb,
                opacity_extra_patches,
                camera_extra_patches,
                kv_cache[i] if kv_cache is not None else None,
                block_crossattn_cache,
                block_neighbor_crossattn_cache,
                current_start,
                frame_seqlen,
                self.prope_cross_attn_src,
                self.prope_cross_attn_tgt,
                block_mask,
            )

            if use_checkpointing and i % self.checkpoint_every_n_blocks == 0:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    *args,
                    use_reentrant=False,
                )
            else:
                hidden_states = block(*args)

        # Context parallelism: gather sharded hidden_states back to full sequence before output projection
        if cp_active:
            gathered = [torch.empty_like(hidden_states) for _ in range(self._cp_world_size)]
            dist.all_gather(gathered, hidden_states, group=self._cp_mesh.get_group())
            hidden_states = torch.cat(gathered, dim=1)

        # 5. Output norm, projection & unpatchify
        shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        # Move the shift and scale tensors to the same device as hidden_states.
        # When using multi-GPU inference via accelerate these will be on the
        # first device rather than the last device, which hidden_states ends up
        # on.
        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        if scale.shape[0] != hidden_states.shape[0]:
            # Using different timesteps for diffusion forcing
            shift = shift.squeeze(1).unflatten(0, (hidden_states.shape[0], shift.shape[0] // hidden_states.shape[0]))
            shift = shift.repeat_interleave(hidden_states.shape[1] // shift.shape[1], dim=1)

            scale = scale.squeeze(1).unflatten(0, (hidden_states.shape[0], scale.shape[0] // hidden_states.shape[0]))
            scale = scale.repeat_interleave(hidden_states.shape[1] // scale.shape[1], dim=1)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def enable_context_parallel(self, mesh: torch.distributed.DeviceMesh) -> None:
        """Enable context parallelism for inference.

        The sequence is split at frame boundaries before the block loop; each rank
        processes its local shard.  Self-attention all-gathers K/V across the CP
        group to reconstruct full context without ring rotation.

        Compatible with block-causal attention (diffusion forcing) when
        frames_per_rank is divisible by frames_per_block.  The block mask is
        rewritten per-rank to map local Q indices to global positions.
        """
        self._cp_mesh = mesh
        self._cp_rank = mesh.get_local_rank()
        self._cp_world_size = mesh.size()
        for block in self.blocks:
            block.attn1.processor._cp_mesh = mesh

    def disable_context_parallel(self) -> None:
        """Disable context parallelism."""
        self._cp_mesh = None
        self._cp_rank = 0
        self._cp_world_size = 1
        for block in self.blocks:
            block.attn1.processor._cp_mesh = None


class ArtifixerTransformerBlock(nn.Module):

    def __init__(
        self,
        base_block: WanTransformerBlock,
        opacity_embedding_dim: int,
        camera_embedding_dim: int,
        local_attn_size: int | None,
        sink_size: int,
    ):
        super().__init__()

        if opacity_embedding_dim > 0:
            self.opacity_embedding = nn.Linear(
                opacity_embedding_dim,
                base_block.attn1.inner_dim,
                dtype=base_block.attn1.to_q.weight.dtype,
                device=base_block.attn1.to_q.weight.device,
                bias=True,
            )
            self.opacity_embedding.weight.data.zero_()
            self.opacity_embedding.bias.data.zero_()
        else:
            self.opacity_embedding = None

        if camera_embedding_dim > 0:
            self.camera_embedding = nn.Linear(
                camera_embedding_dim,
                base_block.attn1.inner_dim,
                dtype=base_block.attn1.to_q.weight.dtype,
                device=base_block.attn1.to_q.weight.device,
                bias=True,
            )
            self.camera_embedding.weight.data.zero_()
            self.camera_embedding.bias.data.zero_()
        else:
            self.camera_embedding = None

        # 1. Self-attention
        self.norm1 = base_block.norm1
        self.attn1 = base_block.attn1

        attn1_backend = base_block.attn1.processor._attention_backend
        if local_attn_size is not None:
            self.attn1.processor = KvCacheWanSelfAttnProcessor(local_attn_size, sink_size)
        else:
            self.attn1.processor = ArtifixerSelfAttnProcessor()
        self.attn1.processor._attention_backend = attn1_backend

        # 2. Cross-attention
        self.attn2 = base_block.attn2
        self.norm2 = base_block.norm2

        # 3. Add projections for neighbor cross-attention
        self.attn2.add_k_proj = nn.Linear(
            self.attn2.inner_dim,
            self.attn2.inner_dim,
            bias=True,
            dtype=self.attn2.to_k.weight.dtype,
            device=self.attn2.to_k.weight.device,
        )

        # Zero-initialize to not affect the pretrained model initialization
        self.attn2.add_v_proj = nn.Linear(
            self.attn2.inner_dim,
            self.attn2.inner_dim,
            bias=True,
            dtype=self.attn2.to_v.weight.dtype,
            device=self.attn2.to_v.weight.device,
        )
        self.attn2.add_v_proj.weight.data.zero_()
        self.attn2.add_v_proj.bias.data.zero_()
        self.attn2.norm_added_k = nn.RMSNorm(
            self.attn2.norm_k.normalized_shape,
            eps=self.attn2.norm_k.eps,
            elementwise_affine=True,
            dtype=self.attn2.norm_k.weight.dtype,
            device=self.attn2.norm_k.weight.device,
        )

        attn2_backend = base_block.attn2.processor._attention_backend
        self.attn2.processor = ArtifixerCrossAttnProcessor()
        self.attn2.processor._attention_backend = attn2_backend

        # 4. Feed-forward
        self.ffn = base_block.ffn
        self.norm3 = base_block.norm3
        self.scale_shift_table = base_block.scale_shift_table

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        neighbor_hidden_states: torch.Tensor | None,
        ignore_neighbors: bool,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        opacity_extra_patches: torch.Tensor,
        camera_extra_patches: torch.Tensor,
        kv_cache: dict[str, torch.Tensor] | None,
        crossattn_cache: dict[str, torch.Tensor | bool] | None,
        neighbor_crossattn_cache: dict[str, torch.Tensor | bool] | None,
        current_start: int,
        frame_seqlen: int,
        cross_attn_src: PropeDotProductAttention,
        cross_attn_tgt: PropeDotProductAttention,
        block_mask: BlockMask | None,
    ) -> torch.Tensor:
        # temb: batch_size, 6, inner_dim (wan2.1/wan2.2 14B)
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table + temb.float()
        ).chunk(6, dim=1)

        if scale_msa.shape[0] != hidden_states.shape[0]:
            # Using different timesteps for diffusion forcing
            scale_msa = scale_msa.squeeze(1).unflatten(
                0, (hidden_states.shape[0], scale_msa.shape[0] // hidden_states.shape[0])
            )
            scale_msa = scale_msa.repeat_interleave(hidden_states.shape[1] // scale_msa.shape[1], dim=1)

            shift_msa = shift_msa.squeeze(1).unflatten(
                0, (hidden_states.shape[0], shift_msa.shape[0] // hidden_states.shape[0])
            )
            shift_msa = shift_msa.repeat_interleave(hidden_states.shape[1] // shift_msa.shape[1], dim=1)

            gate_msa = gate_msa.squeeze(1).unflatten(
                0, (hidden_states.shape[0], gate_msa.shape[0] // hidden_states.shape[0])
            )
            gate_msa = gate_msa.repeat_interleave(hidden_states.shape[1] // gate_msa.shape[1], dim=1)

            c_shift_msa = c_shift_msa.squeeze(1).unflatten(
                0, (hidden_states.shape[0], c_shift_msa.shape[0] // hidden_states.shape[0])
            )
            c_shift_msa = c_shift_msa.repeat_interleave(hidden_states.shape[1] // c_shift_msa.shape[1], dim=1)

            c_scale_msa = c_scale_msa.squeeze(1).unflatten(
                0, (hidden_states.shape[0], c_scale_msa.shape[0] // hidden_states.shape[0])
            )
            c_scale_msa = c_scale_msa.repeat_interleave(hidden_states.shape[1] // c_scale_msa.shape[1], dim=1)

            c_gate_msa = c_gate_msa.squeeze(1).unflatten(
                0, (hidden_states.shape[0], c_gate_msa.shape[0] // hidden_states.shape[0])
            )
            c_gate_msa = c_gate_msa.repeat_interleave(hidden_states.shape[1] // c_gate_msa.shape[1], dim=1)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        if self.opacity_embedding is not None:
            norm_hidden_states = norm_hidden_states + self.opacity_embedding(opacity_extra_patches)
        if self.camera_embedding is not None:
            norm_hidden_states = norm_hidden_states + self.camera_embedding(camera_extra_patches)

        if kv_cache is not None:
            attn_output = self.attn1(
                norm_hidden_states,
                None,
                None,
                rotary_emb,
                kv_cache=kv_cache,
                current_start=current_start,
                frame_seqlen=frame_seqlen,
            )
        else:
            attn_output = self.attn1(
                norm_hidden_states,
                None,
                None,
                rotary_emb,
                block_mask=block_mask,
            )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)

        if neighbor_hidden_states is not None:
            neighbor_hidden_states = neighbor_hidden_states.type_as(hidden_states)

        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            None,
            None,
            neighbor_hidden_states=neighbor_hidden_states,
            ignore_neighbors=ignore_neighbors,
            crossattn_cache=crossattn_cache,
            neighbor_crossattn_cache=neighbor_crossattn_cache,
            prope_attn_src=cross_attn_src,
            prope_attn_tgt=cross_attn_tgt,
        )
        hidden_states = hidden_states + attn_output

        # 4. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states


def _apply_rotary_emb(
    hidden_states: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]
    out = torch.empty_like(hidden_states)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out.type_as(hidden_states)


class ArtifixerCrossAttnProcessor:
    _attention_backend = None

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        neighbor_hidden_states: torch.Tensor | None = None,
        ignore_neighbors: bool = False,
        crossattn_cache: dict[str, torch.Tensor | bool] | None = None,
        neighbor_crossattn_cache: dict[str, torch.Tensor | bool] | None = None,
        prope_attn_src: PropeDotProductAttention | None = None,
        prope_attn_tgt: PropeDotProductAttention | None = None,
    ) -> torch.Tensor:
        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
                query = attn.norm_q(query)
                key = attn.norm_k(key)
                crossattn_cache["k"] = key
                crossattn_cache["v"] = value
                crossattn_cache["is_init"] = True
            else:
                query = attn.norm_q(attn.to_q(hidden_states))
                key = crossattn_cache["k"]
                value = crossattn_cache["v"]
        else:
            query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)
            query = attn.norm_q(query)
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        # Add neighbor cross-attention
        hidden_states_neighbor = None
        if neighbor_hidden_states is not None:
            if neighbor_crossattn_cache is not None:
                if not neighbor_crossattn_cache["is_init"]:
                    key_neighbor, value_neighbor = _get_added_kv_projections(attn, neighbor_hidden_states)
                    key_neighbor = attn.norm_added_k(key_neighbor)
                    neighbor_crossattn_cache["k"] = key_neighbor
                    neighbor_crossattn_cache["v"] = value_neighbor
                    neighbor_crossattn_cache["is_init"] = True
                else:
                    key_neighbor = neighbor_crossattn_cache["k"]
                    value_neighbor = neighbor_crossattn_cache["v"]
            else:
                key_neighbor, value_neighbor = _get_added_kv_projections(attn, neighbor_hidden_states)
                key_neighbor = attn.norm_added_k(key_neighbor)

            query_neighbor = query
            key_neighbor = key_neighbor.unflatten(2, (attn.heads, -1))
            value_neighbor = value_neighbor.unflatten(2, (attn.heads, -1))
            query_dtype = query.dtype

            query_neighbor = prope_attn_src._apply_to_q(query.transpose(2, 1).float()).transpose(2, 1).to(query_dtype)
            key_neighbor = (
                prope_attn_tgt._apply_to_kv(key_neighbor.transpose(2, 1).float()).transpose(2, 1).to(query_dtype)
            )
            value_neighbor = (
                prope_attn_tgt._apply_to_kv(value_neighbor.transpose(2, 1).float()).transpose(2, 1).to(query_dtype)
            )

            hidden_states_neighbor = dispatch_attention_fn(
                query_neighbor,
                key_neighbor,
                value_neighbor,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )

            hidden_states_neighbor_dtype = hidden_states_neighbor.dtype
            hidden_states_neighbor = (
                prope_attn_src._apply_to_o(hidden_states_neighbor.transpose(2, 1).float())
                .transpose(2, 1)
                .to(hidden_states_neighbor_dtype)
            )

            hidden_states_neighbor = hidden_states_neighbor.flatten(2, 3)

            hidden_states_neighbor = hidden_states_neighbor.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_neighbor is not None:
            hidden_states = hidden_states + hidden_states_neighbor * (0 if ignore_neighbors else 1)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class ArtifixerSelfAttnProcessor:
    _attention_backend = None
    _cp_mesh: torch.distributed.DeviceMesh | None = None

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        block_mask: BlockMask | None,
    ) -> torch.Tensor:
        query, key, value = _get_qkv_projections(attn, hidden_states, None)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        if self._cp_mesh is not None:
            # CP: all-gather K/V so each rank has full context.
            if block_mask is not None:
                if torch.is_grad_enabled():
                    raise RuntimeError(
                        "CP + block_mask uses non-differentiable all_gather for K/V "
                        "and is only valid during inference. Use ring attention for training."
                    )
            cp_group = self._cp_mesh.get_group()
            cp_size = self._cp_mesh.size()
            gathered_k = [torch.empty_like(key) for _ in range(cp_size)]
            gathered_v = [torch.empty_like(value) for _ in range(cp_size)]
            dist.all_gather(gathered_k, key, group=cp_group)
            dist.all_gather(gathered_v, value, group=cp_group)
            full_key = torch.cat(gathered_k, dim=1)
            full_value = torch.cat(gathered_v, dim=1)
        else:
            full_key = key
            full_value = value

        if block_mask is not None:
            # Block-causal attention (training, or CP validation with rewritten mask).
            hidden_states = flex_attention(
                query=query.transpose(2, 1),
                key=full_key.transpose(2, 1),
                value=full_value.transpose(2, 1),
                block_mask=block_mask,
            ).transpose(2, 1)
        else:
            hidden_states = dispatch_attention_fn(
                query,
                full_key,
                full_value,
                attn_mask=attention_mask if self._cp_mesh is None else None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
            )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class KvCacheWanSelfAttnProcessor:
    _attention_backend = None
    _cp_mesh: torch.distributed.DeviceMesh | None = None

    def __init__(self, local_attn_size: int, sink_size: int):
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size

    def __call__(
        self,
        attn: WanAttention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
        kv_cache: dict[str, torch.Tensor],
        current_start: int,
        frame_seqlen: int,
    ) -> torch.Tensor:
        query, key, value = _get_qkv_projections(attn, hidden_states, None)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:
            query = _apply_rotary_emb(query, *rotary_emb)
            key = _apply_rotary_emb(key, *rotary_emb)

        # With CP each rank's cache stores 1/cp_size of the total frames.
        # Adjust current_start to local indexing and scale sink/window sizes.
        cp_active = self._cp_mesh is not None
        if cp_active:
            cp_size = self._cp_mesh.size()
            current_start = current_start // cp_size
            local_sink_size = self.sink_size // cp_size
            local_attn_size = self.local_attn_size // cp_size if self.local_attn_size != -1 else -1
        else:
            local_sink_size = self.sink_size
            local_attn_size = self.local_attn_size

        num_new_tokens = query.shape[1]
        current_end = current_start + num_new_tokens
        sink_tokens = local_sink_size * frame_seqlen
        # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
        kv_cache_size = kv_cache["k"].shape[1]

        global_end_index = kv_cache["global_end_index"].item()
        local_end_index = kv_cache["local_end_index"].item()
        if (
            local_attn_size != -1
            and (current_end > global_end_index)
            and (num_new_tokens + local_end_index > kv_cache_size)
        ):
            # Calculate the number of new tokens added in this step
            # Shift existing cache content left to discard oldest tokens
            # Clone the source slice to avoid overlapping memory error
            num_evicted_tokens = num_new_tokens + local_end_index - kv_cache_size
            num_rolled_tokens = local_end_index - num_evicted_tokens - sink_tokens
            kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["k"][
                :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
            ].clone()
            kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["v"][
                :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
            ].clone()
            # Insert the new keys/values at the end
            local_end_index = local_end_index + current_end - global_end_index - num_evicted_tokens
            local_start_index = local_end_index - num_new_tokens
            kv_cache["k"][:, local_start_index:local_end_index] = key
            kv_cache["v"][:, local_start_index:local_end_index] = value
        else:
            # Assign new keys/values directly up to current_end
            local_end_index = local_end_index + current_end - global_end_index
            local_start_index = local_end_index - num_new_tokens
            if torch.is_grad_enabled():
                # this seems to be necessary with FSDP2
                # otherwise you get "RuntimeError: Output 0 of SliceBackward0 is a view and is being modified inplace"
                kv_cache["k"] = torch.cat(
                    [
                        kv_cache["k"][:, :local_start_index],
                        key,
                        kv_cache["k"][:, local_end_index:],
                    ],
                    dim=1,
                )
                kv_cache["v"] = torch.cat(
                    [
                        kv_cache["v"][:, :local_start_index],
                        value,
                        kv_cache["v"][:, local_end_index:],
                    ],
                    dim=1,
                )
            else:
                kv_cache["k"][:, local_start_index:local_end_index] = key
                kv_cache["v"][:, local_start_index:local_end_index] = value

        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)

        if local_attn_size != -1:
            window_tokens = (local_attn_size - local_sink_size) * frame_seqlen
            cached_k = torch.cat(
                [
                    kv_cache["k"][:, :sink_tokens],
                    kv_cache["k"][:, max(sink_tokens, local_end_index - window_tokens) : local_end_index],
                ],
                dim=1,
            )
            cached_v = torch.cat(
                [
                    kv_cache["v"][:, :sink_tokens],
                    kv_cache["v"][:, max(sink_tokens, local_end_index - window_tokens) : local_end_index],
                ],
                dim=1,
            )
        else:
            cached_k = kv_cache["k"][:, :local_end_index]
            cached_v = kv_cache["v"][:, :local_end_index]

        # CP: all-gather cached K/V from all ranks to reconstruct the full
        # attention context, then run standard SDPA with local Q.
        if cp_active:
            cp_group = self._cp_mesh.get_group()
            cp_size = self._cp_mesh.size()
            gathered_k = [torch.empty_like(cached_k) for _ in range(cp_size)]
            gathered_v = [torch.empty_like(cached_v) for _ in range(cp_size)]
            dist.all_gather(gathered_k, cached_k, group=cp_group)
            dist.all_gather(gathered_v, cached_v, group=cp_group)
            cached_k = torch.cat(gathered_k, dim=1)
            cached_v = torch.cat(gathered_v, dim=1)

        hidden_states = dispatch_attention_fn(
            query,
            cached_k,
            cached_v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
        )

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class OffsetableWanRotaryPosEmbed(WanRotaryPosEmbed):

    def forward(self, hidden_states: torch.Tensor, frame_offset: int) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [
            self.attention_head_dim - 2 * (self.attention_head_dim // 3),
            self.attention_head_dim // 3,
            self.attention_head_dim // 3,
        ]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][frame_offset : frame_offset + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][frame_offset : frame_offset + ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin
