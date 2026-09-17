# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Inference script for eval.
Distributes scenes across GPUs for parallel inference.

Usage:
    python -m model_eval.run_inference \
        --checkpoint_dir /path/to/checkpoints/checkpoint_10000/pytorch_model_fsdp_0 \
        --save_dir /path/to/output \
        --evalset 3dgrut_dl3dv_ours
"""

import argparse
import gc
import math
import os
import shutil
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.distributed import DeviceMesh
from tqdm import tqdm

from model_eval.checkpoint_loading import (
    add_checkpoint_args,
    checkpoint_output_name,
    load_transformer_checkpoint,
    validate_checkpoint_args,
)
from model_eval.datasets.dl3dv_hdf5_eval import DL3DVHDF5EvalDataset
from model_eval.datasets.nerfbusters_eval import NerfbustersEvalDataset
from model_eval.dl3dv_reconstruction_evalsets import (
    DL3DV_RECONSTRUCTION_EVALSETS,
    add_dl3dv_reconstruction_args,
    create_dl3dv_reconstruction_dataset,
    is_dl3dv_reconstruction_evalset,
)
from model_eval.reconstructed_colmap_evalsets import (
    DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS,
    RECONSTRUCTED_COLMAP_EVALSETS,
    create_reconstructed_colmap_dataset,
    is_reconstructed_colmap_evalset,
)
from model_training.data.utils import NeighborSelectionMode
from model_training.utils.train_utils import (
    barrier_if_distributed,
    get_eval_common_opts,
    get_kv_cache_pipe,
    get_pipe,
)
from model_training.utils.video_io import save_video

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "YES"


EVALSETS = (
    "3dgrut_dl3dv_ours",
    *DL3DV_RECONSTRUCTION_EVALSETS,
    *RECONSTRUCTED_COLMAP_EVALSETS,
    "nerfbusters",
)


def validate_evalset_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.evalset not in EVALSETS:
        parser.error(f"invalid --evalset {args.evalset!r}. Choices: {', '.join(EVALSETS)}")

    if args.bidirectional_chunk_size <= 0:
        parser.error("--bidirectional_chunk_size must be positive")
    if args.output_fps <= 0:
        parser.error("--output_fps must be positive")
    if args.num_views is not None and args.num_views <= 0:
        parser.error("--num_views must be positive")

    if is_dl3dv_reconstruction_evalset(args.evalset):
        required_args = ("split_path", "dl3dv_dir", "prompt_dir", "recon_results_dir")
    else:
        required_args_by_evalset = {
            "nerfbusters": ("nerfbusters_dir", "nerfbusters_recon_results_dir", "nerfbusters_captions_dir"),
            "3dgrut_dl3dv_ours": ("split_path", "dl3dv_dir", "prompt_dir"),
            "reconstructed_colmap": ("split_path",),
        }
        required_args = required_args_by_evalset[args.evalset]

    for name in required_args:
        if getattr(args, name) is None:
            parser.error(f"--{name} is required for --evalset {args.evalset}")

    if args.render_trajectory == "trajectory" and not is_reconstructed_colmap_evalset(args.evalset):
        parser.error("--render_trajectory=trajectory is currently supported only for --evalset reconstructed_colmap")

    if (
        args.num_views is None
        and not is_reconstructed_colmap_evalset(args.evalset)
    ):
        args.num_views = 6


def init_distributed(timeout_minutes: int = 60):
    """Initialize distributed evaluation if available."""
    if "RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=torch.device(f"cuda:{local_rank}"),
            timeout=timedelta(minutes=timeout_minutes),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    return rank, world_size, local_rank


def compute_frame_padding(num_frames: int, frames_per_block: int, vae_temporal_scale: int = 4) -> int:
    """Compute how many frames to pad to satisfy VAE and frames_per_block requirements."""
    original = num_frames
    num_frames += (vae_temporal_scale - (num_frames - 1) % vae_temporal_scale) % vae_temporal_scale
    latent_frames = (num_frames - 1) // vae_temporal_scale + 1
    padded_latent = math.ceil(latent_frames / frames_per_block) * frames_per_block
    padded_num_frames = (padded_latent - 1) * vae_temporal_scale + 1
    return padded_num_frames - original


def compute_context_parallel_padding(num_frames: int, cp_size: int, vae_temporal_scale: int = 4) -> int:
    """Pad further so latent frame count is divisible by the CP group size."""
    if cp_size <= 1:
        return 0

    padded_num_frames = num_frames
    while (((padded_num_frames - 1) // vae_temporal_scale) + 1) % cp_size != 0:
        padded_num_frames += vae_temporal_scale
    return padded_num_frames - num_frames


def latent_num_frames_from_rgb_num_frames(num_frames: int, vae_temporal_scale: int = 4) -> int:
    return ((num_frames - 1) // vae_temporal_scale) + 1


def pad_temporal(tensor: torch.Tensor, pad_frames: int, dim: int = 1) -> torch.Tensor:
    """Pad tensor along temporal dimension by repeating last frame."""
    if pad_frames <= 0:
        return tensor
    last_frame = tensor.select(dim, -1).unsqueeze(dim)
    padding = last_frame.expand(*[-1 if i != dim else pad_frames for i in range(tensor.dim())])
    return torch.cat([tensor, padding], dim=dim)


def pipeline_frames_per_block(pipe) -> int:
    frames_per_block = getattr(pipe, "frames_per_block", None)
    if frames_per_block is None:
        frames_per_block = pipe.transformer.frames_per_block
    return frames_per_block if frames_per_block is not None else 1


def save_image(frame: torch.Tensor, path: Path):
    """Save a single CHW tensor image."""
    image = (frame.cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype("uint8")
    Image.fromarray(image).save(path)


def _to_list(value) -> list:
    if torch.is_tensor(value):
        return value.cpu().tolist()
    return list(value)


def _valid_mask_list(item: dict) -> list[bool]:
    valid_mask = item.get("valid_frames_mask")
    if valid_mask is None:
        return [True] * int(item["rgb_rendered"].shape[0])
    return [bool(value) for value in _to_list(valid_mask)]


def _values_for_valid_frames(value, valid_mask: list[bool], name: str) -> list[int]:
    valid_count = sum(valid_mask)
    if value is None:
        return list(range(valid_count))

    values = [int(entry) for entry in _to_list(value)]
    if len(values) == len(valid_mask):
        return [entry for entry, keep in zip(values, valid_mask) if keep]
    assert len(values) == valid_count, f"{name} has {len(values)} entries; expected {len(valid_mask)} or {valid_count}"
    return values


def _eval_frame_metadata(item: dict) -> tuple[list[bool], list[int], list[int]]:
    valid_mask = _valid_mask_list(item)
    frame_indices = _values_for_valid_frames(item.get("frame_indices"), valid_mask, "frame_indices")
    output_source = item.get("gt_index")
    if output_source is None:
        output_source = item.get("frame_indices")
    output_indices = _values_for_valid_frames(output_source, valid_mask, "output_indices")
    return valid_mask, output_indices, frame_indices


def expected_output_frame_names(item: dict) -> list[str]:
    _, output_indices, _ = _eval_frame_metadata(item)
    return [f"{idx:05d}.png" for idx in output_indices if idx != -1]


def expected_video_frame_names(item: dict) -> list[str]:
    _, _, frame_indices = _eval_frame_metadata(item)
    return [f"{idx:05d}.png" for idx in frame_indices]


def output_kinds(item: dict) -> tuple[str, ...]:
    return ("pred", "gt", "rendered") if "rgb_gt" in item else ("pred", "rendered")


def output_frames_complete(scene_dir: Path, item: dict) -> bool:
    names = expected_output_frame_names(item)
    frames_dir = scene_dir / "frames" / "batch_0000"
    return all((frames_dir / kind / name).is_file() for kind in output_kinds(item) for name in names)


def output_video_frames_complete(scene_dir: Path, item: dict) -> bool:
    names = expected_video_frame_names(item)
    video_frames_dir = scene_dir / "video_frames" / "batch_0000"
    return all((video_frames_dir / kind / name).is_file() for kind in output_kinds(item) for name in names)


def create_dataset(args, rank: int):
    """Create the appropriate dataset based on evalset argument."""
    num_frames = None
    max_test_frames = args.bidirectional_chunk_size if args.inference_pipeline == "bidirectional" else None
    include_all_frames = args.render_trajectory == "all_frames"

    if args.evalset == "nerfbusters":
        selection_mode = NeighborSelectionMode(args.neighbor_selection_mode)
        return NerfbustersEvalDataset(
            nerfbusters_dir=args.nerfbusters_dir,
            recon_results_dir=args.nerfbusters_recon_results_dir,
            num_frames=num_frames,
            num_views=args.num_views,
            dataset_scaling_factor=args.dataset_scaling_factor,
            neighbor_selection_mode=selection_mode,
            max_test_frames=max_test_frames,
            include_all_frames=include_all_frames,
            filter_scene_id=args.scene_id,
            checkpoint=args.nerfbusters_checkpoint,
            recon_experiment_name=args.nerfbusters_recon_experiment_name,
            image_folder=args.nerfbusters_image_folder,
            captions_dir=args.nerfbusters_captions_dir,
            generator=None,
            verbose=(rank == 0),
        )
    elif args.evalset == "3dgrut_dl3dv_ours":
        selection_mode = NeighborSelectionMode(args.neighbor_selection_mode)
        return DL3DVHDF5EvalDataset(
            split="test",
            split_path=args.split_path,
            dl3dv_dir=args.dl3dv_dir,
            prompt_dir=args.prompt_dir,
            num_frames=num_frames,
            num_views=args.num_views,
            dataset_scaling_factor=args.dataset_scaling_factor,
            frames_per_block=args.frames_per_block,
            neighbor_selection_mode=selection_mode,
            max_test_frames=max_test_frames,
            include_all_frames=include_all_frames,
            filter_scene_id=args.scene_id,
            generator=None,
            verbose=(rank == 0),
        )
    elif is_dl3dv_reconstruction_evalset(args.evalset):
        selection_mode = NeighborSelectionMode(args.neighbor_selection_mode)
        return create_dl3dv_reconstruction_dataset(
            args=args,
            selection_mode=selection_mode,
            max_test_frames=max_test_frames,
            include_all_frames=include_all_frames,
            rank=rank,
        )
    elif is_reconstructed_colmap_evalset(args.evalset):
        selection_mode = NeighborSelectionMode(args.neighbor_selection_mode)
        return create_reconstructed_colmap_dataset(
            args=args,
            selection_mode=selection_mode,
            max_test_frames=max_test_frames,
            include_all_frames=include_all_frames,
            rank=rank,
        )
    assert False, f"Invalid evalset: {args.evalset}"


def get_output_dir(args) -> Path:
    """Get output directory based on evalset."""
    ckpt_full_name = checkpoint_output_name(args)
    sink_suffix = f"_sink{args.sink_size}" if args.sink_size > 0 else ""
    if args.render_trajectory == "val_frames":
        trajectory_suffix = ""
    elif args.render_trajectory == "trajectory":
        trajectory_suffix = "_trajectory"
    else:
        trajectory_suffix = f"_{args.render_trajectory}"
    num_views = (
        f"auto{DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS}"
        if args.num_views is None and is_reconstructed_colmap_evalset(args.evalset)
        else str(args.num_views)
    )
    if (
        args.evalset == "3dgrut_dl3dv_ours"
        or is_dl3dv_reconstruction_evalset(args.evalset)
        or is_reconstructed_colmap_evalset(args.evalset)
    ):
        mode_name = (
            f"distilled_views_{args.evalset}_{num_views}_"
            f"{args.neighbor_selection_mode}{sink_suffix}{trajectory_suffix}"
        )
        return args.save_dir / ckpt_full_name / mode_name
    if args.evalset == "nerfbusters":
        output_suffix = f"_{args.output_suffix}" if args.output_suffix else ""
        return (
            args.save_dir
            / ckpt_full_name
            / f"nerfbusters_{num_views}_{args.neighbor_selection_mode}{sink_suffix}{trajectory_suffix}{output_suffix}"
        )
    assert False, f"Invalid evalset: {args.evalset}"


def get_eval_pipe(args: argparse.Namespace, device: torch.device):
    if args.inference_pipeline == "bidirectional":
        return (
            get_pipe(
                args,
                load_pretrained_transformer_weights=False,
                frames_per_block=None,
                device=device,
            )
            .to(torch.bfloat16)
            .to(device)
        )
    return get_kv_cache_pipe(args, device).to(torch.bfloat16).to(device)


def create_context_parallel_meshes(
    rank: int,
    world_size: int,
    cp_group_size: int,
    device_type: str,
) -> tuple[DeviceMesh | None, int, int, int]:
    if cp_group_size <= 1:
        return None, world_size, 0, 0

    assert world_size % cp_group_size == 0, (
        f"world_size={world_size} must be divisible by context_parallel_size={cp_group_size}. "
        "This eval path does not allow leftover idle ranks."
    )

    num_groups = world_size // cp_group_size
    all_group_ranks = [list(range(g * cp_group_size, (g + 1) * cp_group_size)) for g in range(num_groups)]

    my_mesh = None
    # All ranks must construct the same sequence of meshes.
    for group_ranks in all_group_ranks:
        mesh = DeviceMesh(device_type, group_ranks)
        if rank in group_ranks:
            my_mesh = mesh

    group_index = rank // cp_group_size
    group_rank = rank % cp_group_size

    return my_mesh, num_groups, group_index, group_rank


def save_comparison_output(
    output: torch.Tensor,
    rgb_gt: torch.Tensor | None,
    rgb_rendered: torch.Tensor,
    rgb_neighbors: torch.Tensor,
    save_dir: Path,
    key: str = "comparison",
    fps: int = 15,
    save_frames: bool = True,
    output_indices: torch.Tensor | None = None,
):
    """
    Save comparison video/images with pred, optional gt, rendered side by side and neighbors below.

    Args:
        output: (T, C, H, W) predicted frames
        rgb_gt: optional (T, C, H, W) ground truth frames
        rgb_rendered: (T, C, H, W) rendered frames
        rgb_neighbors: (N, C, H, W) neighbor frames
        save_dir: directory to save outputs
        key: filename prefix
        fps: video framerate
        save_frames: whether to save individual PNG frames
        output_indices: optional (T,) tensor of dataset frame indices for metric PNG names;
            -1 means the frame stays in videos but has no metric PNG output
    """
    output = output.cpu().float().clamp(0, 1)
    if rgb_gt is not None:
        rgb_gt = rgb_gt.cpu().float().clamp(0, 1)
    rgb_rendered = rgb_rendered.cpu().float().clamp(0, 1)
    rgb_neighbors = rgb_neighbors.cpu().float().clamp(0, 1)

    first_row = [output]
    if rgb_gt is not None:
        first_row.append(rgb_gt)
    first_row.append(rgb_rendered)
    result_rows = [torch.cat(first_row, dim=-1)]
    columns = len(first_row)

    num_frames = output.shape[0]
    for neighbor_index in range(0, rgb_neighbors.shape[0], columns):
        neighbors_in_row = rgb_neighbors[neighbor_index : neighbor_index + columns]
        if neighbors_in_row.shape[0] < columns:
            padding = torch.zeros(
                columns - neighbors_in_row.shape[0], *neighbors_in_row.shape[1:], dtype=rgb_neighbors.dtype
            )
            neighbors_in_row = torch.cat([neighbors_in_row, padding], dim=0)
        row = torch.cat([n for n in neighbors_in_row], dim=-1)
        row = row.unsqueeze(0).expand(num_frames, -1, -1, -1)
        result_rows.append(row)

    result = torch.cat(result_rows, dim=-2)

    save_dir.mkdir(parents=True, exist_ok=True)

    if save_frames:
        frames_dir = save_dir / f"{key}_all"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for frame_idx in range(num_frames):
            if output_indices is not None:
                output_idx = int(output_indices[frame_idx])
                if output_idx == -1:
                    continue
                fname = f"{output_idx:05d}.png"
            else:
                fname = f"{frame_idx:05d}.png"
            img = (result[frame_idx].permute(1, 2, 0) * 255).round().byte().cpu().numpy()
            Image.fromarray(img).save(frames_dir / fname)

    save_video(result, save_dir / f"{key}.mp4", fps=fps)


def process_item(pipe, item, args, output_dir, rank, device, vae_temporal_scale, save_outputs: bool = True):
    """
    Unified processing function for all evalsets.

    Item dict expected keys:
        - rgb_rendered: (T, C, H, W) rendered frames
        - rgb_gt (optional): (T, C, H, W) ground truth frames
        - rgb_neighbors: (N, C, H, W) neighbor frames
        - encoded_prompt: encoded text prompt
        - scene_id: scene identifier
        - split (optional): nested output group
        - frame_indices (optional): non-padding frame indices in output video order
        - valid_frames_mask (optional): mask selecting non-padding frames
        - gt_index (optional): output frame indices; -1 skips frame PNG output
        - target_h, target_w (optional): output PNG/video frame size
        - opacity, camera_rays, w2cs, Ks, etc.
    """
    scene_id = item["scene_id"]
    split = item.get("split")
    has_gt = "rgb_gt" in item

    if split is not None:
        scene_dir = output_dir / scene_id / str(split)
    else:
        scene_dir = output_dir / scene_id

    if not args.replace_if_exists and output_frames_complete(scene_dir, item):
        if args.save_frame_outputs_only or output_video_frames_complete(scene_dir, item):
            return

    rgb_rendered = item["rgb_rendered"].unsqueeze(0).to(device)
    rgb_gt = item["rgb_gt"].unsqueeze(0).to(device) if has_gt else None
    rgb_neighbors = item["rgb_neighbors"].unsqueeze(0)
    if args.max_neighbors_per_encode is None:
        rgb_neighbors = rgb_neighbors.to(device)
    rgb_neighbors_cpu = item["rgb_neighbors"].cpu()
    encoded_prompt = item["encoded_prompt"].unsqueeze(0).to(device)

    original_num_frames = rgb_rendered.shape[1]
    pad_frames = compute_frame_padding(original_num_frames, pipeline_frames_per_block(pipe), vae_temporal_scale)
    pad_frames += compute_context_parallel_padding(
        original_num_frames + pad_frames,
        getattr(pipe.transformer, "_cp_world_size", 1),
        vae_temporal_scale,
    )
    target_num_frames = original_num_frames + pad_frames
    target_latent_num_frames = latent_num_frames_from_rgb_num_frames(target_num_frames, vae_temporal_scale)

    if pad_frames > 0:
        rgb_rendered = pad_temporal(rgb_rendered, pad_frames, dim=1)
        if rgb_gt is not None:
            rgb_gt = pad_temporal(rgb_gt, pad_frames, dim=1)

    opacity = item["opacity"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    if pad_frames > 0:
        opacity = pad_temporal(opacity, pad_frames, dim=1)

    camera_rays = item["camera_rays"].unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    latent_pad_frames = target_latent_num_frames - camera_rays.shape[1]
    if latent_pad_frames > 0:
        camera_rays = pad_temporal(camera_rays, latent_pad_frames, dim=1)

    w2cs = item["w2cs"].unsqueeze(0).to(device)
    if latent_pad_frames > 0:
        w2cs = pad_temporal(w2cs, latent_pad_frames, dim=1)

    Ks = item["Ks"].unsqueeze(0).to(device)
    if latent_pad_frames > 0:
        Ks = pad_temporal(Ks, latent_pad_frames, dim=1)

    neighbor_w2cs = item["neighbor_w2cs"].unsqueeze(0).to(device)
    neighbor_Ks = item["neighbor_Ks"].unsqueeze(0).to(device)

    kwargs = {
        "rendered_rgb": rgb_rendered,
        "rendered_opacity": opacity,
        "neighbors": rgb_neighbors,
        "camera_rays": camera_rays,
        "w2cs": w2cs,
        "neighbor_w2cs": neighbor_w2cs,
        "Ks": Ks,
        "neighbor_Ks": neighbor_Ks,
        "prompt": encoded_prompt,
        "num_inference_steps": args.num_inference_steps,
        "show_progress": rank == 0,
        "max_neighbors_per_encode": args.max_neighbors_per_encode,
    }

    if args.inference_pipeline == "kv_cache":
        latents = pipe.denoise_to_latents(**kwargs)
        if not save_outputs:
            pipe.clear_inference_caches()
            del latents, rgb_gt, rgb_rendered, rgb_neighbors, rgb_neighbors_cpu, encoded_prompt
            del opacity, camera_rays, w2cs, Ks, neighbor_w2cs, neighbor_Ks
            return
        out = pipe.decode_latents_to_video(latents)
        del latents
    else:
        out = pipe.forward_inference(**kwargs)

    if not save_outputs:
        del out, rgb_gt, rgb_rendered, rgb_neighbors, rgb_neighbors_cpu, encoded_prompt
        del opacity, camera_rays, w2cs, Ks, neighbor_w2cs, neighbor_Ks
        return

    out = out[:, :original_num_frames].cpu()
    rgb_gt_out = rgb_gt[:, :original_num_frames].cpu() if rgb_gt is not None else None
    rgb_rendered_out = rgb_rendered[:, :original_num_frames].cpu()

    valid_mask_list, output_indices_list, frame_indices = _eval_frame_metadata(item)
    assert (
        len(valid_mask_list) == original_num_frames
    ), f"valid_frames_mask has {len(valid_mask_list)} entries, expected {original_num_frames}"
    valid_mask = torch.tensor(valid_mask_list, dtype=torch.bool)

    pred_all = out[0, valid_mask].float().clamp(0, 1)
    gt_all = rgb_gt_out[0, valid_mask].float().clamp(0, 1) if rgb_gt_out is not None else None
    rendered_all = rgb_rendered_out[0, valid_mask].float().clamp(0, 1)

    scene_dir.mkdir(parents=True, exist_ok=True)

    output_indices = torch.tensor(output_indices_list, dtype=torch.long)
    if not args.save_frame_outputs_only and args.inference_pipeline != "bidirectional":
        save_comparison_output(
            output=pred_all,
            rgb_gt=gt_all,
            rgb_rendered=rendered_all,
            rgb_neighbors=rgb_neighbors_cpu,
            save_dir=scene_dir,
            key="default" if split is not None else "comparison",
            fps=args.output_fps,
            save_frames=True,
            output_indices=output_indices,
        )

    # Save individual frames
    frames_dir = scene_dir / "frames" / "batch_0000"
    video_frames_dir = scene_dir / "video_frames" / "batch_0000"
    for subdir in output_kinds(item):
        (frames_dir / subdir).mkdir(parents=True, exist_ok=True)
        if not args.save_frame_outputs_only:
            (video_frames_dir / subdir).mkdir(parents=True, exist_ok=True)

    target_h = item.get("target_h")
    target_w = item.get("target_w")
    assert (target_h is None) == (target_w is None), "target_h and target_w must be provided together"

    for frame_idx in range(pred_all.shape[0]):
        pred_frame = pred_all[frame_idx]
        gt_frame = gt_all[frame_idx] if has_gt else None
        rendered_frame = rendered_all[frame_idx]

        if target_h is not None:
            th = target_h.item() if torch.is_tensor(target_h) else target_h
            tw = target_w.item() if torch.is_tensor(target_w) else target_w
            if pred_frame.shape[-2:] != (th, tw):
                pred_frame = F.interpolate(
                    pred_frame.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False
                ).squeeze(0)
                if gt_frame is not None:
                    gt_frame = F.interpolate(
                        gt_frame.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False
                    ).squeeze(0)
                rendered_frame = F.interpolate(
                    rendered_frame.unsqueeze(0), size=(th, tw), mode="bilinear", align_corners=False
                ).squeeze(0)

        output_idx = int(output_indices[frame_idx])
        if output_idx != -1:
            fname = f"{output_idx:05d}.png"
            save_image(pred_frame, frames_dir / "pred" / fname)
            if gt_frame is not None:
                save_image(gt_frame, frames_dir / "gt" / fname)
            save_image(rendered_frame, frames_dir / "rendered" / fname)

        if not args.save_frame_outputs_only:
            fname = f"{frame_indices[frame_idx]:05d}.png"
            save_image(pred_frame, video_frames_dir / "pred" / fname)
            if gt_frame is not None:
                save_image(gt_frame, video_frames_dir / "gt" / fname)
            save_image(rendered_frame, video_frames_dir / "rendered" / fname)

    if args.render_diagnostics:
        for diag_name, diag_kwargs in [
            (
                "no_rendered_rgb",
                {
                    "rendered_rgb": torch.zeros_like(kwargs["rendered_rgb"]),
                    "rendered_opacity": (
                        torch.zeros_like(kwargs["rendered_opacity"]) if kwargs["rendered_opacity"] is not None else None
                    ),
                },
            ),
            (
                "text_only",
                {
                    "rendered_rgb": torch.zeros_like(kwargs["rendered_rgb"]),
                    "rendered_opacity": (
                        torch.zeros_like(kwargs["rendered_opacity"]) if kwargs["rendered_opacity"] is not None else None
                    ),
                    "neighbors": None,
                },
            ),
            ("no_text", {"prompt": torch.zeros_like(kwargs["prompt"])}),
        ]:
            diag_output_path = scene_dir / f"{diag_name}.mp4"
            if args.replace_if_exists or not diag_output_path.exists():
                diag_full_kwargs = dict(kwargs)
                diag_full_kwargs.update(diag_kwargs)
                diag_out = pipe.forward_inference(**diag_full_kwargs)
                diag_out = diag_out[0, :original_num_frames].cpu()[valid_mask].float().clamp(0, 1)
                save_comparison_output(
                    output=diag_out,
                    rgb_gt=gt_all,
                    rgb_rendered=rendered_all,
                    rgb_neighbors=rgb_neighbors_cpu,
                    save_dir=scene_dir,
                    key=diag_name,
                    fps=args.output_fps,
                    save_frames=False,
                    output_indices=output_indices,
                )
                del diag_out

    del out, rgb_gt_out, rgb_rendered_out, rgb_gt, rgb_rendered, rgb_neighbors, rgb_neighbors_cpu, encoded_prompt
    del opacity, camera_rays, w2cs, Ks, neighbor_w2cs, neighbor_Ks


def finalize_video_outputs(output_dir: Path, fps: int, replace: bool = False) -> None:
    for batch_dir in sorted(output_dir.glob("**/video_frames/batch_*")):
        scene_dir = batch_dir.parent.parent
        videos_dir = scene_dir / "videos"
        videos_dir.mkdir(parents=True, exist_ok=True)
        for kind in ("pred", "gt", "rendered"):
            paths = sorted((batch_dir / kind).glob("*.png"))
            output_path = videos_dir / f"{batch_dir.name}_{kind}.mp4"
            if paths and (replace or not output_path.exists()):
                save_video(paths, output_path, fps=fps)


def dataset_scene_ids(dataset) -> list[str]:
    scene_ids = getattr(dataset, "scene_ids", None)
    if scene_ids is not None:
        return sorted(str(scene_id) for scene_id in scene_ids)

    inference_items = getattr(dataset, "inference_items", None)
    assert inference_items is not None, f"{type(dataset).__name__} does not expose scene ids for replacement cleanup"
    scene_ids = {str(pair.scene_id) for _, pair in inference_items}
    return sorted(scene_ids)


def clear_existing_scene_outputs(output_dir: Path, dataset) -> None:
    for scene_id in dataset_scene_ids(dataset):
        scene_dir = output_dir / scene_id
        if scene_dir.exists():
            assert scene_dir.is_dir() and not scene_dir.is_symlink(), f"Expected eval scene directory, got {scene_dir}"
            shutil.rmtree(scene_dir)


def process_items_with_context_parallel(
    pipe,
    dataset,
    args,
    output_dir: Path,
    rank: int,
    world_size: int,
    device: torch.device,
):
    vae_temporal_scale = pipe.vae.config.scale_factor_temporal

    if len(dataset) == 0:
        return

    if args.context_parallel_size <= 1:
        for item_idx in tqdm(range(rank, len(dataset), world_size), disable=(rank != 0), desc=f"GPU {rank}"):
            torch.cuda.empty_cache()
            gc.collect()
            item = dataset[item_idx]
            process_item(pipe, item, args, output_dir, rank, device, vae_temporal_scale)
        return

    probe_item = dataset[0]
    total_num_frames = probe_item["rgb_rendered"].shape[0]
    pad_frames = compute_frame_padding(total_num_frames, pipeline_frames_per_block(pipe), vae_temporal_scale)
    pad_frames += compute_context_parallel_padding(
        total_num_frames + pad_frames,
        args.context_parallel_size,
        vae_temporal_scale,
    )
    latent_num_frames = ((total_num_frames + pad_frames) - 1) // vae_temporal_scale + 1
    cp_frame_divisor = pipe.cp_frame_divisor(latent_num_frames)

    assert cp_frame_divisor % args.context_parallel_size == 0, (
        f"context_parallel_size={args.context_parallel_size} must divide the valid CP frame divisor "
        f"{cp_frame_divisor} for {total_num_frames} frames ({latent_num_frames} latent frames)"
    )

    my_mesh, num_groups, group_index, group_rank = create_context_parallel_meshes(
        rank, world_size, args.context_parallel_size, device.type
    )

    if rank == 0:
        print(f"Eval CP: cp_size={args.context_parallel_size}, groups={num_groups}, items={len(dataset)}")

    if my_mesh is not None:
        pipe.transformer.enable_context_parallel(my_mesh)

    try:
        rounds = math.ceil(len(dataset) / num_groups)
        for round_idx in tqdm(range(rounds), disable=(rank != 0), desc=f"GPU {rank}"):
            item_idx = round_idx * num_groups + group_index
            if item_idx >= len(dataset):
                continue

            torch.cuda.empty_cache()
            gc.collect()
            item = dataset[item_idx]
            process_item(
                pipe,
                item,
                args,
                output_dir,
                rank,
                device,
                vae_temporal_scale,
                save_outputs=(group_rank == 0),
            )
    finally:
        if my_mesh is not None:
            pipe.transformer.disable_context_parallel()


@torch.inference_mode()
def main(args: argparse.Namespace, dataset_factory=create_dataset, output_dir_factory=get_output_dir):
    rank, world_size, local_rank = init_distributed(args.distributed_timeout_minutes)
    device = torch.device(f"cuda:{local_rank}")

    try:
        if rank == 0:
            print(f"Running on {world_size} GPUs, evalset: {args.evalset}")

        pipe = get_eval_pipe(args, device)
        barrier_if_distributed()

        if rank == 0:
            print("Initialized pipeline")

        load_transformer_checkpoint(pipe.transformer, args)
        pipe.transformer.eval()

        if rank == 0:
            print("Loaded transformer checkpoint")

        dataset = dataset_factory(args, rank)
        if rank == 0:
            print(f"Dataset has {len(dataset)} items")

        output_dir = output_dir_factory(args)
        if rank == 0:
            if args.replace_if_exists:
                clear_existing_scene_outputs(output_dir, dataset)
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Writing outputs to {output_dir}")

        barrier_if_distributed()

        process_items_with_context_parallel(pipe, dataset, args, output_dir, rank, world_size, device)

        barrier_if_distributed()

        if rank == 0 and not args.save_frame_outputs_only:
            finalize_video_outputs(output_dir, fps=args.output_fps, replace=args.replace_if_exists)

        if rank == 0:
            print("Done!")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    add_checkpoint_args(parser)
    parser.add_argument("--save_dir", type=Path, required=True)
    parser.add_argument(
        "--inference_pipeline",
        type=str,
        default="kv_cache",
        choices=["kv_cache", "bidirectional"],
        help="Inference pipeline to use. 'bidirectional' runs the full model over the whole chunk "
        "and supports context parallelism; 'kv_cache' uses the chunked local-attention pipeline.",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs per context-parallel group for inference.",
    )
    parser.add_argument(
        "--save_frame_outputs_only",
        action="store_true",
        help="Only save pred/gt/rendered PNG frames needed for metrics; skip comparison videos and diagnostic videos.",
    )
    parser.add_argument(
        "--distributed_timeout_minutes",
        default=60,
        type=int,
        help="NCCL process-group timeout used by distributed eval barriers.",
    )
    parser.add_argument("--num_inference_steps", default=4, type=int)
    parser.add_argument("--frames_per_block", default=7, type=int)
    parser.add_argument("--local_attn_size", default=21, type=int)
    parser.add_argument("--sink_size", default=7, type=int)
    parser.add_argument("--render_diagnostics", action="store_true")
    parser.add_argument("--replace_if_exists", action="store_true")
    parser.add_argument(
        "--render_trajectory",
        default="val_frames",
        choices=["val_frames", "all_frames", "trajectory"],
        help=(
            "Camera trajectory to render: held-out validation frames, the full source clip, "
            "or the prepared trajectory in the split."
        ),
    )
    parser.add_argument(
        "--bidirectional_chunk_size",
        default=81,
        type=int,
        help="Frames per eval chunk when --inference_pipeline=bidirectional; ignored by kv_cache inference.",
    )
    parser.add_argument(
        "--max_neighbors_per_encode",
        default=None,
        type=int,
        help=(
            "Maximum neighbor RGB frames to VAE-encode at once. Defaults to encoding all neighbors "
            "together for speed - set to 1 on memory-constrained runs."
        ),
    )
    parser.add_argument(
        "--output_suffix",
        default="",
        type=str,
        help="Optional suffix appended to eval output directory names.",
    )
    parser.add_argument("--output_fps", default=15, type=int, help="FPS for final pred/gt/rendered MP4 outputs.")


def add_dl3dv_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--split_path", type=Path, default=None)
    parser.add_argument("--dl3dv_dir", type=Path, default=None)
    parser.add_argument("--prompt_dir", type=Path, default=None)
    parser.add_argument("--recon_results_dir", type=Path, default=None)
    parser.add_argument(
        "--num_views",
        default=None,
        type=int,
        help="Number of reconstruction views to condition on. Defaults to 6 for benchmark evalsets; "
        f"for reconstructed_colmap, defaults to up to {DEFAULT_RECONSTRUCTED_COLMAP_NUM_VIEWS} selected views.",
    )
    parser.add_argument("--scene_id", type=str, default=None)
    parser.add_argument(
        "--neighbor_selection_mode",
        type=str,
        default="evenly_spaced",
        choices=["consecutive", "evenly_spaced", "covisibility"],
    )


def add_nerfbusters_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nerfbusters_dir", type=Path, default=None, help="Path to nerfbusters dataset directory")
    parser.add_argument(
        "--nerfbusters_recon_results_dir",
        type=Path,
        default=None,
        help="Path to nerfbusters reconstruction results directory",
    )
    parser.add_argument(
        "--nerfbusters_checkpoint",
        type=str,
        default="30000",
        help="Checkpoint step to use for nerfbusters reconstruction renders",
    )
    parser.add_argument(
        "--nerfbusters_recon_experiment_name",
        type=str,
        default=None,
        help="Experiment name for nerfbusters reconstruction (3DGRUT output structure)",
    )
    parser.add_argument(
        "--nerfbusters_image_folder",
        type=str,
        default=None,
        help="Image folder for nerfbusters (if None, uses per-scene mapping for height=960)",
    )
    parser.add_argument(
        "--nerfbusters_captions_dir",
        type=Path,
        default=None,
        help="Path to directory containing nerfbusters caption HDF5 files",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = get_eval_common_opts()
    add_inference_args(parser)
    parser.add_argument(
        "--evalset",
        default="3dgrut_dl3dv_ours",
        type=str,
        choices=EVALSETS,
        help="Evaluation dataset.",
    )
    add_dl3dv_args(parser)
    add_dl3dv_reconstruction_args(parser)
    add_nerfbusters_args(parser)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_checkpoint_args(parser, args)
    validate_evalset_args(parser, args)
    if args.max_neighbors_per_encode is not None and args.max_neighbors_per_encode <= 0:
        parser.error("--max_neighbors_per_encode must be positive when set")
    return args


if __name__ == "__main__":
    main(parse_args())
