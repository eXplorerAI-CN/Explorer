# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import subprocess
from argparse import Namespace
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_MODEL_ID = "Wan-AI/Wan2.1-T2V-14B-Diffusers"


def metadata_path_for_output(output_pt: Path) -> Path:
    return output_pt.with_suffix(".metadata.json")


def _git_value(repo_dir: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_dir,
        check=False,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def build_metadata(
    args: Namespace,
    *,
    branch: str | None,
    commit: str | None,
    export_date: datetime,
) -> dict[str, str | int]:
    source_path = args.source_path if args.source_path is not None else args.checkpoint_dir
    return {
        "run_id": args.run_id,
        "checkpoint": int(args.checkpoint),
        "slot": args.slot,
        "source_path": str(source_path),
        "output_pt": str(args.output_pt),
        "model_id": args.model_id,
        "branch": branch or "",
        "commit": commit or "",
        "export_date": export_date.isoformat(),
    }


def _state_dict_to_cpu(state_dict: dict, torch_module=None):
    if torch_module is None:
        import torch as torch_module

    return {
        key: value.detach().cpu() if torch_module.is_tensor(value) else value
        for key, value in state_dict.items()
    }


def create_export_transformer(args: Namespace):
    from model_training.utils.train_utils import get_kv_cache_pipe

    pipe = get_kv_cache_pipe(args, device="cpu")
    return pipe.transformer


def export_checkpoint(
    args: Namespace,
    *,
    transformer_factory=None,
    dcp_loader=None,
    torch_module=None,
) -> tuple[Path, Path]:
    if args.output_pt.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {args.output_pt}")

    if torch_module is None:
        import torch as torch_module
    if dcp_loader is None:
        from model_training.utils.train_utils import load_model_weights_from_dcp

        dcp_loader = load_model_weights_from_dcp

    if transformer_factory is None:
        transformer_factory = create_export_transformer

    transformer = transformer_factory(args)
    dcp_loader(transformer, args.checkpoint_dir)

    args.output_pt.parent.mkdir(parents=True, exist_ok=True)
    torch_module.save(_state_dict_to_cpu(transformer.state_dict(), torch_module=torch_module), args.output_pt)

    repo_dir = Path(__file__).resolve().parents[1]
    branch = args.branch or _git_value(repo_dir, "branch", "--show-current")
    commit = args.commit or _git_value(repo_dir, "rev-parse", "HEAD")
    metadata = build_metadata(
        args,
        branch=branch,
        commit=commit,
        export_date=datetime.now(timezone.utc),
    )
    metadata_path = args.metadata_path if args.metadata_path is not None else metadata_path_for_output(args.output_pt)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    return args.output_pt, metadata_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a DCP/FSDP transformer checkpoint to a single .pt state dict.")
    parser.add_argument("--checkpoint_dir", type=Path, required=True)
    parser.add_argument("--output_pt", type=Path, required=True)
    parser.add_argument("--metadata_path", type=Path, default=None)
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID, type=str)
    parser.add_argument("--frames_per_block", default=7, type=int)
    parser.add_argument("--local_attn_size", default=21, type=int)
    parser.add_argument("--sink_size", default=7, type=int)
    parser.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    parser.add_argument("--checkpoint_every_n_blocks", default=1, type=int)
    parser.add_argument("--attention_backend", default=None, type=str)
    parser.add_argument("--run_id", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=int)
    parser.add_argument("--slot", required=True, type=str)
    parser.add_argument("--source_path", type=Path, default=None)
    parser.add_argument("--branch", default=None, type=str)
    parser.add_argument("--commit", default=None, type=str)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    output_pt, metadata_path = export_checkpoint(parse_args(argv))
    print(f"Wrote {output_pt}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
