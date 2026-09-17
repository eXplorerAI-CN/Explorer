# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from model_training.data.dl3dv_base import DL3DVPairedDatasetBase


class DL3DVPairedDatasetTrain(DL3DVPairedDatasetBase):

    def __init__(
        self,
        split: str,
        split_path: Path,
        dl3dv_dir: Path,
        prompt_dir: Path,
        num_frames: int,
        frames_per_block: int,
        num_fetches: int,
        dataset_scaling_factor: float,
        max_neighbors: int = 12,
        verbose: bool = False,
    ):
        """
        num_fetches is the TOTAL number of DataLoader fetches the training run will perform
        across every rank (num_fetches = max_iterations × gradient_accumulation_steps ×
        num_processes). Sizing __len__ to this value guarantees exactly one DataLoader epoch
        per rank under accelerate.prepare()'s shard-on-index behavior, so every (iter_idx)
        slot in the pre-generated plan is consumed exactly once — no wrap-around reuse of the
        same (split, num_neighbors, scene) triple under gradient_accumulation_steps > 1 or
        world_size > 1.
        """
        super().__init__(
            split=split,
            split_path=split_path,
            dl3dv_dir=dl3dv_dir,
            prompt_dir=prompt_dir,
            num_frames=num_frames,
            frames_per_block=frames_per_block,
            return_unencoded_prompt=False,
            dataset_scaling_factor=dataset_scaling_factor,
            max_neighbors=max_neighbors,
            generator=None,
            verbose=verbose,
        )

        assert self.available_splits, "available_splits is empty"
        assert all(s > 0 for s in self.available_splits), f"available_splits must be positive: {self.available_splits}"
        assert all(
            a < b for a, b in zip(self.available_splits, self.available_splits[1:])
        ), f"available_splits must be strictly increasing: {self.available_splits}"

        # intervals[i] = (lo, hi) inclusive range of num_neighbors for split available_splits[i].
        # Each split "owns" neighbor counts between its predecessor (exclusive) and itself (inclusive),
        # with the first split starting at 1. For (2, 3, 6, 12) this gives (1,2), (3,3), (4,6), (7,12).
        prev = [0, *self.available_splits[:-1]]
        self._intervals = torch.tensor([(lo + 1, hi) for lo, hi in zip(prev, self.available_splits)], dtype=torch.int64)

        self.num_fetches = num_fetches
        # Populated by set_random_splits; shape (num_fetches, 2), cols = (split, num_neighbors).
        self.iter_plan: torch.Tensor | None = None
        # Populated by set_random_splits; shape (len(self.data),) permutation of scene indices.
        self.data_perm: torch.Tensor | None = None

    def set_random_splits(self, device: torch.device | str, seed: int) -> None:
        """Pre-sample a (split, num_neighbors) pair per fetch and a one-time scene permutation,
        then broadcast rank-0's values to all ranks so every worker agrees on the plan."""
        rank = dist.get_rank() if dist.is_initialized() else 0
        iter_plan = torch.empty((self.num_fetches, 2), dtype=torch.int64, device=device)
        data_perm = torch.empty((len(self.data),), dtype=torch.int64, device=device)

        if rank == 0:
            generator = torch.Generator().manual_seed(seed)
            split_idx = torch.randint(
                0, len(self.available_splits), (self.num_fetches,), generator=generator, dtype=torch.int64
            )
            lo = self._intervals[split_idx, 0]
            hi = self._intervals[split_idx, 1]
            # Uniform integer sampling from [lo, hi] inclusive: continuous uniform [0, 1) mapped to
            # integer offsets in [0, hi - lo] via floor, then shifted by lo.
            num_neighbors = lo + (torch.rand(self.num_fetches, generator=generator) * (hi - lo + 1)).long()
            iter_plan.copy_(torch.stack([hi, num_neighbors], dim=1))
            # Scene permutation decouples the slight over-representation of scenes in
            # [0, num_fetches mod len(self.data)) from self.data's default sort order so the
            # over-weighted subset rotates with the seed. Same generator → deterministic across
            # ranks' broadcast views.
            data_perm.copy_(torch.randperm(len(self.data), generator=generator))

        if dist.is_initialized():
            dist.broadcast(iter_plan, src=0)
            dist.broadcast(data_perm, src=0)

        self.iter_plan = iter_plan.cpu()
        self.data_perm = data_perm.cpu()

    def __len__(self) -> int:
        return self.num_fetches

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.iter_plan is None or self.data_perm is None:
            raise RuntimeError("set_random_splits() must be called before iterating the dataset")

        chosen_split, num_neighbors = self.iter_plan[idx].tolist()
        data_idx = self.data_perm[idx % len(self.data)].item()
        return self.get_item_inner(data_idx, chosen_split, num_neighbors)
