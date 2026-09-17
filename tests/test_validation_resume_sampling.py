# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import MethodType

import torch

from model_training.data.dl3dv_test import DL3DVPairedDatasetTest


def _fake_get_item_inner(self, data_idx, num_train_views, num_neighbors, generator=None):
    assert generator is not None
    return {
        "data_idx": data_idx,
        "num_train_views": num_train_views,
        "num_neighbors": num_neighbors,
        "draws": torch.randint(0, 1_000_000, size=(8,), generator=generator),
    }


def _make_validation_probe(start_index: int, num_views: int = 3) -> DL3DVPairedDatasetTest:
    dataset = object.__new__(DL3DVPairedDatasetTest)
    dataset.data = [object() for _ in range(5)]
    dataset.num_views = num_views
    dataset.start_index = start_index
    dataset.validation_seed = 42
    dataset.get_item_inner = MethodType(_fake_get_item_inner, dataset)
    return dataset


def test_validation_sampling_is_resume_stable():
    continuous = _make_validation_probe(start_index=0)[2]
    resumed = _make_validation_probe(start_index=2)[0]

    assert continuous["data_idx"] == resumed["data_idx"]
    assert continuous["num_train_views"] == resumed["num_train_views"]
    assert continuous["num_neighbors"] == resumed["num_neighbors"]
    assert torch.equal(continuous["draws"], resumed["draws"])


def test_validation_sampling_changes_across_logical_indices():
    logical_index_2 = _make_validation_probe(start_index=0)[2]
    logical_index_3 = _make_validation_probe(start_index=0)[3]

    assert logical_index_2["data_idx"] != logical_index_3["data_idx"]
    assert not torch.equal(logical_index_2["draws"], logical_index_3["draws"])
