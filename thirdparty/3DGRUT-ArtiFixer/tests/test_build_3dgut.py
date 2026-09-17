# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal smoke test that JIT-compiles the `lib3dgut_cc` C++ extension.

Validates the ARM64/CUDA build pipeline without loading data or running
training. Exercises:
    - `threedgrut.utils.jit.compile_slang_kernel` (slangc -> .cuh)
    - `threedgrut.utils.jit.load` (torch.utils.cpp_extension.load)
    - All CUDA link flags (targets/<arch>/lib{,/stubs})

Run:
    python tests/test_build_3dgut.py
"""
import hydra
from omegaconf import DictConfig, OmegaConf

import threedgrut.utils.misc  # noqa: F401  registers ${div:}, ${eq:}

OmegaConf.register_new_resolver(
    "int_list", lambda l: [int(x) for x in l], replace=True
)


@hydra.main(
    config_path="../configs",
    config_name="apps/colmap_3dgut.yaml",
    version_base=None,
)
def main(conf: DictConfig) -> None:
    OmegaConf.set_struct(conf, False)
    conf.path = "/tmp/unused"
    conf.out_dir = "/tmp/unused"
    conf.experiment_name = "build_test"
    conf.selected_indices_file = None
    conf.num_selected_indices = None
    conf.train_test_split_file = None
    conf.image_path_override = None

    print("[1/2] Importing setup_3dgut")
    from threedgut_tracer.setup_3dgut import setup_3dgut

    print("[2/2] JIT-compiling lib3dgut_cc (slangc + nvcc + ld)")
    tdgut = setup_3dgut(conf)
    print(f"PASS: loaded {tdgut!r}")


if __name__ == "__main__":
    main()
