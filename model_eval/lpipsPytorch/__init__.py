# SPDX-FileCopyrightText: Copyright (c) 2025 Sibo Wu
# SPDX-FileCopyrightText: Copyright (c) 2018 Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT AND BSD-2-Clause AND Apache-2.0

import torch

from .modules.lpips import LPIPS

# adapted from genfusion codebase: https://github.com/Inception3D/GenFusion/tree/main/Reconstruction/lpipsPyTorch
# added a cache to avoid re-initializing the LPIPS network for each call
_lpips_cache = {}


def lpips(x: torch.Tensor, y: torch.Tensor, net_type: str = "alex", version: str = "0.1"):
    r"""Function that measures
    Learned Perceptual Image Patch Similarity (LPIPS).

    Arguments:
        x, y (torch.Tensor): the input tensors to compare.
        net_type (str): the network type to compare the features:
                        'alex' | 'squeeze' | 'vgg'. Default: 'alex'.
        version (str): the version of LPIPS. Default: 0.1.
    """
    device = x.device
    cache_key = (net_type, version, device)
    if cache_key not in _lpips_cache:
        _lpips_cache[cache_key] = LPIPS(net_type, version).to(device)
    return _lpips_cache[cache_key](x, y)
