# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import unittest

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


class HydraConfigTests(unittest.TestCase):
    def test_sparse_colmap_configs_compose_in_global_package(self) -> None:
        OmegaConf.register_new_resolver(
            "int_list", lambda values: [int(value) for value in values], replace=True
        )

        expected_strategies = {
            "apps/colmap_3dgut_sparse": "GSStrategy",
            "apps/colmap_3dgut_sparse_lpips": "GSStrategy",
            "apps/colmap_3dgut_sparse_mcmc": "MCMCStrategy",
            "apps/colmap_3dgut_sparse_mcmc_lpips": "MCMCStrategy",
        }

        for config_name, strategy_method in expected_strategies.items():
            with self.subTest(config_name=config_name):
                with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
                    config = compose(config_name=config_name, overrides=[])

                self.assertNotIn("apps", config)
                self.assertIn("dataset", config)
                self.assertIn("render", config)
                self.assertEqual(config.strategy.method, strategy_method)


if __name__ == "__main__":
    unittest.main()
