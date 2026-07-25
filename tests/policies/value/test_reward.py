#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
from importlib import util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = ROOT / "src" / "opentau" / "policies" / "value" / "reward.py"


def load_reward_module():
    spec = util.spec_from_file_location("value_reward", MODULE_PATH)
    module = util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RewardNormalizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reward = load_reward_module()

    def test_equal_width_return_includes_c_neg_in_normalization_factor(self):
        bin_idx, normalized = self.reward.calculate_return_bins_with_equal_width(
            success=False,
            b=200,
            episode_end_idx=401,
            reward_normalizer=600,
            current_idx=200,
            c_neg=-1000.0,
        )

        self.assertAlmostEqual(normalized, -1200 / 1600)
        self.assertEqual(bin_idx, 49)
        self.assertGreaterEqual(bin_idx, 0)
        self.assertLess(bin_idx, 200)

    def test_n_step_return_uses_the_same_normalization_factor(self):
        normalized = self.reward.calculate_n_step_return(
            success=False,
            n_steps_look_ahead=50,
            episode_end_idx=401,
            reward_normalizer=600,
            current_idx=390,
            c_neg=-1000.0,
        )

        self.assertAlmostEqual(normalized, -1010 / 1600)
        self.assertGreaterEqual(normalized, -1)
        self.assertLessEqual(normalized, 0)

    def test_success_and_nonterminal_failure_share_the_same_scale(self):
        successful = self.reward.calculate_n_step_return(True, 50, 401, 600, 200, -1000.0)
        failure_before_terminal = self.reward.calculate_n_step_return(False, 50, 401, 600, 200, -1000.0)

        self.assertAlmostEqual(successful, -50 / 1600)
        self.assertAlmostEqual(failure_before_terminal, -50 / 1600)


if __name__ == "__main__":
    unittest.main()
