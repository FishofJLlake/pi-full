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

import numpy as np

from opentau.datasets.utils import merge_missing_stats_quantiles


def test_sidecar_only_fills_missing_quantiles():
    stats = {
        "state": {
            "mean": np.array([1.0]),
            "std": np.array([2.0]),
            "min": np.array([-1.0]),
            "max": np.array([3.0]),
            "q01": np.array([-0.5]),
        }
    }
    sidecar = {
        "state": {
            "mean": np.array([99.0]),
            "q01": np.array([-9.0]),
            "q99": np.array([2.5]),
        }
    }

    merged = merge_missing_stats_quantiles(stats, sidecar)

    np.testing.assert_array_equal(merged["state"]["mean"], [1.0])
    np.testing.assert_array_equal(merged["state"]["std"], [2.0])
    np.testing.assert_array_equal(merged["state"]["min"], [-1.0])
    np.testing.assert_array_equal(merged["state"]["max"], [3.0])
    np.testing.assert_array_equal(merged["state"]["q01"], [-0.5])
    np.testing.assert_array_equal(merged["state"]["q99"], [2.5])


def test_missing_sidecar_keeps_existing_stats_object():
    stats = {"state": {"mean": np.array([1.0])}}
    assert merge_missing_stats_quantiles(stats, None) is stats
