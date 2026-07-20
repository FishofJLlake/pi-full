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

from unittest.mock import patch

from opentau.scripts.compute_max_token_length import make_unsplit_dataset_mixture


def test_make_unsplit_dataset_mixture_disables_validation_split(train_pipeline_config):
    train_pipeline_config.val_freq = 100
    expected_mixture = object()

    with patch(
        "opentau.scripts.compute_max_token_length.make_dataset_mixture",
        return_value=expected_mixture,
    ) as make_mixture:
        mixture = make_unsplit_dataset_mixture(train_pipeline_config)

    stats_cfg = make_mixture.call_args.args[0]
    assert mixture is expected_mixture
    assert stats_cfg is not train_pipeline_config
    assert stats_cfg.val_freq == 0
    assert train_pipeline_config.val_freq == 100
