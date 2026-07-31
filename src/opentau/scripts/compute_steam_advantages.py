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
"""CLI and compatibility exports for the STEAM advantage pipeline."""

from opentau.scripts.steam_advantage_pipeline import (
    DatasetScores,
    _collect_datasets,
    _finalize_and_persist,
    _score_member,
    apply_threshold,
    main,
    normalized_dataset_root,
    paper_advantage_from_expected_bin,
    parse_args,
    rlinf_advantage_from_probabilities,
    shard_bounds,
    source_quantile_thresholds,
    validate_checkpoint_count,
    validate_tag,
    validate_unique_dataset_roots,
)
from opentau.utils.utils import init_logging

__all__ = [
    "DatasetScores",
    "_collect_datasets",
    "_finalize_and_persist",
    "_score_member",
    "apply_threshold",
    "main",
    "normalized_dataset_root",
    "paper_advantage_from_expected_bin",
    "rlinf_advantage_from_probabilities",
    "shard_bounds",
    "source_quantile_thresholds",
    "validate_checkpoint_count",
    "validate_tag",
    "validate_unique_dataset_roots",
]


def cli() -> None:
    init_logging()
    main(parse_args())


if __name__ == "__main__":
    cli()
