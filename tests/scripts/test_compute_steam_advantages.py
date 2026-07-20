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

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from opentau.scripts import compute_steam_advantages as steam_advantages
from opentau.scripts.compute_steam_advantages import (
    DatasetScores,
    _score_member,
    paper_advantage_from_expected_bin,
    source_quantile_thresholds,
    validate_checkpoint_count,
)


def test_paper_advantage_is_an_affine_expected_bin_shift():
    expected_bin = torch.tensor([0.0, 3.0])
    result = paper_advantage_from_expected_bin(expected_bin, num_bins=4)
    torch.testing.assert_close(result, torch.tensor([-1.5, 0.0]))


def test_exactly_three_checkpoints_are_required():
    validate_checkpoint_count([Path("a"), Path("b"), Path("c")])
    for count in (0, 1, 2, 4):
        with pytest.raises(ValueError, match="exactly three"):
            validate_checkpoint_count([Path(str(index)) for index in range(count)])
    with pytest.raises(ValueError, match="three distinct paths"):
        validate_checkpoint_count([Path("same"), Path("same"), Path("same")])


def test_main_continues_into_dataset_collection_after_checkpoint_validation(monkeypatch):
    config = SimpleNamespace(
        policy=steam_advantages.SteamConfig(),
        dataset_mixture=SimpleNamespace(),
        seed=None,
        dataloader_batch_size=1,
        batch_size=1,
        num_workers=0,
        prefetch_factor=None,
    )
    monkeypatch.setattr(
        steam_advantages.TrainPipelineConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(
        steam_advantages,
        "auto_torch_device",
        lambda: torch.device("cpu"),
    )

    def reached_collection(*_args, **_kwargs):
        raise RuntimeError("reached dataset collection")

    monkeypatch.setattr(steam_advantages, "_collect_datasets", reached_collection)
    args = SimpleNamespace(
        checkpoint=[Path("a"), Path("b"), Path("c")],
        config_path=None,
        dataset_mixture=None,
        batch_size=None,
        num_workers=None,
    )
    with pytest.raises(RuntimeError, match="reached dataset collection"):
        steam_advantages.main(args)


def test_source_quantiles_are_computed_independently_and_inclusively():
    thresholds = source_quantile_thresholds(
        {
            "expert": [0.0, 1.0, 2.0, 3.0, 4.0],
            "non_expert": [-4.0, -3.0, -2.0, -1.0, 0.0],
        },
        expert_positive_fraction=0.8,
        non_expert_positive_fraction=0.4,
    )
    assert thresholds["expert"] == np.percentile([0, 1, 2, 3, 4], 20)
    assert thresholds["non_expert"] == np.percentile([-4, -3, -2, -1, 0], 60)

    tied_scores = [1.0, 1.0, 1.0]
    tied_threshold = source_quantile_thresholds(
        {"expert": tied_scores},
        expert_positive_fraction=0.8,
        non_expert_positive_fraction=0.3,
    )["expert"]
    assert [score >= tied_threshold for score in tied_scores] == [True, True, True]


class _FakeInferencePairs(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            "episode_index": 0,
            "frame_index": index,
            "prompt": "task",
        }


class _FakeMember:
    def __init__(self, expected_bins):
        self.expected_bins = torch.tensor(expected_bins, dtype=torch.float32)
        self.config = SimpleNamespace(num_bins=4)

    def predict_temporal_offset(self, batch):
        indices = batch["frame_index"].to(dtype=torch.long)
        return {"expected_bin": self.expected_bins.to(indices.device)[indices]}


def test_members_are_reduced_by_a_pointwise_minimum():
    bundle = DatasetScores(
        config=SimpleNamespace(),
        dataset=_FakeInferencePairs(),
        minimum_advantage={(0, 0): float("inf"), (0, 1): float("inf")},
        tasks={},
    )
    for member in (_FakeMember([3.0, 1.0]), _FakeMember([2.0, 2.0])):
        _score_member(
            member,
            [bundle],
            device=torch.device("cpu"),
            batch_size=2,
            num_workers=0,
            prefetch_factor=None,
        )

    assert bundle.minimum_advantage == {
        (0, 0): -0.5,
        (0, 1): -1.0,
    }
