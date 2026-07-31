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

import torch

from datasets import Dataset as HFDataset
from opentau.datasets.steam_pair_dataset import (
    SteamPairDataset,
    set_global_length_reference,
)
from opentau.policies.steam.binning import scaled_signed_offset_to_bin
from opentau.policies.steam.configuration_steam import SteamConfig


class _FakeHFDataset:
    def __init__(self, episode_indices, frame_indices):
        self.columns = {
            "episode_index": episode_indices,
            "frame_index": frame_indices,
        }

    def with_transform(self, _transform):
        return self

    def with_format(self, format_type):
        assert format_type == "arrow"
        return self

    def __getitem__(self, key):
        return self.columns[key]


class _FakeLeRobotDataset:
    def __init__(self, lengths, repo_id="fake/repo"):
        self.root = Path("fake-root")
        self.repo_id = repo_id
        self.meta = SimpleNamespace()
        self.num_cams = 1
        self.rows = []
        episode_indices = []
        frame_indices = []
        for episode_index, length in enumerate(lengths):
            for frame_index in range(length):
                episode_indices.append(episode_index)
                frame_indices.append(frame_index)
                self.rows.append(
                    {
                        "camera0": torch.full((3, 8, 8), frame_index / 10),
                        "img_is_pad": torch.tensor([False]),
                        "prompt": f"task-{episode_index}",
                    }
                )
        self.hf_dataset = _FakeHFDataset(episode_indices, frame_indices)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def _config():
    return SteamConfig(
        max_temporal_offset=4,
        num_bins=8,
        length_reference_percentile=100,
    )


def test_training_pairs_include_forward_and_reverse_without_crossing_episodes():
    dataset = SteamPairDataset(
        _FakeLeRobotDataset([2, 2]),
        _config(),
        mode="training",
    )
    assert len(dataset) == 4

    forward = dataset[0]
    reverse = dataset[1]
    assert forward["steam_direction"] == 1
    assert reverse["steam_direction"] == -1
    assert (forward["episode_index"], forward["frame_index"], forward["paired_frame_index"]) == (
        0,
        0,
        1,
    )
    assert (reverse["episode_index"], reverse["frame_index"], reverse["paired_frame_index"]) == (
        0,
        1,
        0,
    )
    assert forward["steam_target_bin"].item() >= 4
    assert reverse["steam_target_bin"].item() < 4


def test_metadata_columns_bypass_hugging_face_row_transforms():
    base_dataset = _FakeLeRobotDataset([2, 1])
    base_dataset.hf_dataset = HFDataset.from_dict(base_dataset.hf_dataset.columns)
    base_dataset.hf_dataset.set_transform(lambda batch: batch)

    dataset = SteamPairDataset(base_dataset, _config(), mode="inference")

    assert dataset.frame_records == [(0, 0, 0), (0, 1, 1), (1, 0, 2)]
    assert dataset.terminal_records == [(0, 1, 1), (1, 0, 2)]


def test_inference_uses_paper_length_scaling_without_a_minimum_scale_clamp():
    dataset = SteamPairDataset(
        _FakeLeRobotDataset([8]),
        _config(),
        mode="inference",
    )
    dataset.set_length_reference(4)
    pair = dataset[0]
    assert pair["steam_scaled_offset"].item() == 2.0
    assert pair["steam_target_bin"].item() == scaled_signed_offset_to_bin(2.0, 4, 8)


def test_inference_exposes_all_nonterminal_anchors_and_terminal_records():
    dataset = SteamPairDataset(
        _FakeLeRobotDataset([3, 1]),
        _config(),
        mode="inference",
    )
    assert len(dataset) == 2
    assert dataset.terminal_records == [(0, 2, 2), (1, 0, 3)]
    assert len(dataset.frame_records) == 4


def test_global_reference_is_shared_across_datasets():
    first = SteamPairDataset(_FakeLeRobotDataset([2]), _config())
    second = SteamPairDataset(_FakeLeRobotDataset([5], repo_id="fake/other"), _config())
    reference = set_global_length_reference([first, second], percentile=100)
    assert reference == 5.0
    assert first.length_reference == second.length_reference == 5.0


def test_length_scaling_can_be_disabled_for_rlinf_parity():
    config = _config()
    config.length_scale_enabled = False
    dataset = SteamPairDataset(_FakeLeRobotDataset([8]), config, mode="inference")
    dataset.set_length_reference(4)
    pair = dataset[0]
    assert pair["steam_scaled_offset"].item() == 4.0


def test_target_bin_histogram_counts_all_valid_directions_without_rng():
    dataset = SteamPairDataset(_FakeLeRobotDataset([3]), _config(), mode="training")
    first = dataset.target_bin_histogram()
    second = dataset.target_bin_histogram()
    assert first == second
    assert sum(first) == 6
    assert sum(first[:4]) == sum(first[4:]) == 3
