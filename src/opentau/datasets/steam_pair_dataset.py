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

"""LeRobot frame-pair dataset used to train and run STEAM."""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset, Subset

from opentau.policies.steam.binning import scaled_signed_offset_to_bin
from opentau.policies.steam.configuration_steam import SteamConfig


def _scalar(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.flatten()[0].item())
    if isinstance(value, np.ndarray):
        return int(value.flatten()[0].item())
    return int(value)


class SteamPairDataset(Dataset):
    """Wrap a standardized LeRobot dataset as language-conditioned frame pairs.

    Training duplicates each non-terminal anchor into a progressive and a
    regressive example. Inference emits one fixed-lookahead progressive pair for
    each non-terminal frame; terminal frames are exposed separately so the
    labeling script can still write complete per-frame metadata.
    """

    def __init__(
        self,
        base_dataset,
        config: SteamConfig,
        *,
        mode: Literal["training", "inference"] = "training",
    ) -> None:
        if mode not in ("training", "inference"):
            raise ValueError(f"Unsupported STEAM pair mode: {mode!r}.")
        self.base_dataset = base_dataset
        self.config = config
        self.mode = mode
        self.meta = base_dataset.meta
        self.root = base_dataset.root
        self.repo_id = base_dataset.repo_id

        raw_dataset = getattr(
            base_dataset, "_metadata_hf_dataset", base_dataset.hf_dataset.with_transform(None)
        )
        episode_indices = raw_dataset["episode_index"]
        frame_indices = raw_dataset["frame_index"]
        if len(episode_indices) != len(base_dataset):
            raise ValueError(
                "STEAM requires one metadata row per selected LeRobot frame; "
                f"got metadata={len(episode_indices)}, dataset={len(base_dataset)}."
            )

        by_episode: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row_index, (episode_index, frame_index) in enumerate(
            zip(episode_indices, frame_indices, strict=True)
        ):
            by_episode[_scalar(episode_index)].append((_scalar(frame_index), row_index))

        self._episodes: dict[int, tuple[tuple[int, int], ...]] = {}
        self._anchors: list[tuple[int, int]] = []
        self.frame_records: list[tuple[int, int, int]] = []
        self.terminal_records: list[tuple[int, int, int]] = []
        for episode_index, records in sorted(by_episode.items()):
            records.sort()
            frame_ids = [frame_index for frame_index, _ in records]
            if len(frame_ids) != len(set(frame_ids)):
                raise ValueError(f"Duplicate frame_index in episode {episode_index}.")
            frozen_records = tuple(records)
            self._episodes[episode_index] = frozen_records
            self.frame_records.extend(
                (episode_index, frame_index, row_index)
                for frame_index, row_index in frozen_records
            )
            if frozen_records:
                terminal_frame, terminal_row = frozen_records[-1]
                self.terminal_records.append(
                    (episode_index, terminal_frame, terminal_row)
                )
            self._anchors.extend(
                (episode_index, position)
                for position in range(max(0, len(frozen_records) - 1))
            )

        self.episode_lengths = tuple(len(records) for records in self._episodes.values())
        if not self.episode_lengths:
            raise ValueError("STEAM pair dataset contains no episodes.")
        self.length_reference = float(
            np.percentile(
                np.asarray(self.episode_lengths, dtype=np.float64),
                config.length_reference_percentile,
            )
        )

    def set_length_reference(self, length_reference: float) -> None:
        if length_reference <= 0:
            raise ValueError("STEAM length_reference must be positive.")
        self.length_reference = float(length_reference)

    def __len__(self) -> int:
        multiplier = 2 if self.mode == "training" else 1
        return len(self._anchors) * multiplier

    def shallow_copy_with_dropout(
        self, *, enable_dropout: bool, enable_prompt_substitution: bool
    ) -> "SteamPairDataset":
        """Copy only the wrapper and delegate optional-key toggles to the base dataset."""
        cloned = copy.copy(self)
        cloned.base_dataset = self.base_dataset.shallow_copy_with_dropout(
            enable_dropout=enable_dropout,
            enable_prompt_substitution=enable_prompt_substitution,
        )
        return cloned

    def _make_pair(self, anchor_index: int, reverse: bool) -> dict:
        episode_index, position = self._anchors[anchor_index]
        records = self._episodes[episode_index]
        remaining = len(records) - position - 1
        if self.mode == "training":
            stride = int(
                torch.randint(
                    low=1,
                    high=min(self.config.max_temporal_offset, remaining) + 1,
                    size=(),
                ).item()
            )
        else:
            stride = min(self.config.max_temporal_offset, remaining)

        future_position = position + stride
        frame_a, row_a = records[position]
        frame_b, row_b = records[future_position]
        if reverse:
            frame_t, row_t, frame_tk, row_tk = frame_b, row_b, frame_a, row_a
        else:
            frame_t, row_t, frame_tk, row_tk = frame_a, row_a, frame_b, row_b

        sample_t = self.base_dataset[row_t]
        sample_tk = self.base_dataset[row_tk]
        camera_keys = [f"camera{index}" for index in range(self.base_dataset.num_cams)]
        images_t = {key: sample_t[key] for key in camera_keys}
        images_tk = {key: sample_tk[key] for key in camera_keys}
        masks_t = {
            key: ~sample_t["img_is_pad"][camera_index].to(dtype=torch.bool)
            for camera_index, key in enumerate(camera_keys)
        }
        masks_tk = {
            key: ~sample_tk["img_is_pad"][camera_index].to(dtype=torch.bool)
            for camera_index, key in enumerate(camera_keys)
        }

        signed_offset = frame_tk - frame_t
        episode_length = len(records)
        scaled_offset = signed_offset * self.length_reference / episode_length
        target_bin = scaled_signed_offset_to_bin(
            scaled_offset,
            self.config.max_temporal_offset,
            self.config.num_bins,
        )
        return {
            "steam_images_t": images_t,
            "steam_images_tk": images_tk,
            "steam_image_masks_t": masks_t,
            "steam_image_masks_tk": masks_tk,
            "steam_target_bin": torch.tensor(target_bin, dtype=torch.long),
            "steam_scaled_offset": torch.tensor(scaled_offset, dtype=torch.float32),
            "steam_direction": -1 if reverse else 1,
            "prompt": sample_t["prompt"],
            "episode_index": episode_index,
            "frame_index": frame_t,
            "paired_frame_index": frame_tk,
        }

    def __getitem__(self, index: int) -> dict:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self.mode == "training":
            anchor_index, direction = divmod(index, 2)
            return self._make_pair(anchor_index, reverse=bool(direction))
        return self._make_pair(index, reverse=False)


def get_steam_pair_dataset(dataset) -> SteamPairDataset:
    """Unwrap a validation subset and return its underlying STEAM dataset."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    if not isinstance(dataset, SteamPairDataset):
        raise TypeError(f"Expected SteamPairDataset, got {type(dataset).__name__}.")
    return dataset


def set_global_length_reference(
    datasets: list,
    percentile: float,
) -> float:
    """Set one episode-length reference across every STEAM dataset in a mixture."""
    wrappers: list[SteamPairDataset] = []
    seen: set[int] = set()
    lengths: list[int] = []
    for dataset in datasets:
        wrapper = get_steam_pair_dataset(dataset)
        if id(wrapper) in seen:
            continue
        seen.add(id(wrapper))
        wrappers.append(wrapper)
        lengths.extend(wrapper.episode_lengths)
    if not lengths:
        raise ValueError("Cannot derive a STEAM length reference from zero episodes.")
    reference = float(np.percentile(np.asarray(lengths, dtype=np.float64), percentile))
    for wrapper in wrappers:
        wrapper.set_length_reference(reference)
    return reference
