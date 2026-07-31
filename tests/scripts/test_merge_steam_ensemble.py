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

import json

import pytest
import torch
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors.torch import load_file, save_file
from torch import nn

from opentau.policies.steam import ensemble_modeling_steam
from opentau.policies.steam.configuration_steam import SteamConfig
from opentau.scripts import merge_steam_ensemble


class _FakeTrainConfig:
    def __init__(self):
        self.policy = SteamConfig(
            vision_pretrained_path="vision",
            language_pretrained_path="language",
            tokenizer_path="tokenizer",
            image_resolution=(8, 8),
        )

    def save_pretrained(self, output):
        (output / "train_config.json").write_text(
            json.dumps({"ensemble_size": self.policy.ensemble_size}),
            encoding="utf-8",
        )


def _write_checkpoint(path, value: float, *, ensemble_member: int | None = None):
    path.mkdir()
    key = "weight" if ensemble_member is None else f"members.{ensemble_member}.weight"
    save_file({key: torch.tensor([value])}, str(path / SAFETENSORS_SINGLE_FILE))


def test_merge_creates_one_arbitrary_size_members_checkpoint(tmp_path, monkeypatch):
    checkpoints = [tmp_path / f"member_{index}" for index in range(3)]
    for index, checkpoint in enumerate(checkpoints):
        _write_checkpoint(checkpoint, float(index))
    monkeypatch.setattr(
        merge_steam_ensemble.TrainPipelineConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeTrainConfig(),
    )

    output = tmp_path / "merged"
    merge_steam_ensemble.merge_checkpoints([str(path) for path in checkpoints], output)

    state = load_file(str(output / SAFETENSORS_SINGLE_FILE), device="cpu")
    assert set(state) == {"members.0.weight", "members.1.weight", "members.2.weight"}
    assert [state[f"members.{index}.weight"].item() for index in range(3)] == [0.0, 1.0, 2.0]
    manifest = json.loads((output / "merge_manifest.json").read_text(encoding="utf-8"))
    assert manifest["ensemble_size"] == 3
    assert json.loads((output / "train_config.json").read_text(encoding="utf-8")) == {"ensemble_size": 3}


def test_merge_can_extract_one_member_from_an_existing_ensemble(tmp_path, monkeypatch):
    source = tmp_path / "existing_ensemble"
    source.mkdir()
    save_file(
        {
            "members.0.weight": torch.tensor([10.0]),
            "members.1.weight": torch.tensor([20.0]),
        },
        str(source / SAFETENSORS_SINGLE_FILE),
    )
    monkeypatch.setattr(
        merge_steam_ensemble.TrainPipelineConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: _FakeTrainConfig(),
    )

    output = tmp_path / "extracted"
    merge_steam_ensemble.merge_checkpoints([f"{source}:1"], output)
    state = load_file(str(output / SAFETENSORS_SINGLE_FILE), device="cpu")
    assert state["members.0.weight"].item() == 20.0


def test_merge_refuses_to_overwrite_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite"):
        merge_steam_ensemble.merge_checkpoints(["unused"], output)


class _TinySteamMember(nn.Module):
    def __init__(self, _config, **_kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs):
        raise AssertionError("Merged size-one checkpoints must use the ensemble wrapper.")


def test_single_member_merged_checkpoint_still_uses_ensemble_axis(tmp_path, monkeypatch):
    checkpoint = tmp_path / "merged_one"
    checkpoint.mkdir()
    save_file(
        {"members.0.weight": torch.tensor([7.0])},
        str(checkpoint / SAFETENSORS_SINGLE_FILE),
    )
    monkeypatch.setattr(ensemble_modeling_steam, "SteamPolicy", _TinySteamMember)

    loaded = ensemble_modeling_steam.load_steam_inference_checkpoint(
        checkpoint,
        SteamConfig(ensemble_size=1),
        local_files_only=True,
    )
    assert isinstance(loaded, ensemble_modeling_steam.SteamEnsemblePolicy)
    assert loaded.members[0].weight.item() == 7.0
