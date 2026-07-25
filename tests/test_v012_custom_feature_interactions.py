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

"""Regression tests for v0.12.0 interactions with the custom branch."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from opentau.datasets.lerobot_dataset import BaseDataset
from opentau.policies.normalize import LEGACY_EPS, OPENPI_EPS
from opentau.policies.pi05.modeling_pi05 import PI05Policy
from opentau.policies.value import modeling_value
from opentau.policies.value.configuration_value import ValueConfig

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 project runtime
    import tomli as tomllib


class _InteractionDataset(BaseDataset):
    def _get_feature_mapping_key(self) -> str:
        return "test/v012-interactions"

    def _emit_optional_keys(self, item: dict, standard_item: dict) -> None:
        self.optional_snapshot = {
            "real_action_dim": standard_item["real_action_dim"].item(),
            "state_shape": tuple(standard_item["state"].shape),
            "action_shape": tuple(standard_item["actions"].shape),
            "actions": standard_item["actions"].clone(),
        }
        self._emit_intervention(item, standard_item)


def test_index_delta_and_intervention_share_the_standardization_pipeline():
    dataset = object.__new__(_InteractionDataset)
    dataset._data_features_name_mapping = {
        "state": "raw_state",
        "actions": "raw_actions",
        "prompt": "task",
        "intervention": "human_intervention",
    }
    dataset.num_cams = 0
    dataset.resolution = (2, 2)
    dataset.state_index = [2, 0]
    dataset.action_index = [2, 0]
    dataset.delta_action_state_map = {0: 1, 1: 0}
    dataset.max_state_dim = 4
    dataset.max_action_dim = 4
    dataset.n_obs_history = None

    output = dataset._to_standard_data_format(
        {
            "raw_state": torch.tensor([10.0, 20.0, 30.0]),
            "raw_actions": torch.tensor(
                [
                    [31.0, 99.0, 12.0],
                    [33.0, 100.0, 13.0],
                ]
            ),
            "raw_actions_is_pad": torch.tensor([False, False]),
            "task": "move",
            "human_intervention": torch.tensor(1.0),
            "intervention_raw": 1.0,
            "episode_index": 4,
            "frame_index": 7,
        }
    )

    # Reindex first: state [30, 10], actions [[12, 31], [13, 33]]. Delta is
    # then computed in raw units against that one chunk-start state.
    expected_actions = torch.tensor(
        [
            [2.0, 1.0, 0.0, 0.0],
            [3.0, 3.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(output["actions"], expected_actions)
    torch.testing.assert_close(
        output["state"],
        torch.tensor([30.0, 10.0, 0.0, 0.0], dtype=torch.bfloat16),
    )
    assert output["real_action_dim"].item() == 2
    assert output["intervention"].item() is True
    assert output["intervention_is_pad"].item() is False
    assert dataset.optional_snapshot["real_action_dim"] == 2
    assert dataset.optional_snapshot["state_shape"] == (4,)
    assert dataset.optional_snapshot["action_shape"] == (2, 4)


def test_pi05_cfg_delta_inverse_crop_and_queue_prefix_are_composed_once():
    policy = object.__new__(PI05Policy)
    policy.config = SimpleNamespace(
        actual_action_dim=2,
        action_feature=SimpleNamespace(shape=(4,)),
        max_action_dim=4,
        chunk_size=2,
        max_delay=1,
        state_type="discrete",
        advantage="use",
        guidance_scale=2.5,
        delta_action_state_map={0: 0, 1: 1},
    )
    policy.training = False
    policy._resolve_dataset_index = lambda batch: 0
    policy.normalize_inputs = lambda batch, dataset_index: batch
    policy.prepare_images = lambda batch: (
        [torch.ones((1, 1), dtype=torch.float32)],
        [torch.ones((1, 1), dtype=torch.bool)],
    )

    def prepare_language(batch, force_uncond=False):
        token = 9 if force_uncond else 3
        return (
            torch.tensor([[token]], dtype=torch.long),
            torch.ones((1, 1), dtype=torch.bool),
        )

    policy.prepare_language = prepare_language
    seen = {}

    def normalize_targets(outputs, dataset_index):
        seen["prefix_before_norm"] = outputs["actions"].clone()
        return {"actions": outputs["actions"] + 100.0}

    policy.normalize_targets = normalize_targets

    def model_sample_actions(
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        action_prefix,
        delay,
        *,
        noise,
        state,
        guidance_scale,
    ):
        seen["image_batch"] = images[0].shape[0]
        seen["language_tokens"] = lang_tokens.clone()
        seen["action_prefix"] = action_prefix.clone()
        seen["guidance_scale"] = guidance_scale
        assert state is None
        return torch.tensor(
            [
                [
                    [1.0, 2.0, 30.0, 40.0],
                    [3.0, 4.0, 31.0, 41.0],
                ]
            ]
        )

    policy.model = SimpleNamespace(sample_actions=model_sample_actions)

    def unnormalize_outputs(outputs, dataset_index):
        seen["unnormalize_shape"] = tuple(outputs["actions"].shape)
        return {"actions": outputs["actions"] + 10.0}

    policy.unnormalize_outputs = unnormalize_outputs
    raw_state = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    absolute_prefix = torch.tensor([[[11.0, 22.0, 50.0, 60.0]]])

    actions = PI05Policy.sample_actions(
        policy,
        {"state": raw_state},
        action_prefix=absolute_prefix,
        delay=torch.tensor(1),
    )

    torch.testing.assert_close(
        seen["prefix_before_norm"],
        torch.tensor([[[1.0, 2.0, 50.0, 60.0]]]),
    )
    torch.testing.assert_close(
        seen["action_prefix"],
        torch.tensor(
            [
                [
                    [101.0, 102.0, 150.0, 160.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
    )
    assert seen["image_batch"] == 2
    assert seen["language_tokens"].tolist() == [[3], [9]]
    assert seen["guidance_scale"] == 2.5
    assert seen["unnormalize_shape"] == (1, 2, 4)
    torch.testing.assert_close(
        actions,
        torch.tensor([[[21.0, 32.0], [23.0, 34.0]]]),
    )


class _CapturedNormalize(nn.Module):
    calls: list[dict] = []

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.calls.append(kwargs)

    def forward(self, batch, dataset_index=None):
        return batch


@pytest.mark.parametrize(
    ("config_version", "expected_center", "expected_eps"),
    [(0, False, LEGACY_EPS), (1, True, OPENPI_EPS)],
)
def test_value_policy_threads_versioned_normalization_options(
    monkeypatch,
    config_version,
    expected_center,
    expected_eps,
):
    _CapturedNormalize.calls.clear()
    monkeypatch.setattr(modeling_value, "Normalize", _CapturedNormalize)
    monkeypatch.setattr(
        modeling_value.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        modeling_value,
        "ValueModel",
        lambda *args, **kwargs: nn.Identity(),
    )

    modeling_value.ValueFunction(ValueConfig(config_version=config_version))

    assert len(_CapturedNormalize.calls) == 1
    assert _CapturedNormalize.calls[0]["zero_range_center"] is expected_center
    assert _CapturedNormalize.calls[0]["eps"] == expected_eps


def test_steam_and_so101_entrypoints_match_the_lockfile():
    repo_root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((repo_root / "uv.lock").read_text(encoding="utf-8"))

    expected_scripts = {
        "opentau-steam-advantages",
        "opentau-so101-calibrate",
        "opentau-so101-teleoperate",
        "opentau-so101-record",
        "opentau-so101-setup-motors",
        "opentau-so101-find-port",
        "opentau-so101-find-cameras",
    }
    assert expected_scripts <= project["project"]["scripts"].keys()
    assert project["project"]["optional-dependencies"]["so101"] == [
        "feetech-servo-sdk>=1.0.0"
    ]

    locked_packages = {package["name"]: package for package in lock["package"]}
    assert locked_packages["opentau"]["version"] == project["project"]["version"] == "0.12.0"
    assert "feetech-servo-sdk" in locked_packages
