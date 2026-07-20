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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from opentau.configs.types import FeatureType, PolicyFeature
from opentau.policies.steam import modeling_steam
from opentau.policies.steam.binning import (
    bin_centers,
    expected_signed_offset,
    scaled_signed_offset_to_bin,
    signed_offset_to_bin,
)
from opentau.policies.steam.configuration_steam import SteamConfig


def test_signed_offset_bins_preserve_the_sign_split():
    assert signed_offset_to_bin(-4, 4, 8) == 0
    assert signed_offset_to_bin(-1, 4, 8) == 3
    assert signed_offset_to_bin(1, 4, 8) == 4
    assert signed_offset_to_bin(4, 4, 8) == 7
    assert scaled_signed_offset_to_bin(0.2, 4, 8) == 4
    assert scaled_signed_offset_to_bin(-0.2, 4, 8) == 3


def test_bin_centers_and_expectation_match_the_discrete_layout():
    centers = bin_centers(4, 4)
    torch.testing.assert_close(
        centers,
        torch.tensor([-3.5, -1.5, 1.5, 3.5]),
    )
    probabilities = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    torch.testing.assert_close(
        expected_signed_offset(probabilities, 4, 4),
        torch.tensor([1.5]),
    )


def test_steam_config_rejects_an_invalid_bin_layout():
    with pytest.raises(ValueError, match="divisible"):
        SteamConfig(max_temporal_offset=3, num_bins=4)


class _FakeVision(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4)
        self.config = SimpleNamespace(projection_dim=4)

    def get_image_features(self, pixel_values):
        pooled = pixel_values.mean(dim=(-2, -1))
        return self.projection(pooled)


class _FakeLanguage(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32, 5)
        self.config = SimpleNamespace(hidden_size=5)

    def forward(self, input_ids, attention_mask, return_dict):
        del attention_mask, return_dict
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class _FakeTokenizer:
    def __call__(self, prompts, **kwargs):
        del kwargs
        batch_size = len(prompts)
        return {
            "input_ids": torch.ones(batch_size, 3, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 3, dtype=torch.long),
        }


class _FakeImageProcessor:
    size = {"height": 8, "width": 8}
    image_mean = [0.5, 0.5, 0.5]
    image_std = [0.5, 0.5, 0.5]


def test_steam_policy_trains_and_predicts_without_loading_real_backbones(monkeypatch):
    def load_model(path, **_kwargs):
        return _FakeVision() if path == "vision" else _FakeLanguage()

    monkeypatch.setattr(modeling_steam.AutoModel, "from_pretrained", load_model)
    monkeypatch.setattr(
        modeling_steam.AutoTokenizer,
        "from_pretrained",
        lambda _path, **_kwargs: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        modeling_steam.AutoImageProcessor,
        "from_pretrained",
        lambda _path, **_kwargs: _FakeImageProcessor(),
    )

    config = SteamConfig(
        vision_pretrained_path="vision",
        language_pretrained_path="language",
        tokenizer_path="tokenizer",
        num_bins=4,
        max_temporal_offset=4,
        fusion_hidden_dim=8,
        dropout=0.0,
        use_gradient_checkpointing=False,
        input_features={
            "camera0": PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 8, 8),
            )
        },
    )
    policy = modeling_steam.SteamPolicy(config)
    batch = {
        "steam_images_t": {"camera0": torch.rand(2, 3, 8, 8)},
        "steam_images_tk": {"camera0": torch.rand(2, 3, 8, 8)},
        "steam_image_masks_t": {"camera0": torch.tensor([True, True])},
        "steam_image_masks_tk": {"camera0": torch.tensor([True, True])},
        "steam_target_bin": torch.tensor([2, 3]),
        "prompt": ["pick object", "place object"],
    }

    losses = policy(batch)
    assert losses["CE"].ndim == 0
    assert losses["CE"].requires_grad
    prediction = policy.predict_temporal_offset(batch)
    assert prediction["logits"].shape == (2, 4)
    assert prediction["signed_score"].shape == (2,)
    torch.testing.assert_close(
        prediction["probabilities"].sum(dim=-1),
        torch.ones(2),
    )
