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
    expected_rlinf_signed_score,
    expected_signed_offset,
    scaled_signed_offset_to_bin,
    signed_offset_to_bin,
)
from opentau.policies.steam.configuration_steam import SteamConfig
from opentau.policies.steam.ensemble_modeling_steam import SteamEnsemblePolicy


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

    torch.testing.assert_close(
        expected_rlinf_signed_score(torch.eye(4), 4),
        torch.tensor([-1.0, -0.5, 0.5, 1.0]),
    )


def test_steam_config_rejects_zero_ensemble_members():
    with pytest.raises(ValueError, match="ensemble_size"):
        SteamConfig(ensemble_size=0)


def test_steam_config_rejects_an_invalid_bin_layout():
    with pytest.raises(ValueError, match="divisible"):
        SteamConfig(max_temporal_offset=3, num_bins=4)


def test_steam_training_preset_uses_requested_learning_rate_schedule():
    config = SteamConfig()

    optimizer = config.get_optimizer_preset()
    scheduler = config.get_scheduler_preset()

    assert optimizer.lr == 1e-4
    assert scheduler.num_warmup_steps == 1_000
    assert scheduler.num_decay_steps == 20_000
    assert scheduler.peak_lr == 1e-4
    assert scheduler.decay_lr == 1e-5


def test_gradient_checkpointing_uses_non_reentrant_path():
    calls = []
    module = SimpleNamespace(gradient_checkpointing_enable=lambda **kwargs: calls.append(kwargs))

    modeling_steam._enable_gradient_checkpointing(module)  # noqa: SLF001

    assert calls == [{"gradient_checkpointing_kwargs": {"use_reentrant": False}}]


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
        image_resolution=(8, 8),
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
    assert losses["NeighborAccuracy"].ndim == 0
    assert losses["CE"].requires_grad
    prediction = policy.predict_temporal_offset(batch)
    assert prediction["logits"].shape == (2, 4)
    assert prediction["rlinf_signed_score"].shape == (2,)
    assert prediction["signed_score"].shape == (2,)
    torch.testing.assert_close(
        prediction["probabilities"].sum(dim=-1),
        torch.ones(2),
    )


class _FixedMember(nn.Module):
    def __init__(self, scores: list[float], probability_bin: int):
        super().__init__()
        self.register_buffer("scores", torch.tensor(scores, dtype=torch.float32))
        self.probability_bin = probability_bin

    def predict_temporal_offset(self, batch):
        batch_size = len(batch["prompt"])
        scores = self.scores[:batch_size]
        probabilities = torch.zeros(batch_size, 4)
        probabilities[:, self.probability_bin] = 1.0
        return {
            "logits": probabilities,
            "probabilities": probabilities,
            "expected_bin": torch.full((batch_size,), float(self.probability_bin)),
            "temporal_offset": scores * 4,
            "signed_score": scores,
            "rlinf_signed_score": scores,
        }


def test_arbitrary_size_ensemble_uses_rlinf_pointwise_minimum():
    config = SteamConfig(ensemble_size=3)
    policy = SteamEnsemblePolicy(
        config,
        [
            _FixedMember([0.1, -0.2], 2),
            _FixedMember([-0.7, 0.4], 0),
            _FixedMember([-0.1, 0.2], 1),
        ],
    )
    result = policy.predict_temporal_offset({"prompt": ["a", "b"]})
    torch.testing.assert_close(result["prediction_min"], torch.tensor([-0.7, -0.2]))
    assert result["member_rlinf_signed_scores"].shape == (3, 2)
    assert result["probabilities"].argmax(dim=-1).tolist() == [0, 2]


def test_seeded_fake_training_smoke_is_bit_identical(monkeypatch):
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

    def run(seed: int) -> torch.Tensor:
        torch.manual_seed(seed)
        config = SteamConfig(
            vision_pretrained_path="vision",
            language_pretrained_path="language",
            image_resolution=(8, 8),
            tokenizer_path="tokenizer",
            num_bins=4,
            max_temporal_offset=4,
            fusion_hidden_dim=8,
            dropout=0.1,
            use_gradient_checkpointing=False,
            input_features={
                "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            },
        )
        policy = modeling_steam.SteamPolicy(config)
        optimizer = torch.optim.SGD(policy.parameters(), lr=1e-3)
        batch = {
            "steam_images_t": {"camera0": torch.rand(2, 3, 8, 8)},
            "steam_images_tk": {"camera0": torch.rand(2, 3, 8, 8)},
            "steam_image_masks_t": {"camera0": torch.tensor([True, True])},
            "steam_image_masks_tk": {"camera0": torch.tensor([True, True])},
            "steam_target_bin": torch.tensor([2, 3]),
            "prompt": ["pick object", "place object"],
        }
        losses = []
        for _ in range(2):
            optimizer.zero_grad()
            loss = policy(batch)["CE"]
            losses.append(loss.detach().clone())
            loss.backward()
            optimizer.step()
        return torch.stack(losses)

    first = run(1234)
    second = run(1234)
    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)


def test_steam_rejects_non_native_input_resolution(monkeypatch):
    monkeypatch.setattr(
        modeling_steam.AutoModel,
        "from_pretrained",
        lambda path, **_kwargs: _FakeVision() if path == "vision" else _FakeLanguage(),
    )
    monkeypatch.setattr(modeling_steam.AutoTokenizer, "from_pretrained", lambda *_a, **_k: _FakeTokenizer())
    monkeypatch.setattr(
        modeling_steam.AutoImageProcessor,
        "from_pretrained",
        lambda *_a, **_k: _FakeImageProcessor(),
    )
    policy = modeling_steam.SteamPolicy(
        SteamConfig(
            vision_pretrained_path="vision",
            language_pretrained_path="language",
            tokenizer_path="tokenizer",
            image_resolution=(8, 8),
            use_gradient_checkpointing=False,
            input_features={
                "camera0": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
            },
        )
    )
    with pytest.raises(ValueError, match="native resolution"):
        policy._preprocess_images(torch.zeros(1, 3, 4, 4))  # noqa: SLF001
