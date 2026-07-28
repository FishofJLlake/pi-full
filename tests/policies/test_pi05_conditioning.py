# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from opentau.policies.pi05.configuration_pi05 import PI05Config
from opentau.policies.pi05.modeling_pi05 import (
    PI05FlowMatching,
    PI05Policy,
    apply_classifier_free_guidance,
    sample_training_delay,
)


def _config(**overrides) -> PI05Config:
    values = {
        "n_obs_steps": 1,
        "chunk_size": 4,
        "n_action_steps": 4,
        "max_delay": 3,
        "max_state_dim": 4,
        "max_action_dim": 4,
    }
    values.update(overrides)
    return PI05Config(**values)


def test_conditioning_defaults_preserve_existing_behavior():
    config = _config(max_delay=0)
    assert config.advantage == "ignore"
    assert config.advantage_threshold == 0.0
    assert config.cfg_dropout == 0.0
    assert config.guidance_scale == 1.0
    assert config.delay_sampling == "uniform"
    assert config.delay_exponential_decay == 0.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("cfg_dropout", -0.1),
        ("advantage_threshold", -0.1),
        ("cfg_dropout", 1.1),
        ("guidance_scale", -0.1),
        ("delay_exponential_decay", 0.0),
        ("delay_exponential_decay", 1.1),
        ("advantage", "invalid"),
    ],
)
def test_conditioning_config_validation(field, value):
    with pytest.raises(ValueError):
        _config(**{field: value})


def test_uniform_delay_keeps_legacy_randint_sequence():
    config = _config(delay_sampling="uniform")
    torch.manual_seed(123)
    expected = torch.randint(0, config.max_delay + 1, (32,))
    torch.manual_seed(123)
    actual = sample_training_delay(config, 32, "cpu")
    assert torch.equal(actual, expected)


def test_exponential_delay_uses_multiplicative_weights(monkeypatch):
    config = _config(delay_sampling="exponential", delay_exponential_decay=0.5)
    captured = {}

    def fake_multinomial(weights, num_samples, replacement):
        captured["weights"] = weights.detach().cpu()
        captured["num_samples"] = num_samples
        captured["replacement"] = replacement
        return torch.zeros(num_samples, dtype=torch.long, device=weights.device)

    monkeypatch.setattr(torch, "multinomial", fake_multinomial)

    delays = sample_training_delay(config, 3, "cpu")

    assert torch.equal(delays, torch.zeros(3, dtype=torch.long))
    assert captured["num_samples"] == 3
    assert captured["replacement"] is True
    assert torch.equal(captured["weights"], torch.tensor([1.0, 0.5, 0.25, 0.125]))


def test_cfg_formula_supports_interpolation_and_extrapolation():
    cond = torch.tensor([1.0, 3.0])
    uncond = torch.tensor([-1.0, 1.0])
    assert torch.equal(apply_classifier_free_guidance(cond, uncond, 0.0), uncond)
    assert torch.equal(apply_classifier_free_guidance(cond, uncond, 1.0), cond)
    assert torch.equal(
        apply_classifier_free_guidance(cond, uncond, 2.0),
        torch.tensor([3.0, 5.0]),
    )


class _RecordingTokenizer:
    def __init__(self):
        self.prompts = None

    def __call__(self, prompts, **_kwargs):
        self.prompts = prompts
        batch_size = len(prompts)
        return {
            "input_ids": torch.ones(batch_size, 4, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
        }


class _LanguageHarness:
    def __init__(self, config, *, training):
        self.config = config
        self.training = training
        self.language_tokenizer = _RecordingTokenizer()

    def prepare_discrete_state(self, batch):
        return ["0 1"] * len(batch["prompt"])


@pytest.mark.parametrize(
    ("advantages", "labels"),
    [
        ([0.2, -0.2, 0.05, -0.05], ["positive", "negative", "none", "none"]),
        (None, ["none", "none", "none", "none"]),
    ],
)
def test_advantage_prompt_three_way_training_labels(advantages, labels):
    harness = _LanguageHarness(
        _config(max_delay=0, advantage="use", advantage_threshold=0.1),
        training=True,
    )
    batch = {"prompt": ["a", "b", "c", "d"], "state": torch.zeros(4, 4)}
    if advantages is not None:
        batch["advantage"] = torch.tensor(advantages)

    PI05Policy.prepare_language(harness, batch)

    for prompt, label in zip(harness.language_tokenizer.prompts, labels, strict=True):
        assert f"Advantage: {label}" in prompt


def test_missing_inference_advantage_selects_positive_and_uncond_selects_none():
    harness = _LanguageHarness(
        _config(max_delay=0, advantage="use", advantage_threshold=0.1),
        training=False,
    )
    batch = {"prompt": ["task"], "state": torch.zeros(1, 4)}
    PI05Policy.prepare_language(harness, batch)
    assert "Advantage: positive" in harness.language_tokenizer.prompts[0]

    PI05Policy.prepare_language(harness, batch, force_uncond=True)
    assert "Advantage: none" in harness.language_tokenizer.prompts[0]


def test_cfg_dropout_uses_torch_rng_per_sample():
    harness = _LanguageHarness(
        _config(max_delay=0, advantage="use", cfg_dropout=1.0),
        training=True,
    )
    batch = {
        "prompt": ["a", "b"],
        "state": torch.zeros(2, 4),
        "advantage": torch.ones(2),
    }
    PI05Policy.prepare_language(harness, batch)
    assert all("Advantage: none" in prompt for prompt in harness.language_tokenizer.prompts)


class _EmbeddingStub(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_inputs = []
        self.forward_calls = []

    def embed_image(self, image):
        return torch.zeros(image.shape[0], 2, 4)

    def embed_language_tokens(self, tokens):
        self.language_inputs.append(tokens.detach().clone())
        return torch.zeros(tokens.shape[0], tokens.shape[1], 4)

    def embed_discrete_actions(self, tokens):
        return torch.zeros(tokens.shape[0], tokens.shape[1], 4)

    def forward(self, **kwargs):
        self.forward_calls.append(kwargs)
        embeddings = kwargs["inputs_embeds"][0]
        return (torch.zeros_like(embeddings), None), "updated-cache"


class _IndicatorTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [7, 8, 9] if text == "Action: " else [1, 2]


def test_action_indicator_is_valid_context_for_variable_prompts_and_responses():
    model = object.__new__(PI05FlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        state_type="discrete",
        predict_response=True,
        use_modality_embedding=False,
    )
    model.paligemma_with_expert = _EmbeddingStub()
    model.language_tokenizer = _IndicatorTokenizer()

    prompt_masks = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.bool)
    response_masks = torch.tensor([[1, 0, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
    action_masks = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    _, pad_masks, _, _ = PI05FlowMatching.embed_prefix(
        model,
        images=[torch.zeros(2, 3, 8, 8)],
        img_masks=[torch.ones(2, dtype=torch.bool)],
        lang_tokens=torch.ones(2, 5, dtype=torch.long),
        lang_masks=prompt_masks,
        response_tokens=torch.ones(2, 4, dtype=torch.long),
        response_masks=response_masks,
        discrete_actions=torch.ones(2, 4, dtype=torch.long),
        discrete_action_masks=action_masks,
    )

    action_indicator = pad_masks[:, -7:-4]
    ce_context = pad_masks[:, -5:-1]
    assert torch.all(action_indicator)
    assert torch.all(ce_context[:, 0])
    assert torch.equal(ce_context[:, 1:], action_masks[:, :-1])


def test_flow_prefix_includes_action_indicator_but_not_discrete_targets():
    model = object.__new__(PI05FlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        state_type="discrete",
        predict_response=False,
        use_modality_embedding=False,
    )
    model.paligemma_with_expert = _EmbeddingStub()
    model.language_tokenizer = _IndicatorTokenizer()

    _, inference_masks, inference_att_masks, inference_segments = PI05FlowMatching.embed_prefix(
        model,
        images=[torch.zeros(2, 3, 8, 8)],
        img_masks=[torch.ones(2, dtype=torch.bool)],
        lang_tokens=torch.ones(2, 5, dtype=torch.long),
        lang_masks=torch.ones(2, 5, dtype=torch.bool),
    )
    assert inference_masks.shape[1] == 2 + 5 + 3
    assert torch.all(inference_masks[:, -3:])
    assert not torch.any(inference_att_masks[:, -3:])
    assert torch.all(
        inference_segments[:, -3:] == PI05FlowMatching._MODALITY_LANGUAGE
    )
    assert torch.equal(
        model.paligemma_with_expert.language_inputs[-1],
        torch.tensor([[7, 8, 9], [7, 8, 9]]),
    )

    action_masks = torch.ones(2, 4, dtype=torch.bool)
    _, pad_masks, att_masks, segment_ids = PI05FlowMatching.embed_prefix(
        model,
        images=[torch.zeros(2, 3, 8, 8)],
        img_masks=[torch.ones(2, dtype=torch.bool)],
        lang_tokens=torch.ones(2, 5, dtype=torch.long),
        lang_masks=torch.ones(2, 5, dtype=torch.bool),
        discrete_actions=torch.ones(2, 4, dtype=torch.long),
        discrete_action_masks=action_masks,
    )

    flow_prefix_length = pad_masks.shape[1] - action_masks.shape[1]
    action_indicator = slice(flow_prefix_length - 3, flow_prefix_length)
    discrete_targets = slice(flow_prefix_length, None)
    assert torch.equal(
        model.paligemma_with_expert.language_inputs[-1],
        torch.tensor([[7, 8, 9], [7, 8, 9]]),
    )
    assert torch.all(pad_masks[:, action_indicator])
    assert not torch.any(att_masks[:, action_indicator])
    assert torch.all(segment_ids[:, action_indicator] == PI05FlowMatching._MODALITY_LANGUAGE)
    assert torch.all(att_masks[:, discrete_targets])
    assert torch.all(
        segment_ids[:, discrete_targets] == PI05FlowMatching._MODALITY_DISCRETE_ACTION
    )


def test_response_inference_appends_action_indicator_to_flow_cache():
    model = object.__new__(PI05FlowMatching)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(use_modality_embedding=False)
    model.paligemma_with_expert = _EmbeddingStub()
    model.language_tokenizer = _IndicatorTokenizer()

    prefix_pad_masks = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    prefix_offsets = prefix_pad_masks.long().sum(dim=1, keepdim=True) - 1
    updated_masks, updated_offsets, updated_cache = model._append_action_indicator_to_cache(
        prefix_pad_masks, "old-cache", prefix_offsets
    )

    assert updated_cache == "updated-cache"
    assert torch.equal(updated_masks[:, -3:], torch.ones(2, 3, dtype=torch.bool))
    assert torch.equal(updated_offsets, prefix_offsets + 3)
    forward_call = model.paligemma_with_expert.forward_calls[-1]
    assert forward_call["past_key_values"] == "old-cache"
    assert forward_call["n_cross_att_tokens"] == prefix_pad_masks.shape[1] + 3
    assert forward_call["inputs_embeds"][0].shape[1] == 3
