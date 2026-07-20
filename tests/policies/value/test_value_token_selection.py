from types import SimpleNamespace

import pytest
import torch
from torch import nn

from opentau.policies.value.modeling_value import ValueFunction, ValueModel


def test_classification_indices_select_last_valid_prompt_token_per_sample():
    masks = torch.tensor(
        [
            [True, True, False, False],
            [True, False, True, False],
            [False, True, True, True],
        ]
    )

    indices = ValueModel._get_classification_indices(masks, num_image_tokens=6)

    assert torch.equal(indices, torch.tensor([7, 8, 9]))


def test_classification_indices_reject_all_padding_prompt():
    with pytest.raises(ValueError, match="at least one valid token"):
        ValueModel._get_classification_indices(
            torch.zeros((1, 4), dtype=torch.bool),
            num_image_tokens=6,
        )


def test_value_bins_are_float32_centers_used_by_the_scalar_expectation():
    policy = object.__new__(ValueFunction)
    nn.Module.__init__(policy)
    policy.config = SimpleNamespace(
        reward_config=SimpleNamespace(number_of_bins=4),
    )

    bins = policy.value_bins("cpu")
    assert bins.dtype == torch.float32
    torch.testing.assert_close(
        bins,
        torch.tensor([-0.875, -0.625, -0.375, -0.125]),
    )
    logits = torch.tensor([[-100.0, -100.0, -100.0, 100.0]])
    torch.testing.assert_close(policy.calculate_value(logits), torch.tensor([-0.125]))
