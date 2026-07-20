# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch
from accelerate import DistributedType
from torch import nn

from opentau.optim.ema import NamedParameterEMA
from opentau.scripts.train import _validate_ema_backend


def _linear(value: float) -> nn.Linear:
    module = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        module.weight.fill_(value)
    return module


def test_ema_update_and_count():
    module = _linear(0.0)
    ema = NamedParameterEMA(module, decay=0.5)
    with torch.no_grad():
        module.weight.fill_(2.0)
    ema.update()
    assert ema.num_updates == 1
    assert torch.equal(ema.shadow["weight"], torch.ones_like(ema.shadow["weight"]))


def test_average_parameter_context_restores_live_values_after_error():
    module = _linear(1.0)
    ema = NamedParameterEMA(module, decay=0.5)
    with torch.no_grad():
        module.weight.fill_(3.0)
    original = module.weight.detach().clone()

    with pytest.raises(RuntimeError, match="sentinel"):
        with ema.average_parameters():
            assert torch.equal(module.weight, torch.ones_like(module.weight))
            raise RuntimeError("sentinel")

    assert torch.equal(module.weight, original)


def test_resume_matches_uninterrupted_ema_update():
    uninterrupted_module = _linear(0.0)
    uninterrupted = NamedParameterEMA(uninterrupted_module, decay=0.75)
    with torch.no_grad():
        uninterrupted_module.weight.fill_(2.0)
    uninterrupted.update()
    checkpoint = uninterrupted.state_dict()
    with torch.no_grad():
        uninterrupted_module.weight.fill_(6.0)
    uninterrupted.update()

    resumed_module = _linear(2.0)
    resumed = NamedParameterEMA(resumed_module, decay=0.75)
    resumed.load_state_dict(checkpoint)
    with torch.no_grad():
        resumed_module.weight.fill_(6.0)
    resumed.update()

    assert resumed.num_updates == uninterrupted.num_updates
    assert torch.equal(resumed.shadow["weight"], uninterrupted.shadow["weight"])


def test_load_rejects_parameter_name_mismatch():
    module = _linear(0.0)
    ema = NamedParameterEMA(module, decay=0.5)
    state = ema.state_dict()
    state["shadow"] = {"wrong": torch.zeros(1)}
    with pytest.raises(ValueError, match="parameter names differ"):
        ema.load_state_dict(state)


def test_ema_backend_validation_accepts_ddp_and_rejects_sharded_backends():
    _validate_ema_backend(0.99, DistributedType.NO)
    _validate_ema_backend(0.99, DistributedType.MULTI_GPU)
    _validate_ema_backend(None, DistributedType.FSDP)

    with pytest.raises(ValueError, match="single-process and replicated DDP"):
        _validate_ema_backend(0.99, DistributedType.FSDP)
    with pytest.raises(ValueError, match="single-process and replicated DDP"):
        _validate_ema_backend(0.99, DistributedType.DEEPSPEED)


@pytest.mark.parametrize("decay", [-0.1, 1.0])
def test_invalid_decay_is_rejected(decay):
    with pytest.raises(ValueError):
        NamedParameterEMA(_linear(0.0), decay)
