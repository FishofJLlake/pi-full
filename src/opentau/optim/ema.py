# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Named-parameter exponential moving averages for replicated training."""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn


class NamedParameterEMA:
    """Maintain fp32 EMA shadows for trainable parameters by stable name."""

    def __init__(self, module: nn.Module, decay: float):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1). Got {decay}.")
        self.decay = float(decay)
        self.num_updates = 0
        self._parameters = {
            name: parameter for name, parameter in module.named_parameters() if parameter.requires_grad
        }
        if not self._parameters:
            raise ValueError("EMA requires at least one trainable named parameter.")
        self.shadow = {
            name: parameter.detach().to(dtype=torch.float32).clone()
            for name, parameter in self._parameters.items()
        }
        self._active = False

    @torch.no_grad()
    def update(self) -> None:
        """Update the shadow once after a completed optimizer step."""
        alpha = 1.0 - self.decay
        for name, parameter in self._parameters.items():
            value = parameter.detach().to(device=self.shadow[name].device, dtype=torch.float32)
            self.shadow[name].mul_(self.decay).add_(value, alpha=alpha)
        self.num_updates += 1

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        """Temporarily expose EMA parameters and restore live values exactly."""
        if self._active:
            raise RuntimeError("Nested EMA parameter contexts are not supported.")
        self._active = True
        with torch.no_grad():
            backup = {name: parameter.detach().clone() for name, parameter in self._parameters.items()}
            try:
                for name, parameter in self._parameters.items():
                    parameter.copy_(self.shadow[name].to(device=parameter.device, dtype=parameter.dtype))
                yield
            finally:
                for name, parameter in self._parameters.items():
                    parameter.copy_(backup[name])
                self._active = False

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        decay = float(state_dict["decay"])
        if decay != self.decay:
            raise ValueError(f"EMA decay mismatch: checkpoint has {decay}, config has {self.decay}.")
        loaded = state_dict["shadow"]
        expected_names = set(self._parameters)
        loaded_names = set(loaded)
        if loaded_names != expected_names:
            missing = sorted(expected_names - loaded_names)
            extra = sorted(loaded_names - expected_names)
            raise ValueError(f"EMA parameter names differ; missing={missing}, extra={extra}.")
        self.shadow = {
            name: loaded[name].detach().to(device=parameter.device, dtype=torch.float32).clone()
            for name, parameter in self._parameters.items()
        }
        self.num_updates = int(state_dict["num_updates"])
