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

"""Configuration for Self-supervised Temporal Ensemble Advantage Modeling."""

from dataclasses import dataclass, field

from opentau.configs.policies import PreTrainedConfig
from opentau.configs.types import NormalizationMode
from opentau.optim.optimizers import AdamWConfig
from opentau.optim.schedulers import (
    CosineDecayWithWarmupSchedulerConfig,
    LRSchedulerConfig,
)
from opentau.policies.steam.binning import validate_binning


@PreTrainedConfig.register_subclass("steam")
@dataclass
class SteamConfig(PreTrainedConfig):
    """Configuration for one independently trained STEAM ensemble member."""

    n_obs_steps: int = 1
    chunk_size: int = 50
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
        }
    )

    vision_pretrained_path: str = ""
    language_pretrained_path: str = ""
    tokenizer_path: str = ""
    prompt_max_length: int = 128

    num_bins: int = 32
    max_temporal_offset: int = 32
    length_reference_percentile: float = 90.0
    fusion_hidden_dim: int = 512
    dropout: float = 0.1
    label_smoothing: float = 0.05
    freeze_vision_encoder: bool = False
    freeze_language_model: bool = False
    use_gradient_checkpointing: bool = True

    optimizer_lr: float = 5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_obs_steps != 1:
            raise ValueError(f"STEAM supports n_obs_steps=1, got {self.n_obs_steps}.")
        validate_binning(self.max_temporal_offset, self.num_bins)
        if not 0 < self.length_reference_percentile <= 100:
            raise ValueError(
                f"length_reference_percentile must be in (0, 100], got {self.length_reference_percentile}."
            )
        if self.fusion_hidden_dim < 1:
            raise ValueError("fusion_hidden_dim must be positive.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if not 0 <= self.label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1).")

    def validate_features(self) -> None:
        if not self.image_features:
            raise ValueError("STEAM requires at least one visual input feature.")

    def validate_backbone_paths(self) -> None:
        if not self.vision_pretrained_path:
            raise ValueError("vision_pretrained_path must be set for STEAM.")
        if not self.language_pretrained_path:
            raise ValueError("language_pretrained_path must be set for STEAM.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> LRSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        # LeRobot's standard formatter expects an action chunk even though the
        # STEAM wrapper discards it.
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
