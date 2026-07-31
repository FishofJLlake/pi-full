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

"""Inference-only deep-ensemble wrapper for STEAM checkpoints."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from safetensors import safe_open
from safetensors.torch import load_file
from torch import Tensor, nn

from opentau.policies.steam.configuration_steam import SteamConfig
from opentau.policies.steam.modeling_steam import SteamPolicy


class SteamEnsemblePolicy(nn.Module):
    """Aggregate one or more STEAM members with RLinf's worst-of-N rule."""

    def __init__(self, config: SteamConfig, members: list[SteamPolicy]) -> None:
        super().__init__()
        if not members:
            raise ValueError("SteamEnsemblePolicy requires at least one member.")
        if len(members) != config.ensemble_size:
            raise ValueError(
                f"Config ensemble_size={config.ensemble_size} but received {len(members)} members."
            )
        self.config = config
        self.members = nn.ModuleList(members)

    @staticmethod
    def _gather_worst(member_tensor: Tensor, member_indices: Tensor) -> Tensor:
        batch_indices = torch.arange(member_tensor.shape[1], device=member_tensor.device)
        return member_tensor[member_indices, batch_indices]

    @torch.no_grad()
    def predict_temporal_offset(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        outputs = [member.predict_temporal_offset(batch) for member in self.members]
        member_logits = torch.stack([output["logits"] for output in outputs], dim=0)
        member_probabilities = torch.stack([output["probabilities"] for output in outputs], dim=0)
        member_expected_bins = torch.stack([output["expected_bin"] for output in outputs], dim=0)
        member_temporal_offsets = torch.stack([output["temporal_offset"] for output in outputs], dim=0)
        member_expected_stride_scores = torch.stack([output["signed_score"] for output in outputs], dim=0)
        member_rlinf_signed_scores = torch.stack([output["rlinf_signed_score"] for output in outputs], dim=0)
        prediction_min, worst_member_indices = member_rlinf_signed_scores.min(dim=0)
        probabilities = self._gather_worst(member_probabilities, worst_member_indices)
        return {
            "logits": self._gather_worst(member_logits, worst_member_indices),
            "probabilities": probabilities,
            "expected_bin": self._gather_worst(member_expected_bins, worst_member_indices),
            "temporal_offset": self._gather_worst(member_temporal_offsets, worst_member_indices),
            "signed_score": self._gather_worst(member_expected_stride_scores, worst_member_indices),
            "rlinf_signed_score": prediction_min,
            "member_logits": member_logits,
            "member_probabilities": member_probabilities,
            "member_expected_bins": member_expected_bins,
            "member_temporal_offsets": member_temporal_offsets,
            "member_expected_stride_scores": member_expected_stride_scores,
            "member_rlinf_signed_scores": member_rlinf_signed_scores,
            "prediction_mean": member_rlinf_signed_scores.mean(dim=0),
            "prediction_min": prediction_min,
            "prediction_variance": member_rlinf_signed_scores.var(dim=0, unbiased=False),
        }


def load_steam_inference_checkpoint(
    checkpoint: Path,
    config: SteamConfig,
    *,
    local_files_only: bool,
) -> SteamPolicy | SteamEnsemblePolicy:
    """Load either a legacy single-member or merged ``members.*`` checkpoint."""
    weights_path = checkpoint / SAFETENSORS_SINGLE_FILE
    if not weights_path.is_file():
        raise FileNotFoundError(f"STEAM weights not found: {weights_path}")
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        weight_keys = handle.keys()
        is_merged_ensemble = any(key.startswith("members.") for key in weight_keys)

    if not is_merged_ensemble:
        if config.ensemble_size != 1:
            raise ValueError(
                f"Config requests ensemble_size={config.ensemble_size}, but {weights_path} "
                "contains legacy single-member weights."
            )
        return SteamPolicy.from_pretrained(
            checkpoint,
            config=config,
            strict=True,
            local_files_only=local_files_only,
            backbone_local_files_only=local_files_only,
        )

    member_config = copy.deepcopy(config)
    member_config.ensemble_size = 1
    base_member = SteamPolicy(
        member_config,
        backbone_local_files_only=local_files_only,
    )
    members = [base_member] + [copy.deepcopy(base_member) for _ in range(config.ensemble_size - 1)]
    ensemble = SteamEnsemblePolicy(config, members)
    state_dict = load_file(str(weights_path), device="cpu")
    incompatible = ensemble.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Merged STEAM checkpoint is incompatible: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}."
        )
    ensemble.eval()
    return ensemble


def split_member_predictions(prediction: dict[str, Tensor]) -> dict[str, Tensor]:
    """Normalize single- and multi-member predictions to ``[member, batch, ...]``."""
    if "member_expected_bins" in prediction:
        return {
            "expected_bins": prediction["member_expected_bins"],
            "probabilities": prediction["member_probabilities"],
            "expected_stride_scores": prediction["member_expected_stride_scores"],
            "rlinf_signed_scores": prediction["member_rlinf_signed_scores"],
        }
    return {
        "expected_bins": rearrange(prediction["expected_bin"], "batch -> 1 batch"),
        "probabilities": rearrange(prediction["probabilities"], "batch bin -> 1 batch bin"),
        "expected_stride_scores": rearrange(prediction["signed_score"], "batch -> 1 batch"),
        "rlinf_signed_scores": rearrange(prediction["rlinf_signed_score"], "batch -> 1 batch"),
    }
