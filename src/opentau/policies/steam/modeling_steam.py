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

"""STEAM temporal progress classifier for OpenTau training and inference."""

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from einops import einsum, rearrange, reduce
from torch import Tensor, nn
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from opentau.policies.pretrained import PreTrainedPolicy
from opentau.policies.steam.binning import expected_signed_offset
from opentau.policies.steam.configuration_steam import SteamConfig


def _hidden_size(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    for candidate in (
        config,
        getattr(config, "vision_config", None),
        getattr(config, "text_config", None),
    ):
        if candidate is None:
            continue
        for name in ("projection_dim", "hidden_size", "d_model"):
            value = getattr(candidate, name, None)
            if value is not None:
                return int(value)
    raise ValueError(f"Cannot infer feature dimension from {type(model).__name__}.")


def _image_size(processor: Any) -> tuple[int, int]:
    size = getattr(processor, "size", None)
    if isinstance(size, int):
        return size, size
    if isinstance(size, Mapping):
        if "height" in size and "width" in size:
            return int(size["height"]), int(size["width"])
        edge = size.get("shortest_edge") or size.get("longest_edge")
        if edge is not None:
            return int(edge), int(edge)
    return 224, 224


def _module_dtype(module: nn.Module, fallback: torch.dtype = torch.float32) -> torch.dtype:
    parameter = next(module.parameters(), None)
    return parameter.dtype if parameter is not None else fallback


def _enable_gradient_checkpointing(module: nn.Module) -> None:
    enable = getattr(module, "gradient_checkpointing_enable", None)
    if callable(enable):
        enable()


class SteamPolicy(PreTrainedPolicy):
    """Predict signed temporal progress for a pair of frames and a task prompt."""

    config_class = SteamConfig
    name = "steam"

    def __init__(
        self,
        config: SteamConfig,
        per_dataset_stats: list[dict[str, dict[str, Tensor]]] | None = None,
        dataset_names: list[str] | None = None,
        *,
        backbone_local_files_only: bool = False,
    ) -> None:
        del per_dataset_stats, dataset_names
        super().__init__(config)
        config.validate_features()
        config.validate_backbone_paths()

        self.vision_encoder = AutoModel.from_pretrained(
            config.vision_pretrained_path,
            local_files_only=backbone_local_files_only,
        )
        self.language_model = AutoModel.from_pretrained(
            config.language_pretrained_path,
            local_files_only=backbone_local_files_only,
        )
        tokenizer_source = config.tokenizer_path or config.language_pretrained_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            local_files_only=backbone_local_files_only,
        )
        image_processor = AutoImageProcessor.from_pretrained(
            config.vision_pretrained_path,
            local_files_only=backbone_local_files_only,
        )

        height, width = _image_size(image_processor)
        self.image_size = (height, width)
        mean = getattr(image_processor, "image_mean", [0.5, 0.5, 0.5])
        std = getattr(image_processor, "image_std", [0.5, 0.5, 0.5])
        self.register_buffer(
            "image_mean",
            rearrange(torch.tensor(mean, dtype=torch.float32), "channel -> 1 channel 1 1"),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            rearrange(torch.tensor(std, dtype=torch.float32), "channel -> 1 channel 1 1"),
            persistent=False,
        )

        hidden_dim = config.fusion_hidden_dim
        self.image_projector = nn.Sequential(
            nn.Linear(_hidden_size(self.vision_encoder), hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.language_projector = nn.Sequential(
            nn.Linear(_hidden_size(self.language_model), hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim * 3)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden_dim, config.num_bins),
        )

        if config.use_gradient_checkpointing:
            _enable_gradient_checkpointing(self.vision_encoder)
            _enable_gradient_checkpointing(self.language_model)
        if config.freeze_vision_encoder:
            self.vision_encoder.requires_grad_(False)
        if config.freeze_language_model:
            self.language_model.requires_grad_(False)

    def reset(self) -> None:
        """STEAM has no episode-local inference state."""

    def get_optim_params(self):
        return self.parameters()

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        del batch
        raise NotImplementedError("STEAM predicts temporal progress, not actions.")

    def sample_actions(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        del batch, noise
        raise NotImplementedError("STEAM predicts temporal progress, not actions.")

    def _preprocess_images(self, images: Tensor) -> Tensor:
        images = images.to(dtype=torch.float32)
        if tuple(images.shape[-2:]) != self.image_size:
            images = F.interpolate(
                images,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
        images = (images - self.image_mean) / self.image_std
        return images.to(dtype=_module_dtype(self.vision_encoder))

    def _encode_vision(self, images: Tensor) -> Tensor:
        if hasattr(self.vision_encoder, "get_image_features"):
            features = self.vision_encoder.get_image_features(pixel_values=images)
            if isinstance(features, Tensor):
                return features
            pooler = getattr(features, "pooler_output", None)
            if pooler is not None:
                return pooler
        outputs = self.vision_encoder(pixel_values=images, return_dict=True)
        pooler = getattr(outputs, "pooler_output", None)
        if pooler is not None:
            return pooler
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is not None:
            return reduce(hidden, "batch token hidden -> batch hidden", "mean")
        raise TypeError("Vision encoder must expose pooled or last-hidden-state features.")

    def _encode_frame(
        self,
        images_by_camera: Mapping[str, Tensor],
        masks_by_camera: Mapping[str, Tensor] | None,
    ) -> Tensor:
        camera_keys = [
            key for key in self.config.image_features if key in images_by_camera
        ]
        if not camera_keys:
            camera_keys = sorted(images_by_camera)
        if not camera_keys:
            raise ValueError("STEAM received no camera images.")

        image_batches = [images_by_camera[key] for key in camera_keys]
        batch_size = image_batches[0].shape[0]
        if any(image.shape[0] != batch_size for image in image_batches):
            raise ValueError("All STEAM cameras must have the same batch size.")

        flat_images = torch.cat(image_batches, dim=0)
        image_context = (
            torch.no_grad() if self.config.freeze_vision_encoder else nullcontext()
        )
        with image_context:
            encoded = self._encode_vision(self._preprocess_images(flat_images))
        projected = self.image_projector(
            encoded.to(dtype=_module_dtype(self.image_projector))
        )
        projected = rearrange(
            projected, "(camera batch) hidden -> batch camera hidden", camera=len(camera_keys)
        )

        masks = []
        for key in camera_keys:
            if masks_by_camera is None or key not in masks_by_camera:
                masks.append(
                    torch.ones(
                        batch_size,
                        device=projected.device,
                        dtype=torch.bool,
                    )
                )
            else:
                masks.append(
                    masks_by_camera[key].to(
                        device=projected.device,
                        dtype=torch.bool,
                    )
                )
        mask = torch.stack(masks, dim=1)
        if not torch.all(mask.any(dim=1)):
            raise ValueError("Every STEAM sample must contain at least one valid camera.")
        weights = rearrange(mask, "batch camera -> batch camera 1").to(dtype=projected.dtype)
        numerator = reduce(projected * weights, "batch camera hidden -> batch hidden", "sum")
        return numerator / reduce(weights, "batch camera 1 -> batch 1", "sum").clamp(min=1)

    def _encode_language(self, prompts: Sequence[str] | str, device: torch.device) -> Tensor:
        if isinstance(prompts, str):
            prompts = [prompts]
        formatted_prompts = [f"Task: {prompt}\nValue: " for prompt in prompts]
        tokenized = self.tokenizer(
            formatted_prompts,
            padding=True,
            truncation=True,
            max_length=self.config.prompt_max_length,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        language_context = (
            torch.no_grad() if self.config.freeze_language_model else nullcontext()
        )
        with language_context:
            outputs = self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None:
            raise TypeError("Language model must expose last_hidden_state.")
        weights = rearrange(attention_mask, "batch token -> batch token 1").to(dtype=hidden.dtype)
        pooled = reduce(hidden * weights, "batch token hidden -> batch hidden", "sum")
        pooled = pooled / reduce(weights, "batch token 1 -> batch 1", "sum").clamp(min=1)
        return self.language_projector(
            pooled.to(dtype=_module_dtype(self.language_projector))
        )

    def _logits(self, batch: dict[str, Any]) -> Tensor:
        frame_t = self._encode_frame(
            batch["steam_images_t"],
            batch.get("steam_image_masks_t"),
        )
        frame_tk = self._encode_frame(
            batch["steam_images_tk"],
            batch.get("steam_image_masks_tk"),
        )
        language = self._encode_language(batch["prompt"], frame_t.device)
        fused = torch.cat([frame_t, frame_tk, language], dim=-1)
        return self.classifier(self.fusion_norm(fused)).to(dtype=torch.float32)

    def forward(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        logits = self._logits(batch)
        labels = batch["steam_target_bin"].to(device=logits.device, dtype=torch.long)
        ce_loss = F.cross_entropy(
            logits,
            labels,
            label_smoothing=self.config.label_smoothing,
        )
        accuracy = (logits.argmax(dim=-1) == labels).to(torch.float32).mean()
        zero = torch.zeros_like(ce_loss, requires_grad=False)
        return {
            "MSE": zero,
            "CE": ce_loss,
            "L1": zero,
            "Accuracy": accuracy,
        }

    @torch.no_grad()
    def predict_temporal_offset(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        self.eval()
        logits = self._logits(batch)
        probabilities = torch.softmax(logits, dim=-1)
        bins = torch.arange(
            self.config.num_bins,
            device=probabilities.device,
            dtype=probabilities.dtype,
        )
        expected_bin = einsum(probabilities, bins, "batch bin, bin -> batch")
        temporal_offset = expected_signed_offset(
            probabilities,
            self.config.max_temporal_offset,
            self.config.num_bins,
        )
        return {
            "logits": logits,
            "probabilities": probabilities,
            "expected_bin": expected_bin,
            "temporal_offset": temporal_offset,
            "signed_score": temporal_offset / self.config.max_temporal_offset,
        }
