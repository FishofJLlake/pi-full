#!/usr/bin/env python
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

"""Compute conservative STEAM advantages from independent checkpoints."""

from __future__ import annotations

import argparse
import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import draccus
import numpy as np
import torch
from torch.utils.data import DataLoader

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.configs.refs import resolve_refs_to_tempfile
from opentau.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from opentau.datasets.advantage_metadata import persist_advantage_bundle
from opentau.datasets.factory import make_dataset
from opentau.datasets.steam_pair_dataset import SteamPairDataset, set_global_length_reference
from opentau.policies.factory import get_policy_class
from opentau.policies.steam.configuration_steam import SteamConfig
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import auto_torch_device, init_logging


@dataclass
class DatasetScores:
    config: DatasetConfig
    dataset: SteamPairDataset
    minimum_advantage: dict[tuple[int, int], float]
    tasks: dict[tuple[int, int], str]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score datasets with independent STEAM members, take the pointwise "
            "minimum, and write the active frame-keyed advantage bundle."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=True,
        help="STEAM checkpoint directory. Repeat for every independent member.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Train config/checkpoint used for dataset settings (defaults to first checkpoint).",
    )
    parser.add_argument(
        "--dataset-mixture",
        type=Path,
        default=None,
        help="Optional DatasetMixtureConfig JSON override.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--expert-positive-fraction",
        type=float,
        default=0.8,
        help="Top expert fraction labeled positive.",
    )
    parser.add_argument(
        "--non-expert-positive-fraction",
        type=float,
        default=0.3,
        help="Top non-expert fraction labeled positive.",
    )
    return parser.parse_args()


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def paper_advantage_from_expected_bin(expected_bin: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Apply STEAM's fixed-lookahead affine baseline to the expected bin."""
    baseline_bin = float(num_bins - 1)
    return (2.0 / num_bins) * (expected_bin.to(torch.float32) - baseline_bin)


def source_quantile_thresholds(
    scores_by_source: dict[str, list[float]],
    *,
    expert_positive_fraction: float,
    non_expert_positive_fraction: float,
) -> dict[str, float]:
    """Derive independent inclusive thresholds for expert and non-expert pools."""
    fractions = {
        "expert": expert_positive_fraction,
        "non_expert": non_expert_positive_fraction,
    }
    thresholds: dict[str, float] = {}
    for source, scores in scores_by_source.items():
        fraction = fractions[source]
        if not 0 < fraction <= 1:
            raise ValueError(
                f"{source} positive fraction must be in (0, 1], got {fraction}."
            )
        if not scores:
            continue
        thresholds[source] = float(
            np.percentile(
                np.asarray(scores, dtype=np.float64),
                (1.0 - fraction) * 100.0,
            )
        )
    return thresholds


def _architecture_signature(config: SteamConfig) -> tuple:
    image_features = tuple(
        (key, str(feature.type), tuple(feature.shape))
        for key, feature in sorted(config.image_features.items())
    )
    return (
        config.num_bins,
        config.max_temporal_offset,
        config.fusion_hidden_dim,
        config.vision_pretrained_path,
        config.language_pretrained_path,
        config.tokenizer_path,
        image_features,
    )


def _core_signature(config: SteamConfig) -> tuple:
    return (
        config.num_bins,
        config.max_temporal_offset,
        config.fusion_hidden_dim,
        config.vision_pretrained_path,
        config.language_pretrained_path,
        config.tokenizer_path,
    )


def _load_member_config(checkpoint: Path, base: SteamConfig) -> SteamConfig:
    train_config_path = checkpoint / TRAIN_CONFIG_NAME
    if train_config_path.is_file():
        member_train_config = TrainPipelineConfig.from_pretrained(
            checkpoint,
            local_files_only=True,
        )
        member_config = member_train_config.policy
    else:
        member_config = copy.deepcopy(base)
    if not isinstance(member_config, SteamConfig):
        raise TypeError(f"Checkpoint {checkpoint} does not contain a STEAM policy config.")
    return member_config


def _make_dataloader(
    dataset: SteamPairDataset,
    *,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
) -> DataLoader:
    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def _collect_datasets(
    cfg: TrainPipelineConfig,
    mixture: DatasetMixtureConfig,
) -> list[DatasetScores]:
    collected = []
    runtime_cfg = copy.deepcopy(cfg)
    runtime_cfg.val_freq = 0
    runtime_cfg.dataset_mixture = mixture
    for configured_dataset in mixture.datasets:
        dataset_config = copy.deepcopy(configured_dataset)
        dataset_config.image_transforms.enable = False
        dataset_config.prompt_substitutions = None
        if dataset_config.steam_source not in ("expert", "non_expert"):
            raise ValueError(
                "Every STEAM labeling dataset must set steam_source to "
                "'expert' or 'non_expert'."
            )
        result = make_dataset(
            dataset_config,
            runtime_cfg,
            return_advantage_input=False,
            local_files_only=True,
            steam_mode="inference",
        )
        if isinstance(result, tuple):
            raise AssertionError("STEAM labeling unexpectedly created a validation split.")
        if not isinstance(result, SteamPairDataset):
            raise TypeError(f"Expected SteamPairDataset, got {type(result).__name__}.")
        nonterminal_keys = {
            (episode_index, frame_index)
            for episode_index, frame_index, _ in result.frame_records
        }
        terminal_keys = {
            (episode_index, frame_index)
            for episode_index, frame_index, _ in result.terminal_records
        }
        nonterminal_keys -= terminal_keys
        collected.append(
            DatasetScores(
                config=dataset_config,
                dataset=result,
                minimum_advantage={key: float("inf") for key in nonterminal_keys},
                tasks={},
            )
        )
    set_global_length_reference(
        [bundle.dataset for bundle in collected],
        cfg.policy.length_reference_percentile,
    )
    return collected


def _score_member(
    policy,
    bundles: list[DatasetScores],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
) -> None:
    for bundle in bundles:
        dataloader = _make_dataloader(
            bundle.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        for batch in dataloader:
            device_batch = _move_to_device(batch, device)
            prediction = policy.predict_temporal_offset(device_batch)
            advantages = paper_advantage_from_expected_bin(
                prediction["expected_bin"],
                policy.config.num_bins,
            )
            for episode_index, frame_index, advantage, task in zip(
                batch["episode_index"],
                batch["frame_index"],
                advantages.detach().cpu(),
                batch["prompt"],
                strict=True,
            ):
                key = (int(episode_index.item()), int(frame_index.item()))
                if key not in bundle.minimum_advantage:
                    raise KeyError(f"Unexpected STEAM inference key: {key!r}.")
                bundle.minimum_advantage[key] = min(
                    bundle.minimum_advantage[key],
                    float(advantage.item()),
                )
                bundle.tasks[key] = str(task)


def _finalize_and_persist(
    bundles: list[DatasetScores],
    *,
    expert_positive_fraction: float,
    non_expert_positive_fraction: float,
) -> None:
    scores_by_source: dict[str, list[float]] = {
        "expert": [],
        "non_expert": [],
    }
    for bundle in bundles:
        source = bundle.config.steam_source
        values = list(bundle.minimum_advantage.values())
        if any(not np.isfinite(value) for value in values):
            raise RuntimeError(
                f"Not every non-terminal frame was scored for {bundle.dataset.root}."
            )
        scores_by_source[source].extend(values)

    thresholds = source_quantile_thresholds(
        scores_by_source,
        expert_positive_fraction=expert_positive_fraction,
        non_expert_positive_fraction=non_expert_positive_fraction,
    )
    logging.info("STEAM source-specific thresholds: %s", thresholds)

    for bundle in bundles:
        source = bundle.config.steam_source
        threshold = thresholds.get(source, float("inf"))
        processed_keys: list[tuple[int, int]] = []
        advantages: dict[tuple[int, int], float] = {}
        raw_advantages: dict[tuple[int, int], float] = {}
        advantage_sources: dict[tuple[int, int], str] = {}
        terminal_keys = {
            (episode_index, frame_index)
            for episode_index, frame_index, _ in bundle.dataset.terminal_records
        }

        for episode_index, frame_index, _row_index in bundle.dataset.frame_records:
            key = (episode_index, frame_index)
            processed_keys.append(key)
            if key in terminal_keys:
                base = bundle.dataset.base_dataset
                task_index = base.episode_to_task_index[episode_index]
                task = str(base.meta.tasks[task_index])
                bundle.tasks[key] = task
                raw_advantages[key] = 0.0
                advantages[key] = 0.0
                advantage_sources[key] = "steam_terminal_default"
                continue
            raw = bundle.minimum_advantage[key]
            raw_advantages[key] = raw
            advantages[key] = float(raw >= threshold)
            advantage_sources[key] = f"steam_{source}_quantile"

        report = persist_advantage_bundle(
            root=Path(bundle.dataset.root),
            processed_keys=processed_keys,
            advantages=advantages,
            raw_advantages=raw_advantages,
            advantage_sources=advantage_sources,
            key_to_task=bundle.tasks,
        )
        positive_count = int(sum(advantages.values()))
        logging.info(
            "Wrote STEAM advantages to %s: frames=%d, positives=%d, coverage=%s",
            bundle.dataset.root,
            len(processed_keys),
            positive_count,
            report["coverage"],
        )


def validate_checkpoint_count(checkpoints: list[Path]) -> None:
    if len(checkpoints) != 3:
        raise ValueError(
            "STEAM advantage generation requires exactly three independent "
            f"checkpoints; got {len(checkpoints)}."
        )
    canonical_paths = {checkpoint.resolve() for checkpoint in checkpoints}
    if len(canonical_paths) != 3:
        raise ValueError(
            "STEAM ensemble checkpoints must be three distinct paths; "
            f"got {checkpoints!r}."
        )


def main(args: argparse.Namespace) -> None:
    config_source = args.config_path or args.checkpoint[0]
    validate_checkpoint_count(args.checkpoint)

    cfg = TrainPipelineConfig.from_pretrained(config_source, local_files_only=True)
    if not isinstance(cfg.policy, SteamConfig):
        raise TypeError("compute_steam_advantages requires policy.type='steam'.")
    if args.dataset_mixture is not None:
        resolved_mixture = resolve_refs_to_tempfile(args.dataset_mixture)
        try:
            mixture = draccus.parse(
                config_class=DatasetMixtureConfig,
                config_path=resolved_mixture,
                args=[],
            )
        finally:
            resolved_mixture.unlink(missing_ok=True)
    else:
        mixture = cfg.dataset_mixture

    if cfg.seed is not None:
        set_seed(cfg.seed)
    device = auto_torch_device()
    batch_size = args.batch_size or cfg.dataloader_batch_size or cfg.batch_size
    if batch_size is None:
        raise ValueError("A labeling batch size must be configured.")
    num_workers = cfg.num_workers if args.num_workers is None else args.num_workers

    bundles = _collect_datasets(cfg, mixture)
    configured_core_signature = _core_signature(cfg.policy)
    member_signature = None
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    for member_index, checkpoint in enumerate(args.checkpoint):
        member_config = _load_member_config(checkpoint, cfg.policy)
        current_signature = _architecture_signature(member_config)
        if _core_signature(member_config) != configured_core_signature:
            raise ValueError(
                f"STEAM member {checkpoint} is incompatible with the labeling config."
            )
        if member_signature is not None and current_signature != member_signature:
            raise ValueError(
                f"STEAM member {checkpoint} is architecture-incompatible with the first member."
            )
        member_signature = current_signature
        member_config.device = str(device)
        policy_class = get_policy_class("steam")
        logging.info(
            "Loading STEAM member %d/%d: %s",
            member_index + 1,
            len(args.checkpoint),
            checkpoint,
        )
        policy = policy_class.from_pretrained(
            checkpoint,
            config=member_config,
            strict=True,
            local_files_only=True,
            backbone_local_files_only=True,
        )
        policy.to(device=device, dtype=dtype)
        policy.eval()
        _score_member(
            policy,
            bundles,
            device=device,
            batch_size=int(batch_size),
            num_workers=int(num_workers),
            prefetch_factor=cfg.prefetch_factor,
        )
        del policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _finalize_and_persist(
        bundles,
        expert_positive_fraction=args.expert_positive_fraction,
        non_expert_positive_fraction=args.non_expert_positive_fraction,
    )


def cli() -> None:
    init_logging()
    main(_parse_args())


if __name__ == "__main__":
    cli()
