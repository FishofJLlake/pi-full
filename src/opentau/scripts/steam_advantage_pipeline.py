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

"""STEAM advantage scoring, ensemble aggregation, and artifact persistence."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import draccus
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.configs.refs import resolve_refs_to_tempfile
from opentau.configs.train import TRAIN_CONFIG_NAME, TrainPipelineConfig
from opentau.constants import HF_OPENTAU_HOME
from opentau.datasets.advantage_metadata import persist_advantage_bundle
from opentau.datasets.factory import make_dataset
from opentau.datasets.steam_pair_dataset import SteamPairDataset, set_global_length_reference
from opentau.policies.steam.binning import expected_rlinf_signed_score
from opentau.policies.steam.configuration_steam import SteamConfig
from opentau.policies.steam.ensemble_modeling_steam import (
    load_steam_inference_checkpoint,
    split_member_predictions,
)
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import auto_torch_device

ScoreMode = Literal["paper_baseline", "rlinf_signed"]
LabelMode = Literal["quantile", "threshold"]
ThresholdComparison = Literal["strict", "inclusive"]
_TAG_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass
class DatasetScores:
    config: DatasetConfig
    dataset: SteamPairDataset
    minimum_advantage: dict[tuple[int, int], float]
    tasks: dict[tuple[int, int], str]
    paper_member_scores: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    rlinf_member_scores: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    expected_stride_member_scores: dict[tuple[int, int], list[float]] = field(default_factory=dict)
    member_entropies: dict[tuple[int, int], list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for key in self.minimum_advantage:
            self.paper_member_scores.setdefault(key, [])
            self.rlinf_member_scores.setdefault(key, [])
            self.expected_stride_member_scores.setdefault(key, [])
            self.member_entropies.setdefault(key, [])


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device
    initialized_here: bool

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score datasets with one or more STEAM checkpoints (single or merged ensembles), "
            "take the pointwise minimum, and write JSON + RLinf-compatible Parquet artifacts."
        )
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        type=Path,
        required=True,
        help="Single-member or merged STEAM checkpoint. Repeat to concatenate members.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="Train config/checkpoint used for dataset settings (defaults to first checkpoint).",
    )
    parser.add_argument("--dataset-mixture", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--tag", default="steam")
    parser.add_argument(
        "--score-mode",
        choices=("paper_baseline", "rlinf_signed"),
        default="rlinf_signed",
        help="Score used for bool labels and legacy JSON. Both scores are always persisted.",
    )
    parser.add_argument("--label-mode", choices=("quantile", "threshold"), default="quantile")
    parser.add_argument("--positive-threshold", type=float, default=0.0)
    parser.add_argument(
        "--threshold-comparison",
        choices=("strict", "inclusive"),
        default="strict",
        help="Use RLinf's strict '>' (default) or legacy-compatible inclusive '>=' labels.",
    )
    parser.add_argument("--expert-positive-fraction", type=float, default=0.8)
    parser.add_argument("--non-expert-positive-fraction", type=float, default=0.3)
    return parser.parse_args()


def setup_distributed() -> DistributedContext:
    """Initialize torchrun data parallelism, or return a single-process context."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    initialized_here = False
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=backend, init_method="env://")
            initialized_here = True
        return DistributedContext(
            rank=dist.get_rank(),
            world_size=dist.get_world_size(),
            device=device,
            initialized_here=initialized_here,
        )
    return DistributedContext(rank=0, world_size=1, device=auto_torch_device(), initialized_here=False)


def cleanup_distributed(context: DistributedContext) -> None:
    if context.initialized_here and dist.is_initialized():
        dist.destroy_process_group()


def shard_bounds(total: int, rank: int, world_size: int) -> tuple[int, int]:
    """Return RLinf-style contiguous, balanced shard bounds."""
    if world_size < 1 or not 0 <= rank < world_size:
        raise ValueError(f"Invalid distributed shard: rank={rank}, world_size={world_size}.")
    quotient, remainder = divmod(total, world_size)
    start = rank * quotient + min(rank, remainder)
    end = start + quotient + int(rank < remainder)
    return start, end


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


def rlinf_advantage_from_probabilities(probabilities: torch.Tensor, num_bins: int) -> torch.Tensor:
    """Apply RLinf's exact normalized signed-bin expectation."""
    return expected_rlinf_signed_score(probabilities.to(torch.float32), num_bins)


def source_quantile_thresholds(
    scores_by_source: dict[str, list[float]],
    *,
    expert_positive_fraction: float,
    non_expert_positive_fraction: float,
) -> dict[str, float]:
    """Derive independent thresholds for expert and non-expert pools."""
    fractions = {"expert": expert_positive_fraction, "non_expert": non_expert_positive_fraction}
    thresholds: dict[str, float] = {}
    for source, scores in scores_by_source.items():
        fraction = fractions[source]
        if not 0 < fraction <= 1:
            raise ValueError(f"{source} positive fraction must be in (0, 1], got {fraction}.")
        if scores:
            thresholds[source] = float(
                np.percentile(np.asarray(scores, dtype=np.float64), 100.0 - fraction * 100.0)
            )
    return thresholds


def apply_threshold(score: float, threshold: float, comparison: ThresholdComparison) -> bool:
    if comparison == "strict":
        return score > threshold
    if comparison == "inclusive":
        return score >= threshold
    raise ValueError(f"Unsupported threshold comparison: {comparison!r}.")


def validate_tag(tag: str) -> str:
    if not _TAG_PATTERN.fullmatch(tag):
        raise ValueError(
            "tag must start with an alphanumeric character and contain only "
            f"letters, numbers, '.', '_' or '-'; got {tag!r}."
        )
    return tag


def normalized_dataset_root(dataset_config: DatasetConfig) -> Path:
    configured = (
        Path(dataset_config.root).expanduser()
        if dataset_config.root is not None
        else HF_OPENTAU_HOME / str(dataset_config.repo_id)
    )
    return configured.resolve(strict=False)


def validate_unique_dataset_roots(mixture: DatasetMixtureConfig) -> None:
    seen: dict[str, DatasetConfig] = {}
    for dataset_config in mixture.datasets:
        root = normalized_dataset_root(dataset_config)
        canonical = os.path.normcase(str(root))
        if canonical in seen:
            previous = seen[canonical]
            raise ValueError(
                "STEAM advantage generation refuses duplicate dataset roots because they would "
                f"overwrite the same metadata: {root} (repo_ids={previous.repo_id!r}, "
                f"{dataset_config.repo_id!r})."
            )
        seen[canonical] = dataset_config


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
        tuple(config.image_resolution),
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
        tuple(config.image_resolution),
    )


def _load_member_config(checkpoint: Path, base: SteamConfig) -> SteamConfig:
    train_config_path = checkpoint / TRAIN_CONFIG_NAME
    if train_config_path.is_file():
        member_train_config = TrainPipelineConfig.from_pretrained(checkpoint, local_files_only=True)
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
    rank: int = 0,
    world_size: int = 1,
) -> DataLoader:
    start, end = shard_bounds(len(dataset), rank, world_size)
    shard = dataset if world_size == 1 else Subset(dataset, range(start, end))
    kwargs: dict[str, Any] = {
        "dataset": shard,
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs)


def _collect_datasets(cfg: TrainPipelineConfig, mixture: DatasetMixtureConfig) -> list[DatasetScores]:
    validate_unique_dataset_roots(mixture)
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
                "Every STEAM labeling dataset must set steam_source to 'expert' or 'non_expert'."
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
        terminal_keys = {
            (episode_index, frame_index) for episode_index, frame_index, _ in result.terminal_records
        }
        nonterminal_keys = {
            (episode_index, frame_index) for episode_index, frame_index, _ in result.frame_records
        } - terminal_keys
        collected.append(
            DatasetScores(
                config=dataset_config,
                dataset=result,
                minimum_advantage={key: float("inf") for key in nonterminal_keys},
                tasks={},
            )
        )
    if cfg.policy.length_scale_enabled:
        set_global_length_reference(
            [bundle.dataset for bundle in collected],
            cfg.policy.length_reference_percentile,
        )
    return collected


def _prediction_member_tensors(prediction: dict[str, torch.Tensor], num_bins: int) -> dict[str, torch.Tensor]:
    # Minimal fake policies used by unit tests may expose only expected_bin.
    if "probabilities" not in prediction:
        expected_bins = prediction["expected_bin"].to(torch.float32)[None, :]
        paper = paper_advantage_from_expected_bin(expected_bins, num_bins)
        zeros = torch.zeros_like(paper)
        return {"paper": paper, "rlinf": paper, "expected_stride": paper, "entropy": zeros}

    members = split_member_predictions(prediction)
    probabilities = members["probabilities"].to(torch.float32)
    paper = paper_advantage_from_expected_bin(members["expected_bins"], num_bins)
    rlinf = members.get("rlinf_signed_scores")
    if rlinf is None:
        rlinf = rlinf_advantage_from_probabilities(probabilities, num_bins)
    entropy = -(probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log() * probabilities).sum(
        dim=-1
    )
    return {
        "paper": paper,
        "rlinf": rlinf.to(torch.float32),
        "expected_stride": members["expected_stride_scores"].to(torch.float32),
        "entropy": entropy,
    }


def _apply_score_payload(
    bundles: list[DatasetScores],
    payload: list[dict[tuple[int, int], dict[str, Any]]],
) -> None:
    for bundle, bundle_payload in zip(bundles, payload, strict=True):
        for key, values in bundle_payload.items():
            if key not in bundle.minimum_advantage:
                raise KeyError(f"Unexpected STEAM inference key: {key!r}.")
            paper_values = [float(value) for value in values["paper"]]
            bundle.paper_member_scores[key].extend(paper_values)
            bundle.rlinf_member_scores[key].extend(float(value) for value in values["rlinf"])
            bundle.expected_stride_member_scores[key].extend(
                float(value) for value in values["expected_stride"]
            )
            bundle.member_entropies[key].extend(float(value) for value in values["entropy"])
            bundle.minimum_advantage[key] = min(bundle.minimum_advantage[key], min(paper_values))
            bundle.tasks[key] = str(values["task"])


def _score_member(
    policy,
    bundles: list[DatasetScores],
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int | None,
    rank: int = 0,
    world_size: int = 1,
) -> list[dict[tuple[int, int], dict[str, Any]]]:
    """Score one checkpoint, which may itself contain one or more members."""
    payload: list[dict[tuple[int, int], dict[str, Any]]] = [{} for _ in bundles]
    for bundle_index, bundle in enumerate(bundles):
        dataloader = _make_dataloader(
            bundle.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            rank=rank,
            world_size=world_size,
        )
        for batch in dataloader:
            device_batch = _move_to_device(batch, device)
            prediction = policy.predict_temporal_offset(device_batch)
            member_tensors = _prediction_member_tensors(prediction, policy.config.num_bins)
            paper = member_tensors["paper"].detach().cpu()
            rlinf = member_tensors["rlinf"].detach().cpu()
            expected_stride = member_tensors["expected_stride"].detach().cpu()
            entropy = member_tensors["entropy"].detach().cpu()
            for batch_index, (episode_index, frame_index, task) in enumerate(
                zip(batch["episode_index"], batch["frame_index"], batch["prompt"], strict=True)
            ):
                key = (int(episode_index.item()), int(frame_index.item()))
                if key in payload[bundle_index]:
                    raise ValueError(f"Duplicate STEAM inference key within one shard: {key!r}.")
                payload[bundle_index][key] = {
                    "paper": paper[:, batch_index].tolist(),
                    "rlinf": rlinf[:, batch_index].tolist(),
                    "expected_stride": expected_stride[:, batch_index].tolist(),
                    "entropy": entropy[:, batch_index].tolist(),
                    "task": str(task),
                }
    _apply_score_payload(bundles, payload)
    return payload


def _gather_payload_to_rank0(
    payload: list[dict[tuple[int, int], dict[str, Any]]],
    context: DistributedContext,
) -> list[list[dict[tuple[int, int], dict[str, Any]]]] | None:
    if context.world_size == 1:
        return [payload]
    gathered = [None] * context.world_size if context.is_main_process else None
    dist.gather_object(payload, gathered, dst=0)
    return gathered


def _aggregate(bundle: DatasetScores, key: tuple[int, int], mode: ScoreMode) -> float:
    values = bundle.paper_member_scores[key] if mode == "paper_baseline" else bundle.rlinf_member_scores[key]
    return float(min(values))


def _validate_member_coverage(bundles: list[DatasetScores], member_count: int) -> None:
    for bundle in bundles:
        for key in bundle.minimum_advantage:
            counts = {
                len(bundle.paper_member_scores[key]),
                len(bundle.rlinf_member_scores[key]),
                len(bundle.expected_stride_member_scores[key]),
                len(bundle.member_entropies[key]),
            }
            if counts != {member_count}:
                raise RuntimeError(
                    f"Incomplete STEAM member coverage for {bundle.dataset.root}, key={key}: "
                    f"counts={sorted(counts)}, expected={member_count}."
                )


def _score_pools(bundles: list[DatasetScores], mode: ScoreMode) -> dict[str, list[float]]:
    pools: dict[str, list[float]] = {"expert": [], "non_expert": []}
    for bundle in bundles:
        pools[bundle.config.steam_source].extend(
            _aggregate(bundle, key, mode) for key in bundle.minimum_advantage
        )
    return pools


def _thresholds_for_mode(
    bundles: list[DatasetScores],
    *,
    score_mode: ScoreMode,
    label_mode: LabelMode,
    positive_threshold: float,
    expert_positive_fraction: float,
    non_expert_positive_fraction: float,
) -> dict[str, float]:
    if label_mode == "threshold":
        return {"expert": float(positive_threshold), "non_expert": float(positive_threshold)}
    return source_quantile_thresholds(
        _score_pools(bundles, score_mode),
        expert_positive_fraction=expert_positive_fraction,
        non_expert_positive_fraction=non_expert_positive_fraction,
    )


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return None
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def _comparison_stats(
    bundle: DatasetScores,
    *,
    paper_threshold: float,
    rlinf_threshold: float,
    comparison: ThresholdComparison,
) -> dict[str, Any]:
    keys = sorted(bundle.minimum_advantage)
    paper = [_aggregate(bundle, key, "paper_baseline") for key in keys]
    rlinf = [_aggregate(bundle, key, "rlinf_signed") for key in keys]
    paper_labels = [apply_threshold(value, paper_threshold, comparison) for value in paper]
    rlinf_labels = [apply_threshold(value, rlinf_threshold, comparison) for value in rlinf]
    disagreements = sum(left != right for left, right in zip(paper_labels, rlinf_labels, strict=True))
    count = len(keys)
    return {
        "nonterminal_count": count,
        "spearman": _spearman(paper, rlinf),
        "label_disagreement_count": disagreements,
        "label_disagreement_rate": disagreements / count if count else 0.0,
        "paper_positive_rate": sum(paper_labels) / count if count else 0.0,
        "rlinf_positive_rate": sum(rlinf_labels) / count if count else 0.0,
    }


def _atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_parquet(path: Path, dataframe: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        dataframe.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _task_for_terminal(bundle: DatasetScores, episode_index: int) -> str:
    base = bundle.dataset.base_dataset
    task_index = base.episode_to_task_index[episode_index]
    return str(base.meta.tasks[task_index])


def _row_for_key(
    bundle: DatasetScores,
    key: tuple[int, int],
    *,
    is_terminal: bool,
    selected_mode: ScoreMode,
    advantage: bool,
    threshold: float,
    tag: str,
    member_count: int,
) -> dict[str, Any]:
    if is_terminal:
        paper_members = [0.0] * member_count
        rlinf_members = [0.0] * member_count
        stride_members = [0.0] * member_count
        entropies = [0.0] * member_count
    else:
        paper_members = bundle.paper_member_scores[key]
        rlinf_members = bundle.rlinf_member_scores[key]
        stride_members = bundle.expected_stride_member_scores[key]
        entropies = bundle.member_entropies[key]
    selected_members = paper_members if selected_mode == "paper_baseline" else rlinf_members
    worst_index = int(np.argmin(selected_members))
    selected_min = float(selected_members[worst_index])
    fps = float(bundle.dataset.meta.fps)
    height, width = bundle.dataset.config.image_resolution
    return {
        "episode_index": int(key[0]),
        "frame_index": int(key[1]),
        "advantage": bool(advantage),
        "advantage_continuous": selected_min,
        "ensemble_signed_score": selected_min,
        "paper_baseline_score": float(min(paper_members)),
        "rlinf_signed_score": float(min(rlinf_members)),
        "p_progress_mean": float(np.mean(selected_members)),
        "p_progress_min": selected_min,
        "p_progress_variance": float(np.var(selected_members)),
        "member_values": [float(value) for value in selected_members],
        "member_paper_baseline_scores": [float(value) for value in paper_members],
        "member_rlinf_signed_scores": [float(value) for value in rlinf_members],
        "expected_stride_normalized": float(stride_members[worst_index]),
        "entropy_aggregated": float(entropies[worst_index]),
        "entropy_member_mean": float(np.mean(entropies)),
        "entropy_member_variance": float(np.var(entropies)),
        "is_terminal": bool(is_terminal),
        "steam_source": str(bundle.config.steam_source),
        "tag": tag,
        "score_mode": selected_mode,
        "threshold": float(threshold),
        "ensemble_size": member_count,
        "num_bins": bundle.dataset.config.num_bins,
        "max_temporal_offset": bundle.dataset.config.max_temporal_offset,
        "fps": fps,
        "resolution_height": int(height),
        "resolution_width": int(width),
        "center_padding": True,
        "length_scale_enabled": bundle.dataset.config.length_scale_enabled,
        "length_reference": float(bundle.dataset.length_reference),
        "paper_baseline_reference_bin": bundle.dataset.config.num_bins - 1,
    }


def _finalize_and_persist(
    bundles: list[DatasetScores],
    *,
    member_count: int,
    tag: str = "steam",
    score_mode: ScoreMode = "rlinf_signed",
    label_mode: LabelMode = "quantile",
    positive_threshold: float = 0.0,
    threshold_comparison: ThresholdComparison = "strict",
    expert_positive_fraction: float,
    non_expert_positive_fraction: float,
) -> None:
    tag = validate_tag(tag)
    if member_count < 1:
        raise ValueError(f"member_count must be >= 1, got {member_count}.")
    if score_mode not in ("paper_baseline", "rlinf_signed"):
        raise ValueError(f"Unsupported STEAM score mode: {score_mode!r}.")
    if label_mode not in ("quantile", "threshold"):
        raise ValueError(f"Unsupported STEAM label mode: {label_mode!r}.")
    if threshold_comparison not in ("strict", "inclusive"):
        raise ValueError(f"Unsupported threshold comparison: {threshold_comparison!r}.")
    if label_mode == "threshold" and score_mode == "rlinf_signed" and not -1 <= positive_threshold <= 1:
        raise ValueError(
            f"RLinf signed-score positive_threshold must be in [-1, 1], got {positive_threshold}."
        )

    _validate_member_coverage(bundles, member_count)
    paper_thresholds = _thresholds_for_mode(
        bundles,
        score_mode="paper_baseline",
        label_mode=label_mode,
        positive_threshold=positive_threshold,
        expert_positive_fraction=expert_positive_fraction,
        non_expert_positive_fraction=non_expert_positive_fraction,
    )
    rlinf_thresholds = _thresholds_for_mode(
        bundles,
        score_mode="rlinf_signed",
        label_mode=label_mode,
        positive_threshold=positive_threshold,
        expert_positive_fraction=expert_positive_fraction,
        non_expert_positive_fraction=non_expert_positive_fraction,
    )
    selected_thresholds = paper_thresholds if score_mode == "paper_baseline" else rlinf_thresholds
    logging.info(
        "STEAM thresholds: selected_mode=%s selected=%s paper=%s rlinf=%s",
        score_mode,
        selected_thresholds,
        paper_thresholds,
        rlinf_thresholds,
    )
    source_totals = {"expert": [0, 0], "non_expert": [0, 0]}

    for bundle in bundles:
        source = bundle.config.steam_source
        threshold = selected_thresholds.get(source, float("inf"))
        terminal_keys = {
            (episode_index, frame_index) for episode_index, frame_index, _ in bundle.dataset.terminal_records
        }
        processed_keys: list[tuple[int, int]] = []
        advantages: dict[tuple[int, int], float] = {}
        raw_advantages: dict[tuple[int, int], float] = {}
        advantage_sources: dict[tuple[int, int], str] = {}
        rows: list[dict[str, Any]] = []
        diagnostics_member_scores: dict[str, list[float]] = {}

        for episode_index, frame_index, _row_index in bundle.dataset.frame_records:
            key = (episode_index, frame_index)
            processed_keys.append(key)
            is_terminal = key in terminal_keys
            if is_terminal:
                bundle.tasks[key] = _task_for_terminal(bundle, episode_index)
                raw = 0.0
                positive = False
                advantage_sources[key] = "steam_terminal_default"
            else:
                raw = _aggregate(bundle, key, score_mode)
                positive = apply_threshold(raw, threshold, threshold_comparison)
                advantage_sources[key] = f"steam_{source}_{label_mode}_{score_mode}_{threshold_comparison}"
            raw_advantages[key] = raw
            advantages[key] = float(positive)
            row = _row_for_key(
                bundle,
                key,
                is_terminal=is_terminal,
                selected_mode=score_mode,
                advantage=positive,
                threshold=threshold,
                tag=tag,
                member_count=member_count,
            )
            rows.append(row)
            diagnostics_member_scores[f"{episode_index},{frame_index}"] = row["member_values"]

        report = persist_advantage_bundle(
            root=Path(bundle.dataset.root),
            processed_keys=processed_keys,
            advantages=advantages,
            raw_advantages=raw_advantages,
            advantage_sources=advantage_sources,
            key_to_task=bundle.tasks,
        )
        comparison = _comparison_stats(
            bundle,
            paper_threshold=paper_thresholds.get(source, float("inf")),
            rlinf_threshold=rlinf_thresholds.get(source, float("inf")),
            comparison=threshold_comparison,
        )
        parquet_path = Path(bundle.dataset.root) / "meta" / f"advantages_{tag}.parquet"
        diagnostics_path = Path(bundle.dataset.root) / "meta" / "steam_advantage_diagnostics.json"
        _atomic_write_parquet(parquet_path, pd.DataFrame(rows))
        _atomic_write_json(
            diagnostics_path,
            {
                "schema_version": 1,
                "key_format": "episode_index,frame_index",
                "aggregation": "minimum",
                "ensemble_size": member_count,
                "tag": tag,
                "score_mode": score_mode,
                "label_mode": label_mode,
                "threshold_comparison": threshold_comparison,
                "threshold": threshold,
                "terminal_policy": "excluded_from_threshold_and_forced_zero",
                "member_scores": diagnostics_member_scores,
                "comparison": comparison,
            },
        )
        positive_count = int(sum(advantages.values()))
        source_totals[source][0] += positive_count
        source_totals[source][1] += len(processed_keys)
        logging.info(
            "Wrote STEAM advantages to %s: frames=%d positives=%d rate=%.4f "
            "coverage=%s parquet=%s comparison=%s",
            bundle.dataset.root,
            len(processed_keys),
            positive_count,
            positive_count / len(processed_keys) if processed_keys else 0.0,
            report["coverage"],
            parquet_path,
            comparison,
        )

    for source, (positive_count, frame_count) in source_totals.items():
        if frame_count:
            logging.info(
                "STEAM source-level labels: source=%s positives=%d frames=%d positive_rate=%.4f",
                source,
                positive_count,
                frame_count,
                positive_count / frame_count,
            )


def validate_checkpoint_count(checkpoints: list[Path]) -> None:
    """Accept arbitrary N>=1 checkpoint containers; each may hold multiple members."""
    if not checkpoints:
        raise ValueError("STEAM advantage generation requires at least one checkpoint.")
    canonical_paths = [checkpoint.expanduser().resolve(strict=False) for checkpoint in checkpoints]
    if len(set(canonical_paths)) != len(canonical_paths):
        raise ValueError(f"STEAM checkpoint paths must be distinct; got {checkpoints!r}.")


def _parse_mixture(path: Path) -> DatasetMixtureConfig:
    resolved_mixture = resolve_refs_to_tempfile(path)
    try:
        return draccus.parse(config_class=DatasetMixtureConfig, config_path=resolved_mixture, args=[])
    finally:
        resolved_mixture.unlink(missing_ok=True)


def main(args: argparse.Namespace) -> None:
    """Run RLinf-compatible worst-of-N scoring over one or more checkpoint containers."""
    checkpoints = args.checkpoint
    validate_checkpoint_count(checkpoints)
    config_source = getattr(args, "config_path", None) or checkpoints[0]
    cfg = TrainPipelineConfig.from_pretrained(config_source, local_files_only=True)
    if not isinstance(cfg.policy, SteamConfig):
        raise TypeError("compute_steam_advantages requires policy.type='steam'.")
    if tuple(cfg.policy.image_resolution) != tuple(cfg.resolution):
        raise ValueError(
            f"policy.image_resolution={tuple(cfg.policy.image_resolution)} != "
            f"resolution={tuple(cfg.resolution)}. Advantage scoring refuses to silently "
            "upsample dataset frames before the STEAM vision tower."
        )
    mixture_path = getattr(args, "dataset_mixture", None)
    mixture = _parse_mixture(mixture_path) if mixture_path is not None else cfg.dataset_mixture

    if cfg.seed is not None:
        set_seed(cfg.seed)
    context = setup_distributed()
    batch_size = getattr(args, "batch_size", None) or cfg.dataloader_batch_size or cfg.batch_size
    if batch_size is None:
        raise ValueError("A labeling batch size must be configured.")
    requested_workers = getattr(args, "num_workers", None)
    num_workers = cfg.num_workers if requested_workers is None else requested_workers
    tag = getattr(args, "tag", "steam")
    score_mode = getattr(args, "score_mode", "rlinf_signed")
    label_mode = getattr(args, "label_mode", "quantile")
    positive_threshold = float(getattr(args, "positive_threshold", 0.0))
    threshold_comparison = getattr(args, "threshold_comparison", "strict")
    expert_positive_fraction = float(getattr(args, "expert_positive_fraction", 0.8))
    non_expert_positive_fraction = float(getattr(args, "non_expert_positive_fraction", 0.3))

    try:
        bundles = _collect_datasets(cfg, mixture)
        configured_core_signature = _core_signature(cfg.policy)
        member_signature = None
        dtype = torch.bfloat16 if context.device.type == "cuda" else torch.float32
        total_member_count = 0

        for checkpoint_index, checkpoint in enumerate(checkpoints):
            member_config = _load_member_config(checkpoint, cfg.policy)
            current_signature = _architecture_signature(member_config)
            if _core_signature(member_config) != configured_core_signature:
                raise ValueError(f"STEAM checkpoint {checkpoint} is incompatible with the labeling config.")
            if member_signature is not None and current_signature != member_signature:
                raise ValueError(
                    f"STEAM checkpoint {checkpoint} is architecture-incompatible with the first checkpoint."
                )
            member_signature = current_signature
            member_config.device = str(context.device)
            if context.is_main_process:
                logging.info(
                    "Loading STEAM checkpoint %d/%d (%d member(s)): %s",
                    checkpoint_index + 1,
                    len(checkpoints),
                    member_config.ensemble_size,
                    checkpoint,
                )
            policy = load_steam_inference_checkpoint(
                checkpoint,
                member_config,
                local_files_only=True,
            )
            policy.to(device=context.device, dtype=dtype)
            policy.eval()
            local_payload = _score_member(
                policy,
                bundles,
                device=context.device,
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                prefetch_factor=cfg.prefetch_factor,
                rank=context.rank,
                world_size=context.world_size,
            )
            gathered = _gather_payload_to_rank0(local_payload, context)
            if context.is_main_process and gathered is not None:
                for remote_payload in gathered[1:]:
                    _apply_score_payload(bundles, remote_payload)
            total_member_count += member_config.ensemble_size
            del policy
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        persistence_error: BaseException | None = None
        persistence_error_text: str | None = None
        if context.is_main_process:
            try:
                _finalize_and_persist(
                    bundles,
                    member_count=total_member_count,
                    tag=tag,
                    score_mode=score_mode,
                    label_mode=label_mode,
                    positive_threshold=positive_threshold,
                    threshold_comparison=threshold_comparison,
                    expert_positive_fraction=expert_positive_fraction,
                    non_expert_positive_fraction=non_expert_positive_fraction,
                )
            except BaseException as exc:
                persistence_error = exc
                persistence_error_text = f"{type(exc).__name__}: {exc}"
        if context.world_size > 1:
            error_payload = [persistence_error_text]
            dist.broadcast_object_list(error_payload, src=0)
            persistence_error_text = error_payload[0]
        if persistence_error is not None:
            raise persistence_error
        if persistence_error_text is not None:
            raise RuntimeError(f"Rank 0 failed to persist STEAM advantages: {persistence_error_text}")
    finally:
        cleanup_distributed(context)
