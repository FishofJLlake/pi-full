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

"""Evaluate a Value checkpoint on an explicit held-out dataset mixture.

Usage:
    python -m opentau.scripts.evaluate_value \
        --train_config configs/train/value_config.json \
        --dataset_mixture configs/eval/value_held_out.json \
        --output_dir outputs/value_eval \
        --batch_size 16
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import draccus
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from opentau.configs.default import DatasetMixtureConfig
from opentau.configs.train import TrainPipelineConfig
from opentau.constants import HF_OPENTAU_HOME
from opentau.datasets.factory import make_dataset
from opentau.policies.factory import get_policy_class
from opentau.policies.value.metrics import ValueMetricsAccumulator
from opentau.utils.random_utils import set_seed
from opentau.utils.utils import init_logging

METRIC_VERSION = "value-quality-v1"
METRIC_DEFINITION = (
    "MAE compares expected values with continuous return targets; categorical NLL is evaluated at "
    "the target return bin; Spearman measures value ranking; calibration error is the mean absolute "
    "difference between predicted and empirical categorical CDFs."
)


@dataclass(frozen=True)
class EvaluateValueConfig:
    """Inputs for a read-only held-out Value evaluation."""

    train_config: Path
    dataset_mixture: Path
    output_dir: Path
    batch_size: int
    num_calibration_thresholds: int = 20


def validate_held_out_mixture(mixture: DatasetMixtureConfig) -> None:
    """Require datasets whose configured episodes are enforceable and non-empty."""
    if not mixture.datasets:
        raise ValueError("Held-out Value evaluation requires at least one dataset.")

    for index, dataset in enumerate(mixture.datasets):
        dataset_name = dataset.repo_id or dataset.vqa or f"dataset {index}"
        if dataset.vqa is not None:
            raise ValueError(
                f"VQA dataset '{dataset_name}' cannot be used for held-out Value evaluation because "
                "its dataset factory does not apply explicit episode selections."
            )
        if not dataset.episodes:
            raise ValueError(
                f"Dataset '{dataset_name}' must specify explicit held-out episodes; "
                "refusing to evaluate the training dataset in full."
            )


def validate_output_dir_does_not_overlap_dataset_roots(
    output_dir: Path, mixture: DatasetMixtureConfig
) -> None:
    """Reject report destinations that could mutate a held-out dataset root."""
    output_path = Path(output_dir).resolve()
    for index, dataset in enumerate(mixture.datasets):
        dataset_root = _resolved_dataset_root(dataset)
        if _paths_overlap(output_path, dataset_root):
            dataset_name = dataset.repo_id or dataset.vqa or f"dataset {index}"
            raise ValueError(
                f"Evaluation output directory {output_path} must not overlap dataset root "
                f"{dataset_root} for {dataset_name!r}."
            )


def validate_output_dir_does_not_overlap_training(
    output_dir: Path, train_config: TrainPipelineConfig
) -> None:
    """Keep evaluation reports outside training outputs and local checkpoints."""
    output_path = Path(output_dir).resolve()
    candidates: list[tuple[str, Path]] = []
    training_output = getattr(train_config, "output_dir", None)
    if training_output is not None:
        candidates.append(("training output", Path(training_output).resolve()))
    checkpoint = getattr(train_config.policy, "pretrained_path", None)
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint)
        if checkpoint_path.is_absolute() or checkpoint_path.exists():
            candidates.append(("checkpoint", checkpoint_path.resolve()))

    for label, path in candidates:
        if _paths_overlap(output_path, path):
            raise ValueError(
                f"Evaluation output directory {output_path} must not overlap the {label} path {path}."
            )


def _resolved_dataset_root(dataset: Any) -> Path:
    root = Path(dataset.root) if dataset.root is not None else HF_OPENTAU_HOME / dataset.repo_id
    return root.resolve()


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first.relative_to(second)
    except ValueError:
        try:
            second.relative_to(first)
        except ValueError:
            return False
    return True


def render_value_metrics_markdown(report: Mapping[str, object]) -> str:
    """Render a deterministic human-readable view of a JSON-safe metrics report."""
    evaluation_config = _mapping(report["evaluation_config"])
    lines = [
        "# Held-out Value evaluation",
        "",
        f"- Checkpoint: `{report['checkpoint_path']}`",
        f"- Generated at: {report['generated_at']}",
        f"- Metric version: {report['metric_version']}",
        f"- Metric definition: {report['metric_definition']}",
        f"- Train config: `{evaluation_config['train_config']}`",
        f"- Dataset mixture: `{evaluation_config['dataset_mixture']}`",
        f"- Batch size: {evaluation_config['batch_size']}",
        f"- Seed: {evaluation_config['seed']}",
        f"- Calibration thresholds: {evaluation_config['num_calibration_thresholds']}",
        "",
        "## Dataset roots",
        "",
        "| Identity | Dataset | Root | Episodes |",
        "| --- | --- | --- | --- |",
    ]
    dataset_roots = report["dataset_roots"]
    if not isinstance(dataset_roots, Sequence) or isinstance(dataset_roots, (str, bytes)):
        raise TypeError("dataset_roots must be a sequence")
    for item in sorted(dataset_roots, key=lambda entry: str(_mapping(entry)["dataset"])):
        dataset = _mapping(item)
        episodes = ", ".join(str(episode) for episode in dataset["episodes"])
        lines.append(f"| {dataset['identity']} | {dataset['dataset']} | {dataset['root']} | {episodes} |")

    lines.extend(["", "## Overall", "", "| Metric | Value |", "| --- | --- |"])
    overall = _mapping(report["overall"])
    for metric_name in _metric_names(overall):
        lines.append(f"| {metric_name} | {_format_metric_value(overall[metric_name])} |")
    lines.extend(
        [
            "",
            "### Target and prediction summaries",
            "",
            "| Series | Mean | Std | Min | Max |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    _append_summary_row(lines, "Target", _mapping(overall["target_summary"]))
    _append_summary_row(lines, "Prediction", _mapping(overall["prediction_summary"]))

    lines.extend(
        [
            "",
            "## By dataset",
            "",
            "| Dataset | Count | MAE | NLL | Spearman | Calibration error | Target mean | Target std | Target min | Target max | Prediction mean | Prediction std | Prediction min | Prediction max |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    _append_group_rows(lines, _mapping(report["by_dataset"]))

    lines.extend(
        [
            "",
            "## By task",
            "",
            "| Task | Count | MAE | NLL | Spearman | Calibration error | Target mean | Target std | Target min | Target max | Prediction mean | Prediction std | Prediction min | Prediction max |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    _append_group_rows(lines, _mapping(report["by_task"]))
    return "\n".join(lines) + "\n"


def main(config: EvaluateValueConfig) -> None:
    """Evaluate ``config.train_config``'s checkpoint over the held-out episodes only."""
    if config.batch_size < 1:
        raise ValueError("batch_size must be positive")

    train_config = TrainPipelineConfig.from_pretrained(config.train_config, local_files_only=True)
    if train_config.policy.type != "value":
        raise ValueError(
            "Held-out Value evaluation requires a Value policy "
            f"(train_config.policy.type == 'value'); got {train_config.policy.type!r}."
        )
    mixture = draccus.parse(
        config_class=DatasetMixtureConfig,
        config_path=config.dataset_mixture,
        args=[],
    )
    validate_held_out_mixture(mixture)
    validate_output_dir_does_not_overlap_dataset_roots(config.output_dir, mixture)
    validate_output_dir_does_not_overlap_training(config.output_dir, train_config)
    _configure_read_only_evaluation(train_config, mixture)

    if train_config.seed is not None:
        set_seed(train_config.seed)

    checkpoint_path = train_config.policy.pretrained_path
    if checkpoint_path is None:
        raise ValueError("train_config.policy.pretrained_path must name the Value checkpoint")

    accelerator = Accelerator()
    device = accelerator.device
    logging.info("Loading Value checkpoint from %s", checkpoint_path)
    policy_class = get_policy_class(train_config.policy.type)
    policy = policy_class.from_pretrained(
        checkpoint_path,
        config=train_config.policy,
        local_files_only=True,
        backbone_local_files_only=True,
    )
    policy.to(device=device, dtype=torch.bfloat16)
    policy.eval()

    accumulator = ValueMetricsAccumulator(config.num_calibration_thresholds)
    with torch.inference_mode():
        for dataset_index, dataset_config in enumerate(mixture.datasets):
            dataset = make_dataset(
                dataset_config,
                train_config,
                return_advantage_input=True,
                local_files_only=True,
            )
            if isinstance(dataset, tuple):
                raise RuntimeError("held-out evaluation must not create a training/validation split")
            task_source_names = _task_source_names(dataset_index, dataset)
            dataloader = accelerator.prepare(_make_dataloader(dataset, train_config, config.batch_size))
            for batch in dataloader:
                source_indices = _task_source_indices(dataset_index, dataset, batch)
                _move_tensor_values_to_device(batch, device)
                prediction = policy.predict_value_distribution(batch)
                local_batch_size = prediction["value"].shape[0]
                gathered = accelerator.gather_for_metrics(
                    {
                        "logits": prediction["logits"].to(torch.float32),
                        "predicted_values": prediction["value"].to(torch.float32),
                        "target_bins": batch["return_bin_idx"].to(torch.long),
                        "target_values": batch["return_continuous"].to(torch.float32),
                        "dataset_index": torch.full(
                            (local_batch_size,), dataset_index, dtype=torch.long, device=device
                        ),
                        "source_index": source_indices.to(device=device, dtype=torch.long),
                    }
                )
                gathered_dataset_names = [
                    dataset_identity(int(index), mixture.datasets[int(index)])
                    for index in gathered["dataset_index"].cpu().tolist()
                ]
                gathered_task_names = [
                    task_source_names[int(index)] for index in gathered["source_index"].cpu().tolist()
                ]
                accumulator.update(
                    logits=gathered["logits"],
                    predicted_values=gathered["predicted_values"],
                    target_bins=gathered["target_bins"],
                    target_values=gathered["target_values"],
                    dataset_names=gathered_dataset_names,
                    task_names=gathered_task_names,
                )

    metrics = accumulator.compute()
    report = {
        "checkpoint_path": str(checkpoint_path),
        "dataset_roots": _dataset_provenance(mixture),
        "metric_version": METRIC_VERSION,
        "metric_definition": METRIC_DEFINITION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "evaluation_config": {
            "train_config": str(Path(config.train_config).resolve()),
            "dataset_mixture": str(Path(config.dataset_mixture).resolve()),
            "batch_size": config.batch_size,
            "seed": train_config.seed,
            "num_calibration_thresholds": config.num_calibration_thresholds,
        },
        **metrics,
    }
    if accelerator.is_main_process:
        _write_reports(config.output_dir, report)
    accelerator.wait_for_everyone()


def _configure_read_only_evaluation(train_config: TrainPipelineConfig, mixture: DatasetMixtureConfig) -> None:
    """Disable train-time splits and stochastic augmentation before opening dataset roots."""
    train_config.dataset_mixture = mixture
    train_config.val_freq = 0
    train_config.policy.action_decoder_latency_std = 0.0
    train_config.policy.cloud_vlm_latency_std = 0.0
    for dataset in mixture.datasets:
        dataset.prompt_substitutions = None
        dataset.image_transforms.enable = False


def _make_dataloader(dataset: Any, train_config: TrainPipelineConfig, batch_size: int) -> DataLoader:
    kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": train_config.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if train_config.num_workers > 0 and train_config.prefetch_factor is not None:
        kwargs["prefetch_factor"] = train_config.prefetch_factor
    return DataLoader(dataset, **kwargs)


def _move_tensor_values_to_device(batch: dict[str, Any], device: torch.device) -> None:
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)


def _task_names(batch: Mapping[str, Any]) -> list[str]:
    tasks = batch["prompt"]
    if (
        isinstance(tasks, str)
        or not isinstance(tasks, Sequence)
        or not all(isinstance(task, str) for task in tasks)
    ):
        raise TypeError("Value evaluation batch must include one string prompt per sample")
    return list(tasks)


def _base_lerobot_dataset(dataset: Any) -> Any:
    """Unwrap the light wrappers used by dataset splitting/tagging."""
    current = dataset
    for _ in range(3):
        if hasattr(current, "episode_to_task_index") and hasattr(current, "meta"):
            return current
        next_dataset = getattr(current, "dataset", None)
        if next_dataset is None:
            next_dataset = getattr(current, "_base", None)
        if next_dataset is None or next_dataset is current:
            break
        current = next_dataset
    raise TypeError("Value evaluation requires a LeRobot dataset with task metadata")


def _task_source_key(dataset_index: int, task_index: int) -> int:
    if dataset_index < 0 or task_index < 0 or task_index >= 2**32:
        raise ValueError("dataset/task indices must fit the non-negative task source key contract")
    return (dataset_index << 32) | task_index


def _task_source_names(dataset_index: int, dataset: Any) -> dict[int, str]:
    base = _base_lerobot_dataset(dataset)
    return {
        _task_source_key(dataset_index, int(task_index)): str(task)
        for task_index, task in base.meta.tasks.items()
    }


def _task_source_indices(dataset_index: int, dataset: Any, batch: Mapping[str, Any]) -> torch.Tensor:
    base = _base_lerobot_dataset(dataset)
    episode_indices = torch.as_tensor(batch["episode_index"], dtype=torch.long).reshape(-1).cpu().tolist()
    try:
        task_indices = [base.episode_to_task_index[int(episode)] for episode in episode_indices]
    except KeyError as exc:
        raise ValueError(f"No task index metadata for held-out episode {exc.args[0]}") from exc
    return torch.tensor(
        [_task_source_key(dataset_index, int(task_index)) for task_index in task_indices],
        dtype=torch.long,
    )


def dataset_identity(index: int, dataset: Any) -> str:
    """Return a stable identity for one configured dataset entry."""
    dataset_name = dataset.repo_id or dataset.vqa or "dataset"
    return f"{index}:{dataset_name}@{_resolved_dataset_root(dataset)}"


def _dataset_provenance(mixture: DatasetMixtureConfig) -> list[dict[str, object]]:
    return [
        {
            "dataset": dataset.repo_id or dataset.vqa,
            "identity": dataset_identity(index, dataset),
            "root": str(_resolved_dataset_root(dataset)),
            "episodes": list(dataset.episodes or []),
        }
        for index, dataset in enumerate(mixture.datasets)
    ]


def _write_reports(output_dir: Path, report: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "value_metrics.json").open("w", encoding="utf-8") as json_file:
        json.dump(report, json_file, indent=2, sort_keys=True, allow_nan=False)
        json_file.write("\n")
    with (output_dir / "value_metrics.md").open("w", encoding="utf-8") as markdown_file:
        markdown_file.write(render_value_metrics_markdown(report))


def _append_summary_row(lines: list[str], name: str, summary: Mapping[str, object]) -> None:
    lines.append(
        "| {name} | {mean} | {std} | {minimum} | {maximum} |".format(
            name=name,
            mean=_format_metric_value(summary["mean"]),
            std=_format_metric_value(summary["std"]),
            minimum=_format_metric_value(summary["min"]),
            maximum=_format_metric_value(summary["max"]),
        )
    )


def _append_group_rows(lines: list[str], grouped_metrics: Mapping[str, object]) -> None:
    for group_name in sorted(grouped_metrics):
        metrics = _mapping(grouped_metrics[group_name])
        target = _mapping(metrics["target_summary"])
        prediction = _mapping(metrics["prediction_summary"])
        lines.append(
            "| {name} | {count} | {mae} | {nll} | {spearman} | {calibration} | "
            "{target_mean} | {target_std} | {target_min} | {target_max} | "
            "{prediction_mean} | {prediction_std} | {prediction_min} | {prediction_max} |".format(
                name=group_name,
                count=_format_metric_value(metrics["count"]),
                mae=_format_metric_value(metrics["mae"]),
                nll=_format_metric_value(metrics["nll"]),
                spearman=_format_metric_value(metrics["spearman"]),
                calibration=_format_metric_value(metrics["calibration_error"]),
                target_mean=_format_metric_value(target["mean"]),
                target_std=_format_metric_value(target["std"]),
                target_min=_format_metric_value(target["min"]),
                target_max=_format_metric_value(target["max"]),
                prediction_mean=_format_metric_value(prediction["mean"]),
                prediction_std=_format_metric_value(prediction["std"]),
                prediction_min=_format_metric_value(prediction["min"]),
                prediction_max=_format_metric_value(prediction["max"]),
            )
        )


def _metric_names(metrics: Mapping[str, object]) -> tuple[str, ...]:
    preferred = ("count", "mae", "nll", "spearman", "spearman_reason", "calibration_error")
    return tuple(name for name in preferred if name in metrics)


def _format_metric_value(value: object) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError("report sections must be mappings")
    return value


def _parse_args() -> EvaluateValueConfig:
    parser = argparse.ArgumentParser(
        description="Evaluate a Value checkpoint on explicit held-out episodes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train_config",
        type=Path,
        required=True,
        help="Training config containing pretrained_path.",
    )
    parser.add_argument(
        "--dataset_mixture",
        type=Path,
        required=True,
        help="Held-out dataset mixture configuration.",
    )
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for value_metrics reports.")
    parser.add_argument("--batch_size", type=int, required=True, help="Evaluation dataloader batch size.")
    parser.add_argument(
        "--num_calibration_thresholds",
        type=int,
        default=20,
        help="Number of nontrivial CDF thresholds used for calibration error.",
    )
    args = parser.parse_args()
    return EvaluateValueConfig(**vars(args))


if __name__ == "__main__":
    init_logging()
    main(_parse_args())
