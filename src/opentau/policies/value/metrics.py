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

"""Non-differentiable quality metrics for Value policy predictions."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.special import log_softmax
from scipy.stats import spearmanr


def compute_value_metrics(
    logits: np.ndarray,
    predicted_values: np.ndarray,
    target_bins: np.ndarray,
    target_values: np.ndarray,
    num_calibration_thresholds: int = 20,
) -> dict[str, object]:
    """Compute JSON-safe quality metrics for a batch of Value predictions.

    Categorical NLL is evaluated at the target bin. Calibration compares the predicted categorical
    CDF against the empirical target-bin CDF at evenly spaced nontrivial thresholds.
    """
    logits, predicted_values, target_bins, target_values = _validate_metric_inputs(
        logits,
        predicted_values,
        target_bins,
        target_values,
    )
    _validate_calibration_threshold_count(num_calibration_thresholds)

    log_probabilities = log_softmax(logits, axis=-1)
    probabilities = np.exp(log_probabilities)
    example_indices = np.arange(target_bins.size)

    metrics: dict[str, object] = {
        "mae": float(np.mean(np.abs(predicted_values - target_values))),
        "nll": float(-log_probabilities[example_indices, target_bins].mean()),
        "calibration_error": _compute_calibration_error(
            probabilities,
            target_bins,
            num_calibration_thresholds,
        ),
        "count": int(target_bins.size),
        "target_summary": _summary_statistics(target_values),
        "prediction_summary": _summary_statistics(predicted_values),
    }
    spearman, reason = _compute_spearman(predicted_values, target_values)
    metrics["spearman"] = spearman
    if reason is not None:
        metrics["spearman_reason"] = reason

    _ensure_finite_metric_values(metrics)
    return metrics


class ValueMetricsAccumulator:
    """Accumulate detached Value predictions and compute overall and grouped metrics."""

    def __init__(self, num_calibration_thresholds: int = 20) -> None:
        _validate_calibration_threshold_count(num_calibration_thresholds)
        self._num_calibration_thresholds = num_calibration_thresholds
        self._overall = _MetricInputs()
        self._by_dataset: dict[str, _MetricInputs] = {}
        self._by_task: dict[str, _MetricInputs] = {}

    def update(
        self,
        logits: torch.Tensor,
        predicted_values: torch.Tensor,
        target_bins: torch.Tensor,
        target_values: torch.Tensor,
        dataset_names: Sequence[str],
        task_names: Sequence[str],
    ) -> None:
        """Add one batch without retaining autograd tensors or contributing to loss."""
        logits_np, predicted_values_np, target_bins_np, target_values_np = _validate_metric_inputs(
            _detach_to_numpy(logits, "logits"),
            _detach_to_numpy(predicted_values, "predicted_values"),
            _detach_to_numpy(target_bins, "target_bins"),
            _detach_to_numpy(target_values, "target_values"),
        )
        batch_size = target_bins_np.size
        dataset_names = _validate_group_names(dataset_names, "dataset_names", batch_size)
        task_names = _validate_group_names(task_names, "task_names", batch_size)

        self._overall.add(logits_np, predicted_values_np, target_bins_np, target_values_np)
        for index, (dataset_name, task_name) in enumerate(zip(dataset_names, task_names, strict=True)):
            sample_logits = logits_np[index : index + 1]
            sample_predicted_values = predicted_values_np[index : index + 1]
            sample_target_bins = target_bins_np[index : index + 1]
            sample_target_values = target_values_np[index : index + 1]
            self._by_dataset.setdefault(dataset_name, _MetricInputs()).add(
                sample_logits,
                sample_predicted_values,
                sample_target_bins,
                sample_target_values,
            )
            self._by_task.setdefault(task_name, _MetricInputs()).add(
                sample_logits,
                sample_predicted_values,
                sample_target_bins,
                sample_target_values,
            )

    def compute(self) -> dict[str, object]:
        """Return metrics for all samples and for each dataset and task."""
        return {
            "overall": self._overall.compute(self._num_calibration_thresholds),
            "by_dataset": {
                name: inputs.compute(self._num_calibration_thresholds)
                for name, inputs in self._by_dataset.items()
            },
            "by_task": {
                name: inputs.compute(self._num_calibration_thresholds)
                for name, inputs in self._by_task.items()
            },
        }


@dataclass
class _MetricInputs:
    """Detached arrays needed to recompute metrics over one aggregate group."""

    logits: list[np.ndarray] = field(default_factory=list)
    predicted_values: list[np.ndarray] = field(default_factory=list)
    target_bins: list[np.ndarray] = field(default_factory=list)
    target_values: list[np.ndarray] = field(default_factory=list)
    num_bins: int | None = None

    def add(
        self,
        logits: np.ndarray,
        predicted_values: np.ndarray,
        target_bins: np.ndarray,
        target_values: np.ndarray,
    ) -> None:
        if self.num_bins is None:
            self.num_bins = logits.shape[1]
        elif logits.shape[1] != self.num_bins:
            raise ValueError("all accumulated logits must have the same number of bins")
        self.logits.append(logits)
        self.predicted_values.append(predicted_values)
        self.target_bins.append(target_bins)
        self.target_values.append(target_values)

    def compute(self, num_calibration_thresholds: int) -> dict[str, object]:
        if not self.logits:
            raise ValueError("cannot compute metrics for an empty accumulator")
        return compute_value_metrics(
            logits=np.concatenate(self.logits, axis=0),
            predicted_values=np.concatenate(self.predicted_values, axis=0),
            target_bins=np.concatenate(self.target_bins, axis=0),
            target_values=np.concatenate(self.target_values, axis=0),
            num_calibration_thresholds=num_calibration_thresholds,
        )


def _validate_metric_inputs(
    logits: np.ndarray,
    predicted_values: np.ndarray,
    target_bins: np.ndarray,
    target_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate metric inputs and normalize their dtypes for numerical computation."""
    try:
        logits = np.asarray(logits, dtype=np.float64)
        predicted_values = np.asarray(predicted_values, dtype=np.float64)
        target_bins = np.asarray(target_bins, dtype=np.float64)
        target_values = np.asarray(target_values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("metric inputs must be numeric arrays") from error

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch_size, num_bins]")
    if logits.shape[0] == 0:
        raise ValueError("metric inputs must contain at least one example")
    if logits.shape[1] == 0:
        raise ValueError("logits must contain at least one bin")
    for name, values in (
        ("predicted_values", predicted_values),
        ("target_bins", target_bins),
        ("target_values", target_values),
    ):
        if values.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional")
        if values.size != logits.shape[0]:
            raise ValueError("all metric inputs must have the same batch size")

    if not all(
        np.isfinite(values).all() for values in (logits, predicted_values, target_bins, target_values)
    ):
        raise ValueError("metric inputs must be finite")
    if not np.equal(target_bins, np.floor(target_bins)).all():
        raise ValueError("target_bins must contain integer bin indices")
    if np.any(target_bins < 0) or np.any(target_bins >= logits.shape[1]):
        raise ValueError("target_bins contain an index out of range for logits")

    return logits, predicted_values, target_bins.astype(np.int64), target_values


def _validate_calibration_threshold_count(num_calibration_thresholds: int) -> None:
    if isinstance(num_calibration_thresholds, bool) or not isinstance(
        num_calibration_thresholds, (int, np.integer)
    ):
        raise ValueError("num_calibration_thresholds must be a positive integer")
    if num_calibration_thresholds < 1:
        raise ValueError("num_calibration_thresholds must be a positive integer")


def _compute_calibration_error(
    probabilities: np.ndarray,
    target_bins: np.ndarray,
    num_calibration_thresholds: int,
) -> float:
    if probabilities.shape[1] == 1:
        return 0.0
    threshold_count = min(num_calibration_thresholds, probabilities.shape[1] - 1)
    thresholds = np.linspace(0, probabilities.shape[1] - 2, num=threshold_count, dtype=np.int64)
    predicted_cdf = np.cumsum(probabilities, axis=-1)
    calibration_gaps = [
        abs(predicted_cdf[:, threshold].mean() - (target_bins <= threshold).mean())
        for threshold in thresholds
    ]
    return float(np.mean(calibration_gaps))


def _compute_spearman(
    predicted_values: np.ndarray, target_values: np.ndarray
) -> tuple[float | None, str | None]:
    if predicted_values.size < 2:
        return None, "fewer_than_two_examples"
    if np.all(predicted_values == predicted_values[0]):
        return None, "constant_prediction"
    if np.all(target_values == target_values[0]):
        return None, "constant_target"

    correlation = float(spearmanr(predicted_values, target_values).statistic)
    if not np.isfinite(correlation):
        return None, "undefined"
    return correlation, None


def _summary_statistics(values: np.ndarray) -> dict[str, float]:
    summary = {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }
    if not all(np.isfinite(value) for value in summary.values()):
        raise ValueError("summary statistics must be finite")
    return summary


def _ensure_finite_metric_values(metrics: dict[str, object]) -> None:
    for name in ("mae", "nll", "calibration_error"):
        value = metrics[name]
        if not isinstance(value, float) or not np.isfinite(value):
            raise ValueError(f"{name} must be finite")


def _detach_to_numpy(tensor: torch.Tensor, name: str) -> np.ndarray:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    tensor = tensor.detach().cpu()
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.to(torch.float32)
    return tensor.numpy()


def _validate_group_names(names: Sequence[str], name: str, batch_size: int) -> tuple[str, ...]:
    if isinstance(names, str):
        raise TypeError(f"{name} must be a sequence of strings, not a single string")
    if len(names) != batch_size:
        raise ValueError(f"{name} must have the same batch size as metric inputs")
    if not all(isinstance(group_name, str) for group_name in names):
        raise TypeError(f"{name} must contain only strings")
    return tuple(names)
