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

import json

import numpy as np
import pytest
import torch

from opentau.policies.value.metrics import ValueMetricsAccumulator, compute_value_metrics


def test_perfect_predictions_have_zero_mae_nll_and_calibration_error():
    logits = np.array([[20.0, -20.0], [-20.0, 20.0]])

    metrics = compute_value_metrics(
        logits=logits,
        predicted_values=np.array([-0.75, -0.25]),
        target_bins=np.array([0, 1]),
        target_values=np.array([-0.75, -0.25]),
        num_calibration_thresholds=2,
    )

    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["nll"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["target_summary"] == {
        "mean": pytest.approx(-0.5),
        "std": pytest.approx(0.25),
        "min": pytest.approx(-0.75),
        "max": pytest.approx(-0.25),
    }
    assert metrics["prediction_summary"] == metrics["target_summary"]
    json.dumps(metrics, allow_nan=False)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["calibration_error"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["count"] == 2
    assert "spearman_reason" not in metrics


def test_single_example_summary_statistics_are_finite():
    metrics = compute_value_metrics(
        logits=np.array([[1.0, 0.0]]),
        predicted_values=np.array([-0.5]),
        target_bins=np.array([0]),
        target_values=np.array([-0.75]),
    )

    assert metrics["target_summary"] == {
        "mean": -0.75,
        "std": 0.0,
        "min": -0.75,
        "max": -0.75,
    }
    assert metrics["prediction_summary"]["std"] == 0.0
    json.dumps(metrics, allow_nan=False)


def test_reversed_value_ordering_has_negative_spearman_correlation():
    metrics = compute_value_metrics(
        logits=np.array([[20.0, -20.0], [-20.0, 20.0], [20.0, -20.0]]),
        predicted_values=np.array([3.0, 2.0, 1.0]),
        target_bins=np.array([0, 1, 0]),
        target_values=np.array([1.0, 2.0, 3.0]),
        num_calibration_thresholds=2,
    )

    assert metrics["spearman"] == pytest.approx(-1.0)
    assert "spearman_reason" not in metrics


def test_constant_predictions_return_json_safe_undefined_spearman():
    metrics = compute_value_metrics(
        logits=np.array([[20.0, -20.0], [-20.0, 20.0], [-20.0, 20.0]]),
        predicted_values=np.array([-0.5, -0.5, -0.5]),
        target_bins=np.array([0, 1, 1]),
        target_values=np.array([-0.75, -0.5, -0.25]),
        num_calibration_thresholds=2,
    )

    assert metrics["spearman"] is None
    assert metrics["spearman_reason"] == "constant_prediction"
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    ("predicted_values", "target_values", "expected_reason"),
    [
        (np.array([-0.5]), np.array([-0.5]), "fewer_than_two_examples"),
        (np.array([-0.75, -0.25]), np.array([-0.5, -0.5]), "constant_target"),
    ],
)
def test_undefined_spearman_reasons_are_json_safe(predicted_values, target_values, expected_reason):
    count = predicted_values.size
    metrics = compute_value_metrics(
        logits=np.tile(np.array([[20.0, -20.0]]), (count, 1)),
        predicted_values=predicted_values,
        target_bins=np.zeros(count, dtype=np.int64),
        target_values=target_values,
        num_calibration_thresholds=2,
    )

    assert metrics["spearman"] is None
    assert metrics["spearman_reason"] == expected_reason
    json.dumps(metrics, allow_nan=False)


def test_calibration_uses_only_unique_nontrivial_cdf_thresholds():
    metrics = compute_value_metrics(
        logits=np.log(np.array([[0.9, 0.1], [0.9, 0.1]])),
        predicted_values=np.array([-0.5, -0.5]),
        target_bins=np.array([0, 1]),
        target_values=np.array([-0.75, -0.25]),
        num_calibration_thresholds=20,
    )

    assert metrics["calibration_error"] == pytest.approx(0.4)


def test_single_bin_calibration_is_json_safe():
    metrics = compute_value_metrics(
        logits=np.array([[0.0]]),
        predicted_values=np.array([-0.5]),
        target_bins=np.array([0]),
        target_values=np.array([-0.5]),
    )

    assert metrics["calibration_error"] == pytest.approx(0.0)
    assert metrics["spearman"] is None
    assert metrics["spearman_reason"] == "fewer_than_two_examples"
    json.dumps(metrics, allow_nan=False)


@pytest.mark.parametrize(
    ("logits", "predicted_values", "target_bins", "target_values", "match"),
    [
        (
            np.empty((0, 2)),
            np.empty(0),
            np.empty(0, dtype=np.int64),
            np.empty(0),
            "at least one example",
        ),
        (
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([0.0]),
            np.array([0, 1]),
            np.array([0.0, 1.0]),
            "same batch size",
        ),
        (
            np.array([[1.0, 0.0]]),
            np.array([0.0]),
            np.array([2]),
            np.array([0.0]),
            "out of range",
        ),
        (
            np.array([1.0, 0.0]),
            np.array([0.0]),
            np.array([0]),
            np.array([0.0]),
            "logits must have shape",
        ),
        (
            np.array([[np.nan, 0.0]]),
            np.array([0.0]),
            np.array([0]),
            np.array([0.0]),
            "must be finite",
        ),
    ],
)
def test_invalid_metric_inputs_fail_fast(logits, predicted_values, target_bins, target_values, match):
    with pytest.raises(ValueError, match=match):
        compute_value_metrics(logits, predicted_values, target_bins, target_values)


def test_accumulator_returns_overall_dataset_and_task_metrics_from_detached_tensors():
    accumulator = ValueMetricsAccumulator(num_calibration_thresholds=2)
    logits = torch.tensor([[20.0, -20.0], [-20.0, 20.0]], requires_grad=True)
    predicted_values = torch.tensor([-0.75, -0.25], requires_grad=True)
    target_bins = torch.tensor([0, 1])
    target_values = torch.tensor([-0.75, -0.25], requires_grad=True)

    accumulator.update(
        logits=logits,
        predicted_values=predicted_values,
        target_bins=target_bins,
        target_values=target_values,
        dataset_names=["dataset_a", "dataset_b"],
        task_names=["task_1", "task_1"],
    )
    accumulator.update(
        logits=torch.tensor([[20.0, -20.0]]),
        predicted_values=torch.tensor([-0.5]),
        target_bins=torch.tensor([0]),
        target_values=torch.tensor([-0.5]),
        dataset_names=["dataset_a"],
        task_names=["task_2"],
    )

    metrics = accumulator.compute()

    assert metrics["overall"]["count"] == 3
    assert metrics["by_dataset"]["dataset_a"]["count"] == 2
    assert metrics["by_dataset"]["dataset_b"]["count"] == 1
    assert metrics["by_task"]["task_1"]["count"] == 2
    assert metrics["by_task"]["task_2"]["count"] == 1
    assert metrics["overall"]["mae"] == pytest.approx(0.0)
    json.dumps(metrics, allow_nan=False)


def test_accumulator_upcasts_bfloat16_logits_before_numpy_conversion():
    accumulator = ValueMetricsAccumulator(num_calibration_thresholds=2)

    accumulator.update(
        logits=torch.tensor([[20.0, -20.0], [-20.0, 20.0]], dtype=torch.bfloat16),
        predicted_values=torch.tensor([-0.75, -0.25]),
        target_bins=torch.tensor([0, 1]),
        target_values=torch.tensor([-0.75, -0.25]),
        dataset_names=["dataset_a", "dataset_b"],
        task_names=["task_1", "task_2"],
    )

    metrics = accumulator.compute()["overall"]

    assert metrics["count"] == 2
    for metric_name in ("mae", "nll", "calibration_error"):
        assert np.isfinite(metrics[metric_name])


def test_accumulator_rejects_group_name_length_mismatch():
    accumulator = ValueMetricsAccumulator()

    with pytest.raises(ValueError, match="dataset_names must have the same batch size"):
        accumulator.update(
            logits=torch.tensor([[20.0, -20.0]]),
            predicted_values=torch.tensor([-0.75]),
            target_bins=torch.tensor([0]),
            target_values=torch.tensor([-0.75]),
            dataset_names=[],
            task_names=["task_1"],
        )
