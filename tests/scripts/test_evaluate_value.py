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

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.datasets import factory
from opentau.scripts import evaluate_value
from opentau.scripts.evaluate_value import (
    EvaluateValueConfig,
    render_value_metrics_markdown,
    validate_held_out_mixture,
)


def make_mixture(episodes: list[int] | None) -> DatasetMixtureConfig:
    return DatasetMixtureConfig(
        datasets=[DatasetConfig(repo_id="org/held-out", root="/datasets/held-out", episodes=episodes)]
    )


def sample_metrics(count: int) -> dict[str, object]:
    return {
        "count": count,
        "mae": 0.125,
        "nll": 0.25,
        "spearman": 0.75,
        "calibration_error": 0.0625,
        "target_summary": {"mean": -0.5, "std": 0.1, "min": -0.75, "max": -0.25},
        "prediction_summary": {"mean": -0.45, "std": 0.2, "min": -0.7, "max": -0.2},
    }


def sample_report() -> dict[str, object]:
    return {
        "checkpoint_path": "/checkpoints/value",
        "dataset_roots": [
            {
                "dataset": "org/held-out",
                "identity": "0:org/held-out@/datasets/held-out",
                "root": "/datasets/held-out",
                "episodes": [4, 7],
            }
        ],
        "metric_version": "value-quality-v1",
        "metric_definition": "MAE, categorical NLL, Spearman, and CDF calibration error.",
        "generated_at": "2026-07-10T00:00:00Z",
        "evaluation_config": {
            "train_config": "/configs/value.json",
            "dataset_mixture": "/configs/held-out.json",
            "batch_size": 16,
            "seed": 42,
            "num_calibration_thresholds": 20,
        },
        "overall": sample_metrics(3),
        "by_dataset": {"zeta": sample_metrics(1), "alpha": sample_metrics(2)},
        "by_task": {"task-z": sample_metrics(1), "task-a": sample_metrics(2)},
    }


def test_validation_requires_explicit_held_out_episodes():
    mixture = make_mixture(episodes=None)

    with pytest.raises(ValueError, match="explicit held-out episodes"):
        validate_held_out_mixture(mixture)


def test_validation_accepts_explicit_held_out_episodes():
    validate_held_out_mixture(make_mixture(episodes=[4, 7]))


@pytest.mark.parametrize("episodes", [None, []])
def test_validation_rejects_missing_or_empty_held_out_episodes(episodes):
    with pytest.raises(ValueError, match="explicit held-out episodes"):
        validate_held_out_mixture(make_mixture(episodes=episodes))


def test_validation_rejects_an_empty_held_out_mixture():
    with pytest.raises(ValueError, match="at least one"):
        validate_held_out_mixture(DatasetMixtureConfig(datasets=[]))


def test_validation_rejects_vqa_because_it_cannot_select_held_out_episodes():
    mixture = DatasetMixtureConfig(datasets=[DatasetConfig(vqa="cocoqa", episodes=[4, 7])])

    with pytest.raises(ValueError, match="VQA"):
        validate_held_out_mixture(mixture)


def test_markdown_contains_global_dataset_and_task_sections():
    markdown = render_value_metrics_markdown(sample_report())

    assert "## Overall" in markdown
    assert "## By dataset" in markdown
    assert "## By task" in markdown


def test_task_names_use_standardized_prompt_field():
    assert evaluate_value._task_names({"prompt": ["held-out task"]}) == ["held-out task"]


def test_markdown_tables_are_deterministic_and_sorted_by_group_name():
    markdown = render_value_metrics_markdown(sample_report())

    assert markdown.index("| alpha |") < markdown.index("| zeta |")
    assert markdown.index("| task-a |") < markdown.index("| task-z |")
    assert markdown == render_value_metrics_markdown(sample_report())


def test_markdown_exposes_evaluation_provenance_and_summary_statistics():
    markdown = render_value_metrics_markdown(sample_report())

    assert "Train config: `/configs/value.json`" in markdown
    assert "Dataset mixture: `/configs/held-out.json`" in markdown
    assert "Batch size: 16" in markdown
    assert "Seed: 42" in markdown
    assert "Calibration thresholds: 20" in markdown
    assert "Target mean" in markdown
    assert "Prediction max" in markdown


def test_dataset_identity_distinguishes_duplicate_repo_ids_with_different_roots(tmp_path):
    mixture = DatasetMixtureConfig(
        datasets=[
            DatasetConfig(repo_id="org/shared", root=str(tmp_path / "root-a"), episodes=[1]),
            DatasetConfig(repo_id="org/shared", root=str(tmp_path / "root-b"), episodes=[2]),
        ]
    )

    identities = [
        evaluate_value.dataset_identity(index, dataset) for index, dataset in enumerate(mixture.datasets)
    ]
    provenance = evaluate_value._dataset_provenance(mixture)

    assert len(set(identities)) == 2
    assert [item["identity"] for item in provenance] == identities

    accumulator = evaluate_value.ValueMetricsAccumulator(num_calibration_thresholds=1)
    accumulator.update(
        logits=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        predicted_values=torch.tensor([-0.75, -0.25]),
        target_bins=torch.tensor([0, 1]),
        target_values=torch.tensor([-0.75, -0.25]),
        dataset_names=identities,
        task_names=["task", "task"],
    )
    assert set(accumulator.compute()["by_dataset"]) == set(identities)


def test_main_rejects_non_value_policy_before_checkpoint_loading(monkeypatch, tmp_path):
    train_config = SimpleNamespace(policy=SimpleNamespace(type="pi05"))

    monkeypatch.setattr(
        evaluate_value.TrainPipelineConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: train_config),
    )
    monkeypatch.setattr(
        evaluate_value.draccus,
        "parse",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("mixture must not be loaded")),
    )
    monkeypatch.setattr(
        evaluate_value,
        "get_policy_class",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("checkpoint must not be loaded")),
    )

    with pytest.raises(ValueError, match="Value policy"):
        evaluate_value.main(
            EvaluateValueConfig(
                train_config=tmp_path / "train_config.json",
                dataset_mixture=tmp_path / "held_out.json",
                output_dir=tmp_path / "reports",
                batch_size=1,
            )
        )


@pytest.mark.parametrize(
    "output_suffix",
    ["datasets/held-out", "datasets/held-out/reports", "datasets"],
)
def test_output_directory_must_not_overlap_any_dataset_root(tmp_path, output_suffix):
    dataset_root = tmp_path / "datasets" / "held-out"
    mixture = DatasetMixtureConfig(
        datasets=[DatasetConfig(repo_id="org/held-out", root=str(dataset_root), episodes=[4, 7])]
    )

    with pytest.raises(ValueError, match="overlap"):
        evaluate_value.validate_output_dir_does_not_overlap_dataset_roots(tmp_path / output_suffix, mixture)


def test_output_directory_must_not_overlap_training_output_or_checkpoint(tmp_path):
    train_config = SimpleNamespace(
        output_dir=tmp_path / "training",
        policy=SimpleNamespace(pretrained_path=tmp_path / "training" / "checkpoints" / "10"),
    )

    with pytest.raises(ValueError, match="training output"):
        evaluate_value.validate_output_dir_does_not_overlap_training(
            tmp_path / "training" / "evaluation",
            train_config,
        )
    with pytest.raises(ValueError, match="checkpoint"):
        evaluate_value.validate_output_dir_does_not_overlap_training(
            tmp_path / "training-other" / "reports",
            SimpleNamespace(
                output_dir=tmp_path / "separate-training",
                policy=SimpleNamespace(pretrained_path=tmp_path / "training-other"),
            ),
        )


def test_factory_forwards_local_only_to_metadata_and_dataset(monkeypatch, tmp_path):
    captured = {}
    dataset_config = DatasetConfig(
        repo_id="org/held-out",
        root=str(tmp_path / "held-out"),
        episodes=[4],
        use_imagenet_stats=False,
    )
    train_config = SimpleNamespace(
        dataset_mixture=DatasetMixtureConfig(datasets=[dataset_config]),
        val_freq=0,
    )

    class FakeMetadata:
        def __init__(self, *args, **kwargs):
            captured["metadata_local_files_only"] = kwargs["local_files_only"]

    class FakeDataset:
        def __init__(self, *args, **kwargs):
            captured["dataset_local_files_only"] = kwargs["local_files_only"]

    monkeypatch.setattr(factory, "LeRobotDatasetMetadata", FakeMetadata)
    monkeypatch.setattr(factory, "LeRobotDataset", FakeDataset)
    monkeypatch.setattr(factory, "resolve_delta_timestamps", lambda *args: (({}, {}, {}, {}), {}))

    dataset = factory.make_dataset(dataset_config, train_config, local_files_only=True)

    assert isinstance(dataset, FakeDataset)
    assert captured == {
        "metadata_local_files_only": True,
        "dataset_local_files_only": True,
    }


def test_main_rejects_output_overlap_before_opening_policy_or_dataset(monkeypatch, tmp_path):
    dataset_root = tmp_path / "held-out"
    mixture = DatasetMixtureConfig(
        datasets=[DatasetConfig(repo_id="org/held-out", root=str(dataset_root), episodes=[4, 7])]
    )
    train_config = SimpleNamespace(
        seed=None,
        policy=SimpleNamespace(
            pretrained_path="/checkpoints/value",
            type="value",
            action_decoder_latency_std=1.0,
            cloud_vlm_latency_std=1.0,
        ),
        dataset_mixture=None,
        val_freq=3,
        num_workers=0,
        prefetch_factor=None,
    )

    def fail_if_policy_or_dataset_is_opened(*args, **kwargs):
        raise AssertionError(
            "the overlapping output directory must be rejected before opening a policy or dataset"
        )

    monkeypatch.setattr(
        evaluate_value.TrainPipelineConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: train_config),
    )
    monkeypatch.setattr(evaluate_value.draccus, "parse", lambda **kwargs: mixture)
    monkeypatch.setattr(evaluate_value, "get_policy_class", fail_if_policy_or_dataset_is_opened)
    monkeypatch.setattr(evaluate_value, "make_dataset", fail_if_policy_or_dataset_is_opened)

    with pytest.raises(ValueError, match="overlap"):
        evaluate_value.main(
            EvaluateValueConfig(
                train_config=tmp_path / "train_config.json",
                dataset_mixture=tmp_path / "held_out.json",
                output_dir=dataset_root,
                batch_size=1,
            )
        )

    assert not dataset_root.exists()


def test_main_uses_local_dataset_and_writes_selected_episode_reports(monkeypatch, tmp_path):
    dataset_root = tmp_path / "datasets" / "held-out"
    dataset_root.mkdir(parents=True)
    output_dir = tmp_path / "reports"
    mixture = DatasetMixtureConfig(
        datasets=[DatasetConfig(repo_id="org/held-out", root=str(dataset_root), episodes=[4, 7])]
    )
    policy_config = SimpleNamespace(
        pretrained_path="/checkpoints/value",
        type="value",
        action_decoder_latency_std=1.0,
        cloud_vlm_latency_std=1.0,
    )
    train_config = SimpleNamespace(
        seed=None,
        policy=policy_config,
        dataset_mixture=None,
        val_freq=3,
        num_workers=0,
        prefetch_factor=None,
    )
    seen = {}

    class FakeAccelerator:
        device = torch.device("cpu")
        is_main_process = True

        def prepare(self, dataloader):
            return dataloader

        def gather_for_metrics(self, payload):
            seen["gather_keys"] = set(payload)
            assert payload["dataset_index"].dtype == torch.long
            assert payload["source_index"].dtype == torch.long
            return payload

        def wait_for_everyone(self):
            seen["waited"] = True

    class FakeDataset(list):
        def __init__(self, items):
            super().__init__(items)
            self.meta = SimpleNamespace(tasks={3: "held-out task"})
            self.episode_to_task_index = {4: 3}

    class FakePolicy:
        def to(self, *, device, dtype):
            return self

        def eval(self):
            return self

        def predict_value_distribution(self, batch):
            assert batch["prompt"] == ["held-out task"]
            assert isinstance(batch["prompt"][0], str)
            return {
                "logits": torch.tensor([[2.0, -2.0]]),
                "probabilities": torch.tensor([[0.98, 0.02]]),
                "value": torch.tensor([-0.75]),
            }

    class FakePolicyClass:
        @classmethod
        def from_pretrained(
            cls,
            checkpoint_path,
            config,
            local_files_only=False,
            backbone_local_files_only=False,
        ):
            assert local_files_only is True
            assert backbone_local_files_only is True
            seen["checkpoint_path"] = checkpoint_path
            return FakePolicy()

    def fake_make_dataset(dataset_config, config, *, return_advantage_input, local_files_only):
        assert return_advantage_input is True
        assert local_files_only is True
        seen["episodes"] = dataset_config.episodes
        return FakeDataset(
            [
                {
                    "prompt": "held-out task",
                    "return_bin_idx": torch.tensor(0, dtype=torch.long),
                    "return_continuous": torch.tensor(-0.75),
                    "episode_index": 4,
                }
            ]
        )

    monkeypatch.setattr(
        evaluate_value.TrainPipelineConfig,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: train_config),
    )
    monkeypatch.setattr(evaluate_value.draccus, "parse", lambda **kwargs: mixture)
    monkeypatch.setattr(evaluate_value, "get_policy_class", lambda policy_type: FakePolicyClass)
    monkeypatch.setattr(evaluate_value, "make_dataset", fake_make_dataset)
    monkeypatch.setattr(evaluate_value, "Accelerator", FakeAccelerator)

    evaluate_value.main(
        EvaluateValueConfig(
            train_config=tmp_path / "train_config.json",
            dataset_mixture=tmp_path / "held_out.json",
            output_dir=output_dir,
            batch_size=1,
        )
    )

    report = json.loads((output_dir / "value_metrics.json").read_text(encoding="utf-8"))
    markdown = (output_dir / "value_metrics.md").read_text(encoding="utf-8")
    assert seen == {
        "checkpoint_path": "/checkpoints/value",
        "episodes": [4, 7],
        "gather_keys": {
            "logits",
            "predicted_values",
            "target_bins",
            "target_values",
            "dataset_index",
            "source_index",
        },
        "waited": True,
    }
    assert report["dataset_roots"] == [
        {
            "dataset": "org/held-out",
            "episodes": [4, 7],
            "identity": f"0:org/held-out@{dataset_root.resolve()}",
            "root": str(dataset_root),
        }
    ]
    assert report["evaluation_config"] == {
        "batch_size": 1,
        "dataset_mixture": str((tmp_path / "held_out.json").resolve()),
        "num_calibration_thresholds": 20,
        "seed": None,
        "train_config": str((tmp_path / "train_config.json").resolve()),
    }
    assert "| held-out task |" in markdown
