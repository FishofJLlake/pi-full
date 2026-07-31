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
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.scripts import steam_advantage_pipeline as pipeline
from opentau.scripts.steam_advantage_pipeline import (
    DatasetScores,
    _finalize_and_persist,
    _score_member,
    apply_threshold,
    paper_advantage_from_expected_bin,
    rlinf_advantage_from_probabilities,
    shard_bounds,
    source_quantile_thresholds,
    validate_checkpoint_count,
    validate_tag,
    validate_unique_dataset_roots,
)


def test_paper_advantage_is_an_affine_expected_bin_shift():
    expected_bin = torch.tensor([0.0, 3.0])
    result = paper_advantage_from_expected_bin(expected_bin, num_bins=4)
    torch.testing.assert_close(result, torch.tensor([-1.5, 0.0]))


def test_rlinf_advantage_uses_exact_signed_bin_values():
    probabilities = torch.eye(4)
    result = rlinf_advantage_from_probabilities(probabilities, num_bins=4)
    torch.testing.assert_close(result, torch.tensor([-1.0, -0.5, 0.5, 1.0]))


def test_any_positive_checkpoint_count_is_accepted():
    for count in (1, 2, 4):
        validate_checkpoint_count([Path(str(index)) for index in range(count)])
    with pytest.raises(ValueError, match="at least one"):
        validate_checkpoint_count([])
    with pytest.raises(ValueError, match="distinct"):
        validate_checkpoint_count([Path("same"), Path("same")])


def test_threshold_comparison_supports_rlinf_strict_and_legacy_inclusive():
    assert not apply_threshold(0.5, 0.5, "strict")
    assert apply_threshold(0.5, 0.5, "inclusive")


def test_tag_validation_rejects_paths():
    assert validate_tag("steam-k32.v2") == "steam-k32.v2"
    with pytest.raises(ValueError, match="tag must"):
        validate_tag("../escape")


def test_duplicate_normalized_dataset_roots_are_rejected(tmp_path):
    mixture = DatasetMixtureConfig(
        datasets=[
            DatasetConfig(repo_id="a/one", root=str(tmp_path), steam_source="expert"),
            DatasetConfig(repo_id="b/two", root=str(tmp_path / "."), steam_source="non_expert"),
        ]
    )
    with pytest.raises(ValueError, match="duplicate dataset roots"):
        validate_unique_dataset_roots(mixture)


def test_finalize_rejects_invalid_ensemble_size_and_rlinf_threshold():
    common = {
        "tag": "steam",
        "score_mode": "rlinf_signed",
        "label_mode": "threshold",
        "threshold_comparison": "strict",
        "expert_positive_fraction": 0.8,
        "non_expert_positive_fraction": 0.3,
    }
    with pytest.raises(ValueError, match="member_count"):
        _finalize_and_persist(
            [],
            member_count=0,
            positive_threshold=0.0,
            **common,
        )
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        _finalize_and_persist(
            [],
            member_count=1,
            positive_threshold=1.1,
            **common,
        )


def test_contiguous_shards_are_balanced_like_rlinf():
    assert [shard_bounds(10, rank, 3) for rank in range(3)] == [(0, 4), (4, 7), (7, 10)]


def test_main_continues_into_dataset_collection_after_checkpoint_validation(monkeypatch):
    config = SimpleNamespace(
        policy=pipeline.SteamConfig(),
        resolution=(384, 384),
        dataset_mixture=SimpleNamespace(),
        seed=None,
        dataloader_batch_size=1,
        batch_size=1,
        num_workers=0,
        prefetch_factor=None,
    )
    monkeypatch.setattr(
        pipeline.TrainPipelineConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: config,
    )
    monkeypatch.setattr(pipeline, "auto_torch_device", lambda: torch.device("cpu"))

    def reached_collection(*_args, **_kwargs):
        raise RuntimeError("reached dataset collection")

    monkeypatch.setattr(pipeline, "_collect_datasets", reached_collection)
    args = SimpleNamespace(
        checkpoint=[Path("a")],
        config_path=None,
        dataset_mixture=None,
        batch_size=None,
        num_workers=None,
    )
    with pytest.raises(RuntimeError, match="reached dataset collection"):
        pipeline.main(args)


def test_source_quantiles_are_computed_independently():
    thresholds = source_quantile_thresholds(
        {
            "expert": [0.0, 1.0, 2.0, 3.0, 4.0],
            "non_expert": [-4.0, -3.0, -2.0, -1.0, 0.0],
        },
        expert_positive_fraction=0.8,
        non_expert_positive_fraction=0.4,
    )
    assert thresholds["expert"] == np.percentile([0, 1, 2, 3, 4], 20)
    assert thresholds["non_expert"] == np.percentile([-4, -3, -2, -1, 0], 60)


class _FakeInferencePairs(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {"episode_index": 0, "frame_index": index, "prompt": "task"}


class _FakeMember:
    def __init__(self, expected_bins):
        self.expected_bins = torch.tensor(expected_bins, dtype=torch.float32)
        self.config = SimpleNamespace(num_bins=4)

    def predict_temporal_offset(self, batch):
        indices = batch["frame_index"].to(dtype=torch.long)
        return {"expected_bin": self.expected_bins.to(indices.device)[indices]}


class _FakeMergedEnsemble:
    def __init__(self):
        self.config = SimpleNamespace(num_bins=4)

    def predict_temporal_offset(self, batch):
        batch_size = len(batch["frame_index"])
        probabilities = torch.tensor(
            [
                [[0.0, 0.0, 1.0, 0.0]] * batch_size,
                [[0.0, 1.0, 0.0, 0.0]] * batch_size,
            ],
            dtype=torch.float32,
        )
        return {
            "probabilities": probabilities[0],
            "member_probabilities": probabilities,
            "member_expected_bins": torch.tensor([[2.0] * batch_size, [1.0] * batch_size]),
            "member_expected_stride_scores": torch.tensor([[0.5] * batch_size, [-0.5] * batch_size]),
            "member_rlinf_signed_scores": torch.tensor([[0.5] * batch_size, [-0.5] * batch_size]),
        }


def _bundle():
    return DatasetScores(
        config=SimpleNamespace(),
        dataset=_FakeInferencePairs(),
        minimum_advantage={(0, 0): float("inf"), (0, 1): float("inf")},
        tasks={},
    )


def test_independent_members_are_reduced_by_a_pointwise_minimum():
    bundle = _bundle()
    for member in (_FakeMember([3.0, 1.0]), _FakeMember([2.0, 2.0])):
        _score_member(
            member,
            [bundle],
            device=torch.device("cpu"),
            batch_size=2,
            num_workers=0,
            prefetch_factor=None,
        )
    assert bundle.minimum_advantage == {(0, 0): -0.5, (0, 1): -1.0}
    assert all(len(values) == 2 for values in bundle.paper_member_scores.values())


def test_single_checkpoint_ensemble_emits_every_member_in_one_pass():
    bundle = _bundle()
    _score_member(
        _FakeMergedEnsemble(),
        [bundle],
        device=torch.device("cpu"),
        batch_size=2,
        num_workers=0,
        prefetch_factor=None,
    )
    assert bundle.rlinf_member_scores[(0, 0)] == [0.5, -0.5]
    assert bundle.paper_member_scores[(0, 0)] == [-0.5, -1.0]


def test_persistence_writes_rlinf_parquet_diagnostics_and_neutral_terminal(tmp_path):
    policy_config = pipeline.SteamConfig(
        image_resolution=(8, 8),
        num_bins=4,
        max_temporal_offset=4,
        length_scale_enabled=False,
    )
    dataset = SimpleNamespace(
        root=tmp_path,
        meta=SimpleNamespace(fps=5.0),
        config=policy_config,
        length_reference=2.0,
        frame_records=[(0, 0, 0), (0, 1, 1)],
        terminal_records=[(0, 1, 1)],
        base_dataset=SimpleNamespace(
            episode_to_task_index={0: 0},
            meta=SimpleNamespace(tasks={0: "task"}),
        ),
    )
    bundle = DatasetScores(
        config=SimpleNamespace(steam_source="non_expert"),
        dataset=dataset,
        minimum_advantage={(0, 0): -0.5},
        tasks={(0, 0): "task"},
        paper_member_scores={(0, 0): [0.1, -0.2]},
        rlinf_member_scores={(0, 0): [0.5, -0.25]},
        expected_stride_member_scores={(0, 0): [0.4, -0.2]},
        member_entropies={(0, 0): [0.3, 0.4]},
    )

    _finalize_and_persist(
        [bundle],
        member_count=2,
        tag="rlinf",
        score_mode="rlinf_signed",
        label_mode="threshold",
        positive_threshold=0.0,
        threshold_comparison="strict",
        expert_positive_fraction=0.8,
        non_expert_positive_fraction=0.3,
    )

    raw = json.loads((tmp_path / "meta" / "raw_advantages.json").read_text(encoding="utf-8"))
    effective = json.loads((tmp_path / "meta" / "advantages.json").read_text(encoding="utf-8"))
    diagnostics = json.loads(
        (tmp_path / "meta" / "steam_advantage_diagnostics.json").read_text(encoding="utf-8")
    )
    parquet = pipeline.pd.read_parquet(tmp_path / "meta" / "advantages_rlinf.parquet")

    assert raw == {"0,0": "-0.250000", "0,1": "0.000000"}
    assert effective == {"0,0": "0.000000", "0,1": "0.000000"}
    assert diagnostics["member_scores"]["0,0"] == [0.5, -0.25]
    assert diagnostics["member_scores"]["0,1"] == [0.0, 0.0]
    assert diagnostics["threshold_comparison"] == "strict"
    assert parquet["frame_index"].tolist() == [0, 1]
    assert parquet["ensemble_signed_score"].tolist() == [-0.25, 0.0]
    assert parquet["is_terminal"].tolist() == [False, True]
