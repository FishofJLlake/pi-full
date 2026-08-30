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
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from opentau.configs.default import DatasetConfig
from opentau.datasets.advantage_metadata import persist_advantage_bundle
from opentau.datasets.utils import (
    ADVANTAGE_REPORT_PATH,
    ADVANTAGE_SOURCES_PATH,
    RAW_ADVANTAGES_PATH,
    load_advantage_sources_from_path,
    load_advantages,
    load_advantages_from_path,
)
from opentau.scripts import relabel_steam_advantages as relabel_module
from opentau.scripts.relabel_steam_advantages import (
    RelabelDataset,
    _apply_positive_overrides,
    _assign_base_labels,
    _assign_preserved_labels,
    _compute_actual_lookahead,
    _output_dataframe,
    _persist_effective_bundle,
)


def _args(**overrides) -> Namespace:
    values = {
        "expert_mode": "quantile",
        "non_expert_mode": "quantile",
        "expert_threshold": 0.0,
        "non_expert_threshold": 0.0,
        "threshold_comparison": "strict",
        "expert_positive_fraction": 0.5,
        "non_expert_positive_fraction": 0.5,
        "quantile_grouping": "actual_lookahead",
        "tail_positive_frames": 0,
        "tail_positive_sources": ["expert"],
        "force_intervention_positive": False,
        "intervention_positive_sources": ["non_expert"],
        "intervention_value_threshold": 0.0,
    }
    values.update(overrides)
    return Namespace(**values)


def _dataset(
    tmp_path: Path,
    scores: dict[tuple[int, int], float],
    *,
    source: str = "expert",
    index: int = 0,
) -> RelabelDataset:
    keys = sorted(scores)
    dataframe = pd.DataFrame(
        {
            "episode_index": [key[0] for key in keys],
            "frame_index": [key[1] for key in keys],
            "advantage_continuous": [scores[key] for key in keys],
            "advantage": [scores[key] > 0 for key in keys],
            "max_temporal_offset": [3] * len(keys),
        }
    )
    config = DatasetConfig(
        repo_id=f"test/relabel-{index}",
        root=str(tmp_path / f"dataset-{index}"),
        steam_source=source,
    )
    old_advantages = {key: float(scores[key] > 0) for key in keys}
    return RelabelDataset(
        index=index,
        config=config,
        root=Path(config.root),
        source=source,
        dataframe=dataframe,
        old_advantages=old_advantages,
        raw_advantages=scores.copy(),
        old_sources=dict.fromkeys(keys, "steam_quantile"),
        last_frames=relabel_module._episode_last_frames(set(keys)),
    )


def _tail_scores() -> dict[tuple[int, int], float]:
    return {
        (0, 0): 0.9,
        (0, 1): 0.8,
        (0, 2): 0.1,
        (0, 3): 0.0,
        (1, 0): 0.7,
        (1, 1): 0.6,
        (1, 2): 0.5,
        (1, 3): 0.0,
    }


def test_actual_lookahead_quantiles_keep_tail_strata(tmp_path: Path):
    stratified = _dataset(tmp_path, _tail_scores())
    _compute_actual_lookahead([stratified], max_temporal_offset=3)
    _assign_base_labels([stratified], _args(quantile_grouping="actual_lookahead"))

    assert stratified.advantages[(0, 3)] == stratified.advantages[(1, 3)] == 0.0
    assert sum(stratified.advantages[key] for key in ((0, 2), (1, 2))) == 1.0
    assert stratified.advantages[(1, 2)] == 1.0

    global_pool = _dataset(tmp_path, _tail_scores(), index=1)
    _compute_actual_lookahead([global_pool], max_temporal_offset=3)
    _assign_base_labels([global_pool], _args(quantile_grouping="global"))

    assert sum(global_pool.advantages[key] for key in ((0, 2), (1, 2))) == 0.0


def test_all_positive_expert_still_excludes_terminal_without_override(tmp_path: Path):
    dataset = _dataset(tmp_path, _tail_scores())
    _compute_actual_lookahead([dataset], max_temporal_offset=3)
    _assign_base_labels([dataset], _args(expert_mode="all_positive"))

    assert all(
        dataset.advantages[key] == float(dataset.actual_lookahead[key] > 0) for key in dataset.advantages
    )


def test_tail_and_intervention_overrides_have_expected_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scores = {(0, frame): float(frame) / 10 for frame in range(4)}
    dataset = _dataset(tmp_path, scores)
    _compute_actual_lookahead([dataset], max_temporal_offset=3)
    args = _args(
        expert_mode="threshold",
        expert_threshold=1.0,
        tail_positive_frames=2,
        tail_positive_sources=["expert"],
        force_intervention_positive=True,
        intervention_positive_sources=["expert"],
    )
    _assign_base_labels([dataset], args)
    monkeypatch.setattr(
        relabel_module,
        "_load_intervention_keys",
        lambda dataset, *, value_threshold: {(0, 0), (0, 2)},
    )
    _apply_positive_overrides([dataset], args)

    assert dataset.advantages == {
        (0, 0): 1.0,
        (0, 1): 0.0,
        (0, 2): 1.0,
        (0, 3): 1.0,
    }
    assert dataset.advantage_sources[(0, 2)] == "relabel_intervention_positive"
    assert dataset.tail_overrides == {(0, 2), (0, 3)}
    assert dataset.intervention_overrides == {(0, 0), (0, 2)}


def test_preserved_labels_only_change_requested_positive_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scores = {(0, frame): float(frame) / 10 for frame in range(5)}
    dataset = _dataset(tmp_path, scores)
    dataset.old_advantages = {
        (0, 0): 0.0,
        (0, 1): 1.0,
        (0, 2): 0.0,
        (0, 3): 0.0,
        (0, 4): 0.0,
    }
    dataset.old_sources = dict.fromkeys(scores, "steam_expert_quantile_rlinf_signed_strict")
    _compute_actual_lookahead([dataset], max_temporal_offset=3)
    args = _args(
        tail_positive_frames=2,
        tail_positive_sources=["expert"],
        force_intervention_positive=True,
        intervention_positive_sources=["expert"],
    )

    _assign_preserved_labels([dataset])
    monkeypatch.setattr(
        relabel_module,
        "_load_intervention_keys",
        lambda dataset, *, value_threshold: {(0, 2), (0, 3)},
    )
    _apply_positive_overrides([dataset], args)

    assert dataset.base_advantages == dataset.old_advantages
    assert dataset.advantages == {
        (0, 0): 0.0,
        (0, 1): 1.0,
        (0, 2): 1.0,
        (0, 3): 1.0,
        (0, 4): 1.0,
    }
    assert dataset.advantage_sources[(0, 0)] == "steam_expert_quantile_rlinf_signed_strict"
    assert dataset.advantage_sources[(0, 3)] == "relabel_intervention_positive"
    assert dataset.advantage_sources[(0, 4)] == "relabel_tail_positive"
    assert dataset.tail_overrides == {(0, 3), (0, 4)}
    assert dataset.intervention_overrides == {(0, 2), (0, 3)}


def test_intervention_keys_are_read_from_explicit_mapped_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    scores = {(0, 0): 0.2, (0, 1): 0.1}
    dataset = _dataset(tmp_path, scores)
    dataset.config.data_features_name_mapping = {"intervention": "human_intervention"}
    dataset.root.mkdir(parents=True)
    data_path = dataset.root / "data.parquet"
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "human_intervention": [0.0, 1.0],
        }
    ).to_parquet(data_path, index=False)

    class _FakeMetadata:
        def __init__(self, *args, **kwargs):
            assert kwargs["local_files_only"] is True

        def get_data_file_path(self, ep_index: int) -> Path:
            assert ep_index == 0
            return Path("data.parquet")

    monkeypatch.setattr(relabel_module, "LeRobotDatasetMetadata", _FakeMetadata)

    assert relabel_module._load_intervention_keys(dataset, value_threshold=0.0) == {(0, 1)}


def test_output_parquet_columns_preserve_raw_score_and_audit_overrides(tmp_path: Path):
    scores = {(0, 0): 0.2, (0, 1): 0.0}
    dataset = _dataset(tmp_path, scores)
    _compute_actual_lookahead([dataset], max_temporal_offset=3)
    args = _args(expert_mode="all_positive", tail_positive_frames=1)
    _assign_base_labels([dataset], args)
    _apply_positive_overrides([dataset], args)

    output = _output_dataframe(dataset, "recovery")

    assert output["advantage_continuous"].tolist() == [0.2, 0.0]
    assert output["advantage"].tolist() == [True, True]
    assert output["tail_positive_override"].tolist() == [False, True]
    assert output["actual_lookahead"].tolist() == [1, 0]
    assert output["tag"].unique().tolist() == ["recovery"]


def test_effective_bundle_update_preserves_raw_scores_and_report(tmp_path: Path):
    root = tmp_path / "dataset"
    keys = [(0, 0), (0, 1)]
    raw = {(0, 0): 0.2, (0, 1): 0.0}
    old = {(0, 0): 0.0, (0, 1): 0.0}
    old_sources = dict.fromkeys(keys, "steam_quantile")
    persist_advantage_bundle(
        root,
        keys,
        old,
        raw,
        old_sources,
        dict.fromkeys(keys, "task"),
    )
    raw_before = (root / RAW_ADVANTAGES_PATH).read_bytes()
    report_before = (root / ADVANTAGE_REPORT_PATH).read_bytes()
    dataset = _dataset(tmp_path, raw)
    dataset.root = root
    dataset.advantages = {(0, 0): 1.0, (0, 1): 1.0}
    dataset.advantage_sources = dict.fromkeys(keys, "relabel_tail_positive")

    _persist_effective_bundle(dataset)

    assert load_advantages(root) == {(0, 0): 1.0, (0, 1): 1.0}
    assert load_advantage_sources_from_path(root / ADVANTAGE_SOURCES_PATH) == dict.fromkeys(
        keys, "relabel_tail_positive"
    )
    assert load_advantages_from_path(root / RAW_ADVANTAGES_PATH) == raw
    assert (root / RAW_ADVANTAGES_PATH).read_bytes() == raw_before
    assert (root / ADVANTAGE_REPORT_PATH).read_bytes() == report_before
    assert json.loads((root / ADVANTAGE_REPORT_PATH).read_text(encoding="utf-8"))["coverage"] == 1.0


def test_relabel_end_to_end_supports_dry_run_then_persist(tmp_path: Path):
    root = tmp_path / "dataset"
    keys = [(0, 0), (0, 1)]
    raw = {(0, 0): 0.2, (0, 1): 0.0}
    old = {(0, 0): 1.0, (0, 1): 0.0}
    persist_advantage_bundle(
        root,
        keys,
        old,
        raw,
        dict.fromkeys(keys, "steam_quantile"),
        dict.fromkeys(keys, "task"),
    )
    pd.DataFrame(
        {
            "episode_index": [0, 0],
            "frame_index": [0, 1],
            "advantage": [False, False],
            "advantage_continuous": [0.2, 0.0],
            "max_temporal_offset": [3, 3],
        }
    ).to_parquet(root / "meta" / "advantages_steam.parquet", index=False)
    mixture_path = tmp_path / "mixture.json"
    mixture_path.write_text(
        json.dumps(
            {
                "datasets": [
                    {
                        "repo_id": "test/relabel-e2e",
                        "root": str(root),
                        "steam_source": "expert",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    arguments = [
        "--dataset-mixture",
        str(mixture_path),
        "--output-tag",
        "recovery",
        "--preserve-existing-labels",
        "--tail-positive-frames",
        "1",
    ]

    dry_run = relabel_module.relabel(relabel_module.parse_args([*arguments, "--dry-run"]))

    assert dry_run["settings"]["preserve_existing_labels"] is True
    assert dry_run["datasets"][0]["positive_before_count"] == 1
    assert dry_run["datasets"][0]["base_positive_count"] == 1
    assert dry_run["datasets"][0]["final_positive_count"] == 2
    assert load_advantages(root) == old
    assert not (root / "meta" / "advantages_recovery.parquet").exists()

    result = relabel_module.relabel(relabel_module.parse_args(arguments))

    assert result["dry_run"] is False
    assert load_advantages(root) == dict.fromkeys(keys, 1.0)
    assert load_advantages_from_path(root / RAW_ADVANTAGES_PATH) == raw
    assert (root / "meta" / "advantages_recovery.parquet").is_file()
    assert (root / "meta" / "advantage_relabel_recovery.json").is_file()


def test_cli_defaults_are_safe_for_non_expert_tails():
    args = relabel_module.parse_args(["--dataset-mixture", "mixture.json", "--output-tag", "recovery"])

    assert args.expert_mode == "all_positive"
    assert args.non_expert_mode == "quantile"
    assert not args.preserve_existing_labels
    assert args.quantile_grouping == "actual_lookahead"
    assert args.tail_positive_sources == ["expert"]
    assert args.intervention_positive_sources == ["non_expert"]
    assert not args.force_intervention_positive
