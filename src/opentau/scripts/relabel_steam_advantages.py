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

"""Relabel existing STEAM scores without model inference.

The script preserves raw_advantages.json, derives new binary labels separately
for expert and non-expert sources, then optionally forces trajectory-tail and
explicit human-intervention frames positive. The effective JSON bundle used by
PI0.5 is updated in place while a new tagged Parquet keeps the relabel audit.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import draccus
import pandas as pd
import pyarrow.parquet as pq

from opentau.configs.default import DatasetConfig, DatasetMixtureConfig
from opentau.configs.refs import resolve_refs_to_tempfile
from opentau.datasets.lerobot_dataset import LeRobotDatasetMetadata
from opentau.datasets.standard_data_format_mapping import resolve_feature_mapping
from opentau.datasets.utils import (
    ADVANTAGE_SOURCES_PATH,
    ADVANTAGES_PATH,
    RAW_ADVANTAGES_PATH,
    AdvantageKey,
    load_advantage_sources_from_path,
    load_advantages,
    load_advantages_from_path,
    serialize_advantage_key,
    validate_advantage_bundle_files,
)
from opentau.scripts.steam_advantage_pipeline import (
    _atomic_write_json,
    _atomic_write_parquet,
    normalized_dataset_root,
    validate_tag,
    validate_unique_dataset_roots,
)

SteamSource = Literal["expert", "non_expert"]
BaseMode = Literal["all_positive", "threshold", "quantile"]
QuantileGrouping = Literal["global", "actual_lookahead"]


@dataclass
class RelabelDataset:
    index: int
    config: DatasetConfig
    root: Path
    source: SteamSource
    dataframe: pd.DataFrame
    old_advantages: dict[AdvantageKey, float]
    raw_advantages: dict[AdvantageKey, float]
    old_sources: dict[AdvantageKey, str]
    last_frames: dict[int, int]
    actual_lookahead: dict[AdvantageKey, int] = field(default_factory=dict)
    advantages: dict[AdvantageKey, float] = field(default_factory=dict)
    advantage_sources: dict[AdvantageKey, str] = field(default_factory=dict)
    base_thresholds: dict[AdvantageKey, float] = field(default_factory=dict)
    base_advantages: dict[AdvantageKey, float] = field(default_factory=dict)
    base_sources: dict[AdvantageKey, str] = field(default_factory=dict)
    tail_overrides: set[AdvantageKey] = field(default_factory=set)
    intervention_overrides: set[AdvantageKey] = field(default_factory=set)


def _parse_dataset_mixture(path: Path) -> DatasetMixtureConfig:
    resolved = resolve_refs_to_tempfile(path)
    try:
        return draccus.parse(config_class=DatasetMixtureConfig, config_path=resolved, args=[])
    finally:
        resolved.unlink(missing_ok=True)


def _episode_last_frames(keys: set[AdvantageKey]) -> dict[int, int]:
    last_frames: dict[int, int] = {}
    for episode_index, frame_index in keys:
        last_frames[episode_index] = max(last_frames.get(episode_index, -1), frame_index)
    return last_frames


def _frame_mapping(dataframe: pd.DataFrame, column: str) -> dict[AdvantageKey, object]:
    return {
        (int(episode_index), int(frame_index)): value
        for episode_index, frame_index, value in zip(
            dataframe["episode_index"],
            dataframe["frame_index"],
            dataframe[column],
            strict=True,
        )
    }


def _infer_max_temporal_offset(datasets: list[RelabelDataset], requested: int | None) -> int:
    if requested is not None:
        if requested < 1:
            raise ValueError(f"max_temporal_offset must be >= 1, got {requested}.")
        return requested
    values = set()
    for dataset in datasets:
        if "max_temporal_offset" not in dataset.dataframe:
            raise ValueError(
                f"{dataset.root} source Parquet has no max_temporal_offset column; "
                "pass --max-temporal-offset explicitly."
            )
        values.update(int(value) for value in dataset.dataframe["max_temporal_offset"].dropna().unique())
    if len(values) != 1:
        raise ValueError(
            "Could not infer one max_temporal_offset across source Parquets; "
            f"found {sorted(values)}. Pass --max-temporal-offset explicitly."
        )
    return values.pop()


def _load_dataset(
    index: int,
    config: DatasetConfig,
    *,
    source_tag: str,
) -> RelabelDataset:
    if config.steam_source not in ("expert", "non_expert"):
        raise ValueError(f"Dataset {config.repo_id!r} must set steam_source to expert or non_expert.")
    root = normalized_dataset_root(config)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")

    old_advantages = load_advantages(root)
    validate_advantage_bundle_files(root, old_advantages)
    assert old_advantages is not None
    raw_advantages = load_advantages_from_path(
        root / RAW_ADVANTAGES_PATH,
        "raw advantage",
    )
    old_sources = load_advantage_sources_from_path(root / ADVANTAGE_SOURCES_PATH)

    source_path = root / "meta" / f"advantages_{source_tag}.parquet"
    if not source_path.is_file():
        raise FileNotFoundError(f"Source STEAM Parquet not found: {source_path}")
    dataframe = pd.read_parquet(source_path)
    required = {"episode_index", "frame_index", "advantage_continuous"}
    missing = required - set(dataframe.columns)
    if missing:
        raise ValueError(f"{source_path} is missing required columns {sorted(missing)}.")
    if dataframe.empty:
        raise ValueError(f"{source_path} is empty.")
    if dataframe.duplicated(subset=["episode_index", "frame_index"]).any():
        raise ValueError(f"{source_path} contains duplicate frame keys.")

    parquet_scores = {
        key: float(value) for key, value in _frame_mapping(dataframe, "advantage_continuous").items()
    }
    expected_keys = set(old_advantages)
    for name, mapping in (
        ("raw_advantages.json", raw_advantages),
        ("advantage_sources.json", old_sources),
        (source_path.name, parquet_scores),
    ):
        if set(mapping) != expected_keys:
            missing_keys = sorted(expected_keys - set(mapping))
            unexpected = sorted(set(mapping) - expected_keys)
            raise ValueError(
                f"{name} keys differ from advantages.json at {root}: "
                f"missing={missing_keys[:10]!r}, unexpected={unexpected[:10]!r}"
            )
    for key, raw_score in raw_advantages.items():
        if not math.isclose(raw_score, parquet_scores[key], rel_tol=0.0, abs_tol=5e-6):
            raise ValueError(
                f"Source Parquet score disagrees with raw_advantages.json at {root}, "
                f"key={key}: parquet={parquet_scores[key]}, json={raw_score}."
            )

    return RelabelDataset(
        index=index,
        config=config,
        root=root,
        source=config.steam_source,
        dataframe=dataframe,
        old_advantages=old_advantages,
        raw_advantages=raw_advantages,
        old_sources=old_sources,
        last_frames=_episode_last_frames(expected_keys),
    )


def _compute_actual_lookahead(datasets: list[RelabelDataset], max_temporal_offset: int) -> None:
    """Attach the real frame-pair offset, including the shortened episode tail."""
    for dataset in datasets:
        dataset.actual_lookahead = {
            key: min(max_temporal_offset, dataset.last_frames[key[0]] - key[1])
            for key in dataset.raw_advantages
        }


def _mode_for_source(args: argparse.Namespace, source: SteamSource) -> BaseMode:
    return args.expert_mode if source == "expert" else args.non_expert_mode


def _threshold_for_source(args: argparse.Namespace, source: SteamSource) -> float:
    return args.expert_threshold if source == "expert" else args.non_expert_threshold


def _fraction_for_source(args: argparse.Namespace, source: SteamSource) -> float:
    return args.expert_positive_fraction if source == "expert" else args.non_expert_positive_fraction


def _quantile_group(
    dataset: RelabelDataset,
    key: AdvantageKey,
    grouping: QuantileGrouping,
) -> tuple[SteamSource, int | None]:
    lookahead = dataset.actual_lookahead[key] if grouping == "actual_lookahead" else None
    return dataset.source, lookahead


def _assign_base_labels(datasets: list[RelabelDataset], args: argparse.Namespace) -> None:
    """Assign source-specific base labels before tail/intervention overrides."""
    quantile_groups: dict[tuple[SteamSource, int | None], list[tuple[RelabelDataset, AdvantageKey]]] = (
        defaultdict(list)
    )

    for dataset in datasets:
        mode = _mode_for_source(args, dataset.source)
        threshold = _threshold_for_source(args, dataset.source)
        for key, score in dataset.raw_advantages.items():
            lookahead = dataset.actual_lookahead[key]
            if lookahead == 0:
                dataset.advantages[key] = 0.0
                dataset.advantage_sources[key] = "relabel_terminal_default"
                dataset.base_thresholds[key] = math.nan
            elif mode == "all_positive":
                dataset.advantages[key] = 1.0
                dataset.advantage_sources[key] = f"relabel_{dataset.source}_all_positive"
                dataset.base_thresholds[key] = math.nan
            elif mode == "threshold":
                positive = score > threshold if args.threshold_comparison == "strict" else score >= threshold
                dataset.advantages[key] = float(positive)
                dataset.advantage_sources[key] = f"relabel_{dataset.source}_threshold"
                dataset.base_thresholds[key] = threshold
            else:
                quantile_groups[_quantile_group(dataset, key, args.quantile_grouping)].append((dataset, key))

    # Select an exact top fraction in every stratum. A deterministic key
    # tiebreak avoids strict-percentile ties silently selecting too few rows.
    for (source, lookahead), items in quantile_groups.items():
        fraction = _fraction_for_source(args, source)
        target_count = math.ceil(len(items) * fraction)
        ranked = sorted(
            items,
            key=lambda item: (
                -item[0].raw_advantages[item[1]],
                item[0].index,
                item[1][0],
                item[1][1],
            ),
        )
        selected = {(dataset.index, key) for dataset, key in ranked[:target_count]}
        boundary = (
            ranked[target_count - 1][0].raw_advantages[ranked[target_count - 1][1]]
            if target_count
            else math.nan
        )
        for dataset, key in items:
            dataset.advantages[key] = float((dataset.index, key) in selected)
            suffix = "actual_lookahead" if lookahead is not None else "global"
            dataset.advantage_sources[key] = f"relabel_{source}_quantile_{suffix}"
            dataset.base_thresholds[key] = float(boundary)

    for dataset in datasets:
        dataset.base_advantages = dataset.advantages.copy()
        dataset.base_sources = dataset.advantage_sources.copy()


def _resolve_intervention_column(dataset: RelabelDataset) -> str:
    mapping = dataset.config.data_features_name_mapping
    if mapping is None:
        assert dataset.config.repo_id is not None
        mapping = resolve_feature_mapping(dataset.config.repo_id, dataset.config.control_mode)
    column = mapping.get("intervention")
    if not column:
        raise ValueError(
            f"Dataset {dataset.config.repo_id!r} is selected for intervention overrides but "
            "has no explicit data_features_name_mapping['intervention'] role."
        )
    return column


def _load_intervention_keys(
    dataset: RelabelDataset,
    *,
    value_threshold: float,
) -> set[AdvantageKey]:
    """Read the explicitly mapped per-frame intervention signal from local Parquet."""
    assert dataset.config.repo_id is not None
    column = _resolve_intervention_column(dataset)
    meta = LeRobotDatasetMetadata(
        dataset.config.repo_id,
        root=dataset.root,
        revision=dataset.config.revision,
        local_files_only=True,
    )
    paths = sorted({dataset.root / meta.get_data_file_path(ep_index) for ep_index in dataset.last_frames})
    tables = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Intervention data Parquet not found: {path}")
        schema_names = pq.read_schema(path).names
        required = {"episode_index", "frame_index", column}
        missing = required - set(schema_names)
        if missing:
            raise ValueError(f"{path} is missing intervention columns {sorted(missing)}.")
        tables.append(pq.read_table(path, columns=sorted(required)).to_pandas())
    frame_table = pd.concat(tables, ignore_index=True)
    keys = set(dataset.raw_advantages)
    frame_table["_key"] = list(
        zip(
            frame_table["episode_index"].astype(int),
            frame_table["frame_index"].astype(int),
            strict=True,
        )
    )
    frame_table = frame_table[frame_table["_key"].isin(keys)]
    if frame_table["_key"].duplicated().any():
        raise ValueError(f"Duplicate intervention rows found under {dataset.root}.")
    available = set(frame_table["_key"])
    if available != keys:
        missing_keys = sorted(keys - available)
        raise ValueError(
            f"Intervention data does not cover all advantage keys at {dataset.root}; "
            f"missing={missing_keys[:10]!r}."
        )
    numeric = pd.to_numeric(frame_table[column], errors="raise")
    if not numeric.map(math.isfinite).all():
        raise ValueError(f"Mapped intervention column {column!r} contains non-finite values.")
    return set(frame_table.loc[numeric > value_threshold, "_key"])


def _apply_positive_overrides(datasets: list[RelabelDataset], args: argparse.Namespace) -> None:
    tail_sources = set(args.tail_positive_sources)
    intervention_sources = set(args.intervention_positive_sources)
    for dataset in datasets:
        if args.tail_positive_frames > 0 and dataset.source in tail_sources:
            for key in dataset.advantages:
                distance_from_end = dataset.last_frames[key[0]] - key[1]
                if distance_from_end < args.tail_positive_frames:
                    dataset.advantages[key] = 1.0
                    dataset.advantage_sources[key] = "relabel_tail_positive"
                    dataset.tail_overrides.add(key)

        if args.force_intervention_positive and dataset.source in intervention_sources:
            intervention_keys = _load_intervention_keys(
                dataset,
                value_threshold=args.intervention_value_threshold,
            )
            for key in intervention_keys:
                dataset.advantages[key] = 1.0
                dataset.advantage_sources[key] = "relabel_intervention_positive"
                dataset.intervention_overrides.add(key)


def _write_json_temp(final_path: Path, payload: object) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=final_path.parent,
        prefix=f".{final_path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _serialize_advantages(values: dict[AdvantageKey, float]) -> dict[str, str]:
    return {serialize_advantage_key(*key): f"{float(values[key]):.6f}" for key in sorted(values)}


def _serialize_sources(values: dict[AdvantageKey, str]) -> dict[str, str]:
    return {serialize_advantage_key(*key): values[key] for key in sorted(values)}


def _persist_effective_bundle(dataset: RelabelDataset) -> None:
    """Atomically replace only effective labels and sources; keep raw scores/report."""
    payloads = (
        (dataset.root / ADVANTAGES_PATH, _serialize_advantages(dataset.advantages)),
        (dataset.root / ADVANTAGE_SOURCES_PATH, _serialize_sources(dataset.advantage_sources)),
    )
    staged: list[tuple[Path, Path]] = []
    backups: dict[Path, Path] = {}
    originally_present = {path for path, _ in payloads if path.exists()}
    try:
        for path, payload in payloads:
            staged.append((path, _write_json_temp(path, payload)))
        try:
            for path, _ in payloads:
                if path not in originally_present:
                    continue
                fd, backup_name = tempfile.mkstemp(
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".backup",
                )
                os.close(fd)
                backup = Path(backup_name)
                try:
                    path.replace(backup)
                except BaseException:
                    backup.unlink(missing_ok=True)
                    raise
                backups[path] = backup
            for path, temporary in staged:
                temporary.replace(path)
            validate_advantage_bundle_files(dataset.root, load_advantages(dataset.root))
        except BaseException as transaction_error:
            rollback_errors = []
            for path, backup in reversed(backups.items()):
                try:
                    path.unlink(missing_ok=True)
                    backup.replace(path)
                except BaseException as rollback_error:
                    rollback_errors.append((path, rollback_error))
            for path, _ in payloads:
                if path not in originally_present:
                    try:
                        path.unlink(missing_ok=True)
                    except BaseException as rollback_error:
                        rollback_errors.append((path, rollback_error))
            if rollback_errors:
                details = ", ".join(f"{path}: {error}" for path, error in rollback_errors)
                raise RuntimeError(f"Effective advantage rollback failed: {details}") from transaction_error
            raise
        else:
            for backup in backups.values():
                backup.unlink(missing_ok=True)
    finally:
        for _, temporary in staged:
            temporary.unlink(missing_ok=True)
        for backup in backups.values():
            backup.unlink(missing_ok=True)


def _output_dataframe(dataset: RelabelDataset, output_tag: str) -> pd.DataFrame:
    dataframe = dataset.dataframe.copy()
    keys = list(
        zip(
            dataframe["episode_index"].astype(int),
            dataframe["frame_index"].astype(int),
            strict=True,
        )
    )
    dataframe["advantage_before_relabel"] = [bool(dataset.old_advantages[key]) for key in keys]
    dataframe["advantage_base_relabel"] = [bool(dataset.base_advantages[key]) for key in keys]
    dataframe["advantage"] = [bool(dataset.advantages[key]) for key in keys]
    dataframe["actual_lookahead"] = [dataset.actual_lookahead[key] for key in keys]
    dataframe["relabel_base_threshold"] = [dataset.base_thresholds[key] for key in keys]
    dataframe["advantage_source_before_relabel"] = [dataset.old_sources[key] for key in keys]
    dataframe["relabel_base_source"] = [dataset.base_sources[key] for key in keys]
    dataframe["relabel_source"] = [dataset.advantage_sources[key] for key in keys]
    dataframe["tail_positive_override"] = [key in dataset.tail_overrides for key in keys]
    dataframe["intervention_positive_override"] = [key in dataset.intervention_overrides for key in keys]
    dataframe["steam_source"] = dataset.source
    dataframe["tag"] = output_tag
    return dataframe


def _dataset_summary(dataset: RelabelDataset) -> dict[str, object]:
    keys = sorted(dataset.advantages)
    by_lookahead = {}
    for lookahead in sorted(set(dataset.actual_lookahead.values())):
        stratum = [key for key in keys if dataset.actual_lookahead[key] == lookahead]
        by_lookahead[str(lookahead)] = {
            "count": len(stratum),
            "base_positive_count": sum(bool(dataset.base_advantages[key]) for key in stratum),
            "final_positive_count": sum(bool(dataset.advantages[key]) for key in stratum),
        }
    return {
        "dataset_index": dataset.index,
        "repo_id": dataset.config.repo_id,
        "root": str(dataset.root),
        "steam_source": dataset.source,
        "frame_count": len(keys),
        "positive_before_count": sum(bool(dataset.old_advantages[key]) for key in keys),
        "base_positive_count": sum(bool(dataset.base_advantages[key]) for key in keys),
        "final_positive_count": sum(bool(dataset.advantages[key]) for key in keys),
        "tail_override_count": len(dataset.tail_overrides),
        "intervention_override_count": len(dataset.intervention_overrides),
        "by_actual_lookahead": by_lookahead,
    }


def _relabel_settings(args: argparse.Namespace, max_temporal_offset: int) -> dict[str, object]:
    return {
        "source_tag": args.source_tag,
        "output_tag": args.output_tag,
        "expert_mode": args.expert_mode,
        "non_expert_mode": args.non_expert_mode,
        "expert_threshold": args.expert_threshold,
        "non_expert_threshold": args.non_expert_threshold,
        "threshold_comparison": args.threshold_comparison,
        "expert_positive_fraction": args.expert_positive_fraction,
        "non_expert_positive_fraction": args.non_expert_positive_fraction,
        "quantile_grouping": args.quantile_grouping,
        "max_temporal_offset": max_temporal_offset,
        "tail_positive_frames": args.tail_positive_frames,
        "tail_positive_sources": args.tail_positive_sources,
        "force_intervention_positive": args.force_intervention_positive,
        "intervention_positive_sources": args.intervention_positive_sources,
        "intervention_value_threshold": args.intervention_value_threshold,
    }


def _preflight_outputs(
    datasets: list[RelabelDataset],
    *,
    source_tag: str,
    output_tag: str,
    overwrite: bool,
) -> None:
    if output_tag == source_tag:
        raise ValueError("output_tag must differ from source_tag so raw STEAM evidence is preserved.")
    if overwrite:
        return
    conflicts = []
    for dataset in datasets:
        for path in (
            dataset.root / "meta" / f"advantages_{output_tag}.parquet",
            dataset.root / "meta" / f"advantage_relabel_{output_tag}.json",
        ):
            if path.exists():
                conflicts.append(path)
    if conflicts:
        raise FileExistsError(
            f"Relabel outputs already exist: {[str(path) for path in conflicts]!r}; "
            "use --overwrite-output to replace them."
        )


def _persist_outputs(
    datasets: list[RelabelDataset],
    *,
    output_tag: str,
    settings: dict[str, object],
) -> None:
    for dataset in datasets:
        summary = _dataset_summary(dataset)
        _atomic_write_parquet(
            dataset.root / "meta" / f"advantages_{output_tag}.parquet",
            _output_dataframe(dataset, output_tag),
        )
        _atomic_write_json(
            dataset.root / "meta" / f"advantage_relabel_{output_tag}.json",
            {"schema_version": 1, "settings": settings, "summary": summary},
        )
        _persist_effective_bundle(dataset)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("expert_positive_fraction", "non_expert_positive_fraction"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}.")
    for name in ("expert_threshold", "non_expert_threshold", "intervention_value_threshold"):
        value = getattr(args, name)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}.")
    if args.tail_positive_frames < 0:
        raise ValueError(f"tail_positive_frames must be >= 0, got {args.tail_positive_frames}.")


def relabel(args: argparse.Namespace) -> dict[str, object]:
    """Run one CPU-only relabel pass and return its diagnostics."""
    _validate_args(args)
    source_tag = validate_tag(args.source_tag)
    output_tag = validate_tag(args.output_tag)
    mixture = _parse_dataset_mixture(args.dataset_mixture)
    if not mixture.datasets:
        raise ValueError("Dataset mixture must contain at least one dataset.")
    validate_unique_dataset_roots(mixture)
    datasets = [
        _load_dataset(index, config, source_tag=source_tag) for index, config in enumerate(mixture.datasets)
    ]
    max_temporal_offset = _infer_max_temporal_offset(datasets, args.max_temporal_offset)
    _compute_actual_lookahead(datasets, max_temporal_offset)
    _assign_base_labels(datasets, args)
    _apply_positive_overrides(datasets, args)
    _preflight_outputs(
        datasets,
        source_tag=source_tag,
        output_tag=output_tag,
        overwrite=args.overwrite_output,
    )
    settings = _relabel_settings(args, max_temporal_offset)
    diagnostics = {
        "schema_version": 1,
        "dry_run": bool(args.dry_run),
        "settings": settings,
        "datasets": [_dataset_summary(dataset) for dataset in datasets],
    }
    if not args.dry_run:
        _persist_outputs(datasets, output_tag=output_tag, settings=settings)
    return diagnostics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Relabel existing STEAM advantage scores on CPU; no checkpoint or model inference is used."
        )
    )
    parser.add_argument(
        "--dataset-mixture",
        type=Path,
        required=True,
        help="DatasetMixtureConfig JSON used for STEAM generation and training.",
    )
    parser.add_argument(
        "--source-tag",
        default="steam",
        help="Existing meta/advantages_<tag>.parquet tag (default: steam).",
    )
    parser.add_argument(
        "--output-tag",
        required=True,
        help="New audit Parquet/diagnostics tag; must differ from source-tag.",
    )
    parser.add_argument(
        "--expert-mode",
        choices=("all_positive", "threshold", "quantile"),
        default="all_positive",
        help="Base relabel rule for non-terminal expert frames.",
    )
    parser.add_argument(
        "--non-expert-mode",
        choices=("all_positive", "threshold", "quantile"),
        default="quantile",
        help="Base relabel rule for non-terminal rollout/non-expert frames.",
    )
    parser.add_argument("--expert-threshold", type=float, default=0.0)
    parser.add_argument("--non-expert-threshold", type=float, default=0.0)
    parser.add_argument(
        "--threshold-comparison",
        choices=("strict", "inclusive"),
        default="strict",
        help="Use score > threshold (strict) or score >= threshold (inclusive).",
    )
    parser.add_argument(
        "--expert-positive-fraction",
        type=float,
        default=0.8,
        help="Exact top fraction used only when expert-mode=quantile.",
    )
    parser.add_argument(
        "--non-expert-positive-fraction",
        type=float,
        default=0.3,
        help="Exact top fraction used only when non-expert-mode=quantile.",
    )
    parser.add_argument(
        "--quantile-grouping",
        choices=("global", "actual_lookahead"),
        default="actual_lookahead",
        help="Quantile strata; actual_lookahead protects shortened tail offsets.",
    )
    parser.add_argument(
        "--max-temporal-offset",
        type=int,
        default=None,
        help="STEAM K; inferred from source Parquet when omitted.",
    )
    parser.add_argument(
        "--tail-positive-frames",
        type=int,
        default=0,
        help="Force the last N frame keys per selected trajectory positive; 0 disables.",
    )
    parser.add_argument(
        "--tail-positive-sources",
        nargs="+",
        choices=("expert", "non_expert"),
        default=["expert"],
        help="Sources eligible for tail overrides (default: expert only).",
    )
    parser.add_argument(
        "--force-intervention-positive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Force explicitly mapped intervention frames positive.",
    )
    parser.add_argument(
        "--intervention-positive-sources",
        nargs="+",
        choices=("expert", "non_expert"),
        default=["non_expert"],
        help="Sources eligible for intervention overrides (default: non_expert).",
    )
    parser.add_argument(
        "--intervention-value-threshold",
        type=float,
        default=0.0,
        help="Mapped intervention value must be greater than this threshold.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Replace an existing tagged audit Parquet/diagnostics pair.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print counts without writing any file.",
    )
    return parser.parse_args(argv)


def cli(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    diagnostics = relabel(parse_args(argv))
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    cli()
