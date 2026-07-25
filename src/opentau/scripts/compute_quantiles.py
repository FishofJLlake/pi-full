#!/usr/bin/env python

# Copyright 2026 Tensor Auto Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Generate robust q01/q99 sidecar statistics for local LeRobot datasets."""

from __future__ import annotations

import copy
import logging
import sys
from dataclasses import dataclass

import draccus
import numpy as np

from opentau.configs import parser
from opentau.configs.default import DatasetMixtureConfig
from opentau.configs.refs import resolve_refs_to_tempfile
from opentau.configs.train import TrainPipelineConfig
from opentau.datasets.factory import make_dataset
from opentau.datasets.utils import STATS_QUANTILES_PATH, serialize_dict, write_json
from opentau.utils.utils import init_logging

_BATCH_SIZE = 1_000
_DEFAULT_APPROX_MAX_SAMPLES = 100_000
_DEFAULT_APPROX_MAX_BYTES = 512 * 1024 * 1024
_DEFAULT_EXACT_MAX_BYTES = 2 * 1024 * 1024 * 1024


class QuantileMemoryLimitExceeded(MemoryError):  # noqa: N818
    """Raised before allocating a quantile buffer larger than its configured cap."""


@dataclass(frozen=True)
class ScriptOptions:
    dataset_mixture_path: str | None = None
    approximate: bool = False
    sample_rate: float | None = None
    seed: int = 0
    max_samples: int = _DEFAULT_APPROX_MAX_SAMPLES
    approximate_max_bytes: int | None = _DEFAULT_APPROX_MAX_BYTES
    exact_max_bytes: int | None = _DEFAULT_EXACT_MAX_BYTES


def _parse_bool(value: str, option: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{option} expects true/false, got {value!r}.")


def _parse_positive_int(value: str, option: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{option} must be greater than zero.")
    return parsed


def _parse_optional_bytes(value: str, option: str) -> int | None:
    if value.strip().lower() in {"none", "null", "inf", "infinity", "-1"}:
        return None
    return _parse_positive_int(value, option)


def _option_value(argv: list[str], index: int, option: str) -> tuple[str, int]:
    argument = argv[index]
    if "=" in argument:
        return argument.split("=", 1)[1], index + 1
    if index + 1 >= len(argv):
        raise ValueError(f"Missing value for {option}.")
    return argv[index + 1], index + 2


def parse_script_options(argv: list[str]) -> tuple[ScriptOptions, list[str]]:
    values = ScriptOptions().__dict__.copy()
    filtered = [argv[0]] if argv else []
    index = 1
    while index < len(argv):
        argument = argv[index]
        option = argument.split("=", 1)[0]
        if option in {"--dataset_mixture", "--dataset_mixture_path"}:
            values["dataset_mixture_path"], index = _option_value(argv, index, option)
        elif option == "--approx_quantiles":
            if "=" in argument:
                raw, index = _option_value(argv, index, option)
                values["approximate"] = _parse_bool(raw, option)
            elif index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                values["approximate"] = _parse_bool(argv[index + 1], option)
                index += 2
            else:
                values["approximate"] = True
                index += 1
        elif option == "--approx_quantiles_sample_rate":
            raw, index = _option_value(argv, index, option)
            rate = float(raw)
            if not 0.0 < rate <= 1.0:
                raise ValueError(f"{option} must be in (0, 1].")
            values["sample_rate"] = rate
        elif option == "--approx_quantiles_seed":
            raw, index = _option_value(argv, index, option)
            values["seed"] = int(raw)
        elif option == "--approx_quantiles_max_samples":
            raw, index = _option_value(argv, index, option)
            values["max_samples"] = _parse_positive_int(raw, option)
        elif option == "--approx_quantiles_max_bytes":
            raw, index = _option_value(argv, index, option)
            values["approximate_max_bytes"] = _parse_optional_bytes(raw, option)
        elif option == "--exact_quantiles_max_bytes":
            raw, index = _option_value(argv, index, option)
            values["exact_max_bytes"] = _parse_optional_bytes(raw, option)
        else:
            filtered.append(argument)
            index += 1
    return ScriptOptions(**values), filtered


_OPTIONS, _FILTERED_ARGV = parse_script_options(sys.argv)
_ORIGINAL_WRAP = parser.wrap()


def _normalize_batch(values) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array[:, None]
    return array


def _required_bytes(rows: int, feature_shape: tuple[int, ...], dtype: np.dtype) -> int:
    values_per_row = int(np.prod(feature_shape, dtype=np.int64))
    return rows * max(1, values_per_row) * dtype.itemsize


def collect_exact(
    raw_dataset,
    feature_key: str,
    *,
    max_bytes: int | None,
    batch_size: int = _BATCH_SIZE,
) -> np.ndarray | None:
    """Collect every row after checking the exact allocation size."""
    total_rows = len(raw_dataset)
    if total_rows == 0:
        return None
    output = None
    cursor = 0
    for batch in raw_dataset.select_columns([feature_key]).iter(batch_size=batch_size):
        array = _normalize_batch(batch[feature_key])
        if output is None:
            required = _required_bytes(total_rows, array.shape[1:], array.dtype)
            if max_bytes is not None and required > max_bytes:
                raise QuantileMemoryLimitExceeded(
                    f"Exact quantiles for {feature_key!r} require {required} bytes, "
                    f"exceeding the configured limit {max_bytes}."
                )
            output = np.empty((total_rows, *array.shape[1:]), dtype=array.dtype)
        elif array.shape[1:] != output.shape[1:]:
            raise ValueError(f"Inconsistent trailing shape for {feature_key!r}.")
        end = cursor + array.shape[0]
        if end > total_rows:
            raise ValueError(f"Collected more rows than declared for {feature_key!r}.")
        output[cursor:end] = array
        cursor = end
    if output is None or cursor == 0:
        return None
    if cursor != total_rows:
        raise ValueError(f"Collected {cursor}/{total_rows} rows for {feature_key!r}.")
    return output


def collect_reservoir(
    raw_dataset,
    feature_key: str,
    *,
    rng: np.random.Generator,
    max_samples: int,
    max_bytes: int | None,
    sample_rate: float | None = None,
    batch_size: int = _BATCH_SIZE,
) -> np.ndarray | None:
    """Uniformly sample rows with a deterministic fixed-size reservoir."""
    total_rows = len(raw_dataset)
    if total_rows == 0:
        return None
    target = min(total_rows, max_samples)
    if sample_rate is not None:
        target = min(target, max(1, int(np.ceil(total_rows * sample_rate))))

    reservoir = None
    seen = 0
    filled = 0
    for batch in raw_dataset.select_columns([feature_key]).iter(batch_size=batch_size):
        array = _normalize_batch(batch[feature_key])
        if reservoir is None:
            row_bytes = _required_bytes(1, array.shape[1:], array.dtype)
            if max_bytes is not None:
                byte_limited = max_bytes // max(1, row_bytes)
                if byte_limited < 1:
                    raise QuantileMemoryLimitExceeded(
                        f"One {feature_key!r} row requires {row_bytes} bytes, exceeding "
                        f"the approximate limit {max_bytes}."
                    )
                target = min(target, byte_limited)
            reservoir = np.empty((target, *array.shape[1:]), dtype=array.dtype)
        elif array.shape[1:] != reservoir.shape[1:]:
            raise ValueError(f"Inconsistent trailing shape for {feature_key!r}.")

        for row in array:
            seen += 1
            if filled < target:
                reservoir[filled] = row
                filled += 1
                continue
            slot = int(rng.integers(0, seen))
            if slot < target:
                reservoir[slot] = row
    if reservoir is None or filled == 0:
        return None
    return reservoir[:filled]


def compute_feature_quantiles(
    raw_dataset,
    feature_key: str,
    *,
    options: ScriptOptions,
    rng: np.random.Generator,
) -> dict[str, np.ndarray] | None:
    if feature_key not in raw_dataset.column_names:
        return None
    if options.approximate:
        values = collect_reservoir(
            raw_dataset,
            feature_key,
            rng=rng,
            max_samples=options.max_samples,
            max_bytes=options.approximate_max_bytes,
            sample_rate=options.sample_rate,
        )
    else:
        values = collect_exact(raw_dataset, feature_key, max_bytes=options.exact_max_bytes)
    if values is None:
        return None
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        raise ValueError(f"Quantile feature {feature_key!r} must contain only finite numeric values.")
    q01, q99 = np.percentile(values, [1.0, 99.0], axis=0)
    logging.info("Computed %s quantiles from %d rows", feature_key, values.shape[0])
    return {"q01": q01, "q99": q99}


def _filter_script_options(function):
    wrapped = _ORIGINAL_WRAP(function)

    def filtered_wrapper(*args, **kwargs):
        if args:
            return wrapped(*args, **kwargs)
        original = sys.argv
        try:
            sys.argv = _FILTERED_ARGV.copy()
            return wrapped(*args, **kwargs)
        finally:
            sys.argv = original

    return filtered_wrapper


def _unwrap_dataset(dataset):
    current = dataset[0] if isinstance(dataset, tuple) else dataset
    for _ in range(8):
        if all(hasattr(current, name) for name in ("hf_dataset", "_get_name_map", "root")):
            return current
        next_dataset = getattr(current, "dataset", None)
        if next_dataset is None:
            next_dataset = getattr(current, "base_dataset", None)
        if next_dataset is None or next_dataset is current:
            break
        current = next_dataset
    raise TypeError(f"Cannot unwrap quantile dataset from {type(dataset).__name__}.")


@_filter_script_options
def main(cfg: TrainPipelineConfig) -> None:
    if _OPTIONS.dataset_mixture_path:
        resolved = resolve_refs_to_tempfile(_OPTIONS.dataset_mixture_path)
        try:
            mixture = draccus.parse(
                config_class=DatasetMixtureConfig,
                config_path=str(resolved),
                args=[],
            )
        finally:
            resolved.unlink(missing_ok=True)
    else:
        mixture = cfg.dataset_mixture

    # Quantiles describe the complete local dataset, not the training half of
    # a train/validation split. Keep the caller's config immutable.
    runtime_cfg = copy.deepcopy(cfg)
    runtime_cfg.val_freq = 0
    for dataset_index, dataset_config in enumerate(mixture.datasets):
        dataset = _unwrap_dataset(make_dataset(dataset_config, runtime_cfg))
        raw_dataset = dataset.hf_dataset.with_transform(None).with_format("numpy")
        mapping = dataset._get_name_map(strict=False)
        feature_keys = (
            mapping.get("state", "observation.state"),
            mapping.get("actions", "action"),
        )
        rng = np.random.default_rng(_OPTIONS.seed + dataset_index)
        quantiles = {}
        for feature_key in feature_keys:
            result = compute_feature_quantiles(
                raw_dataset,
                feature_key,
                options=_OPTIONS,
                rng=rng,
            )
            if result is not None:
                quantiles[feature_key] = result
        if not quantiles:
            logging.warning("No state/action quantiles were generated for %s", dataset.root)
            continue
        output_path = dataset.root / STATS_QUANTILES_PATH
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_json(serialize_dict(quantiles), output_path)
        logging.info("Saved quantile sidecar to %s", output_path)


if __name__ == "__main__":
    init_logging()
    main()
