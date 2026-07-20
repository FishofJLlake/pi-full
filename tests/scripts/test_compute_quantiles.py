import numpy as np
import pytest

from opentau.scripts.compute_quantiles import (
    QuantileMemoryLimitExceeded,
    ScriptOptions,
    collect_exact,
    collect_reservoir,
    compute_feature_quantiles,
    parse_script_options,
)


class _FakeColumnDataset:
    def __init__(self, key: str, values: np.ndarray):
        self.key = key
        self.values = values
        self.column_names = [key]

    def __len__(self):
        return len(self.values)

    def select_columns(self, columns):
        assert columns == [self.key]
        return self

    def iter(self, batch_size):
        for start in range(0, len(self), batch_size):
            yield {self.key: self.values[start : start + batch_size]}


def test_exact_quantiles_match_numpy():
    values = np.arange(200, dtype=np.float32).reshape(100, 2)
    dataset = _FakeColumnDataset("state", values)

    result = compute_feature_quantiles(
        dataset,
        "state",
        options=ScriptOptions(exact_max_bytes=values.nbytes),
        rng=np.random.default_rng(0),
    )

    expected = np.percentile(values, [1, 99], axis=0)
    np.testing.assert_allclose(result["q01"], expected[0])
    np.testing.assert_allclose(result["q99"], expected[1])


def test_exact_collection_checks_memory_before_allocation():
    values = np.arange(40, dtype=np.float32).reshape(10, 4)
    dataset = _FakeColumnDataset("state", values)

    with pytest.raises(QuantileMemoryLimitExceeded, match="require"):
        collect_exact(dataset, "state", max_bytes=values.nbytes - 1)


def test_reservoir_is_seeded_deterministic_and_bounded():
    values = np.arange(1_000, dtype=np.float32).reshape(500, 2)
    dataset = _FakeColumnDataset("state", values)

    first = collect_reservoir(
        dataset,
        "state",
        rng=np.random.default_rng(17),
        max_samples=23,
        max_bytes=23 * 2 * values.dtype.itemsize,
    )
    second = collect_reservoir(
        dataset,
        "state",
        rng=np.random.default_rng(17),
        max_samples=23,
        max_bytes=23 * 2 * values.dtype.itemsize,
    )

    assert first.shape == (23, 2)
    np.testing.assert_array_equal(first, second)


def test_script_options_support_reservoir_limits_and_preserve_train_args():
    options, filtered = parse_script_options(
        [
            "compute_quantiles.py",
            "--approx_quantiles",
            "--approx_quantiles_max_samples=50",
            "--approx_quantiles_max_bytes",
            "4096",
            "--config_path=train.json",
        ]
    )

    assert options.approximate is True
    assert options.max_samples == 50
    assert options.approximate_max_bytes == 4096
    assert filtered == ["compute_quantiles.py", "--config_path=train.json"]
