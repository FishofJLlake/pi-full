import numpy as np

from opentau.datasets.utils import merge_missing_stats_quantiles


def test_sidecar_only_fills_missing_quantiles():
    stats = {
        "state": {
            "mean": np.array([1.0]),
            "std": np.array([2.0]),
            "min": np.array([-1.0]),
            "max": np.array([3.0]),
            "q01": np.array([-0.5]),
        }
    }
    sidecar = {
        "state": {
            "mean": np.array([99.0]),
            "q01": np.array([-9.0]),
            "q99": np.array([2.5]),
        }
    }

    merged = merge_missing_stats_quantiles(stats, sidecar)

    np.testing.assert_array_equal(merged["state"]["mean"], [1.0])
    np.testing.assert_array_equal(merged["state"]["std"], [2.0])
    np.testing.assert_array_equal(merged["state"]["min"], [-1.0])
    np.testing.assert_array_equal(merged["state"]["max"], [3.0])
    np.testing.assert_array_equal(merged["state"]["q01"], [-0.5])
    np.testing.assert_array_equal(merged["state"]["q99"], [2.5])


def test_missing_sidecar_keeps_existing_stats_object():
    stats = {"state": {"mean": np.array([1.0])}}
    assert merge_missing_stats_quantiles(stats, None) is stats
