#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
from datetime import datetime, timedelta

from scripts.edge_detect import detect_edge
from scripts.data_structures import PreprocessOutput, EdgeDetectionOutput


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_preprocess_output(n_ranges=20, n_times=720, dt_sec=60.0):
    """
    Build a minimal PreprocessOutput covering 12:00–24:00 with `n_times` bins.
    arr is (n_ranges, n_times) with a simple gradient so thresholds can be found.
    """
    date = datetime(2020, 1, 1)
    t0_ns = np.datetime64(date + timedelta(hours=12), 'ns').astype(np.int64)
    step_ns = int(dt_sec * 1e9)
    arr_times = (t0_ns + np.arange(n_times) * step_ns).view('datetime64[ns]')

    ranges_km = np.linspace(500, 2000, n_ranges)

    rng = np.random.default_rng(42)
    arr = rng.integers(1, 30, size=(n_ranges, n_times), dtype=np.uint8)

    return PreprocessOutput(
        arr=arr,
        arr_times=arr_times,
        ranges_km=ranges_km,
        Ts_sec=dt_sec,
        Ts_td=timedelta(seconds=dt_sec),
        intermediate={},
        meta={},
    )


# ---------------------------------------------------------------------------
# detect_edge — integration
# ---------------------------------------------------------------------------

DATE = datetime(2020, 1, 1)
EDGE_PARAMS = dict(
    qs=[0.4, 0.5, 0.6],
    lower_cutoff=2,
    select_min=True,
    exact_thresh=False,
    axis=0,
    lowess_window_size=30,
    max_abs_dev=200,
)


def test_detect_edge_returns_edge_detection_output():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
    assert isinstance(result, EdgeDetectionOutput)


def test_detect_edge_time_window_correct():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
    assert result.edge_times[0] == np.datetime64(DATE + timedelta(hours=12), 'ns')
    assert result.edge_times[-1] <= np.datetime64(DATE + timedelta(hours=24), 'ns')


def test_detect_edge_positions_length_matches_times():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
    assert len(result.edge_positions) == len(result.edge_times)


def test_detect_edge_positions_within_range_bounds():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
    assert result.edge_positions.min() >= pre.ranges_km.min() - 1
    assert result.edge_positions.max() <= pre.ranges_km.max() + 1


def test_detect_edge_quantile_output_correct():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
    n_qs = len(EDGE_PARAMS['qs'])
    assert result.quantile_lines.shape[1] == n_qs
    assert np.allclose(result.quantile_values, EDGE_PARAMS['qs'])


def test_detect_edge_meta_has_required_keys():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
    assert 'date' in result.meta
    assert 'time_limits' in result.meta
    assert 'xlim' in result.meta['time_limits']
    assert 'winlim' in result.meta['time_limits']


def test_detect_edge_intermediate_has_required_keys():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
    for key in ('preprocessed_arr', 'preprocessed_times',
                'edge_0_vals', 'edge_0_times',
                'edge_win_vals', 'edge_win_times', 'window_mask'):
        assert key in result.intermediate, f"Missing intermediate key: {key}"
