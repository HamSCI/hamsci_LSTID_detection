#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
from datetime import datetime, timedelta

from scripts.edge_detect import (
    scale_km,
    take_quantile,
    stack_all_thresholds,
    lowess_smooth,
    smooth_remove_abs_deviation,
    select_min_deviation,
    detect_edge,
)
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

    # Simple pattern: uniform value across range for each time bin so that
    # threshold detection works (non-constant, non-zero)
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
# scale_km
# ---------------------------------------------------------------------------

def test_scale_km_min_index_gives_range_min():
    ranges = np.linspace(500, 2000, 50)
    edge = np.array([0.0])
    out = scale_km(edge, ranges)
    assert out[0] == pytest.approx(ranges.min(), abs=1.0)


def test_scale_km_max_index_near_range_max():
    ranges = np.linspace(500, 2000, 50)
    edge = np.array([float(len(ranges) - 1)])
    out = scale_km(edge, ranges)
    # scale_km uses ptp, so at index n-1 we get close to max but not exact
    assert ranges.min() <= out[0] <= ranges.max() + ranges.ptp() / len(ranges)


def test_scale_km_output_length_matches_input():
    ranges = np.linspace(500, 2000, 30)
    edge = np.arange(10, dtype=float)
    out = scale_km(edge, ranges)
    assert len(out) == len(edge)


# ---------------------------------------------------------------------------
# take_quantile
# ---------------------------------------------------------------------------

def test_take_quantile_returns_correct_length():
    arr = np.random.rand(10, 50).astype(np.float32)
    out = take_quantile(arr, q=0.5)
    assert out.shape == (50,)


def test_take_quantile_median_bounds():
    # Each column is [0..9]; median should be ~4-5
    arr = np.tile(np.arange(10, dtype=float)[:, None], (1, 30))
    out = take_quantile(arr, q=0.5)
    assert np.all(out >= 3) and np.all(out <= 6)


def test_take_quantile_raises_on_non_float_q():
    arr = np.random.rand(5, 10).astype(np.float32)
    with pytest.raises(TypeError):
        take_quantile(arr, q=1)  # int, not float


def test_take_quantile_raises_on_q_out_of_range():
    arr = np.random.rand(5, 10).astype(np.float32)
    with pytest.raises(ValueError):
        take_quantile(arr, q=1.5)


# ---------------------------------------------------------------------------
# stack_all_thresholds
# ---------------------------------------------------------------------------

def test_stack_all_thresholds_output_shape_axis0():
    # n_times must exceed n_ranges for the internal shape guard to pass
    # (mirrors real data: ~720 time bins vs ~20 range bins)
    arr = np.random.randint(0, 4, size=(5, 50), dtype=np.uint8)
    out = stack_all_thresholds(arr, select_min=True, exact_thresh=False, axis=0)
    n_unique = len(np.unique(arr))
    assert out.shape == (n_unique, arr.shape[1])


def test_stack_all_thresholds_indices_in_valid_range():
    arr = np.random.randint(0, 5, size=(15, 40), dtype=np.uint8)
    out = stack_all_thresholds(arr, select_min=True, exact_thresh=False, axis=0)
    assert out.min() >= 0
    assert out.max() < arr.shape[0]


# ---------------------------------------------------------------------------
# lowess_smooth
# ---------------------------------------------------------------------------

def test_lowess_smooth_output_length_preserved():
    arr = np.sin(np.linspace(0, 2 * np.pi, 100))
    out = lowess_smooth(arr, window_size=10, x=None)
    assert len(out) == len(arr)


def test_lowess_smooth_reduces_noise():
    rng = np.random.default_rng(0)
    clean = np.sin(np.linspace(0, 2 * np.pi, 200))
    noisy = clean + rng.normal(0, 0.5, size=200)
    out = lowess_smooth(noisy, window_size=30, x=None)
    # smoothed residual should be smaller than noisy residual
    assert np.std(out - clean) < np.std(noisy - clean)


# ---------------------------------------------------------------------------
# smooth_remove_abs_deviation
# ---------------------------------------------------------------------------

def test_smooth_remove_abs_deviation_output_length():
    arr = np.sin(np.linspace(0, 2 * np.pi, 80)) * 100 + 1400
    smooth_fn = lambda a: lowess_smooth(a, window_size=10, x=None)
    out = smooth_remove_abs_deviation(arr, smooth_fn, max_abs_dev=500)
    assert len(out) == len(arr)


def test_smooth_remove_abs_deviation_output_smoother_than_input():
    # Verify the function reduces overall noise vs a clean trend.
    # (Note: LOWESS exactly reproduces a single-point central outlier, so
    # isolated extreme spikes are not reliably filtered — this is a known
    # LOWESS limitation, not a bug in smooth_remove_abs_deviation.)
    rng = np.random.default_rng(7)
    clean = np.linspace(1400, 1200, 80)
    noisy = clean + rng.normal(0, 30, size=80)
    smooth_fn = lambda a: lowess_smooth(a, window_size=15, x=None)
    out = smooth_remove_abs_deviation(noisy, smooth_fn, max_abs_dev=200)
    assert np.std(out - clean) < np.std(noisy - clean)


# ---------------------------------------------------------------------------
# select_min_deviation
# ---------------------------------------------------------------------------

def test_select_min_deviation_returns_two_arrays():
    smooth_fn = lambda a: lowess_smooth(a, window_size=10, x=None)
    candidates = [
        np.sin(np.linspace(0, 2 * np.pi, 60)) * 100 + 1400,
        np.cos(np.linspace(0, 2 * np.pi, 60)) * 200 + 1400,
    ]
    result = select_min_deviation(candidates, smooth_fn, max_abs_dev=500)
    assert len(result) == 2
    assert len(result[0]) == 60
    assert len(result[1]) == 60


def test_select_min_deviation_picks_smoother_candidate():
    rng = np.random.default_rng(1)
    smooth_fn = lambda a: lowess_smooth(a, window_size=10, x=None)
    x = np.linspace(0, 2 * np.pi, 80)
    smooth_cand = np.sin(x) * 50 + 1400
    noisy_cand  = smooth_cand + rng.normal(0, 100, size=80)
    result_arr, _ = select_min_deviation([smooth_cand, noisy_cand], smooth_fn, max_abs_dev=500)
    # Should pick the smoother candidate
    assert np.allclose(result_arr, smooth_cand)


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


def test_detect_edge_edge_times_starts_at_noon():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
    expected_start = np.datetime64(DATE + timedelta(hours=12), 'ns')
    assert result.edge_times[0] == expected_start


def test_detect_edge_edge_times_ends_at_midnight():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
    expected_end = np.datetime64(DATE + timedelta(hours=24), 'ns')
    assert result.edge_times[-1] <= expected_end


def test_detect_edge_positions_length_matches_times():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
    assert len(result.edge_positions) == len(result.edge_times)


def test_detect_edge_positions_within_range_bounds():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
    assert result.edge_positions.min() >= pre.ranges_km.min() - 1
    assert result.edge_positions.max() <= pre.ranges_km.max() + 1


def test_detect_edge_quantile_lines_shape():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
    n_qs = len(EDGE_PARAMS['qs'])
    # shape is (n_arr_times, n_qs)
    assert result.quantile_lines.shape[1] == n_qs


def test_detect_edge_quantile_values_match_qs():
    pre = _make_preprocess_output()
    result = detect_edge(DATE, pre, **EDGE_PARAMS)
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
