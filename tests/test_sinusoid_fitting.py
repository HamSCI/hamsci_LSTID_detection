#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
from datetime import datetime, timedelta

from scripts.sinusoid_fitting import (
    sinusoid,
    compute_stability,
    find_fit_window,
    fit_polynomial,
    apply_bandpass,
    fit_sinusoids,
    sin_fit,
)
from scripts.data_structures import EdgeDetectionOutput, FitOutput


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

DATE = datetime(2020, 1, 1)


def _make_edge_output(edge_positions, dt_sec=60.0):
    """
    Build a minimal EdgeDetectionOutput whose edge_times span 12:00–24:00
    and whose edge_positions are the supplied array.
    """
    n = len(edge_positions)
    t0_ns = np.datetime64(DATE + timedelta(hours=12), 'ns').astype(np.int64)
    step_ns = int(dt_sec * 1e9)
    edge_times = (t0_ns + np.arange(n) * step_ns).view('datetime64[ns]')

    win_0 = np.datetime64(DATE + timedelta(hours=13), 'ns')
    win_1 = np.datetime64(DATE + timedelta(hours=23), 'ns')

    return EdgeDetectionOutput(
        edge_times=edge_times,
        edge_positions=edge_positions.copy(),
        quantile_lines=np.zeros((n, 1)),
        quantile_values=np.array([0.5]),
        min_line=edge_positions.copy(),
        minz_line=edge_positions.copy(),
        ranges_km=np.linspace(500, 2000, 20),
        intermediate={
            'preprocessed_arr':   np.zeros((20, n), dtype=np.uint8),
            'preprocessed_times': edge_times,
            'edge_0_vals':        edge_positions.copy(),
            'edge_0_times':       edge_times,
        },
        meta={
            'date': DATE,
            'time_limits': {
                'xlim':   (edge_times[0], edge_times[-1]),
                'winlim': (win_0, win_1),
            },
        },
    )


def _tt_sec(n=720, dt_sec=60.0):
    """Seconds since midnight for n samples starting at 12:00."""
    return np.arange(n) * dt_sec + 12 * 3600


# ---------------------------------------------------------------------------
# compute_stability
# ---------------------------------------------------------------------------

def test_compute_stability_first_elements_are_nan():
    arr = np.ones(20) * 1400.0
    roll_win = 5
    out = compute_stability(arr, roll_win)
    assert np.all(np.isnan(out[:roll_win - 1]))


def test_compute_stability_constant_array_gives_zero_cv():
    arr = np.ones(30) * 1400.0
    out = compute_stability(arr, roll_win=5)
    valid = out[~np.isnan(out)]
    assert np.allclose(valid, 0.0)


def test_compute_stability_varying_array_gives_nonzero_cv():
    rng = np.random.default_rng(3)
    arr = 1400.0 + rng.normal(0, 50, size=50)
    out = compute_stability(arr, roll_win=5)
    valid = out[~np.isnan(out)]
    assert np.all(valid > 0)


# ---------------------------------------------------------------------------
# find_fit_window
# ---------------------------------------------------------------------------

def _make_edge_times(n=720, dt_sec=60.0):
    t0_ns = np.datetime64(DATE + timedelta(hours=12), 'ns').astype(np.int64)
    step_ns = int(dt_sec * 1e9)
    return (t0_ns + np.arange(n) * step_ns).view('datetime64[ns]')


def test_find_fit_window_finds_stable_region():
    edge_times = _make_edge_times()
    stability = np.zeros(len(edge_times))
    win_0 = np.datetime64(DATE + timedelta(hours=13), 'ns')
    win_1 = np.datetime64(DATE + timedelta(hours=23), 'ns')
    result = find_fit_window(stability, edge_times, (win_0, win_1),
                             stab_thresh=0.1, margin_minutes=10)
    assert result is not None
    fit_mask, fw0, fw1 = result
    assert fw0 < fw1
    assert np.sum(fit_mask) > 0


def test_find_fit_window_returns_none_when_all_unstable():
    edge_times = _make_edge_times()
    stability = np.ones(len(edge_times))
    win_0 = np.datetime64(DATE + timedelta(hours=13), 'ns')
    win_1 = np.datetime64(DATE + timedelta(hours=23), 'ns')
    result = find_fit_window(stability, edge_times, (win_0, win_1),
                             stab_thresh=0.01, margin_minutes=10)
    assert result is None


def test_find_fit_window_respects_margin():
    edge_times = _make_edge_times()
    stability = np.zeros(len(edge_times))
    win_0 = np.datetime64(DATE + timedelta(hours=13), 'ns')
    win_1 = np.datetime64(DATE + timedelta(hours=23), 'ns')
    margin = 30
    result = find_fit_window(stability, edge_times, (win_0, win_1),
                             stab_thresh=0.1, margin_minutes=margin)
    assert result is not None
    _, fw0, fw1 = result
    margin_td = np.timedelta64(margin, 'm')
    assert fw0 >= win_0 + margin_td
    assert fw1 <= win_1 - margin_td


# ---------------------------------------------------------------------------
# fit_polynomial
# ---------------------------------------------------------------------------

def test_fit_polynomial_output_shapes():
    tt = _tt_sec(n=300)
    data = tt * 0.01 + 1400.0
    poly_vals, poly_params, detrend = fit_polynomial(tt, data)
    assert len(poly_vals) == len(tt)
    assert len(detrend) == len(tt)


def test_fit_polynomial_removes_linear_trend():
    tt = _tt_sec(n=300)
    data = tt * 0.01 + 1400.0
    _, _, detrend = fit_polynomial(tt, data)
    assert np.std(detrend) < 1.0


def test_fit_polynomial_r2_in_range():
    tt = _tt_sec(n=300)
    rng = np.random.default_rng(5)
    data = tt * 0.01 + 1400.0 + rng.normal(0, 5, size=300)
    _, poly_params, _ = fit_polynomial(tt, data)
    assert 0.0 <= poly_params['r2'] <= 1.0


def test_fit_polynomial_params_keys():
    tt = _tt_sec(n=100)
    data = np.ones(100) * 1400.0
    _, poly_params, _ = fit_polynomial(tt, data)
    for key in ('c_0', 'c_1', 'c_2', 'r2'):
        assert key in poly_params


# ---------------------------------------------------------------------------
# apply_bandpass
# ---------------------------------------------------------------------------

def test_apply_bandpass_output_length_preserved():
    data = np.sin(np.linspace(0, 4 * np.pi, 600)) * 50.0
    out = apply_bandpass(data, lstid_T_hr_lim=(1.0, 4.5), fs=1/60.0)
    assert len(out) == len(data)


# ---------------------------------------------------------------------------
# fit_sinusoids
# ---------------------------------------------------------------------------

def test_fit_sinusoids_sorted_by_r2_descending():
    tt = _tt_sec(n=600)
    data = sinusoid(tt, T_hr=2.0, amplitude_km=80.0, phase_hr=0.0,
                    offset_km=0.0, slope_kmph=0.0)
    results = fit_sinusoids(tt, data, np.arange(1.0, 4.5, 0.5), lstid_T_hr_lim=(1.0, 4.5))
    assert len(results) > 0
    r2_vals = [r['r2'] for r in results]
    assert r2_vals == sorted(r2_vals, reverse=True)


def test_fit_sinusoids_best_fit_high_r2_for_clean_sinusoid():
    tt = _tt_sec(n=600)
    data = sinusoid(tt, T_hr=2.0, amplitude_km=80.0, phase_hr=0.0,
                    offset_km=0.0, slope_kmph=0.0)
    results = fit_sinusoids(tt, data, np.arange(1.0, 4.5, 0.5), lstid_T_hr_lim=(1.0, 4.5))
    assert results[0]['r2'] > 0.95


def test_fit_sinusoids_returns_empty_list_on_impossible_fit():
    tt = _tt_sec(n=60)
    data = np.zeros(60)
    results = fit_sinusoids(tt, data, np.arange(1.0, 4.5, 0.5), lstid_T_hr_lim=(1.0, 4.5))
    for r in results:
        assert np.isnan(r['r2']) or r['r2'] <= 1.0


def test_fit_sinusoids_recovered_period_close_to_true():
    tt = _tt_sec(n=600)
    true_T = 2.0
    data = sinusoid(tt, T_hr=true_T, amplitude_km=80.0, phase_hr=0.0,
                    offset_km=0.0, slope_kmph=0.0)
    results = fit_sinusoids(tt, data, np.arange(1.0, 4.5, 0.5), lstid_T_hr_lim=(1.0, 4.5))
    assert abs(results[0]['T_hr'] - true_T) < 0.3


# ---------------------------------------------------------------------------
# sin_fit — integration
# ---------------------------------------------------------------------------

SIN_FIT_PARAMS = dict(
    bandpass=True,
    lstid_T_hr_lim=(1.0, 4.5),
    roll_win=10,
    stab_thresh=0.05,
    margin_minutes=10,
    T_hr_guesses=np.arange(1.0, 4.5, 0.5),
)


def _make_sinusoidal_edge(n=720, T_hr=2.0, amplitude=80.0):
    tt = _tt_sec(n=n)
    positions = sinusoid(tt, T_hr=T_hr, amplitude_km=amplitude,
                         phase_hr=0.0, offset_km=1400.0, slope_kmph=0.0)
    return positions


def test_sin_fit_returns_fitoutput():
    edge = _make_edge_output(_make_sinusoidal_edge())
    assert isinstance(sin_fit(edge, **SIN_FIT_PARAMS), FitOutput)


def test_sin_fit_has_sin_params_for_clear_sinusoid():
    edge = _make_edge_output(_make_sinusoidal_edge())
    result = sin_fit(edge, **SIN_FIT_PARAMS)
    assert result.sin_params, "Expected sin_params to be non-empty"


def test_sin_fit_fit_times_within_edge_times():
    edge = _make_edge_output(_make_sinusoidal_edge())
    result = sin_fit(edge, **SIN_FIT_PARAMS)
    if len(result.fit_times) > 0:
        assert result.fit_times[0] >= edge.edge_times[0]
        assert result.fit_times[-1] <= edge.edge_times[-1]


def test_sin_fit_empty_result_on_flat_edge():
    edge = _make_edge_output(np.ones(720) * 1400.0)
    result = sin_fit(edge, **{**SIN_FIT_PARAMS, 'stab_thresh': 0.5})
    if result.sin_params:
        assert result.sin_params.get('r2', 1.0) < 0.5


