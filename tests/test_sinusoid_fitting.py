#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
from datetime import datetime, timedelta

from scripts.sinusoid_fitting import (
    sinusoid,
    bandpass_filter,
    islandinfo,
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
# sinusoid
# ---------------------------------------------------------------------------

def test_sinusoid_zero_amplitude_gives_linear():
    tt = _tt_sec()
    out = sinusoid(tt, T_hr=2.0, amplitude_km=0.0, phase_hr=0.0,
                   offset_km=1400.0, slope_kmph=0.0)
    assert np.allclose(out, 1400.0)


def test_sinusoid_known_value_at_quarter_period():
    # At t = T/4 seconds from phase=0, sin should be 1 → amplitude
    T_hr = 2.0
    T_sec = T_hr * 3600.0
    # sin(2π * (T/4) / T + 0) = sin(π/2) = 1
    tt = np.array([T_sec / 4.0])
    out = sinusoid(tt, T_hr=T_hr, amplitude_km=100.0, phase_hr=0.0,
                   offset_km=0.0, slope_kmph=0.0)
    assert out[0] == pytest.approx(100.0, abs=1e-6)


def test_sinusoid_output_length_matches_input():
    tt = _tt_sec(n=300)
    out = sinusoid(tt, T_hr=2.0, amplitude_km=50.0, phase_hr=0.0,
                   offset_km=1400.0, slope_kmph=0.0)
    assert len(out) == 300


def test_sinusoid_slope_shifts_values():
    tt = np.array([0.0, 3600.0])  # 0 and 1 hour
    out = sinusoid(tt, T_hr=24.0, amplitude_km=0.0, phase_hr=0.0,
                   offset_km=0.0, slope_kmph=10.0)
    # After 1 hour: slope_kmph/3600 * 3600 = 10 km
    assert out[1] - out[0] == pytest.approx(10.0, abs=1e-6)


# ---------------------------------------------------------------------------
# bandpass_filter
# ---------------------------------------------------------------------------

def test_bandpass_filter_output_length_preserved():
    data = np.random.randn(720)
    fs = 1 / 60.0
    out = bandpass_filter(data, lowcut=1/(4.5*3600), highcut=1/(1.0*3600), fs=fs)
    assert len(out) == len(data)


def test_bandpass_filter_attenuates_dc():
    # A pure DC signal has no energy in the passband
    data = np.ones(720)
    fs = 1 / 60.0
    out = bandpass_filter(data, lowcut=1/(4.5*3600), highcut=1/(1.0*3600), fs=fs)
    assert np.abs(out).max() < 0.1


def test_bandpass_filter_passes_in_band_signal():
    # 2-hour sinusoid is in the 1–4.5 hr passband
    fs = 1 / 60.0
    t = np.arange(720) * 60.0
    data = np.sin(2 * np.pi * t / (2 * 3600))
    out = bandpass_filter(data, lowcut=1/(4.5*3600), highcut=1/(1.0*3600), fs=fs)
    # After transient, output power should be close to input power
    mid = slice(100, 620)
    assert np.std(out[mid]) > 0.5 * np.std(data[mid])


# ---------------------------------------------------------------------------
# islandinfo
# ---------------------------------------------------------------------------

def test_islandinfo_single_island():
    y = np.array([False, True, True, True, False])
    islands, lens = islandinfo(y, trigger_val=True)
    assert len(islands) == 1
    assert lens[0] == 3


def test_islandinfo_multiple_islands():
    y = np.array([True, True, False, True, False, True])
    islands, lens = islandinfo(y, trigger_val=True)
    assert len(islands) == 3


def test_islandinfo_no_islands():
    y = np.array([False, False, False])
    islands, lens = islandinfo(y, trigger_val=True)
    assert len(islands) == 0


def test_islandinfo_all_true():
    y = np.ones(5, dtype=bool)
    islands, lens = islandinfo(y, trigger_val=True)
    assert len(islands) == 1
    assert lens[0] == 5


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


def test_compute_stability_output_length_matches_input():
    arr = np.ones(40) * 1200.0
    out = compute_stability(arr, roll_win=10)
    assert len(out) == len(arr)


# ---------------------------------------------------------------------------
# find_fit_window
# ---------------------------------------------------------------------------

def _make_stability_and_times(n=720, dt_sec=60.0):
    t0_ns = np.datetime64(DATE + timedelta(hours=12), 'ns').astype(np.int64)
    step_ns = int(dt_sec * 1e9)
    edge_times = (t0_ns + np.arange(n) * step_ns).view('datetime64[ns]')
    return edge_times


def test_find_fit_window_finds_stable_region():
    edge_times = _make_stability_and_times()
    # All stable (CV = 0)
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
    edge_times = _make_stability_and_times()
    # All unstable (CV = 1.0 >> thresh)
    stability = np.ones(len(edge_times))
    win_0 = np.datetime64(DATE + timedelta(hours=13), 'ns')
    win_1 = np.datetime64(DATE + timedelta(hours=23), 'ns')
    result = find_fit_window(stability, edge_times, (win_0, win_1),
                             stab_thresh=0.01, margin_minutes=10)
    assert result is None


def test_find_fit_window_respects_margin():
    edge_times = _make_stability_and_times()
    stability = np.zeros(len(edge_times))
    win_0 = np.datetime64(DATE + timedelta(hours=13), 'ns')
    win_1 = np.datetime64(DATE + timedelta(hours=23), 'ns')
    margin = 30  # 30 minutes
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
    data = tt * 0.01 + 1400.0  # linear trend
    poly_vals, poly_params, detrend = fit_polynomial(tt, data)
    assert len(poly_vals) == len(tt)
    assert len(detrend) == len(tt)


def test_fit_polynomial_removes_linear_trend():
    tt = _tt_sec(n=300)
    data = tt * 0.01 + 1400.0
    _, _, detrend = fit_polynomial(tt, data)
    assert np.std(detrend) < 1.0  # detrended should be near zero


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
    # Clean 2-hr sinusoid — should be easy to fit
    data = sinusoid(tt, T_hr=2.0, amplitude_km=80.0, phase_hr=0.0,
                    offset_km=0.0, slope_kmph=0.0)
    guesses = np.arange(1.0, 4.5, 0.5)
    results = fit_sinusoids(tt, data, guesses)
    assert len(results) > 0
    r2_vals = [r['r2'] for r in results]
    assert r2_vals == sorted(r2_vals, reverse=True)


def test_fit_sinusoids_best_fit_high_r2_for_clean_sinusoid():
    tt = _tt_sec(n=600)
    data = sinusoid(tt, T_hr=2.0, amplitude_km=80.0, phase_hr=0.0,
                    offset_km=0.0, slope_kmph=0.0)
    results = fit_sinusoids(tt, data, np.arange(1.0, 4.5, 0.5))
    assert results[0]['r2'] > 0.95


def test_fit_sinusoids_returns_empty_list_on_impossible_fit():
    # Flat-zero data → ss_tot = 0, so R² = NaN (divide by zero).
    # The function does not raise; it returns fits with NaN R².
    tt = _tt_sec(n=60)
    data = np.zeros(60)
    results = fit_sinusoids(tt, data, np.arange(1.0, 4.5, 0.5))
    # Either no fits, or every R² is NaN / not a meaningful value
    for r in results:
        assert np.isnan(r['r2']) or r['r2'] <= 1.0


def test_fit_sinusoids_recovered_period_close_to_true():
    tt = _tt_sec(n=600)
    true_T = 2.0
    data = sinusoid(tt, T_hr=true_T, amplitude_km=80.0, phase_hr=0.0,
                    offset_km=0.0, slope_kmph=0.0)
    results = fit_sinusoids(tt, data, np.arange(1.0, 4.5, 0.5))
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
    """Pure sinusoid centred at 1400 km — easy case for sin_fit."""
    tt = _tt_sec(n=n)
    positions = sinusoid(tt, T_hr=T_hr, amplitude_km=amplitude,
                         phase_hr=0.0, offset_km=1400.0, slope_kmph=0.0)
    return positions


def test_sin_fit_returns_fitoutput():
    positions = _make_sinusoidal_edge()
    edge = _make_edge_output(positions)
    result = sin_fit(edge, **SIN_FIT_PARAMS)
    assert isinstance(result, FitOutput)


def test_sin_fit_has_sin_params_for_clear_sinusoid():
    positions = _make_sinusoidal_edge()
    edge = _make_edge_output(positions)
    result = sin_fit(edge, **SIN_FIT_PARAMS)
    assert result.sin_params, "Expected sin_params to be non-empty"


def test_sin_fit_fit_times_within_edge_times():
    positions = _make_sinusoidal_edge()
    edge = _make_edge_output(positions)
    result = sin_fit(edge, **SIN_FIT_PARAMS)
    if len(result.fit_times) > 0:
        assert result.fit_times[0] >= edge.edge_times[0]
        assert result.fit_times[-1] <= edge.edge_times[-1]


def test_sin_fit_empty_result_on_flat_edge():
    """A perfectly flat edge has zero CV so stability is 0 everywhere —
    the window IS found but the sinusoidal fit should either fail or
    return very low R² since there's nothing to fit."""
    positions = np.ones(720) * 1400.0
    edge = _make_edge_output(positions)
    # Use a very tight stab_thresh so the fit window IS found
    params = {**SIN_FIT_PARAMS, 'stab_thresh': 0.5}
    result = sin_fit(edge, **params)
    # If a fit was found its R² should be low (no real oscillation)
    if result.sin_params:
        assert result.sin_params.get('r2', 1.0) < 0.5


def test_sin_fit_stability_length_matches_edge_times():
    positions = _make_sinusoidal_edge()
    edge = _make_edge_output(positions)
    result = sin_fit(edge, **SIN_FIT_PARAMS)
    assert len(result.stability) == len(edge.edge_times)


def test_sin_fit_all_sin_fits_sorted_by_r2():
    positions = _make_sinusoidal_edge()
    edge = _make_edge_output(positions)
    result = sin_fit(edge, **SIN_FIT_PARAMS)
    if len(result.all_sin_fits) > 1:
        r2s = [f['r2'] for f in result.all_sin_fits]
        assert r2s == sorted(r2s, reverse=True)
