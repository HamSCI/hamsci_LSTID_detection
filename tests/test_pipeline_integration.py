#!/usr/bin/env python3
"""
End-to-end pipeline integration tests using the two synthetic HDF5 files.

Day 1  rsd2025-01-01.01.hdf5  sinusoid + polynomial diurnal trend (LSTID present)
Day 2  rsd2025-01-02.01.hdf5  polynomial trend only               (quiet day)

The tests verify that the full pipeline
    HDF5PolarsLoader → preprocess_heatmap → detect_edge → sin_fit
runs without error and produces scientifically sensible outputs for each case.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
from datetime import datetime

from scripts.hdf5_loader import HDF5PolarsLoader
from scripts.heatmap_preprocess import preprocess_heatmap
from scripts.edge_detect import detect_edge
from scripts.sinusoid_fitting import sin_fit
from scripts.data_structures import PreprocessOutput, EdgeDetectionOutput, FitOutput


# ---------------------------------------------------------------------------
# Shared parameters (mirrors config.json)
# ---------------------------------------------------------------------------

MADRIGAL_DIR = Path("tests/data/madrigal")

LOADER_KWARGS = dict(
    data_dir=str(MADRIGAL_DIR),
    cache_dir="/tmp/pytest_pipeline_cache",
    use_cache=False,
    region_bounds={"lat_lim": (24.5, 49.5), "lon_lim": (-125.0, -66.5)},
    freq_range={"min_freq": 13_000_000, "max_freq": 15_000_000, "label": "14"},
    distance_range={"min_dist": 0, "max_dist": 3000},
)

PRE_PARAMS = dict(
    expected_shape=(1440, 300),
    expected_size=1440,
    x_trim=0.08333,
    y_trim=0.08,
    min_dev=1.0,   # synthetic data is sparse (1 spot/min); min_dev>0 prevents MAD=0 blow-up
    sigma=4.2,
    occurrence_n=60,
    i_max=30,
)

EDGE_PARAMS = dict(
    qs=[0.4, 0.5, 0.6],
    lower_cutoff=10,
    select_min=True,
    exact_thresh=False,
    axis=0,
    lowess_window_size=10,
    max_abs_dev=20,
)

FIT_PARAMS = dict(
    bandpass=True,
    lstid_T_hr_lim=(1.0, 4.5),
    roll_win=15,
    stab_thresh=0.05,
    margin_minutes=30,
    T_hr_guesses=np.arange(1.0, 4.5, 0.5),
)


# ---------------------------------------------------------------------------
# Helper: run the full pipeline for one date, return fit_result + df
# ---------------------------------------------------------------------------

def _run_pipeline(sDate: datetime, eDate: datetime):
    loader = HDF5PolarsLoader(sDate=sDate, eDate=eDate, **LOADER_KWARGS)
    df = loader.get_dataframe()
    hist2d, meta = loader.gen_histogram()

    pre   = preprocess_heatmap(hist2d, meta, **PRE_PARAMS)
    edge  = detect_edge(sDate, pre, **EDGE_PARAMS)
    fit   = sin_fit(edge, **FIT_PARAMS)
    return fit, df


# ---------------------------------------------------------------------------
# Stage-output type tests (both days must produce correct dataclass types)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def day1_result():
    sDate = datetime(2025, 1, 1, 0, 0, 0)
    eDate = datetime(2025, 1, 1, 23, 59, 59)
    return _run_pipeline(sDate, eDate)


@pytest.fixture(scope="module")
def day2_result():
    sDate = datetime(2025, 1, 2, 0, 0, 0)
    eDate = datetime(2025, 1, 2, 23, 59, 59)
    return _run_pipeline(sDate, eDate)


# ---------------------------------------------------------------------------
# Day 1 — LSTID present (sinusoid + polynomial)
# ---------------------------------------------------------------------------

def test_day1_pipeline_produces_fitoutput(day1_result):
    fit, _ = day1_result
    assert isinstance(fit, FitOutput)


def test_day1_edge_data_is_edge_detection_output(day1_result):
    fit, _ = day1_result
    assert isinstance(fit.edge_data, EdgeDetectionOutput)


def test_day1_detects_sinusoid(day1_result):
    fit, _ = day1_result
    assert fit.sin_params, "Expected a sinusoidal fit on the LSTID day"


def test_day1_r2_above_threshold(day1_result):
    fit, _ = day1_result
    r2 = fit.sin_params.get('r2', 0.0)
    assert r2 > 0.3, f"R² = {r2:.3f} is too low for a clear sinusoidal signal"


def test_day1_period_in_lstid_range(day1_result):
    fit, _ = day1_result
    T_hr = fit.sin_params.get('T_hr', 0.0)
    assert 1.0 <= T_hr <= 4.5, f"Recovered period {T_hr:.2f} hr outside LSTID range"


def test_day1_fit_times_nonempty(day1_result):
    fit, _ = day1_result
    assert len(fit.fit_times) > 0


def test_day1_sin_fit_and_poly_fit_same_length(day1_result):
    fit, _ = day1_result
    assert len(fit.sin_fit) == len(fit.poly_fit) == len(fit.fit_times)


def test_day1_stability_length_matches_edge_times(day1_result):
    fit, _ = day1_result
    assert len(fit.stability) == len(fit.edge_data.edge_times)


def test_day1_all_sin_fits_sorted_by_r2(day1_result):
    fit, _ = day1_result
    if len(fit.all_sin_fits) > 1:
        r2s = [f['r2'] for f in fit.all_sin_fits]
        assert r2s == sorted(r2s, reverse=True)


def test_day1_n_spots_matches_df(day1_result):
    fit, df = day1_result
    # The df returned by get_dataframe() is a Polars DataFrame
    assert df.height > 0


# ---------------------------------------------------------------------------
# Day 2 — quiet day (polynomial only, no sinusoid)
# ---------------------------------------------------------------------------

def test_day2_pipeline_produces_fitoutput(day2_result):
    fit, _ = day2_result
    assert isinstance(fit, FitOutput)


def test_day2_edge_data_is_edge_detection_output(day2_result):
    fit, _ = day2_result
    assert isinstance(fit.edge_data, EdgeDetectionOutput)


def test_day2_weaker_fit_than_day1(day1_result, day2_result):
    """Quiet day should produce a lower R² than the LSTID day."""
    fit1, _ = day1_result
    fit2, _ = day2_result
    r2_day1 = fit1.sin_params.get('r2', 0.0) if fit1.sin_params else 0.0
    r2_day2 = fit2.sin_params.get('r2', 0.0) if fit2.sin_params else 0.0
    assert r2_day2 < r2_day1, (
        f"Quiet day R²={r2_day2:.3f} should be < LSTID day R²={r2_day1:.3f}"
    )


def test_day2_stability_length_matches_edge_times(day2_result):
    fit, _ = day2_result
    assert len(fit.stability) == len(fit.edge_data.edge_times)


def test_day2_fit_params_have_required_keys_if_fit_found(day2_result):
    fit, _ = day2_result
    if fit.sin_params:
        for key in ('T_hr', 'amplitude_km', 'phase_hr', 'offset_km', 'slope_kmph', 'r2'):
            assert key in fit.sin_params
