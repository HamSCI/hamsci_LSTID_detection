#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
from datetime import timedelta

from scripts.heatmap_preprocess import preprocess_heatmap
from scripts.data_structures import PreprocessOutput


# ---------------------------------------------------------------------------
# preprocess_heatmap — integration
# ---------------------------------------------------------------------------

def _make_meta(dt_sec=60.0, dy_km=10.0, x0=0.0, y0=500.0):
    return {
        "time_bin_seconds": dt_sec,
        "distance_bin_km":  dy_km,
        "xedge_start":      x0,
        "yedge_start_km":   y0,
    }


def test_preprocess_heatmap_returns_preprocessoutput():
    hist2d = np.random.randint(0, 10, size=(8, 10), dtype=np.uint8)
    result = preprocess_heatmap(
        hist2d, _make_meta(),
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.0, y_trim=0.0,
        sigma=0.5, occurrence_n=1, i_max=30,
    )
    assert isinstance(result, PreprocessOutput)


def test_preprocess_heatmap_arr_dtype_uint8():
    hist2d = np.random.randint(0, 10, size=(8, 10), dtype=np.uint8)
    result = preprocess_heatmap(
        hist2d, _make_meta(),
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.0, y_trim=0.0,
        sigma=0.5, occurrence_n=1, i_max=30,
    )
    assert result.arr.dtype == np.uint8


def test_preprocess_heatmap_arr_shape_after_cut_and_trim():
    # expected_shape=(8,10), expected_size=8 → cut_half → (4,10)
    # x_trim=0.25 → xrt = floor(0.25*4)=1, xl=1 → T_trim = 4-1-1 = 2
    # y_trim=0.0  → H_trim = 10
    hist2d = np.random.randint(0, 5, size=(8, 10), dtype=np.uint8)
    result = preprocess_heatmap(
        hist2d, _make_meta(),
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.25, y_trim=0.0,
        sigma=0.5, occurrence_n=1, i_max=30,
    )
    assert result.arr.shape == (10, 2)


def test_preprocess_heatmap_coord_lengths_match_arr():
    hist2d = np.random.randint(0, 5, size=(8, 10), dtype=np.uint8)
    result = preprocess_heatmap(
        hist2d, _make_meta(),
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.0, y_trim=0.0,
        sigma=0.5, occurrence_n=1, i_max=30,
    )
    assert len(result.arr_times) == result.arr.shape[1]
    assert len(result.ranges_km) == result.arr.shape[0]


def test_preprocess_heatmap_Ts_sec_matches_meta():
    hist2d = np.random.randint(0, 5, size=(8, 10), dtype=np.uint8)
    result = preprocess_heatmap(
        hist2d, _make_meta(dt_sec=120.0),
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.0, y_trim=0.0,
        sigma=0.5, occurrence_n=1, i_max=30,
    )
    assert result.Ts_sec == 120.0
    assert result.Ts_td == timedelta(seconds=120.0)


def test_preprocess_heatmap_intermediate_keys_present():
    hist2d = np.random.randint(0, 5, size=(8, 10), dtype=np.uint8)
    result = preprocess_heatmap(
        hist2d, _make_meta(),
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.0, y_trim=0.0,
        sigma=0.5, occurrence_n=1, i_max=30,
    )
    for key in ('raw_hist2d', 'after_mad', 'after_gaussian', 'after_rescale'):
        assert key in result.intermediate, f"Missing intermediate key: {key}"


def test_preprocess_heatmap_raw_hist2d_is_pre_mad():
    hist2d = np.arange(80, dtype=np.uint8).reshape(8, 10)
    result = preprocess_heatmap(
        hist2d, _make_meta(),
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.0, y_trim=0.0,
        sigma=0.0, occurrence_n=1, i_max=30,
    )
    raw = result.intermediate['raw_hist2d'].astype(float)
    assert np.all(raw == raw.astype(np.uint8)), \
        "raw_hist2d contains non-integer values — MAD was applied before saving"
