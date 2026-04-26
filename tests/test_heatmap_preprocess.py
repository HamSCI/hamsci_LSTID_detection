#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
import numpy as np
from datetime import timedelta

from scripts.heatmap_preprocess import (
    pad_axis, pad_img, cut_half, mad,
    occurrence_max, rescale_to_int, preprocess_heatmap,
)
from scripts.data_structures import PreprocessOutput


# ---------------------------------------------------------------------------
# pad_axis
# ---------------------------------------------------------------------------

def test_pad_axis_pads_when_too_small():
    arr = np.zeros((4, 6), dtype=np.uint8)
    out = pad_axis(arr, expected_size=8, axis=0)
    assert out.shape[0] == 8
    assert out.dtype == np.uint8


def test_pad_axis_crops_when_too_large():
    arr = np.zeros((10, 6), dtype=np.uint8)
    out = pad_axis(arr, expected_size=6, axis=1)
    assert out.shape[1] == 6


def test_pad_axis_noop_when_exact():
    arr = np.ones((6, 6), dtype=np.uint8)
    out = pad_axis(arr, expected_size=6, axis=0)
    assert out.shape == (6, 6)
    assert np.array_equal(out, arr)


def test_pad_axis_fills_with_zeros():
    arr = np.ones((2, 4), dtype=np.uint8)
    out = pad_axis(arr, expected_size=6, axis=0)
    # padded rows should be zero
    assert out[0, 0] == 0
    assert out[-1, 0] == 0


# ---------------------------------------------------------------------------
# pad_img
# ---------------------------------------------------------------------------

def test_pad_img_reaches_target_shape():
    arr = np.zeros((3, 5), dtype=np.uint8)
    out = pad_img(arr, expected_shape=(8, 8))
    assert out.shape == (8, 8)


def test_pad_img_exact_shape_is_noop():
    arr = np.ones((4, 4), dtype=np.uint8)
    out = pad_img(arr, expected_shape=(4, 4))
    assert np.array_equal(out, arr)


# ---------------------------------------------------------------------------
# cut_half
# ---------------------------------------------------------------------------

def test_cut_half_returns_second_half():
    arr = np.arange(8 * 3, dtype=np.uint8).reshape(8, 3)
    out = cut_half(arr, expected_size=8)
    assert out.shape == (4, 3)
    # second half rows should start at row index 4
    assert np.array_equal(out, arr[4:, :])


def test_cut_half_raises_on_size_mismatch():
    arr = np.zeros((6, 4), dtype=np.uint8)
    with pytest.raises(AssertionError):
        cut_half(arr, expected_size=8)


def test_cut_half_raises_on_odd_size():
    arr = np.zeros((7, 4), dtype=np.uint8)
    with pytest.raises(AssertionError):
        cut_half(arr, expected_size=7)


# ---------------------------------------------------------------------------
# mad
# ---------------------------------------------------------------------------

def test_mad_preserves_shape():
    arr = np.random.rand(10, 20).astype(np.float32)
    out = mad(arr, min_dev=1.0)
    assert out.shape == arr.shape


def test_mad_constant_array_returns_zeros():
    arr = np.full((5, 5), 3.0, dtype=np.float32)
    out = mad(arr, min_dev=1.0)
    # median deviation from constant is 0, so (x - median) / max(0, min_dev) = 0
    assert np.allclose(out, 0.0)


def test_mad_scales_with_spread():
    # array with known values: median=5, abs_devs have median=1
    arr = np.array([[4, 5, 6]], dtype=np.float32)
    out = mad(arr, min_dev=0.01)
    # median = 5; abs_devs = [1,0,1]; mad_val = 0.5 (median of [1,0,1])
    # scaled: [4-5, 5-5, 6-5] / 0.5 = [-2, 0, 2]
    assert out.shape == arr.shape
    assert out[0, 1] == pytest.approx(0.0, abs=1e-5)


# ---------------------------------------------------------------------------
# occurrence_max
# ---------------------------------------------------------------------------

def test_occurrence_max_top_bin_included_when_n_not_exceeded():
    # 3 pixels at value 10; n=3 → cumsum reaches 3 at the [10,11) bin,
    # so that bin IS included and the function returns its right edge (11).
    arr = np.array([0, 0, 0, 0, 0, 10, 10, 10], dtype=np.uint16)
    assert occurrence_max(arr, n=3) == 11


def test_occurrence_max_top_bin_excluded_when_n_exceeded():
    # Same array, but n=4 exceeds the 3-count top bin,
    # so the result falls to the right edge of the [0,1) bin (== 1).
    arr = np.array([0, 0, 0, 0, 0, 10, 10, 10], dtype=np.uint16)
    assert occurrence_max(arr, n=4) == 1


def test_occurrence_max_monotonic():
    # Larger n → equal or smaller result
    arr = np.array([0] * 10 + [50] * 5, dtype=np.uint16)
    r1 = occurrence_max(arr, n=1)
    r2 = occurrence_max(arr, n=6)
    assert r1 >= r2


# ---------------------------------------------------------------------------
# rescale_to_int
# ---------------------------------------------------------------------------

def test_rescale_to_int_output_dtype():
    arr = np.random.rand(10, 10).astype(np.float32) * 10
    out = rescale_to_int(arr, occurrence_n=1, i_max=30)
    assert out.dtype == np.uint8


def test_rescale_to_int_max_within_i_max():
    arr = np.random.rand(20, 20).astype(np.float32) * 10
    i_max = 30
    out = rescale_to_int(arr, occurrence_n=1, i_max=i_max)
    assert int(out.max()) <= i_max


def test_rescale_to_int_raises_if_i_max_too_large():
    arr = np.ones((5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="i_max"):
        rescale_to_int(arr, occurrence_n=1, i_max=256)


def test_rescale_to_int_min_is_zero():
    arr = np.array([[5.0, 10.0, 15.0]], dtype=np.float32)
    out = rescale_to_int(arr, occurrence_n=1, i_max=30)
    assert int(out.min()) == 0


# ---------------------------------------------------------------------------
# preprocess_heatmap — integration
# ---------------------------------------------------------------------------

def _make_meta(dt_sec=60.0, dy_km=10.0, x0=0.0, y0=500.0):
    """Minimal meta dict for preprocess_heatmap."""
    return {
        "time_bin_seconds": dt_sec,
        "distance_bin_km":  dy_km,
        "xedge_start":      x0,
        "yedge_start_km":   y0,
    }


def test_preprocess_heatmap_returns_preprocessoutput():
    hist2d = np.random.randint(0, 10, size=(8, 10), dtype=np.uint8)
    meta   = _make_meta()
    result = preprocess_heatmap(
        hist2d, meta,
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.0, y_trim=0.0,
        sigma=0.5, occurrence_n=1, i_max=30,
    )
    assert isinstance(result, PreprocessOutput)


def test_preprocess_heatmap_arr_dtype_uint8():
    hist2d = np.random.randint(0, 10, size=(8, 10), dtype=np.uint8)
    meta   = _make_meta()
    result = preprocess_heatmap(
        hist2d, meta,
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
    meta   = _make_meta()
    result = preprocess_heatmap(
        hist2d, meta,
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.25, y_trim=0.0,
        sigma=0.5, occurrence_n=1, i_max=30,
    )
    # arr is (range, time) after gaussian transpose
    assert result.arr.shape == (10, 2)


def test_preprocess_heatmap_coord_lengths_match_arr():
    hist2d = np.random.randint(0, 5, size=(8, 10), dtype=np.uint8)
    meta   = _make_meta()
    result = preprocess_heatmap(
        hist2d, meta,
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.0, y_trim=0.0,
        sigma=0.5, occurrence_n=1, i_max=30,
    )
    assert len(result.arr_times) == result.arr.shape[1]
    assert len(result.ranges_km) == result.arr.shape[0]


def test_preprocess_heatmap_Ts_sec_matches_meta():
    hist2d = np.random.randint(0, 5, size=(8, 10), dtype=np.uint8)
    meta   = _make_meta(dt_sec=120.0)
    result = preprocess_heatmap(
        hist2d, meta,
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.0, y_trim=0.0,
        sigma=0.5, occurrence_n=1, i_max=30,
    )
    assert result.Ts_sec == 120.0
    assert result.Ts_td == timedelta(seconds=120.0)


def test_preprocess_heatmap_intermediate_keys_present():
    hist2d = np.random.randint(0, 5, size=(8, 10), dtype=np.uint8)
    meta   = _make_meta()
    result = preprocess_heatmap(
        hist2d, meta,
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.0, y_trim=0.0,
        sigma=0.5, occurrence_n=1, i_max=30,
    )
    for key in ('raw_hist2d', 'after_mad', 'after_gaussian', 'after_rescale'):
        assert key in result.intermediate, f"Missing intermediate key: {key}"


def test_preprocess_heatmap_raw_hist2d_is_pre_mad():
    # raw_hist2d should contain raw integer counts, not MAD-normalised floats.
    # The simplest check: its values must be whole numbers (original uint8 counts).
    hist2d = np.arange(80, dtype=np.uint8).reshape(8, 10)
    meta   = _make_meta()
    result = preprocess_heatmap(
        hist2d, meta,
        expected_shape=(8, 10), expected_size=8,
        min_dev=1.0, x_trim=0.0, y_trim=0.0,
        sigma=0.0, occurrence_n=1, i_max=30,
    )
    raw = result.intermediate['raw_hist2d'].astype(float)
    assert np.all(raw == raw.astype(np.uint8)), \
        "raw_hist2d contains non-integer values — MAD was applied before saving"
