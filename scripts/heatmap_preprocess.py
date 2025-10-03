#!/usr/bin/env python3

"""
===============================================================================
Heatmap Preprocessing Utilities
===============================================================================

Purpose
-------
This module provides utility functions for preparing 2D radar/heatmap-like
(time x range) data for downstream analysis. The preprocessing pipeline was
originally developed as part of an attempt to support an image detector
AI model. This standardized pipeline proved useful and is now employed as 
a preprocessing step for the current edge detection method used in this 
project.

Design
------
- `preprocess_heatmap()` orchestrates the pipeline:
  (pad/crop) → (cut-half) → (MAD normalize) → coords → trims → sanitize →
  Gaussian smooth → rescale to uint8.
- Helper utilities: `pad_axis`, `pad_img`, `cut_half`, `mad`,
  `occurrence_max`, `rescale_to_int`.


Dependencies
------------
NumPy, SciPy (ndimage), Python stdlib (datetime, math)

Usage
-----
Call `preprocess_heatmap()` with raw hist2d, metadata, and config; receive:
processed uint8 array, time vector (epoch s), range vector (km), and sampling Δt.

from typing import Tuple, Dict, Any
import numpy as np
import math
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter 

Example
-------
>>> arr, arr_times, ranges_km, Ts_sec, Ts_td = preprocess_heatmap(
...     hist2d, meta,
...     expected_shape=(1440, 300), expected_size=1440,
...     min_dev=1.0, x_trim=0.0833, y_trim=0.08,
...     sigma=2.0, occurrence_n=100, i_max=30
... )

Authors & Contributors
----------------------
- Core utilities (`pad_axis`, `pad_img`, `cut_half`, `mad`,
  `occurrence_max`, `rescale_to_int`)
  Author: Nick Callahan

- Integrated pipeline (`preprocess_heatmap`)
  Assembler: Diego F. Sanchez (@kd2rlm)  
  Notes    : Brought components together; parameterized smoothing/rescale;
             coordinated metadata usage and coordinate reconstruction.

  10/02/2025

===============================================================================
"""

from typing import Tuple, Dict, Any
import numpy as np
import math
from datetime import datetime, timedelta
from scipy.ndimage import gaussian_filter

def preprocess_heatmap(
    hist2d: np.ndarray,
    meta: Dict[str, Any],
    *,
    expected_shape: Tuple[int, int],
    expected_size: int,
    min_dev: float,
    x_trim: float,
    y_trim: float,
    sigma: float,
    occurrence_n: int,
    i_max: int,
    qs=None,
    **_: Any
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, timedelta]:
    """
    Preprocess a 2D heatmap image (time vs. range bins).

    Pipeline:
        1. Pad raw array to a consistent shape.
        2. Cut along time axis to keep latter half (e.g., daily → half-day).
        3. Apply MAD normalization (robust scaling).
        4. Build time/range coordinates aligned with standardized grid.
        5. Apply symmetric trims in both axes.
        6. Replace NaNs with zeros.
        7. Gaussian smooth (on transposed array).
        8. Rescale values to integer-like uint8 range [0, i_max].

    Parameters
    ----------
    hist2d : np.ndarray
        Raw 2D histogram input (time x range).
    meta : dict
        Metadata with required keys:
          - "time_bin_seconds" : float, seconds per time bin
          - "distance_bin_km"  : float, km per range bin
          - "xedge_start"      : float, epoch seconds start of time bins
          - "yedge_start_km"   : float, km start of range bins
          - "sDate"            : str (ISO datetime) reference day for anchoring
    expected_shape : tuple[int, int]
        Target shape for padding (time, range).
    expected_size : int
        Expected time dimension before cut-half (must be even).
    min_dev : float
        Minimum denominator for MAD normalization.
    x_trim : float
        Fraction of time axis to trim symmetrically.
    y_trim : float
        Fraction of range axis to trim symmetrically.
    sigma : float
        Gaussian smoothing sigma.
    occurrence_n : int
        Number of top pixels excluded when computing robust max for rescaling.
    i_max : int
        Upper bound of rescaled intensity range (≤ 255).
    qs : optional
        Ignored; placeholder for API compatibility.
    **_ : Any
        Extra unused arguments are ignored.

    Returns
    -------
    arr : np.ndarray
        Preprocessed array, dtype uint8, shape [T_trim, H_trim].
    arr_times : np.ndarray
        1D array of epoch seconds, len == T_trim.
    ranges_km : np.ndarray
        1D array of range bin centers in km, len == H_trim.
    Ts_sec : float
        Sampling interval in seconds.
    Ts_td : timedelta
        Sampling interval as timedelta.
    """
    # --- extract required metadata ---
    dt_sec = float(meta["time_bin_seconds"])      # time step size in seconds
    dy_km  = float(meta["distance_bin_km"])       # distance step size in km
    x0     = float(meta["xedge_start"])           # left edge (epoch seconds) of time bins
    y0     = float(meta["yedge_start_km"])        # left edge (km) of range bins

    # --- enforce standardized shape ---
    hist2d = pad_img(hist2d, expected_shape=expected_shape, dtype=hist2d.dtype)
    hist2d = cut_half(hist2d, expected_size=expected_size)  # cut time dimension in half
    arr    = mad(hist2d, min_dev=min_dev).astype(np.float32)

    # --- coordinate reconstruction ---
    T, H = arr.shape

    # Anchor times at noon of sDate (local reference)
    s_dt = datetime.fromisoformat(meta["sDate"])
    noon = s_dt.replace(hour=12, minute=0, second=0, microsecond=0)
    t0   = noon.timestamp()

    arr_times_full = t0 + dt_sec * np.arange(T, dtype=np.float64)

    # Uniform range bin centers (km)
    ranges_full = (y0 + dy_km / 2.0) + dy_km * np.arange(expected_shape[1], dtype=np.float32)

    # --- trimming (time and range axes) ---
    xrt = math.floor(x_trim * T)      # left time trim
    xl  = math.floor(x_trim * T)      # right time trim
    yr  = math.floor(y_trim * H)      # bottom range trim
    yl  = math.floor(y_trim * H)      # top range trim

    arr        = arr[xrt: T - xl, yr: H - yl]
    arr_times  = arr_times_full[xrt: T - xl]
    ranges_km  = ranges_full[yr: H - yl]

    # --- sanitize values ---
    arr = np.nan_to_num(arr, nan=0.0)

    # Sampling interval
    Ts_sec = dt_sec
    Ts_td  = timedelta(seconds=Ts_sec)

    # --- post-processing ---
    arr = gaussian_filter(arr.T, sigma=(sigma, sigma))   # smooth in both dims, using transpose
    arr = rescale_to_int(arr, occurrence_n, i_max)     # rescale back to uint8

    return arr, arr_times, ranges_km, Ts_sec, Ts_td


def pad_axis(
    arr: np.ndarray,
    expected_size: int,
    dtype: np.dtype = np.uint8,
    axis: int = 0
) -> np.ndarray:
    """
    Pad or crop an array along a single axis to reach a given size.

    Parameters
    ----------
    arr : np.ndarray
        Input array.
    expected_size : int
        Target size along the specified axis.
    dtype : np.dtype, optional
        Enforced dtype for the result.
    axis : int, default=0
        Axis to pad/crop.

    Returns
    -------
    np.ndarray
        Array of shape == expected_size along that axis, dtype == dtype.
    """
    shape_mismatch = expected_size - arr.shape[axis]
    left_pad = shape_mismatch // 2
    
    if shape_mismatch > 0:
        # pad symmetrically
        right_pad = shape_mismatch - left_pad
        axis_pad = (left_pad, right_pad)
        full_pad = [(0, 0) if i != axis else axis_pad for i in range(arr.ndim)]
        arr = np.pad(arr, tuple(full_pad), mode='constant', constant_values=0)
    elif shape_mismatch < 0:
        # crop symmetrically
        left_pad = -shape_mismatch // 2
        right_pad = -shape_mismatch - left_pad
        arr = arr[:, left_pad:-right_pad]
        
    assert arr.dtype == dtype, dtype
    assert arr.shape[axis] == expected_size, f'{arr.shape[axis]} != expected {expected_size}'
    return arr


def pad_img(
    img: np.ndarray,
    expected_shape: Tuple[int, int],
    dtype: np.dtype = np.uint8,
    **_: Any
) -> np.ndarray:
    """
    Pad/crop a 2D image symmetrically to match an exact shape.

    Parameters
    ----------
    img : np.ndarray
        Raw 2D input (time x range).
    expected_shape : (int, int)
        Target (time, range).
    dtype : np.dtype, optional
        Enforced dtype.

    Returns
    -------
    np.ndarray
        Array of exact expected_shape.
    """
    assert len(expected_shape) == img.ndim
    for i in range(img.ndim):
        img = pad_axis(img, expected_shape[i], axis=i, dtype=dtype)
    return img


def cut_half(
    img: np.ndarray,
    expected_size: int,
    **_: Any
) -> np.ndarray:
    """
    Cut an array along axis 0, keeping only the latter half.

    Parameters
    ----------
    img : np.ndarray
        2D input array.
    expected_size : int
        Expected length of axis 0 before cutting.

    Returns
    -------
    np.ndarray
        Bottom/latter half of the input along axis 0.
    """
    if expected_size:
        assert img.shape[0] == expected_size, f'Mismatch: {img.shape[0]} != {expected_size}'
        assert not expected_size % 2, 'Size must be even'
    return img[expected_size // 2:, :]


def mad(
    t: np.ndarray,
    min_dev: float,
    **_: Any
) -> np.ndarray:
    """
    Median Absolute Deviation (MAD) normalization.

    Each value is scaled by:
        (x - median) / max(global MAD, min_dev)

    Parameters
    ----------
    t : np.ndarray
        Input array.
    min_dev : float
        Minimum denominator to avoid extreme amplification.

    Returns
    -------
    np.ndarray
        Normalized array, same shape as input.
    """
    median = np.median(t, axis=(0, 1), keepdims=True)
    abs_devs = np.abs(t - median)
    mad = abs_devs / max(np.median(abs_devs, axis=(0, 1), keepdims=True), min_dev)
    assert t.shape == mad.shape, f'{t.shape} | {mad.shape}'
    return mad


def occurrence_max(
    arr: np.ndarray,
    n: int
) -> int:
    """
    Compute a robust maximum value by ignoring the top `n` pixels.

    Parameters
    ----------
    arr : np.ndarray
        Input array (integer-like).
    n : int
        Number of top pixels to exclude.

    Returns
    -------
    int
        Robust maximum after outlier exclusion.
    """
    hist, bins = np.histogram(
        arr, 
        bins=np.arange(np.min(arr), np.max(arr) + 2)
    )
    bins = bins[1:]

    hist, bins = hist[::-1], bins[::-1]
    hist = np.cumsum(hist)
    bin_mask = hist >= n

    max_value = np.max(bins[bin_mask])
    return max_value


def rescale_to_int(
    arr: np.ndarray,
    occurrence_n: int,
    i_max: int
) -> np.ndarray:
    """
    Rescale an array to [0, i_max] and cast to uint8.

    Parameters
    ----------
    arr : np.ndarray
        Input array, possibly float.
    occurrence_n : int
        Number of top pixels to exclude when computing robust max.
    i_max : int
        Maximum value for rescaling (≤ 255).

    Returns
    -------
    np.ndarray
        Rescaled array, dtype uint8.
    """
    if i_max > 2**8 - 1:
        raise ValueError(f'`i_max` must be ≤ 255, not {i_max}')
    if (arr_max := np.amax(arr.ravel())) > 2**16 - 1:
        raise ValueError(f'All values must be in 16-bit range, got {arr_max}')

    arr = arr - np.amin(arr)
    max_val = occurrence_max(arr.round().astype(np.uint16), occurrence_n)
    factor = i_max / max_val
    arr = arr * factor
    if (arr_max := np.amax(arr.ravel())) > 2**8 - 1:
        raise ValueError(
            f'End rescaling max {arr_max} out of range for uint8; consider clipping'
        )
    return arr.round().astype(np.uint8)
