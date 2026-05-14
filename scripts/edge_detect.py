#!/usr/bin/env python3

"""
===============================================================================
Edge Detection / Thresholding Utilities
===============================================================================

Purpose
-------
This module implements the first stage of the edge-detection pipeline operating
on preprocessed (time x range) heatmaps. It builds column-wise threshold stacks,
selects representative edges via quantiles, and chooses a single edge with
minimal deviation after smoothing. It also provides helpers for smoothing and
index→km conversion used by downstream steps.

Design
------
- `thresholding()` orchestrates the stage for a single (arr, times, ranges):
  (measure_thresholds) → (index→km) → (extent) → (quantile table)
- Core helpers:
  - `stack_all_thresholds()`  : build threshold stack along an axis
  - `take_quantile()`         : pick edge at a given quantile per column
  - `select_min_deviation()`  : choose edge with least deviation vs smoothed
  - `smooth_remove_abs_deviation()` and `lowess_smooth()` for robust smoothing
  - `scale_km()`              : convert edge indices to physical km

Dependencies
------------
NumPy, Polars, Statsmodels (nonparametric lowess), SciPy (CubicSpline)

Usage
-----
Given a preprocessed array and coordinates:

>>> out = thresholding(
...     arr, arr_times, ranges_km, Ts_sec, Ts_td,
...     qs=[0.4, 0.5, 0.6],
...     lower_cutoff=10,
...     select_min=True,
...     exact_thresh=False,
...     axis=0,
...     lowess_window_size=10,
...     max_abs_dev=20,
... )
>>> out["med_lines_df"].head()

Returns a dict with:
- per-quantile edges (km),
- selected/min-deviation edge and its smoothed version (km),
- extent for plotting,
- Polars DataFrame with quantile lines aligned to time.

Authors & Contributors
----------------------
- Thresholding/edge helpers:
  Author: Nick Callahan

- Integration & refactor (orchestration + config threading):
  Assembler: Diego F. Sanchez (@kd2rlm)

  10/03/2025

===============================================================================
"""

from typing import Tuple, List, Iterable, Dict, Any
import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy.interpolate import CubicSpline
from datetime import datetime, timedelta

from scripts.data_structures import PreprocessOutput, EdgeDetectionOutput


def detect_edge(
    date: datetime,
    preprocess_data: PreprocessOutput,
    *,
    qs: list[float],
    lower_cutoff: int,
    select_min: bool,
    exact_thresh: bool,
    axis: int,
    lowess_window_size: int,
    max_abs_dev: int,
    **_: Any
) -> EdgeDetectionOutput:
    """
    Detect ionospheric edge from preprocessed heatmap.
    
    Pipeline
    --------
    1. Compute threshold lines at multiple quantiles
    2. Select best line (minimum deviation after smoothing)
    3. Interpolate NaNs to create edge_0
    4. Window to analysis period (13:00-23:00)
    5. Interpolate onto uniform time grid (12:00-24:00)
    
    Parameters
    ----------
    date : datetime
        Processing date
    preprocess_data : PreprocessOutput
        Output from preprocess_heatmap
    qs : list[float]
        Quantiles for threshold detection (e.g., [0.4, 0.5, 0.6])
    lower_cutoff : int
        Minimum range index for valid detections
    select_min : bool
        Select minimum (True) or maximum (False) threshold
    exact_thresh : bool
        Use exact threshold matching
    axis : int
        Axis for threshold computation
    lowess_window_size : int
        LOWESS smoothing window size
    max_abs_dev : int
        Maximum absolute deviation for outlier filtering
        
    Returns
    -------
    EdgeDetectionOutput
        Detected edges, quantile lines, and intermediate data
    """
    # Unpack preprocessing data
    arr = preprocess_data.arr
    arr_times = preprocess_data.arr_times
    ranges_km = preprocess_data.ranges_km
    Ts_sec = preprocess_data.Ts_sec
    Ts_td = preprocess_data.Ts_td
    
    intermediate = {
        **preprocess_data.intermediate,  # Has: raw_hist2d, after_mad, after_gaussian, after_rescale
        'preprocessed_arr': arr,  # Duplicate of after_rescale (for plotting compatibility)
        'preprocessed_times': arr_times,
    }
    
    # --- 1) Compute thresholds in index space ---
    med_lines_idx, min_line_idx, minz_line_idx = measure_thresholds(
        arr,
        qs=qs,
        lower_cutoff=lower_cutoff,
        select_min=select_min,
        exact_thresh=exact_thresh,
        axis=axis,
        lowess_window_size=lowess_window_size,
        max_abs_dev=max_abs_dev,
    )
    
    # --- 2) Convert indices to km ---
    med_lines = [scale_km(x, ranges_km) for x in med_lines_idx]
    min_line = scale_km(min_line_idx, ranges_km)
    minz_line = scale_km(minz_line_idx, ranges_km)
    
    # Stack quantile lines: shape (T, n_quantiles)
    quantile_lines = np.vstack(med_lines).T
    quantile_values = np.array(qs)
    
    # --- 3) Build edge_0: interpolate NaNs in min_line ---
    edge_0_vals = np.asarray(min_line, dtype=float).squeeze()
    n = np.arange(edge_0_vals.size, dtype=float)
    m = np.isfinite(edge_0_vals)
    if m.any():
        edge_0_vals[~m] = np.interp(n[~m], n[m], edge_0_vals[m])
    edge_0_vals = np.nan_to_num(edge_0_vals, nan=0.0)
    edge_0_times = np.asarray(arr_times, dtype="datetime64[ns]")
    
    # Store edge_0 in intermediate
    intermediate['edge_0_vals'] = edge_0_vals
    intermediate['edge_0_times'] = edge_0_times
    
    # --- 4) Define time windows ---
    x_0 = np.datetime64(date + timedelta(hours=12), 'ns')    # 12:00
    x_1 = np.datetime64(date + timedelta(hours=24), 'ns')    # 24:00 next day
    win_0 = np.datetime64(date + timedelta(hours=13), 'ns')  # 13:00
    win_1 = np.datetime64(date + timedelta(hours=23), 'ns')  # 23:00
    
    # --- 5) Window edge_0 to analysis period (13:00-23:00) ---
    window_mask = (edge_0_times >= win_0) & (edge_0_times < win_1)
    edge_win_times = edge_0_times[window_mask]
    edge_win_vals = edge_0_vals[window_mask]
    
    intermediate['edge_win_times'] = edge_win_times
    intermediate['edge_win_vals'] = edge_win_vals
    intermediate['window_mask'] = window_mask
    
    # --- 6) Create uniform time grid (12:00-24:00) ---
    step_ns = int(Ts_sec * 1e9)
    t0_ns = x_0.astype('datetime64[ns]').astype(np.int64)
    t1_ns = x_1.astype('datetime64[ns]').astype(np.int64)
    grid_ns = np.arange(t0_ns, t1_ns + step_ns, step_ns, dtype=np.int64)
    
    # Don't exceed end time
    if grid_ns[-1] > t1_ns:
        grid_ns = grid_ns[:-1]
    
    edge_times = grid_ns.view('datetime64[ns]')
    
    # --- 7) Interpolate windowed edge onto uniform grid ---
    xp = edge_win_times.astype('datetime64[ns]').astype(np.int64)
    x = edge_times.astype('datetime64[ns]').astype(np.int64)
    edge_positions = np.interp(
        x, xp, edge_win_vals,
        left=edge_win_vals[0],
        right=edge_win_vals[-1]
    )
    
    # --- 8) Build metadata ---
    meta = {
        **preprocess_data.meta,
        'date': date,
        'edge_params': {
            'qs': qs,
            'lower_cutoff': lower_cutoff,
            'select_min': select_min,
            'exact_thresh': exact_thresh,
            'axis': axis,
            'lowess_window_size': lowess_window_size,
            'max_abs_dev': max_abs_dev,
        },
        'time_limits': {
            'xlim': (x_0, x_1),
            'winlim': (win_0, win_1),
        },
    }
    
    return EdgeDetectionOutput(
        edge_times=edge_times,
        edge_positions=edge_positions,
        quantile_lines=quantile_lines,
        quantile_values=quantile_values,
        min_line=min_line,
        minz_line=minz_line,
        ranges_km=ranges_km,
        intermediate=intermediate,
        meta=meta,
    )



def occurrence_max(arr, n):
    """
    Compute a robust maximum by ignoring the top `n` pixels.

    Parameters
    ----------
    arr : np.ndarray
        Input array (integer-like).
    n : int
        Number of highest pixels to exclude.

    Returns
    -------
    int
        Robust maximum value after outlier trimming.
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


def stack_all_thresholds(
    arr, 
    select_min, 
    exact_thresh, 
    axis
):
    """
    Build a 2D stack of threshold edges across all unique values.

    Parameters
    ----------
    arr : np.ndarray
        Preprocessed input image.
    select_min : bool
        Select argmin (True) or argmax (False) across the mask.
    exact_thresh : bool
        If True, mask = (arr <= threshold); else mask = (arr != threshold).
    axis : int
        Axis to compute indices along.

    Returns
    -------
    np.ndarray
        2D array stacking the per-threshold edge indices.
    """
    thresholds = np.unique(arr)
    thresh_edges = list()
    for threshold in thresholds:
        if exact_thresh:
            thresh_mask = arr <= threshold
        else:
            thresh_mask = arr != threshold
        
        idx_fn = np.argmin if select_min else np.argmax
        thresh_edge = idx_fn(thresh_mask.astype(np.uint8), axis=axis, keepdims=True)
            
        if max(thresh_edge.shape) != max(arr.shape):
            raise ValueError(
                f'Expected largest dim in `thresh_edge` (shape {thresh_edge.shape})'
                + f'to match largest dim for `arr` (shape {arr.shape})'
            )
        
        thresh_edges.append(thresh_edge)
    thresh_edge_arr = np.concatenate(thresh_edges, axis=axis)
    return thresh_edge_arr


def lowess_smooth(arr, window_size, x):
    """
    LOWESS smoothing wrapper around statsmodels.nonparametric.lowess.

    Parameters
    ----------
    arr : np.ndarray
        1D series to smooth.
    window_size : int
        Window length in native array units (converted to fraction).
    x : np.ndarray or None
        Optional 1D x-axis; if None, evenly spaced indices are used.

    Returns
    -------
    np.ndarray
        Smoothed array of same length as input.
    """
    if x is None:
        x = np.linspace(0, len(arr), len(arr))
    frac = window_size/len(arr)
    z = sm.nonparametric.lowess(arr, x, frac=frac, return_sorted=False)    
    return z


def smooth_remove_abs_deviation(arr, smooth_fn, max_abs_dev):
    """
    Smooth `arr`, filter points with excessive |arr - smooth|, then
    interpolate to fill removed points.

    Parameters
    ----------
    arr : np.ndarray
        1D array to process.
    smooth_fn : Callable
        Function mapping arr -> smoothed arr.
    max_abs_dev : int
        Maximum absolute deviation allowed before filtering.

    Returns
    -------
    np.ndarray
        Smoothed + interpolated array.
    """
    x = np.arange(0, arr.shape[0], 1)
    z = smooth_fn(arr)
    if len(x) != len(arr) or len(z) != len(x):
        raise ValueError(
            'Expected lengths of `arr`, `x`, and `z` to match : '
            + f'{len(arr)}, {len(x)}, {len(z)}'
        )
    dev_mask = np.abs(arr - z) < max_abs_dev
    interp = CubicSpline(x[dev_mask], z[dev_mask])
    z = interp(x)
    return z


def select_min_deviation(arrs, smooth_fn, max_abs_dev):
    """
    From candidate edge arrays, pick the one with least std(arr - smooth(arr)).

    Parameters
    ----------
    arrs : list[np.ndarray]
        Candidate 1D edge arrays.
    smooth_fn : Callable
        Smoothing function compatible with `smooth_remove_abs_deviation`.
    max_abs_dev : int
        Maximum deviation threshold for filtering.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (original_array, smoothed_array) of the selected edge.
    """
    min_arrs = None
    min_dev = np.inf
    for arr in arrs:
        z = smooth_remove_abs_deviation(arr, smooth_fn, max_abs_dev=max_abs_dev)
        dev = np.std(arr - z)
        if min_arrs is None or dev < min_dev:
            min_arrs = (arr, z)
            min_dev = dev
    return min_arrs


def take_quantile(thresh_arr, q):
    """
    Select a single per-column edge by column-wise q-quantile.

    Parameters
    ----------
    thresh_arr : np.ndarray
        2D stack of threshold edges.
    q : float
        Quantile in (0,1).

    Returns
    -------
    np.ndarray
        1D edge array.
    """
    if not isinstance(q, float):
        raise TypeError(
            f'Expected float for `q`, recieved {type(q)}'
        )
    if 0 >= q >= 1:
        raise ValueError(
            f'Expected `q` to be between 0 and 1, noninclusive, not {q}'
        )
    
    line = np.nanquantile(thresh_arr, q, axis=0)
    return line


def measure_thresholds(
    arr, 
    qs, 
    lower_cutoff, 
    select_min, 
    exact_thresh, 
    axis, 
    lowess_window_size, 
    max_abs_dev
):
    """
    Stack thresholds, mask low indices, take quantiles, and select a single edge.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D image (time × range).
    qs : iterable[float]
        Quantiles to extract per column.
    lower_cutoff : int
        Minimum allowed index before masking to NaN.
    select_min : bool
        Select minimum or maximum over the mask when building edges.
    exact_thresh : bool
        If True, mask uses "<= threshold"; else "!= threshold".
    axis : int
        Axis along which edges are computed.
    lowess_window_size : int
        LOWESS window (array units).
    max_abs_dev : int
        Absolute deviation threshold for robust smoothing.

    Returns
    -------
    tuple
        (med_lines, min_line, minz_line)
    """
    thresh_edge_arr = stack_all_thresholds(
        arr,
        select_min=select_min,
        exact_thresh=exact_thresh,
        axis=axis,
    )
    
    thresh_edge_arr = thresh_edge_arr.astype(np.float32)
    thresh_edge_arr[thresh_edge_arr < lower_cutoff] = np.nan   
    
    # qs must be an iterable of floats
    if isinstance(qs, float):
        qs = [qs]

    med_lines = [take_quantile(thresh_edge_arr, q) for q in qs]

    # Make a smoothing function that uses the provided window size
    smooth_fn = lambda a: lowess_smooth(a, lowess_window_size, None)

    min_line, minz_line = select_min_deviation(
        med_lines, 
        smooth_fn, 
        max_abs_dev=max_abs_dev
    )
    
    return med_lines, min_line, minz_line


def scale_km(edge, ranges):
    """
    Convert edge indices (array space) to physical distances (km).

    Parameters
    ----------
    edge : np.ndarray
        Edge indices in array space.
    ranges : np.ndarray
        Ground range vector (km).

    Returns
    -------
    np.ndarray
        Edge locations in km.
    """
    ranges  = np.array(ranges) 
    edge_km = (edge / len(ranges) * ranges.ptp()) + ranges.min()
    return edge_km


