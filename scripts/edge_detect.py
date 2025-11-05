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



import numpy as np
import polars as pl
from datetime import datetime, timedelta

def detect_edge(
    date,                                # a datetime.date or datetime.datetime (local day start)
    arr: np.ndarray,                     # preprocessed heatmap (time x range) - only used by measure_thresholds
    arr_times: np.ndarray,               # datetime64[ns], length T, aligned to arr rows (native sampling)
    ranges_km: np.ndarray,               # length H, aligned to arr cols
    Ts_sec: float,                       # sampling interval seconds
    Ts_td: "timedelta",
    *,
    qs: list[float],
    lower_cutoff: int,
    select_min: bool,
    exact_thresh: bool,
    axis: int,
    lowess_window_size: int,
    max_abs_dev: int,
    **_: dict
) -> pl.DataFrame:

    # 1) thresholds in index space
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

    # 2) indices -> km
    med_lines = [scale_km(x, ranges_km) for x in med_lines_idx]
    min_line  = scale_km(min_line_idx, ranges_km)
    minz_line = scale_km(minz_line_idx, ranges_km)

    # 3) time limits
    if isinstance(date, datetime):
        day0 = datetime(date.year, date.month, date.day)
    else:
        day0 = datetime(date.year, date.month, date.day)
    x_0   = day0 + timedelta(hours=12)  # 12:00
    x_1   = day0 + timedelta(hours=24)  # 24:00 (next day 00:00)
    win_0 = day0 + timedelta(hours=13)  # 13:00
    win_1 = day0 + timedelta(hours=23)  # 23:00

    # 4) native-sample DF (wide)
    med_cols = [f"med_{q}" for q in qs]
    base = pl.DataFrame({"Time": arr_times.astype("datetime64[ns]")})
    for c, colvals in zip(med_cols, med_lines):
        base = base.with_columns(pl.Series(c, np.asarray(colvals, dtype=float)))
    base = base.with_columns(
        pl.Series("min_line",  np.asarray(min_line,  dtype=float)),
        pl.Series("minz_line", np.asarray(minz_line, dtype=float)),
    )

    # edge_0: interpolate internal NaNs on native sampling; unmatched remain null -> then 0.0
    base = base.with_columns(
        pl.col("min_line").interpolate().fill_null(0.0).alias("edge_0")
    )

    # --- 5) uniform grid 12:00→24:00 (left-closed) ---
    dt_s = Ts_sec if Ts_sec is not None else float(Ts_td.total_seconds())
    step_ns = int(dt_s * 1e9)

    t0_ns = np.int64(np.datetime64(x_0, 'ns'))
    t1_ns = np.int64(np.datetime64(x_1, 'ns'))

    grid_ns = np.arange(t0_ns, t1_ns, step_ns, dtype=np.int64)  # left-closed
    time_np = grid_ns.view('datetime64[ns]')

    grid = pl.DataFrame({"Time": time_np})

    # --- 6) join + interpolate + window zeroing (clean) ---
    df = (
        grid.join(
            base.select(["Time", "edge_0", "min_line", "minz_line", *med_cols]),
            on="Time",
            how="left",
        )
        .with_columns(
            pl.col("edge_0")
            .interpolate()
            .fill_null(strategy="forward")
            .fill_null(strategy="backward")
            .alias("edge_1")
        )
        .with_columns(
            pl.when((pl.col("Time") >= win_0) & (pl.col("Time") < win_1))
            .then(pl.col("edge_1"))
            .otherwise(0.0)
            .alias("sg_edge"),
            pl.col("Time").cast(pl.Int64).alias("epoch_ns"),
        )
    )

    pl.Config.set_tbl_rows(1_000_000)     # max rows to show
    pl.Config.set_tbl_cols(10_000)        # max cols to show
    pl.Config.set_tbl_width_chars(10_000) # allow super-wide tables
    print(df)
    return df, {"xlim": (x_0, x_1), "winlim": (win_0, win_1), "qs": qs}


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


