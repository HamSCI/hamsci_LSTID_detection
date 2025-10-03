#!/usr/bin/env python3

from typing import Tuple, List, Iterable, Dict, Any
import numpy as np
import polars as pl
import statsmodels.api as sm
from scipy.interpolate import CubicSpline

def edge_detection(
    arr: np.ndarray,
    arr_times: np.ndarray,
    ranges_km: np.ndarray,
    Ts_sec: float,
    Ts_td: "timedelta",
    *,
    qs: List[float],
    occurrence_n: int,
    i_max: int,
    **_: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Main entrypoint for edge detection.

    Pipeline
    --------
        1. Threshold stacking and selection (`measure_thresholds`).
        2. Convert indices to km (`scale_km`).
        3. Build extent, quantile columns, and polars DataFrame.
        4. (Further stages to be added later).

    Parameters
    ----------
    arr : np.ndarray
        Preprocessed 2D heatmap array (time x range).
    arr_times : np.ndarray
        1D array of datetime-like objects or timestamps.
    ranges_km : np.ndarray
        1D array of range bin centers in km.
    Ts_sec : float
        Sampling interval in seconds.
    Ts_td : timedelta
        Sampling interval as timedelta.
    qs : list[float], default=(0.8,)
        Quantiles to extract from threshold stack.
    occurrence_n : int, default=60
        Number of top pixels excluded when computing robust max for rescaling.
    i_max : int, default=30
        Maximum value for rescaling intensities (≤ 255).
    **_ : dict
        Extra unused parameters are ignored.

    Returns
    -------
    dict
        Dictionary containing:
          - med_lines : list[np.ndarray], quantile edges (scaled to km)
          - min_line  : np.ndarray, edge with minimal deviation (scaled to km)
          - minz_line : np.ndarray, smoothed version of min_line
          - extent    : list, [t_start, t_end, range_min, range_max]
          - med_lines_df : polars.DataFrame with quantile lines over time
    """

    # --- compute thresholds ---
    med_lines, min_line, minz_line = measure_thresholds(
        arr,
        qs=qs,
        occurrence_n=occurrence_n,
        i_max=i_max,
    )

    # --- convert edges to km ---
    med_lines = [scale_km(x, ranges_km) for x in med_lines]
    min_line  = scale_km(min_line, ranges_km)
    minz_line = scale_km(minz_line, ranges_km)

    # --- plotting extent (time, range) ---
    extent = [arr_times[0], arr_times[-1], float(ranges_km[0]), float(ranges_km[-1])]

    # --- build quantile DataFrame ---
    med_cols = [str(q) for q in qs]
    med_arr  = np.vstack(med_lines).T

    med_lines_df = (
        pl.DataFrame(med_arr, schema=med_cols)
        .with_columns(pl.Series("Time", arr_times))
        .select(["Time"] + med_cols)
    )

    return {
        "med_lines": med_lines,
        "min_line": min_line,
        "minz_line": minz_line,
        "extent": extent,
        "med_lines_df": med_lines_df,
    }

def stack_all_thresholds(
    arr: np.ndarray,
    select_min: bool = True,
    exact_thresh: bool = False,
    axis: int = 0,
) -> np.ndarray:
    """
    Convert preprocessed array into stacked thresholds.

    Steps:
        1. Iterate over unique values in arr.
        2. Build threshold masks for each value.
        3. Select min/max index along axis.
        4. Concatenate into 2D stack.

    Parameters
    ----------
    arr : np.ndarray
        Preprocessed input image, float or int dtype.
    select_min : bool, default=True
        Whether to select minimum (True) or maximum (False) edge.
    exact_thresh : bool, default=False
        If True, select pixels <= threshold, else use != threshold.
    axis : int, default=0
        Axis to compute edge along.

    Returns
    -------
    np.ndarray
        2D stack of threshold edge arrays.
    """
    thresholds = np.unique(arr)
    thresh_edges = []
    for threshold in thresholds:
        if exact_thresh:
            thresh_mask = arr <= threshold
        else:
            thresh_mask = arr != threshold

        idx_fn = np.argmin if select_min else np.argmax
        thresh_edge = idx_fn(thresh_mask.astype(np.uint8), axis=axis, keepdims=True)

        if max(thresh_edge.shape) != max(arr.shape):
            raise ValueError(
                f"Expected largest dim in thresh_edge {thresh_edge.shape} "
                f"to match arr {arr.shape}"
            )
        thresh_edges.append(thresh_edge)

    return np.concatenate(thresh_edges, axis=axis)

def lowess_smooth(
    arr: np.ndarray,
    window_size: int = 10,
    x: np.ndarray = None
) -> np.ndarray:
    """
    LOWESS smoothing wrapper.

    Parameters
    ----------
    arr : np.ndarray
        1D array of values to smooth.
    window_size : int, default=10
        Window size in array units (converted to fraction).
    x : np.ndarray, optional
        Optional 1D x-axis array. If None, evenly spaced.

    Returns
    -------
    np.ndarray
        Smoothed array.
    """
    if x is None:
        x = np.linspace(0, len(arr), len(arr))
    frac = window_size / len(arr)
    return sm.nonparametric.lowess(arr, x, frac=frac, return_sorted=False)


def smooth_remove_abs_deviation(
    arr: np.ndarray,
    smooth_fn,
    max_abs_dev: int = 20
) -> np.ndarray:
    """
    Smooth an array and remove points with excessive deviation.

    Parameters
    ----------
    arr : np.ndarray
        1D threshold array.
    smooth_fn : Callable
        Function that smooths array (takes arr as input).
    max_abs_dev : int, default=20
        Max absolute deviation threshold.

    Returns
    -------
    np.ndarray
        Smoothed and interpolated array.
    """
    x = np.arange(0, arr.shape[0], 1)
    z = smooth_fn(arr)
    if len(x) != len(arr) or len(z) != len(x):
        raise ValueError("Length mismatch in arr/x/z.")

    dev_mask = np.abs(arr - z) < max_abs_dev
    interp = CubicSpline(x[dev_mask], z[dev_mask])
    return interp(x)


def select_min_deviation(
    arrs: List[np.ndarray],
    smooth_fn,
    max_abs_dev: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select edge with minimal deviation vs smoothed version.

    Parameters
    ----------
    arrs : list[np.ndarray]
        Candidate edge arrays.
    smooth_fn : Callable
        Smoothing function passed to smooth_remove_abs_deviation.
    max_abs_dev : int, default=20
        Max deviation allowed for filtering.

    Returns
    -------
    tuple
        (original array, smoothed array) with minimal deviation.
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

def take_quantile(
    thresh_arr: np.ndarray,
    q: float
) -> np.ndarray:
    """
    Select quantile from threshold stack.

    Parameters
    ----------
    thresh_arr : np.ndarray
        2D stacked thresholds.
    q : float
        Quantile value in (0,1).

    Returns
    -------
    np.ndarray
        Selected edge line.
    """
    if not isinstance(q, float):
        raise TypeError(f"Expected float for q, got {type(q)}")
    if q <= 0 or q >= 1:
        raise ValueError(f"q must be between 0 and 1, got {q}")

    return np.nanquantile(thresh_arr, q, axis=0)


def measure_thresholds(
    arr: np.ndarray,
    qs: Iterable[float] = (0.8,),
    lower_cutoff: int = 10,
    **threshold_kwargs
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """
    Calculate thresholds and select edges.

    Parameters
    ----------
    arr : np.ndarray
        Input 2D image.
    qs : iterable of float, default=(0.8,)
        Quantiles to extract.
    lower_cutoff : int, default=10
        Minimum y-axis allowed for edges.
    **threshold_kwargs : dict
        Passed to stack_all_thresholds.

    Returns
    -------
    med_lines : list[np.ndarray]
        Detected edges for each quantile.
    min_line : np.ndarray
        Line with minimal deviation.
    minz_line : np.ndarray
        Smoothed version of min_line.
    """
    thresh_edge_arr = stack_all_thresholds(arr, **threshold_kwargs)
    thresh_edge_arr = thresh_edge_arr.astype(np.float32)
    thresh_edge_arr[thresh_edge_arr < lower_cutoff] = np.nan

    if isinstance(qs, float):
        qs = [qs]

    med_lines = [take_quantile(thresh_edge_arr, q) for q in qs]
    min_line, minz_line = select_min_deviation(med_lines, lowess_smooth)

    return med_lines, min_line, minz_line

def scale_km(
    edge: np.ndarray,
    ranges: np.ndarray
) -> np.ndarray:
    """
    Convert edge indices to km.

    Parameters
    ----------
    edge : np.ndarray
        Edge indices.
    ranges : np.ndarray
        Ground range vector.

    Returns
    -------
    np.ndarray
        Edge positions in km.
    """
    ranges = np.array(ranges)
    return (edge / len(ranges) * ranges.ptp()) + ranges.min()
