from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Any, List
import numpy as np


@dataclass
class PreprocessOutput:
    """
    Output from heatmap preprocessing pipeline.
    
    Attributes
    ----------
    arr : np.ndarray
        Preprocessed array, dtype uint8, shape (range_bins, time_bins).
    arr_times : np.ndarray
        1D datetime64[ns] array of bin LEFT EDGES.
    ranges_km : np.ndarray
        1D array of range bin centers in km.
    Ts_sec : float
        Sampling interval in seconds.
    Ts_td : timedelta
        Sampling interval as timedelta.
    intermediate : Dict[str, Any]
        Intermediate preprocessing steps for plotting/debugging.
    meta : Dict[str, Any]
        Preprocessing parameters and loader metadata.
    """
    arr: np.ndarray
    arr_times: np.ndarray
    ranges_km: np.ndarray
    Ts_sec: float
    Ts_td: timedelta
    intermediate: Dict[str, Any] = field(default_factory=dict)  
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeDetectionOutput:
    """
    Output from edge detection algorithm.
    
    Contains detected edge positions and all threshold lines for plotting.
    
    Attributes
    ----------
    edge_times : np.ndarray
        Uniform time grid from 12:00-24:00 (datetime64[ns]).
    edge_positions : np.ndarray
        Detected edge positions in km, aligned with edge_times.
    quantile_lines : np.ndarray
        2D array (time, n_quantiles) of threshold lines in km.
    quantile_values : np.ndarray
        Quantile values used (e.g., [0.4, 0.5, 0.6]).
    min_line : np.ndarray
        Selected minimum deviation line in km (before smoothing).
    minz_line : np.ndarray
        Smoothed version of selected line in km.
    ranges_km : np.ndarray
        Range bins (passed through from preprocessing).
    intermediate : Dict[str, Any]
        Intermediate data for plotting:
        - 'preprocessed_arr': input array from preprocessing
        - 'preprocessed_times': original time array
        - 'edge_0_vals': edge before windowing
        - 'edge_0_times': times for edge_0
        - 'edge_win_vals': windowed edge (13:00-23:00)
        - 'edge_win_times': times for windowed edge
        - 'window_mask': boolean mask for analysis window
    meta : Dict[str, Any]
        Detection parameters, time limits, and inherited metadata.
    """
    edge_times: np.ndarray
    edge_positions: np.ndarray
    quantile_lines: np.ndarray
    quantile_values: np.ndarray
    min_line: np.ndarray
    minz_line: np.ndarray
    ranges_km: np.ndarray
    intermediate: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FitOutput:
    """
    Output from sinusoidal fitting with polynomial detrending.
    
    Attributes
    ----------
    fit_times : np.ndarray
        Timestamps for fitted data (datetime64[ns]).
    sin_fit : np.ndarray
        Best sinusoidal fit values in km.
    poly_fit : np.ndarray
        2nd degree polynomial fit for detrending in km.
    data_detrend : np.ndarray
        Detrended data used for sin fitting in km.
    stability : np.ndarray
        Coefficient of variation (rolling std/mean), full edge_times length.
    sin_params : Dict[str, float]
        Best fit parameters: T_hr, amplitude_km, phase_hr, offset_km, 
        slope_kmph, r2, T_hr_guess (initial guess used).
    poly_params : Dict[str, float]
        Polynomial coefficients: c_0, c_1, c_2, r2.
    all_sin_fits : List[Dict[str, float]]
        All attempted fits sorted by R² descending.
    edge_data : EdgeDetectionOutput
        Complete edge detection output (carries all previous pipeline data).
    intermediate : Dict[str, Any]
        All intermediate data from previous stages for plotting.
    meta : Dict[str, Any]
        Fit parameters (fitWinLim, bandpass settings) and inherited metadata.
    """
    fit_times: np.ndarray
    sin_fit: np.ndarray
    poly_fit: np.ndarray
    data_detrend: np.ndarray
    stability: np.ndarray
    sin_params: Dict[str, float]
    poly_params: Dict[str, float]
    all_sin_fits: List[Dict[str, float]]
    edge_data: EdgeDetectionOutput
    intermediate: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)