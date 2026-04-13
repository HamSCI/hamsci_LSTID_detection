import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from operator import itemgetter
from scipy.signal import butter, filtfilt
from numpy.polynomial import polynomial as poly
from scipy.optimize import curve_fit

from scripts.data_structures import EdgeDetectionOutput, FitOutput


def sinusoid(tt_sec, T_hr, amplitude_km, phase_hr, offset_km, slope_kmph):
    """
    Sinusoid function for curve fitting.

    Parameters
    ----------
    tt_sec : np.ndarray
        Time in seconds since midnight.
    T_hr : float
        Period in hours.
    amplitude_km : float
        Amplitude in km.
    phase_hr : float
        Phase in hours.
    offset_km : float
        Vertical offset in km.
    slope_kmph : float
        Linear drift in km/hour.

    Returns
    -------
    np.ndarray
        Sinusoid values.
    """
    phase_rad = (2. * np.pi) * (phase_hr / T_hr)
    freq = 1. / (timedelta(hours=T_hr).total_seconds())
    return (np.abs(amplitude_km) * np.sin((2 * np.pi * tt_sec * freq) + phase_rad) +
            (slope_kmph / 3600.) * tt_sec + offset_km)


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply Butterworth bandpass filter.

    Parameters
    ----------
    data : np.ndarray
        Input signal.
    lowcut : float
        Low cutoff frequency (Hz).
    highcut : float
        High cutoff frequency (Hz).
    fs : float
        Sampling frequency (Hz).
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)


def islandinfo(y, trigger_val, stopind_inclusive=True):
    """
    Find continuous regions (islands) where y == trigger_val.

    Parameters
    ----------
    y : np.ndarray
        Boolean or comparison array.
    trigger_val : bool
        Value to match.
    stopind_inclusive : bool
        Include stop index in island.

    Returns
    -------
    tuple
        (list of (start, stop) tuples, array of island lengths)
    """
    y_ext = np.r_[False, y == trigger_val, False]
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])
    lens = idx[1::2] - idx[:-1:2]
    return list(zip(idx[:-1:2], idx[1::2] - int(stopind_inclusive))), lens


def compute_stability(edge_positions: np.ndarray, roll_win: int) -> np.ndarray:
    """
    Compute rolling coefficient of variation (std/mean) as a stability metric.

    Parameters
    ----------
    edge_positions : np.ndarray
        Detected edge positions in km.
    roll_win : int
        Rolling window size in samples.

    Returns
    -------
    np.ndarray
        Stability array (NaN for first roll_win-1 points).
    """
    n = len(edge_positions)
    stability = np.full(n, np.nan)
    for i in range(roll_win - 1, n):
        window = edge_positions[i - roll_win + 1:i + 1]
        mean_val = np.mean(window)
        std_val = np.std(window)
        stability[i] = std_val / mean_val if mean_val != 0 else np.inf
    return stability


def find_fit_window(
    stability: np.ndarray,
    edge_times: np.ndarray,
    winlim: tuple,
    stab_thresh: float,
    margin_minutes: int,
) -> Optional[Tuple[np.ndarray, np.datetime64, np.datetime64]]:
    """
    Find the longest stable region within the analysis window, with safety margins.

    Parameters
    ----------
    stability : np.ndarray
        Rolling CV stability metric.
    edge_times : np.ndarray
        Uniform time grid (datetime64[ns]).
    winlim : tuple
        (win_0, win_1) analysis window limits.
    stab_thresh : float
        Maximum CV for a point to be considered stable.
    margin_minutes : int
        Safety margin applied inward from window edges.

    Returns
    -------
    (fit_mask, fitWin_0, fitWin_1) or None if no valid window found.
    """
    win_0, win_1 = winlim
    window_mask = (edge_times >= win_0) & (edge_times < win_1)
    stable_mask = (stability < stab_thresh) & window_mask

    islands, island_lengths = islandinfo(stable_mask, True)
    if len(islands) == 0:
        return None

    isl_idx = np.argmax(island_lengths)
    sInx, eInx = islands[isl_idx]

    margin = np.timedelta64(margin_minutes, 'm')
    fitWin_0 = max(edge_times[sInx], win_0 + margin)
    fitWin_1 = min(edge_times[eInx], win_1 - margin)

    if fitWin_0 >= fitWin_1:
        return None

    fit_mask = (edge_times >= fitWin_0) & (edge_times < fitWin_1)
    if np.sum(fit_mask) == 0:
        return None

    return fit_mask, fitWin_0, fitWin_1


def fit_polynomial(
    tt_sec: np.ndarray,
    fit_data: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """
    Fit a 2nd-degree polynomial and detrend the data.

    Parameters
    ----------
    tt_sec : np.ndarray
        Time in seconds since midnight.
    fit_data : np.ndarray
        Edge positions in km within fit window.

    Returns
    -------
    poly_fit_vals : np.ndarray
        Polynomial fit evaluated at tt_sec.
    poly_params : dict
        Polynomial coefficients (c_0, c_1, c_2) and R².
    data_detrend : np.ndarray
        fit_data minus the polynomial trend.
    """
    coefs, [ss_res, *_] = poly.polyfit(tt_sec, fit_data, 2, full=True)
    poly_fit_vals = poly.polyval(tt_sec, coefs)

    ss_tot = np.sum((fit_data - np.mean(fit_data)) ** 2)
    poly_params = {f'c_{i}': coef for i, coef in enumerate(coefs)}
    poly_params['r2'] = 1 - (ss_res[0] / ss_tot)

    data_detrend = fit_data - poly_fit_vals
    return poly_fit_vals, poly_params, data_detrend


def apply_bandpass(
    data_detrend: np.ndarray,
    lstid_T_hr_lim: tuple,
    fs: float,
) -> np.ndarray:
    """
    Apply a bandpass filter to detrended edge data.

    Parameters
    ----------
    data_detrend : np.ndarray
        Detrended edge positions in km.
    lstid_T_hr_lim : tuple
        (min_T_hr, max_T_hr) period limits for bandpass.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Bandpass-filtered detrended data.
    """
    lowcut  = 1 / (lstid_T_hr_lim[1] * 3600)
    highcut = 1 / (lstid_T_hr_lim[0] * 3600)
    return bandpass_filter(data_detrend, lowcut, highcut, fs, order=4)


def fit_sinusoids(
    tt_sec: np.ndarray,
    data_detrend: np.ndarray,
    T_hr_guesses: np.ndarray,
) -> List[Dict[str, float]]:
    """
    Try sinusoidal fits across multiple period guesses, return all sorted by R².

    Parameters
    ----------
    tt_sec : np.ndarray
        Time in seconds since midnight.
    data_detrend : np.ndarray
        Detrended (and optionally bandpass-filtered) edge in km.
    T_hr_guesses : np.ndarray
        Period guesses in hours to try.

    Returns
    -------
    List of fit parameter dicts sorted by R² descending.
    Empty list if all fits fail.
    """
    all_fits = []
    for T_hr_guess in T_hr_guesses:
        p0 = [
            T_hr_guess,
            np.ptp(data_detrend) / 2.,
            0.,
            np.mean(data_detrend),
            0.,
        ]
        try:
            sin_params, *_ = curve_fit(sinusoid, tt_sec, data_detrend, p0=p0, full_output=True)
            fit_result = {
                'T_hr':         sin_params[0],
                'amplitude_km': np.abs(sin_params[1]),
                'phase_hr':     sin_params[2],
                'offset_km':    sin_params[3],
                'slope_kmph':   sin_params[4],
            }
            sin_vals = sinusoid(tt_sec, **fit_result)
            ss_res = np.sum((data_detrend - sin_vals) ** 2)
            ss_tot = np.sum((data_detrend - np.mean(data_detrend)) ** 2)
            fit_result['r2'] = 1 - (ss_res / ss_tot)
            fit_result['T_hr_guess'] = T_hr_guess
            all_fits.append(fit_result)
        except Exception:
            continue

    return sorted(all_fits, key=itemgetter('r2'), reverse=True)


def sin_fit(
    edge_data: EdgeDetectionOutput,
    *,
    bandpass: bool,
    lstid_T_hr_lim: tuple,
    roll_win: int,
    stab_thresh: float,
    margin_minutes: int,
    T_hr_guesses: np.ndarray = None,
    **_: Any,
) -> FitOutput:
    """
    Fit sinusoid to detected edge with stability-based windowing.

    Pipeline
    --------
    1. compute_stability  — rolling CV metric
    2. find_fit_window    — longest stable region + safety margins
    3. fit_polynomial     — 2nd-degree detrend
    4. apply_bandpass     — optional 1–4.5 hr bandpass
    5. fit_sinusoids      — multi-guess curve fit, best R² selected
    """
    if T_hr_guesses is None:
        T_hr_guesses = np.arange(1, 4.5, 0.5)

    edge_times     = edge_data.edge_times
    edge_positions = edge_data.edge_positions
    date           = edge_data.meta['date']
    winlim         = edge_data.meta['time_limits']['winlim']
    intermediate   = {**edge_data.intermediate}

    # --- 1) Stability ---
    stability = compute_stability(edge_positions, roll_win)

    # --- 2) Find fit window ---
    window_result = find_fit_window(stability, edge_times, winlim, stab_thresh, margin_minutes)
    if window_result is None:
        return _empty_fit_result(edge_data, edge_times, stability)

    fit_mask, fitWin_0, fitWin_1 = window_result
    fit_times = edge_times[fit_mask]
    fit_data  = edge_positions[fit_mask]

    t0     = datetime(date.year, date.month, date.day)
    tt_sec = np.array([
        (t - np.datetime64(t0, 'ns')).astype('timedelta64[s]').astype(float)
        for t in fit_times
    ])

    # --- 3) Polynomial detrend ---
    poly_fit_vals, poly_params, data_detrend = fit_polynomial(tt_sec, fit_data)

    # --- 4) Optional bandpass ---
    data_detrend_no_bp = data_detrend.copy()
    if bandpass:
        fs = 1 / 60  # 1 sample per minute
        data_detrend = apply_bandpass(data_detrend, lstid_T_hr_lim, fs)

    intermediate['data_detrend_no_bp'] = data_detrend_no_bp

    # --- 5) Sinusoidal fitting ---
    all_sin_fits = fit_sinusoids(tt_sec, data_detrend, T_hr_guesses)

    if all_sin_fits:
        sin_params   = all_sin_fits[0].copy()
        sin_fit_vals = sinusoid(tt_sec, **{k: sin_params[k] for k in
                                ['T_hr', 'amplitude_km', 'phase_hr', 'offset_km', 'slope_kmph']})
    else:
        sin_params   = {}
        sin_fit_vals = np.full(len(tt_sec), np.nan)

    meta = {
        **edge_data.meta,
        'fit_params': {
            'bandpass':       bandpass,
            'lstid_T_hr_lim': lstid_T_hr_lim,
            'roll_win':       roll_win,
            'stab_thresh':    stab_thresh,
            'margin_minutes': margin_minutes,
        },
        'fitWinLim': (fitWin_0, fitWin_1),
    }

    return FitOutput(
        fit_times    = fit_times,
        sin_fit      = sin_fit_vals,
        poly_fit     = poly_fit_vals,
        data_detrend = data_detrend,
        stability    = stability,
        sin_params   = sin_params,
        poly_params  = poly_params,
        all_sin_fits = all_sin_fits,
        edge_data    = edge_data,
        intermediate = intermediate,
        meta         = meta,
    )


def _empty_fit_result(edge_data, edge_times, stability):
    """Return empty FitOutput when no stable region is found."""
    return FitOutput(
        fit_times    = np.array([], dtype='datetime64[ns]'),
        sin_fit      = np.array([]),
        poly_fit     = np.array([]),
        data_detrend = np.array([]),
        stability    = stability,
        sin_params   = {},
        poly_params  = {},
        all_sin_fits = [],
        edge_data    = edge_data,
        intermediate = {**edge_data.intermediate, 'data_detrend_no_bp': np.array([])},
        meta         = {**edge_data.meta, 'fit_params': {}, 'fitWinLim': (None, None)},
    )
