import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List
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
        Time in seconds since midnight
    T_hr : float
        Period in hours
    amplitude_km : float
        Amplitude in km
    phase_hr : float
        Phase in hours
    offset_km : float
        Vertical offset in km
    slope_kmph : float
        Linear drift in km/hour
        
    Returns
    -------
    np.ndarray
        Sinusoid values
    """
    phase_rad = (2. * np.pi) * (phase_hr / T_hr)
    freq = 1. / (timedelta(hours=T_hr).total_seconds())
    result = (np.abs(amplitude_km) * np.sin((2 * np.pi * tt_sec * freq) + phase_rad) +
              (slope_kmph / 3600.) * tt_sec + offset_km)
    return result


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply Butterworth bandpass filter.
    
    Parameters
    ----------
    data : np.ndarray
        Input signal
    lowcut : float
        Low cutoff frequency (Hz)
    highcut : float
        High cutoff frequency (Hz)
    fs : float
        Sampling frequency (Hz)
    order : int
        Filter order
        
    Returns
    -------
    np.ndarray
        Filtered signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)
    return filtered


def islandinfo(y, trigger_val, stopind_inclusive=True):
    """
    Find continuous regions (islands) where y == trigger_val.
    
    Parameters
    ----------
    y : np.ndarray
        Boolean or comparison array
    trigger_val : bool
        Value to match
    stopind_inclusive : bool
        Include stop index in island
        
    Returns
    -------
    tuple
        (list of (start, stop) tuples, array of island lengths)
    """
    y_ext = np.r_[False, y == trigger_val, False]
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])
    lens = idx[1::2] - idx[:-1:2]
    return list(zip(idx[:-1:2], idx[1::2] - int(stopind_inclusive))), lens


def sin_fit(
    edge_data: EdgeDetectionOutput,
    *,
    bandpass: bool,
    lstid_T_hr_lim: tuple,
    roll_win: int,
    stab_thresh: float,
    margin_minutes: int,
    T_hr_guesses: np.ndarray = None,
    **_: Any
) -> FitOutput:
    """
    Fit sinusoid to detected edge with stability-based windowing.
    
    Pipeline
    --------
    1. Compute stability (coefficient of variation in rolling window)
    2. Find longest stable region meeting threshold
    3. Apply margins to avoid edge effects
    4. Fit 2nd degree polynomial for detrending
    5. Optionally apply bandpass filter
    6. Try multiple sinusoidal periods, select best R²
    
    Parameters
    ----------
    edge_data : EdgeDetectionOutput
        Output from detect_edge
    bandpass : bool
        Apply bandpass filter after detrending
    lstid_T_hr_lim : tuple
        (min_period_hr, max_period_hr) for bandpass cutoffs
    roll_win : int
        Rolling window size for stability calculation (minutes)
    stab_thresh : float
        Stability threshold (coefficient of variation)
    margin_minutes : int
        Safety margin from window edges for fitting
    T_hr_guesses : np.ndarray, optional
        Period guesses for sin fitting (default: np.arange(1, 4.5, 0.5))
        
    Returns
    -------
    FitOutput
        Fitted sinusoid, polynomial, detrended data, and all parameters
    """
    if T_hr_guesses is None:
        T_hr_guesses = np.arange(1, 4.5, 0.5)
    
    # Unpack edge data
    edge_times = edge_data.edge_times
    edge_positions = edge_data.edge_positions
    date = edge_data.meta['date']
    winlim = edge_data.meta['time_limits']['winlim']
    
    # Initialize intermediate dict with all previous data
    intermediate = {**edge_data.intermediate}
    
    # --- 1) Compute stability (coefficient of variation) ---
    # Rolling std / rolling mean
    # Convert to simple indexing (edge_times is uniform grid)
    n = len(edge_positions)
    stability = np.full(n, np.nan)
    
    for i in range(roll_win - 1, n):
        window = edge_positions[i - roll_win + 1:i + 1]
        mean_val = np.mean(window)
        std_val = np.std(window)
        if mean_val != 0:
            stability[i] = std_val / mean_val
        else:
            stability[i] = np.inf
    
    # DEBUG: Print stability info
    print(f"\n=== STABILITY DEBUG ===")
    print(f"Total edge points: {len(edge_positions)}")
    print(f"Edge position range: {np.nanmin(edge_positions):.1f} to {np.nanmax(edge_positions):.1f} km")
    print(f"Stability range: {np.nanmin(stability):.4f} to {np.nanmax(stability):.4f}")
    print(f"Stability threshold: {stab_thresh}")
    print(f"Points below threshold: {np.sum(stability < stab_thresh)}")
    print(f"Window limits: {winlim}")
    
    # --- 2) Find stable regions ---
    # Only consider data within analysis window
    win_0, win_1 = winlim
    window_mask = (edge_times >= win_0) & (edge_times < win_1)
    print(f"Points in analysis window (13:00-23:00): {np.sum(window_mask)}")
    
    # Stability criteria: must be < threshold AND in analysis window
    stable_mask = (stability < stab_thresh) & window_mask
    print(f"Stable points in window: {np.sum(stable_mask)}")
    
    if np.sum(stable_mask) > 0:
        print(f"First stable index: {np.where(stable_mask)[0][0]}")
        print(f"Last stable index: {np.where(stable_mask)[0][-1]}")
    print(f"======================\n")
    
    # Find islands of stability
    islands, island_lengths = islandinfo(stable_mask, True)
    print(f"Number of islands found: {len(islands)}")
    if len(islands) > 0:
        print(f"Island lengths: {island_lengths}")
        print(f"Longest island: {max(island_lengths)} points")
    if len(islands) == 0:
        # No stable region found - return empty fit
        return _empty_fit_result(edge_data, edge_times, stability)
    
    # Get longest stable island
    isl_idx = np.argmax(island_lengths)
    island = islands[isl_idx]
    sInx, eInx = island
    
    fitWin_0 = edge_times[sInx]
    fitWin_1 = edge_times[eInx]

    print(f"Selected island: indices {sInx} to {eInx}")
    print(f"Initial fit window: {fitWin_0} to {fitWin_1}")
    
    # --- 3) Apply safety margins ---
    margin = np.timedelta64(margin_minutes, 'm')
    if fitWin_0 < (win_0 + margin):
        fitWin_0 = win_0 + margin
    if fitWin_1 > (win_1 - margin):
        fitWin_1 = win_1 - margin

    print(f"After margins: {fitWin_0} to {fitWin_1}")

    # Check if window is still valid after margins
    if fitWin_0 >= fitWin_1:
        print(f"ERROR: Fit window invalid after margins! {fitWin_0} >= {fitWin_1}")
        return _empty_fit_result(edge_data, edge_times, stability)

    # Select data in fit window
    fit_mask = (edge_times >= fitWin_0) & (edge_times < fitWin_1)
    print(f"Points in fit window: {np.sum(fit_mask)}")

    if np.sum(fit_mask) == 0:
        print(f"ERROR: No points in fit window!")
        return _empty_fit_result(edge_data, edge_times, stability)

    fit_times = edge_times[fit_mask]
    fit_data = edge_positions[fit_mask]
    
    print(f"fit_times length: {len(fit_times)}")
    print(f"fit_data length: {len(fit_data)}")
    print(f"fit_data range: {fit_data.min():.1f} to {fit_data.max():.1f} km")
    
    # Convert to seconds since midnight
    t0 = datetime(date.year, date.month, date.day)
    tt_sec = np.array([(t - np.datetime64(t0, 'ns')).astype('timedelta64[s]').astype(float)
                       for t in fit_times])
    
    print(f"tt_sec length: {len(tt_sec)}")
    print(f"tt_sec range: {tt_sec.min():.0f} to {tt_sec.max():.0f} seconds")
    print(f"Starting polynomial fitting...")
    
    # --- 4) Polynomial detrending ---
    try:
        coefs, [ss_res, rank, singular_values, rcond] = poly.polyfit(
            tt_sec, fit_data, 2, full=True
        )
        print(f"Polynomial fit successful!")
        print(f"  Coefficients: {coefs}")
        print(f"  R² = {1 - (ss_res[0] / np.sum((fit_data - np.mean(fit_data))**2)):.4f}")
        
        ss_res_poly = ss_res[0]
        poly_fit_vals = poly.polyval(tt_sec, coefs)
        
        poly_params = {f'c_{i}': coef for i, coef in enumerate(coefs)}
        ss_tot_poly = np.sum((fit_data - np.mean(fit_data))**2)
        r_sqrd_poly = 1 - (ss_res_poly / ss_tot_poly)
        poly_params['r2'] = r_sqrd_poly
        
        # Detrend
        data_detrend = fit_data - poly_fit_vals
        print(f"Data detrended, range: {data_detrend.min():.1f} to {data_detrend.max():.1f} km")
        
        # Save pre-bandpass version
        data_detrend_no_bp = data_detrend.copy()
        
        # --- 5) Optional bandpass filter ---
        if bandpass:
            print(f"Applying bandpass filter...")
            lowcut = 1 / (lstid_T_hr_lim[1] * 3600)
            highcut = 1 / (lstid_T_hr_lim[0] * 3600)
            fs = 1 / 60  # 1 sample per minute
            order = 4
            
            print(f"  lowcut={lowcut:.6f} Hz, highcut={highcut:.6f} Hz, fs={fs:.4f} Hz")
            
            data_detrend = bandpass_filter(
                data_detrend, lowcut, highcut, fs, order
            )
            print(f"Bandpass applied, range: {data_detrend.min():.1f} to {data_detrend.max():.1f} km")
        else:
            # If no bandpass, they're the same
            data_detrend_no_bp = data_detrend
        
        # --- 6) Sinusoidal fitting with multiple period guesses ---
        print(f"Trying {len(T_hr_guesses)} period guesses: {T_hr_guesses}")
        all_sin_fits = []
        for T_hr_guess in T_hr_guesses:
            guess = {
                'T_hr': T_hr_guess,
                'amplitude_km': np.ptp(data_detrend) / 2.,
                'phase_hr': 0.,
                'offset_km': np.mean(data_detrend),
                'slope_kmph': 0.,
            }
            
            try:
                sin_params, pcov, infodict, mesg, ier = curve_fit(
                    sinusoid, tt_sec, data_detrend,
                    p0=list(guess.values()),
                    full_output=True
                )
                
                # Extract parameters
                fit_result = {
                    'T_hr': sin_params[0],
                    'amplitude_km': np.abs(sin_params[1]),
                    'phase_hr': sin_params[2],
                    'offset_km': sin_params[3],
                    'slope_kmph': sin_params[4],
                }
                
                # Compute fit and R²
                sin_fit_vals = sinusoid(tt_sec, **fit_result)
                ss_res_sin = np.sum((data_detrend - sin_fit_vals)**2)
                ss_tot_sin = np.sum((data_detrend - np.mean(data_detrend))**2)
                r_sqrd_sin = 1 - (ss_res_sin / ss_tot_sin)
                
                fit_result['r2'] = r_sqrd_sin
                fit_result['T_hr_guess'] = T_hr_guess
                all_sin_fits.append(fit_result)
                print(f"  T={T_hr_guess:.1f}h: R²={r_sqrd_sin:.4f}, Amp={fit_result['amplitude_km']:.1f}km")
                
            except Exception as e:
                print(f"  T={T_hr_guess:.1f}h: FAILED ({e})")
                continue
        
        # Sort by R² and select best
        if len(all_sin_fits) > 0:
            print(f"Successfully fit {len(all_sin_fits)} sinusoids!")
            all_sin_fits = sorted(all_sin_fits, key=itemgetter('r2'), reverse=True)
            sin_params = all_sin_fits[0].copy()
            print(f"Best fit: T={sin_params['T_hr']:.2f}h, R²={sin_params['r2']:.4f}")
            sin_fit_vals = sinusoid(
                tt_sec,
                T_hr=sin_params['T_hr'],
                amplitude_km=sin_params['amplitude_km'],
                phase_hr=sin_params['phase_hr'],
                offset_km=sin_params['offset_km'],
                slope_kmph=sin_params['slope_kmph']
            )
        else:
            print(f"WARNING: No successful sinusoid fits!")
            # No successful fits
            sin_params = {}
            sin_fit_vals = np.full(len(tt_sec), np.nan)
            poly_fit_vals = np.full(len(tt_sec), np.nan)
            data_detrend = np.full(len(tt_sec), np.nan)
            
    except Exception as e:
        # Fitting failed entirely
        print(f"!!! EXCEPTION IN FITTING: {e}")
        import traceback
        traceback.print_exc()
        all_sin_fits = []
        sin_params = {}
        poly_params = {}
        sin_fit_vals = np.full(len(tt_sec), np.nan)
        poly_fit_vals = np.full(len(tt_sec), np.nan)
        data_detrend = np.full(len(tt_sec), np.nan)
    
    print(f"Building final output...")
    print(f"  sin_params: {sin_params}")
    print(f"  Number of fits tried: {len(all_sin_fits) if 'all_sin_fits' in locals() else 0}")
    
    # Store pre-bandpass detrended data in intermediate
    if 'data_detrend_no_bp' in locals():
        intermediate['data_detrend_no_bp'] = data_detrend_no_bp
    else:
        intermediate['data_detrend_no_bp'] = np.array([])
    
    # --- 7) Build metadata ---
    meta = {
        **edge_data.meta,
        'fit_params': {
            'bandpass': bandpass,
            'lstid_T_hr_lim': lstid_T_hr_lim,
            'roll_win': roll_win,
            'stab_thresh': stab_thresh,
            'margin_minutes': margin_minutes,
        },
        'fitWinLim': (fitWin_0, fitWin_1),
    }
    
    print(f"Returning FitOutput...\n")
    
    return FitOutput(
        fit_times=fit_times,
        sin_fit=sin_fit_vals,
        poly_fit=poly_fit_vals,
        data_detrend=data_detrend,
        stability=stability,
        sin_params=sin_params,
        poly_params=poly_params,
        all_sin_fits=all_sin_fits,
        edge_data=edge_data,
        intermediate=intermediate,
        meta=meta,
    )


def _empty_fit_result(edge_data, edge_times, stability):
    """Helper to return empty fit when no stable region found."""
    return FitOutput(
        fit_times=np.array([], dtype='datetime64[ns]'),
        sin_fit=np.array([]),
        poly_fit=np.array([]),
        data_detrend=np.array([]),
        stability=stability,
        sin_params={},
        poly_params={},
        all_sin_fits=[],
        edge_data=edge_data,
        intermediate={**edge_data.intermediate},
        meta={**edge_data.meta, 'fit_params': {}, 'fitWinLim': (None, None)},
    )