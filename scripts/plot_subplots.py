#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple

from scripts.data_structures import FitOutput


def setup_mpl_style():
    """Configure matplotlib style for all plots."""
    plt.rcParams['font.size'] = 18
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.xmargin'] = 0
    plt.rcParams['axes.titlesize'] = 'x-large'


def fmt_xaxis(ax, xlim, label=True):
    """Format x-axis with hour ticks."""
    ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H%M"))
    if label:
        ax.set_xlabel('Time [UTC]')
    ax.set_xlim(xlim)


def plot_intermediate_heatmap(ax, arr, arr_times, ranges_km, title, cmap, 
                              ylim, xlim, cb_label, add_contours):
    """
    Generic heatmap plotter for any intermediate array.
    
    NO DEFAULTS - all parameters must be provided explicitly.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    arr : np.ndarray
        2D array to plot
    arr_times : np.ndarray
        Time coordinates
    ranges_km : np.ndarray
        Range coordinates
    title : str
        Panel title
    cmap : str
        Colormap name
    ylim : tuple
        Y-axis limits
    xlim : tuple
        X-axis limits
    cb_label : str
        Colorbar label
    add_contours : bool
        Whether to overlay contours
    """
    mpbl = ax.pcolormesh(arr_times, ranges_km, arr, cmap=cmap, 
                         shading='nearest', rasterized=True)
    
    if add_contours:
        levels = np.linspace(np.nanmin(arr), np.nanmax(arr), 15)
        ax.contour(arr_times, ranges_km, arr, levels=levels, 
                   colors='k', linewidths=0.5)
    
    plt.colorbar(mpbl, ax=ax, label=cb_label)
    ax.set_title(title, loc='left')
    ax.set_ylabel('Range [km]')
    ax.set_ylim(ylim)
    fmt_xaxis(ax, xlim)


def plot_heatmap(ax, fit_result: FitOutput, cb_pad: float, ylim: Tuple[float, float]):
    """
    Plot preprocessed heatmap.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    fit_result : FitOutput
        Complete pipeline result
    cb_pad : float
        Colorbar padding
    ylim : tuple
        Y-axis limits (min_km, max_km)
    """
    # Extract data from fit_result
    arr = fit_result.intermediate['preprocessed_arr']
    arr_times = fit_result.intermediate['preprocessed_times']
    ranges_km = fit_result.edge_data.ranges_km
    date = fit_result.meta['date']
    xlim = fit_result.meta['time_limits']['xlim']
    
    # Plot
    mpbl = ax.pcolormesh(arr_times, ranges_km, arr, cmap='plasma')
    plt.colorbar(mpbl, ax=ax, aspect=10, pad=cb_pad, 
                 label='Scaled Amateur Radio Data')
    
    ax.set_title(f'| {date.strftime("%Y-%m-%d")} |')
    ax.set_ylabel('Range [km]')
    ax.set_ylim(ylim)
    
    fmt_xaxis(ax, xlim)


def plot_heatmap_with_edge(ax, fit_result: FitOutput, cb_pad: float, ylim: Tuple[float, float]):
    """
    Plot heatmap with detected edge overlay.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    fit_result : FitOutput
        Complete pipeline result
    cb_pad : float
        Colorbar padding
    ylim : tuple
        Y-axis limits (min_km, max_km)
    """
    # Extract data from fit_result
    arr = fit_result.intermediate['preprocessed_arr']
    arr_times = fit_result.intermediate['preprocessed_times']
    ranges_km = fit_result.edge_data.ranges_km
    edge_0_times = fit_result.intermediate['edge_0_times']
    edge_0_vals = fit_result.intermediate['edge_0_vals']
    stability_times = fit_result.edge_data.edge_times
    stability = fit_result.stability
    xlim = fit_result.meta['time_limits']['xlim']
    winlim = fit_result.meta['time_limits']['winlim']
    fitWinLim = fit_result.meta.get('fitWinLim', (None, None))
    
    # Heatmap
    mpbl = ax.pcolormesh(arr_times, ranges_km, arr, cmap='plasma')
    plt.colorbar(mpbl, ax=ax, aspect=10, pad=cb_pad,
                 label='Scaled Amateur Radio Data')
    
    # Detected edge
    ax.plot(edge_0_times, edge_0_vals, lw=2, label='Detected Edge')
    
    # Sin fit if available
    if fit_result.sin_params:
        fit_times = fit_result.fit_times
        # Reconstruct full edge by adding polynomial back
        full_fit = fit_result.sin_fit + fit_result.poly_fit
        ax.plot(fit_times, full_fit, label='Sin Fit', 
                color='white', lw=3, ls='--')
    
    # Stability on secondary axis
    ax2 = ax.twinx()
    ax2.plot(stability_times, stability, lw=2, color='0.5')
    ax2.grid(False)
    ax2.set_ylabel('Edge Coef. of Variation\n(Grey Line)')
    
    # Window limits
    for wl in winlim:
        ax.axvline(wl, color='0.8', ls='--', lw=2)
    
    # Fit window limits
    if fitWinLim[0] is not None:
        for wl in fitWinLim:
            ax.axvline(wl, color='lime', ls='--', lw=2)
    
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)
    ax.set_ylabel('Range [km]')
    ax.set_ylim(ylim)
    
    fmt_xaxis(ax, xlim)


def plot_detrended_fit(ax, fit_result: FitOutput):
    """
    Plot detrended edge data with sinusoidal fit.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    fit_result : FitOutput
        Complete pipeline result
    """
    if len(fit_result.fit_times) == 0:
        ax.text(0.5, 0.5, 'No fit available', 
                ha='center', va='center', transform=ax.transAxes)
        return
    
    # Extract data
    fit_times = fit_result.fit_times
    data_detrend = fit_result.data_detrend
    sin_fit = fit_result.sin_fit
    xlim = fit_result.meta['time_limits']['xlim']
    fitWinLim = fit_result.meta.get('fitWinLim', (None, None))
    
    # Detrended data
    ax.plot(fit_times, data_detrend, 
            label='Detrended Edge', marker='o', alpha=0.6)
    
    # Sin fit
    if fit_result.sin_params:
        ax.plot(fit_times, sin_fit, 
                label='Sin Fit', color='red', lw=3, ls='--')
    
    # Fit window limits
    if fitWinLim[0] is not None:
        for wl in fitWinLim:
            ax.axvline(wl, color='lime', ls='--', lw=2)
    
    ax.set_ylabel('Range [km]')
    ax.legend(loc='lower right', fontsize='x-small', ncols=4)
    
    fmt_xaxis(ax, xlim)


def plot_fit_parameters(ax, fit_result: FitOutput):
    """
    Display fit parameters as text.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    fit_result : FitOutput
        Complete pipeline result
    """
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    fontdict = {'weight': 'normal', 'family': 'monospace'}
    
    # Polynomial fit parameters
    txt = ['2nd Deg Poly Fit', '(Used for Detrending)']
    for key, val in fit_result.poly_params.items():
        if key == 'r2':
            txt.append(f'{key}: {val:0.2f}')
        else:
            txt.append(f'{key}: {val:0.1f}')
    ax.text(0.01, 0.95, '\n'.join(txt), fontdict=fontdict, va='top')
    
    # Sinusoid fit parameters
    txt = ['Sinusoid Fit']
    for key, val in fit_result.sin_params.items():
        if key == 'r2':
            txt.append(f'{key}: {val:0.2f}')
        elif key == 'T_hr_guess':
            txt.append(f'{key}: {val:0.1f}')
        else:
            txt.append(f'{key}: {val:0.1f}')
    ax.text(0.30, 0.95, '\n'.join(txt), fontdict=fontdict, va='top')


def plot_quantile_lines(ax, fit_result: FitOutput):
    """
    Plot quantile threshold lines.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    fit_result : FitOutput
        Complete pipeline result
    """
    times = fit_result.intermediate['edge_0_times']
    quantile_lines = fit_result.edge_data.quantile_lines
    quantile_values = fit_result.edge_data.quantile_values
    xlim = fit_result.meta['time_limits']['xlim']
    
    for i, q in enumerate(quantile_values):
        ax.plot(times, quantile_lines[:, i], label=f'Q={q}', alpha=0.7)
    
    ax.legend(loc='best', fontsize='small')
    ax.set_ylabel('Range [km]')
    ax.set_title('Quantile Threshold Lines')
    
    fmt_xaxis(ax, xlim)


def plot_stability(ax, fit_result: FitOutput):
    """
    Plot stability (coefficient of variation).
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    fit_result : FitOutput
        Complete pipeline result
    """
    times = fit_result.edge_data.edge_times
    stability = fit_result.stability
    xlim = fit_result.meta['time_limits']['xlim']
    fitWinLim = fit_result.meta.get('fitWinLim', (None, None))
    
    ax.plot(times, stability, lw=2, color='blue')
    
    # Threshold line
    stab_thresh = fit_result.meta.get('fit_params', {}).get('stab_thresh', 0.05)
    ax.axhline(stab_thresh, color='red', ls='--', lw=2, 
               label=f'Threshold ({stab_thresh})')
    
    # Fit window
    if fitWinLim[0] is not None:
        for wl in fitWinLim:
            ax.axvline(wl, color='lime', ls='--', lw=2)
    
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Edge Stability')
    ax.legend(loc='best', fontsize='small')
    
    fmt_xaxis(ax, xlim)


def plot_edge_comparison(ax, fit_result: FitOutput):
    """
    Compare edge_0 (full) vs windowed edge.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    fit_result : FitOutput
        Complete pipeline result
    """
    # Full edge
    edge_0_times = fit_result.intermediate['edge_0_times']
    edge_0_vals = fit_result.intermediate['edge_0_vals']
    ax.plot(edge_0_times, edge_0_vals, label='Full Edge', alpha=0.5)
    
    # Windowed edge
    edge_win_times = fit_result.intermediate['edge_win_times']
    edge_win_vals = fit_result.intermediate['edge_win_vals']
    ax.plot(edge_win_times, edge_win_vals, label='Windowed Edge', lw=2)
    
    # Final interpolated edge
    edge_times = fit_result.edge_data.edge_times
    edge_positions = fit_result.edge_data.edge_positions
    ax.plot(edge_times, edge_positions, label='Interpolated', 
            ls='--', alpha=0.7)
    
    xlim = fit_result.meta['time_limits']['xlim']
    
    ax.legend(loc='best', fontsize='small')
    ax.set_ylabel('Range [km]')
    ax.set_title('Edge Extraction Steps')
    
    fmt_xaxis(ax, xlim)


def plot_bandpass_filtered(ax, fit_result: FitOutput):
    """
    Panel (d): 1-4.5 HR Bandpass Filtered Edge
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    fit_result : FitOutput
        Complete pipeline result
    """
    fit_times = fit_result.fit_times
    data_detrend = fit_result.data_detrend
    xlim = fit_result.meta['time_limits']['xlim']
    fitWinLim = fit_result.meta.get('fitWinLim', (None, None))
    
    if len(fit_times) > 0:
        ax.plot(fit_times, data_detrend, label='Bandpass Filtered')
        
        if fitWinLim[0] is not None:
            for wl in fitWinLim:
                ax.axvline(wl, color='lime', ls='--', lw=2)
    
    fmt_xaxis(ax, xlim)
    ax.set_title("(d) 1 - 4.5 HR Bandpass Filtered Edge", loc='left')
    ax.set_ylabel('Range [km]')
    ax.legend(loc='lower right', fontsize='x-small', ncols=4)


def plot_multiple_sin_fits(ax, fit_result: FitOutput):
    """
    Panel (e): Sinusoidal Fit to Bandpass Filtered Edge (multiple fits)
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    fit_result : FitOutput
        Complete pipeline result
    """
    from datetime import datetime
    from scripts.sinusoid_fitting import sinusoid
    import matplotlib.cm as cm
    
    fit_times = fit_result.fit_times
    data_detrend = fit_result.data_detrend
    all_sin_fits = fit_result.all_sin_fits
    date = fit_result.meta['date']
    xlim = fit_result.meta['time_limits']['xlim']
    fitWinLim = fit_result.meta.get('fitWinLim', (None, None))
    
    if len(fit_times) > 0:
        ax.plot(fit_times, data_detrend, label='Bandpass Filtered')
        
        # Plot all sin fits with different colors
        num_fits = len(all_sin_fits)
        
        if num_fits > 0:
            palette = cm.get_cmap('tab10', num_fits)
            
            for i, fit_params in enumerate(all_sin_fits):
                color = palette(i)
                # Reconstruct sin fit from parameters
                date_midnight = datetime(date.year, date.month, date.day)
                tt_sec = np.array([(t - np.datetime64(date_midnight, 'ns')).astype('timedelta64[s]').astype(float)
                                   for t in fit_times])
                sin_vals = sinusoid(tt_sec, 
                                   T_hr=fit_params['T_hr'],
                                   amplitude_km=fit_params['amplitude_km'],
                                   phase_hr=fit_params['phase_hr'],
                                   offset_km=fit_params['offset_km'],
                                   slope_kmph=fit_params['slope_kmph'])
                
                if i == 0:  # Best fit (highest R²)
                    ax.plot(fit_times, sin_vals, color='red', lw=3, ls='--', 
                           alpha=1.0, label=f"Sin Fit {i+1} (SELECTED)")
                else:
                    ax.plot(fit_times, sin_vals, color=color, lw=1.5, ls='--', 
                           alpha=0.6, label=f"Sin Fit {i+1}")
        
        if fitWinLim[0] is not None:
            for wl in fitWinLim:
                ax.axvline(wl, color='lime', ls='--', lw=2)
    
    fmt_xaxis(ax, xlim)
    ax.set_title("(e) Sinusodial Fit to Bandpass Filtered Edge", loc='left')
    ax.set_ylabel('Range [km]')
    ax.legend(loc='lower right', fontsize='small', ncols=4)


def plot_sin_fit_table(ax, fit_result: FitOutput):
    """
    Panel (f): Sin Fit Parameters table
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    fit_result : FitOutput
        Complete pipeline result
    """
    all_sin_fits = fit_result.all_sin_fits
    
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    fontdict = {'weight': 'normal', 'family': 'monospace', 'size': 20}
    
    if all_sin_fits:
        # Collect all parameter keys we want to display
        param_keys = [k for k in all_sin_fits[0].keys() if k not in ['selected', 'T_hr_guess']]
        
        # Build header row
        header = ['Fit #'] + param_keys
        rows = []
        
        for i, fit in enumerate(all_sin_fits):
            row = [f"{i+1}" + ("*" if i == 0 else "")]  # Mark best fit
            for key in param_keys:
                val = fit[key]
                if isinstance(val, float):
                    if key == 'r2':
                        row.append(f"{val:.2f}")
                    else:
                        row.append(f"{val:.1f}")
                else:
                    row.append(str(val))
            rows.append(row)
        
        # Convert rows into aligned text
        col_widths = [max(len(str(cell)) for cell in col) for col in zip(*([header] + rows))]
        fmt_str = "  ".join("{:<" + str(w) + "}" for w in col_widths)
        
        txt_lines = [fmt_str.format(*header)]
        for row in rows:
            txt_lines.append(fmt_str.format(*row))
        
        ax.set_title("(f) Sin Fit Parameters", loc='left')
        ax.text(0.01, 0.95, '\n'.join(txt_lines), fontdict=fontdict, va='top')
    else:
        ax.text(0.5, 0.5, 'No fits available', ha='center', va='center',
                transform=ax.transAxes, fontsize=20)


def plot_heatmap_with_polynomial(ax, ax_cb, fit_result: FitOutput):
    """
    Panel (b): Polynomial Fit - heatmap with polynomial overlay
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Main axis to plot on
    ax_cb : matplotlib.axes.Axes
        Colorbar axis
    fit_result : FitOutput
        Complete pipeline result
    """
    arr = fit_result.intermediate['preprocessed_arr']
    arr_times = fit_result.intermediate['preprocessed_times']
    ranges_km = fit_result.edge_data.ranges_km
    edge_0_times = fit_result.intermediate['edge_0_times']
    edge_0_vals = fit_result.intermediate['edge_0_vals']
    fit_times = fit_result.fit_times
    poly_fit = fit_result.poly_fit
    data_detrend = fit_result.data_detrend
    xlim = fit_result.meta['time_limits']['xlim']
    fitWinLim = fit_result.meta.get('fitWinLim', (None, None))
    
    mpbl = ax.pcolormesh(arr_times, ranges_km, arr, cmap='plasma', 
                         shading='nearest', rasterized=True, antialiased=False)
    
    # Plot polynomial fit reconstructed
    if len(fit_times) > 0:
        ax.plot(fit_times, data_detrend + poly_fit, 
                label='Polynomial Fit', color='white', lw=3, ls='--')
    ax.plot(edge_0_times, edge_0_vals, lw=2, label='Detected Edge', color='cyan')
    
    if fitWinLim[0] is not None:
        for wl in fitWinLim:
            ax.axvline(wl, color='lime', ls='--', lw=2)
    
    ax.set_title("(b) Polynomial Fit", loc='left')
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Normalized Spot Count')
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.set_ylim(900, 1600)


def plot_heatmap_with_variance(ax, ax_cb, fit_result: FitOutput):
    """
    Panel (a): Coefficient of Variance Selection
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Main axis to plot on
    ax_cb : matplotlib.axes.Axes
        Colorbar axis
    fit_result : FitOutput
        Complete pipeline result
    """
    date = fit_result.meta['date']
    arr = fit_result.intermediate['preprocessed_arr']
    arr_times = fit_result.intermediate['preprocessed_times']
    ranges_km = fit_result.edge_data.ranges_km
    edge_0_times = fit_result.intermediate['edge_0_times']
    edge_0_vals = fit_result.intermediate['edge_0_vals']
    stability_times = fit_result.edge_data.edge_times
    stability = fit_result.stability
    xlim = fit_result.meta['time_limits']['xlim']
    winlim = fit_result.meta['time_limits']['winlim']
    fitWinLim = fit_result.meta.get('fitWinLim', (None, None))
    
    mpbl = ax.pcolormesh(arr_times, ranges_km, arr, cmap='plasma', 
                         shading='nearest', rasterized=True, antialiased=False)
    ax.plot(edge_0_times, edge_0_vals, lw=2, label='Detected Edge', color='cyan')
    
    # Add stability on twin axis
    ax2 = ax.twinx()
    ax2.plot(stability_times, stability, lw=2, color='0.5', label='Coef. of Variance')
    ax2.grid(False)
    ax2.set_ylabel('Edge Coef. of Variation')
    
    # Window limits
    for wl in winlim:
        ax.axvline(wl, color='0.8', ls='--', lw=2)
    if fitWinLim[0] is not None:
        for wl in fitWinLim:
            ax.axvline(wl, color='lime', ls='--', lw=2)
    
    ax.set_title("(a) Coefficient of Variance Selection", loc='left')
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Normalized Spot Count')
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.set_ylim(900, 1600)
    ax.text(0.5, 1.12, date.strftime('%Y-%m-%d'), transform=ax.transAxes,
            ha='center', va='bottom', fontsize=25, fontweight='bold')


def plot_heatmap_with_final_fit(ax, ax_cb, fit_result: FitOutput):
    """
    Panel (g): Polynomial Detrend + Sin Fit
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Main axis to plot on
    ax_cb : matplotlib.axes.Axes
        Colorbar axis
    fit_result : FitOutput
        Complete pipeline result
    """
    arr = fit_result.intermediate['preprocessed_arr']
    arr_times = fit_result.intermediate['preprocessed_times']
    ranges_km = fit_result.edge_data.ranges_km
    fit_times = fit_result.fit_times
    sin_fit = fit_result.sin_fit
    poly_fit = fit_result.poly_fit
    xlim = fit_result.meta['time_limits']['xlim']
    fitWinLim = fit_result.meta.get('fitWinLim', (None, None))
    
    mpbl = ax.pcolormesh(arr_times, ranges_km, arr, cmap='plasma', 
                         shading='nearest', rasterized=True, antialiased=False)
    
    if len(fit_times) > 0 and len(sin_fit) > 0:
        ax.plot(fit_times, poly_fit + sin_fit, 
                label='Final Sin Fit', color='white', lw=3, ls='--')
    
    if fitWinLim[0] is not None:
        for wl in fitWinLim:
            ax.axvline(wl, color='lime', ls='--', lw=2)
    
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Normalized Spot Count')
    fmt_xaxis(ax, xlim)
    ax.set_title("(g) Polynomial Detrend + Sin Fit", loc='left')
    ax.set_ylabel('Range [km]')
    ax.set_ylim(900, 1600)
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)


def plot_detrended_before_bandpass(ax, fit_result: FitOutput):
    """
    Panel (c): Edge Detrended using Polynomial Fit (before bandpass)
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    fit_result : FitOutput
        Complete pipeline result
    """
    fit_times = fit_result.fit_times
    xlim = fit_result.meta['time_limits']['xlim']
    fitWinLim = fit_result.meta.get('fitWinLim', (None, None))
    
    # Use pre-bandpass detrended data if available
    data_detrend_no_bp = fit_result.intermediate.get('data_detrend_no_bp', None)
    
    if len(fit_times) > 0 and data_detrend_no_bp is not None and len(data_detrend_no_bp) > 0:
        ax.plot(fit_times, data_detrend_no_bp, label='Detrended Edge')
        
        if fitWinLim[0] is not None:
            for wl in fitWinLim:
                ax.axvline(wl, color='lime', ls='--', lw=2)
    
    ax.set_title("(c) Edge Detrended using Polynomial Fit", loc='left')
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.legend(loc='lower right', fontsize='x-small', ncols=4)