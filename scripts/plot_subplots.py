#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm

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


def plot_bandpass_filtered(ax, fit_result: FitOutput, *, label='d'):
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
    ax.set_title(f"({label}) 1 - 4.5 HR Bandpass Filtered Edge", loc='left')
    ax.set_ylabel('Range [km]')
    ax.legend(loc='lower right', fontsize='x-small', ncols=4)


def plot_multiple_sin_fits(ax, ax_leg, fit_result: FitOutput, *, label='e'):
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
    ax.set_title(f"({label}) Sinusoidal Fit to Bandpass Filtered Edge", loc='left')
    ax.set_ylabel('Range [km]')
    handles, labels = ax.get_legend_handles_labels()
    ax_leg.axis('off')
    ax_leg.legend(handles, labels, loc='center', fontsize=12)

def plot_selected_sin_fit(ax, fit_result: FitOutput, *, label='d'):
    """
    Panel (e): Selected sinusoidal fit to bandpass filtered edge
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    ax_leg : matplotlib.axes.Axes
        Legend axis
    fit_result : FitOutput
        Complete pipeline result
    """
    from datetime import datetime
    from scripts.sinusoid_fitting import sinusoid

    fit_times = fit_result.fit_times
    bandpass_edge = fit_result.data_detrend
    selected_sin_fit = fit_result.all_sin_fits
    date = fit_result.meta['date']
    xlim = fit_result.meta['time_limits']['xlim']
    fit_window_limits = fit_result.meta.get('fitWinLim', (None, None))

    if len(fit_times) > 0:
        ax.plot(
            fit_times,
            bandpass_edge,
            color='C0',
            lw=2,
            label='Bandpass Filtered Edge'
        )

        if len(selected_sin_fit) > 0:
            fit_params = selected_sin_fit[0]

            date_midnight = datetime(date.year, date.month, date.day)
            tt_sec = np.array([
                (t - np.datetime64(date_midnight, 'ns')).astype('timedelta64[s]').astype(float)
                for t in fit_times
            ])

            selected_fit_vals = sinusoid(
                tt_sec,
                T_hr=fit_params['T_hr'],
                amplitude_km=fit_params['amplitude_km'],
                phase_hr=fit_params['phase_hr'],
                offset_km=fit_params['offset_km'],
                slope_kmph=fit_params['slope_kmph']
            )

            ax.plot(
                fit_times,
                selected_fit_vals,
                color='red',
                lw=3,
                ls='--',
                alpha=1.0,
                label='Selected Sinusoidal Fit'
            )

        if fit_window_limits[0] is not None:
            for wl in fit_window_limits:
                ax.axvline(wl, color='lime', ls='--', lw=2)

    fmt_xaxis(ax, xlim)
    ax.set_title(f"({label}) Selected Sinusoidal Fit to Bandpass Filtered Edge", loc='left')
    ax.set_ylabel('Range [km]')

    ax.legend(loc='lower right', fontsize='x-small', ncols=4)

def plot_sin_fit_table(ax, fit_result: FitOutput, *, label='f'):
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
        
        ax.set_title(f"({label}) Sin Fit Parameters", loc='left')
        ax.text(0.01, 0.95, '\n'.join(txt_lines), fontdict=fontdict, va='top')
    else:
        ax.text(0.5, 0.5, 'No fits available', ha='center', va='center',
                transform=ax.transAxes, fontsize=20)


def plot_heatmap_with_polynomial(ax, ax_cb, fit_result: FitOutput, *, label='b', ylim=None):
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
        ax.plot(fit_times, poly_fit, 
                label='Polynomial Fit', color='white', lw=3, ls='--')
    ax.plot(edge_0_times, edge_0_vals, lw=2, label='Detected Edge', color='cyan')
    
    if fitWinLim[0] is not None:
        for wl in fitWinLim:
            ax.axvline(wl, color='lime', ls='--', lw=2)
    
    ax.set_title(f"({label}) Polynomial Fit", loc='left')
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Normalized Spot Density (a.u.)')
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_heatmap_with_variance(ax, ax_cb, fit_result: FitOutput, *, label='a', ylim=None):
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
    
    ax.set_title(f"({label}) Coefficient of Variance Selection", loc='left')
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Normalized Spot Density (a.u.)')
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    if ylim is not None:
        ax.set_ylim(ylim)


def plot_heatmap_with_final_fit(ax, ax_cb, fit_result: FitOutput, *, label='e', ylim=None):
    """
    Polynomial Detrend + Sin Fit
    
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
    
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Normalized Spot Density (a.u.)')
    fmt_xaxis(ax, xlim)
    ax.set_title(f"({label}) Polynomial Detrend + Sin Fit", loc='left')
    ax.set_ylabel('Range [km]')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)


def plot_detrended_before_bandpass(ax, fit_result: FitOutput, *, label='c'):
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
    
    ax.set_title(f"({label}) Edge Detrended using Polynomial Fit", loc='left')
    fmt_xaxis(ax, xlim)
    ax.set_ylabel('Range [km]')
    ax.legend(loc='lower right', fontsize='x-small', ncols=4)

def plot_spot_location_heatmap(ax, ax_cb, df, date, *, label='a'):
    """
    Spot density by TX-RX midpoint location.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Cartopy GeoAxes (must be created with projection=ccrs.PlateCarree())
    ax_cb : matplotlib.axes.Axes
        Colorbar axis
    df : polars.DataFrame
        Spot dataframe with mid_lat and mid_long columns
    date : datetime
        Processing date for title
    label : str
        Panel label character (e.g. 'a', 'b')
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    lats = df['mid_lat'].to_numpy()
    lons = df['mid_long'].to_numpy()

    lat_bins = np.linspace(lats.min(), lats.max(), 200)
    lon_bins = np.linspace(lons.min(), lons.max(), 200)

    hist, lat_edges, lon_edges = np.histogram2d(lats, lons, bins=[lat_bins, lon_bins])

    mpbl = ax.pcolormesh(lon_edges[:-1], lat_edges[:-1], hist,
                         cmap='inferno', shading='nearest', rasterized=True,
                         norm=LogNorm(vmin=1, clip=True),
                         transform=ccrs.PlateCarree())

    ax.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='white')
    ax.add_feature(cfeature.BORDERS, linewidth=0.5, edgecolor='white')
    ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor='white')

    avg_lat = float(lats.mean())
    avg_lon = float(lons.mean())
    ax.plot(avg_lon, avg_lat, marker='*', markersize=14, color='cyan',
            markeredgecolor='black', markeredgewidth=0.5,
            transform=ccrs.PlateCarree(),
            label=f'Mean ({avg_lat:.1f}°N, {avg_lon:.1f}°E)')

    ax.set_extent([lons.min(), lons.max(), lats.min(), lats.max()],
                  crs=ccrs.PlateCarree())

    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label=r'log(spots bin$^{-1}$)')

    ax.set_title(f"({label}) Spot Density by Midpoint Location",
                 loc='left', fontweight='bold')
    ax.set_xlabel('Longitude [°]')
    ax.set_ylabel('Latitude [°]')
    ax.legend(loc='lower right', fontsize='x-small')


def plot_raw_histogram(ax, ax_cb, fit_result: FitOutput, ylim, *, label='b'):
    """
    Raw trimmed 2D histogram panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ax_cb : matplotlib.axes.Axes
        Colorbar axis
    fit_result : FitOutput
    ylim : tuple
    label : str
        Panel label character
    """
    raw_hist  = fit_result.intermediate['raw_hist2d']
    arr_times = fit_result.intermediate['preprocessed_times']
    ranges_km = fit_result.edge_data.ranges_km
    date      = fit_result.meta['date']
    xlim      = fit_result.meta['time_limits']['xlim']

    mpbl = ax.pcolormesh(arr_times, ranges_km, raw_hist, cmap='plasma',
                         shading='nearest', rasterized=True)
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label=r'spots bin$^{-1}$')
    ax.set_title(f"({label}) Raw Trimmed 2D Histogram",
                 loc='left', fontweight='bold')
    ax.set_ylabel('Range [km]')
    ax.set_ylim(ylim)
    fmt_xaxis(ax, xlim)


def plot_gaussian_filtered(ax, ax_cb, fit_result: FitOutput, ylim, *, label='c'):
    """
    Gaussian filtered heatmap panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ax_cb : matplotlib.axes.Axes
        Colorbar axis
    fit_result : FitOutput
    ylim : tuple
    label : str
        Panel label character
    """
    gaussian_arr = fit_result.intermediate['after_gaussian']
    arr_times    = fit_result.intermediate['preprocessed_times']
    ranges_km    = fit_result.edge_data.ranges_km
    xlim         = fit_result.meta['time_limits']['xlim']

    mpbl = ax.pcolormesh(arr_times, ranges_km, gaussian_arr, cmap='plasma',
                         shading='nearest', rasterized=True)
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Normalized Spot Density (a.u.)')
    ax.set_title(f"({label}) Smoothed with 2D Gaussian Filter (σ = 4.2)",
                 loc='left', fontweight='bold')
    ax.set_ylabel('Range [km]')
    ax.set_ylim(ylim)
    fmt_xaxis(ax, xlim)


def plot_rescaled_contours(ax, ax_cb, fit_result: FitOutput, ylim, *, label='d'):
    """
    8-bit rescaled heatmap with structural contours panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ax_cb : matplotlib.axes.Axes
        Colorbar axis
    fit_result : FitOutput
    ylim : tuple
    label : str
        Panel label character
    """
    rescale_arr = fit_result.intermediate['after_rescale']
    arr_times   = fit_result.intermediate['preprocessed_times']
    ranges_km   = fit_result.edge_data.ranges_km
    xlim        = fit_result.meta['time_limits']['xlim']

    mpbl = ax.pcolormesh(arr_times, ranges_km, rescale_arr, cmap='plasma',
                         shading='nearest', rasterized=True)
    levels = np.linspace(np.nanmin(rescale_arr), np.nanmax(rescale_arr), 15)
    ax.contour(arr_times, ranges_km, rescale_arr, levels=levels,
               colors='k', linewidths=0.5)
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Normalized Spot Density (a.u.)')
    ax.set_title(f"({label}) Normalized 8-bit Image with Structural Contours",
                 loc='left', fontweight='bold')
    ax.set_ylabel('Range [km]')
    ax.set_ylim(ylim)
    fmt_xaxis(ax, xlim)


def plot_quantile_lines(ax, ax_cb, fit_result: FitOutput, ylim, *, label='e'):
    """
    Heatmap with all quantile threshold lines, LOWESS smoothed line, and detected edge.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ax_cb : matplotlib.axes.Axes
        Colorbar axis
    fit_result : FitOutput
    ylim : tuple
    label : str
    """
    import matplotlib.cm as cm

    arr             = fit_result.intermediate['preprocessed_arr']
    arr_times       = fit_result.intermediate['preprocessed_times']
    ranges_km       = fit_result.edge_data.ranges_km
    edge_times      = fit_result.edge_data.edge_times
    edge_positions  = fit_result.edge_data.edge_positions
    quantile_lines  = fit_result.edge_data.quantile_lines   # (n_times, n_quantiles)
    quantile_values = fit_result.edge_data.quantile_values
    xlim            = fit_result.meta['time_limits']['xlim']

    mpbl = ax.pcolormesh(arr_times, ranges_km, arr, cmap='Greys_r',
                         shading='nearest', rasterized=True, antialiased=False)

    palette = cm.get_cmap('cool', len(quantile_values))
    for i, q in enumerate(quantile_values):
        ax.plot(arr_times, quantile_lines[:, i], color=palette(i),
                lw=1.5, alpha=0.8, label=f'Q{q:.2f}')

    ax.plot(edge_times, edge_positions, color='cyan', lw=2.5, ls='-', label='Detected Edge')

    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Normalized Spot Density (a.u.)')
    ax.set_title(f"({label}) Quantile Threshold Lines", loc='left', fontweight='bold')
    ax.set_ylabel('Range [km]')
    ax.set_ylim(ylim)
    ax.legend(loc='upper center', fontsize='x-small', ncols=4)
    fmt_xaxis(ax, xlim)


def plot_edge_overlay(ax, ax_cb, fit_result: FitOutput, ylim, *, label='e'):
    """
    Greyscale heatmap with detected edge overlay panel.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    ax_cb : matplotlib.axes.Axes
        Colorbar axis
    fit_result : FitOutput
    ylim : tuple
    label : str
        Panel label character
    """
    rescale_arr  = fit_result.intermediate['after_rescale']
    arr_times    = fit_result.intermediate['preprocessed_times']
    ranges_km    = fit_result.edge_data.ranges_km
    edge_0_times = fit_result.intermediate['edge_0_times']
    edge_0_vals  = fit_result.intermediate['edge_0_vals']
    xlim         = fit_result.meta['time_limits']['xlim']

    mpbl = ax.pcolormesh(arr_times, ranges_km, rescale_arr, cmap='Greys_r',
                         shading='nearest', rasterized=True)
    ax.plot(edge_0_times, edge_0_vals, lw=2, color='cyan', label='Detected Edge')
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Normalized Spot Density (a.u.)')
    ax.set_title(f"({label}) Edge Detection Overlay", loc='left', fontweight='bold')
    ax.set_ylabel('Range [km]')
    ax.set_ylim(ylim)
    ax.legend(loc='upper center', fontsize='x-small')
    fmt_xaxis(ax, xlim)