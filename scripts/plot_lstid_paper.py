import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string

from scripts.data_structures import FitOutput
from scripts.plot_subplots import setup_mpl_style, fmt_xaxis, plot_intermediate_heatmap

letters = string.ascii_lowercase


def stack_plot_1(
    fit_result: FitOutput,
    *,
    output_dir: str,
    cb_pad: float,
    ylim: tuple,
    **_
):
    """
    Create 4-panel preprocessing stackplot.
    
    Panels
    ------
    (a) Raw trimmed histogram
    (b) Gaussian filtered (after smoothing)
    (c) 8-bit rescaled with contours overlay
    (d) Greyscale 8-bit with detected edge overlay
    
    Parameters
    ----------
    fit_result : FitOutput
        Complete pipeline output
    output_dir : str
        Directory for saving figure (from config)
    cb_pad : float
        Colorbar padding (from config)
    ylim : tuple
        Y-axis limits (min_km, max_km) (from config)
    **_
        Additional parameters (ignored for future expansion)
    """
    setup_mpl_style()
    
    date = fit_result.meta['date']
    xlim = fit_result.meta['time_limits']['xlim']
    
    # Extract intermediate preprocessing data
    raw_hist = fit_result.intermediate['raw_hist2d']
    gaussian_arr = fit_result.intermediate['after_gaussian']
    rescale_arr = fit_result.intermediate['after_rescale']
    
    # Get coordinates
    arr_times = fit_result.intermediate['preprocessed_times']
    ranges_km = fit_result.edge_data.ranges_km
    
    # Edge data for panel (d)
    edge_0_times = fit_result.intermediate['edge_0_times']
    edge_0_vals = fit_result.intermediate['edge_0_vals']
    
    # Create figure with GridSpec for colorbar column
    fig = plt.figure(figsize=(19, 20))
    gs = gridspec.GridSpec(4, 2, width_ratios=[20, 0.5], height_ratios=[1]*4)
    
    # Panel (a): Raw histogram
    ax = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    
    mpbl = ax.pcolormesh(arr_times, ranges_km, raw_hist, cmap='plasma', 
                         shading='nearest', rasterized=True)
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Raw Data')
    ax.set_title(f"(a) Raw Trimmed 2D Histogram | {date.strftime('%Y-%m-%d')}", 
                 loc='left', fontweight='bold')
    ax.set_ylabel('Range [km]')
    ax.set_ylim(ylim)
    fmt_xaxis(ax, xlim)
    
    # Panel (b): Gaussian filtered
    ax = fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_subplot(gs[1, 1])
    
    mpbl = ax.pcolormesh(arr_times, ranges_km, gaussian_arr, cmap='plasma',
                         shading='nearest', rasterized=True)
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='Gaussian Filtered')
    ax.set_title("(b) Smoothed with 2D Gaussian Filter (σ = 4.2)", 
                 loc='left', fontweight='bold')
    ax.set_ylabel('Range [km]')
    ax.set_ylim(ylim)
    fmt_xaxis(ax, xlim)
    
    # Panel (c): 8-bit rescale with contours
    ax = fig.add_subplot(gs[2, 0])
    ax_cb = fig.add_subplot(gs[2, 1])
    
    mpbl = ax.pcolormesh(arr_times, ranges_km, rescale_arr, cmap='plasma',
                         shading='nearest', rasterized=True)
    levels = np.linspace(np.nanmin(rescale_arr), np.nanmax(rescale_arr), 15)
    ax.contour(arr_times, ranges_km, rescale_arr, levels=levels, 
               colors='k', linewidths=0.5)
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='8-bit Quantized')
    ax.set_title("(c) Normalized 8-bit Image with Structural Contours", 
                 loc='left', fontweight='bold')
    ax.set_ylabel('Range [km]')
    ax.set_ylim(ylim)
    fmt_xaxis(ax, xlim)
    
    # Panel (d): Greyscale with edge
    ax = fig.add_subplot(gs[3, 0])
    ax_cb = fig.add_subplot(gs[3, 1])
    
    mpbl = ax.pcolormesh(arr_times, ranges_km, rescale_arr, cmap='Greys_r',
                         shading='nearest', rasterized=True)
    ax.plot(edge_0_times, edge_0_vals, lw=2, color='cyan', label='Detected Edge')
    plt.colorbar(mpbl, cax=ax_cb, orientation='vertical', label='8-bit Quantized')
    ax.set_title("(d) Edge Detection Overlay", loc='left', fontweight='bold')
    ax.set_ylabel('Range [km]')
    ax.set_ylim(ylim)
    ax.legend(loc='upper center', fontsize='x-small')
    fmt_xaxis(ax, xlim)
    
    fig.tight_layout()
    
    # Save
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    date_str = date.strftime('%Y%m%d')
    png_fname = f'{date_str}_preprocessing.png'
    png_fpath = os.path.join(output_dir, png_fname)
    
    print(f'   Saving: {png_fpath}')
    fig.savefig(png_fpath, bbox_inches='tight', dpi=150)
    plt.close()
    
    return png_fpath


def stack_plot_2(
    fit_result: FitOutput,
    *,
    output_dir: str,
    cb_pad: float,
    ylim: tuple,
    **_
):
    """
    Create comprehensive 7-panel diagnostic plot matching original figure 2.
    
    Panels
    ------
    (a) Coefficient of Variance Selection - heatmap with edge and stability
    (b) Polynomial Fit - heatmap with polynomial overlay
    (c) Edge Detrended using Polynomial Fit
    (d) 1-4.5 HR Bandpass Filtered Edge
    (e) Sinusoidal Fit to Bandpass Filtered Edge (multiple fits)
    (f) Sin Fit Parameters table
    (g) Polynomial Detrend + Sin Fit overlay on heatmap
    
    Parameters
    ----------
    fit_result : FitOutput
        Complete pipeline output
    output_dir : str
        Directory for saving figure (from config)
    cb_pad : float
        Colorbar padding (from config)
    ylim : tuple
        Y-axis limits (min_km, max_km) (from config)
    **_
        Additional parameters (ignored for future expansion)
    """
    from scripts.plot_subplots import (
        plot_heatmap_with_variance,
        plot_heatmap_with_polynomial,
        plot_detrended_before_bandpass,
        plot_bandpass_filtered,
        plot_multiple_sin_fits,
        plot_sin_fit_table,
        plot_heatmap_with_final_fit
    )
    
    setup_mpl_style()
    
    date = fit_result.meta['date']
    
    # Create figure with GridSpec for colorbar column
    fig = plt.figure(figsize=(19, 28))
    gs = gridspec.GridSpec(7, 2, width_ratios=[20, 0.5], height_ratios=[1]*7)
    
    # Panel (a) — Coefficient of Variance Selection
    ax = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    plot_heatmap_with_variance(ax, ax_cb, fit_result)
    
    # Panel (b) — Polynomial Fit
    ax = fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_subplot(gs[1, 1])
    plot_heatmap_with_polynomial(ax, ax_cb, fit_result)
    
    # Panel (c) — Detrended edge
    ax = fig.add_subplot(gs[2, 0])
    plot_detrended_before_bandpass(ax, fit_result)
    
    # Panel (d) — Bandpass-filtered edge
    ax = fig.add_subplot(gs[3, 0])
    plot_bandpass_filtered(ax, fit_result)
    
    # Panel (e) — Multiple sin fits
    ax = fig.add_subplot(gs[4, 0])
    plot_multiple_sin_fits(ax, fit_result)
    
    # Panel (f) — Sin fit parameters table
    ax = fig.add_subplot(gs[5, 0])
    plot_sin_fit_table(ax, fit_result)
    
    # Panel (g) — Final fit overlay
    ax = fig.add_subplot(gs[6, 0])
    ax_cb = fig.add_subplot(gs[6, 1])
    plot_heatmap_with_final_fit(ax, ax_cb, fit_result)
    
    fig.tight_layout()
    
    # Save figure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    date_str = date.strftime('%Y%m%d')
    png_fname = f'{date_str}_diagnostic.png'
    png_fpath = os.path.join(output_dir, png_fname)
    
    print(f'   Saving: {png_fpath}')
    fig.savefig(png_fpath, bbox_inches='tight', dpi=150)
    plt.close()
    
    return png_fpath