import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import string

from scripts.data_structures import FitOutput
#from scripts.plot_subplots import setup_mpl_style, fmt_xaxis, plot_intermediate_heatmap
from scripts.plot_subplots import *

letters = string.ascii_lowercase


def stack_plot_preprocess(
    fit_result: FitOutput,
    df,
    *,
    output_dir: str,
    cb_pad: float,
    ylim: tuple,
    **_
):
    """
    Create 5-panel preprocessing stackplot.

    Panels
    ------
    (a) Spot density by TX-RX midpoint location (geographic heatmap) — narrower
    (b) Raw trimmed histogram
    (c) Gaussian filtered
    (d) 8-bit rescaled with contours overlay
    (e) Greyscale 8-bit with detected edge overlay

    Panel labels are driven by `labels` — reorder for publication without
    touching the subplot functions.
    """
    import cartopy.crs as ccrs

    setup_mpl_style()

    date   = fit_result.meta['date']
    labels = list(letters[:5])  # ['a', 'b', 'c', 'd', 'e']

    fig = plt.figure(figsize=(19, 28))

    # Panel (a): geographic — narrower, left ~55% of figure width
    gs_top = gridspec.GridSpec(1, 2,
                                left=0.25, right=0.75,
                                bottom=0.76, top=0.93,
                                width_ratios=[12, 0.5],
                                wspace=0.05)

    # Panels (b-e): full width
    gs_bot = gridspec.GridSpec(4, 2,
                                left=0.06, right=0.95,
                                bottom=0.02, top=0.73,
                                width_ratios=[20, 0.5],
                                hspace=0.35,
                                wspace=0.05)

    ax    = fig.add_subplot(gs_top[0, 0], projection=ccrs.PlateCarree())
    ax.set_aspect('auto')
    ax_cb = fig.add_subplot(gs_top[0, 1])
    plot_spot_location_heatmap(ax, ax_cb, df, date, label=labels[0])

    ax    = fig.add_subplot(gs_bot[0, 0])
    ax_cb = fig.add_subplot(gs_bot[0, 1])
    plot_raw_histogram(ax, ax_cb, fit_result, ylim, label=labels[1])

    ax    = fig.add_subplot(gs_bot[1, 0])
    ax_cb = fig.add_subplot(gs_bot[1, 1])
    plot_gaussian_filtered(ax, ax_cb, fit_result, ylim, label=labels[2])

    ax    = fig.add_subplot(gs_bot[2, 0])
    ax_cb = fig.add_subplot(gs_bot[2, 1])
    plot_rescaled_contours(ax, ax_cb, fit_result, ylim, label=labels[3])

    ax    = fig.add_subplot(gs_bot[3, 0])
    ax_cb = fig.add_subplot(gs_bot[3, 1])
    plot_edge_overlay(ax, ax_cb, fit_result, ylim, label=labels[4])

    fig.suptitle(
        f'Image Preprocessing \n{date.strftime("%Y-%m-%d")}',
        fontsize=25,
        fontweight='bold',
        y=0.99
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    date_str  = date.strftime('%Y%m%d')
    png_fpath = os.path.join(output_dir, f'{date_str}_preprocessing.png')
    print(f'   Saving: {png_fpath}')
    fig.savefig(png_fpath, bbox_inches='tight', dpi=150)
    plt.close()

    return png_fpath


def stack_plot_sinfit_v1(
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
    labels = list(letters[:7])  # ['a', 'b', 'c', 'd', 'e', 'f', 'g']

    # Create figure with GridSpec for colorbar column
    fig = plt.figure(figsize=(20, 35))
    gs = gridspec.GridSpec(7, 2, width_ratios=[20, 0.5], height_ratios=[1]*7)

    # Panel (a) — Coefficient of Variance Selection
    ax = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    plot_heatmap_with_variance(ax, ax_cb, fit_result, label=labels[0])

    # Panel (b) — Polynomial Fit
    ax = fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_subplot(gs[1, 1])
    plot_heatmap_with_polynomial(ax, ax_cb, fit_result, label=labels[1])

    # Panel (c) — Detrended edge
    ax = fig.add_subplot(gs[2, 0])
    plot_detrended_before_bandpass(ax, fit_result, label=labels[2])

    # Panel (d) — Bandpass-filtered edge
    ax = fig.add_subplot(gs[3, 0])
    plot_bandpass_filtered(ax, fit_result, label=labels[3])

    # Panel (e) — Multiple sin fits
    ax = fig.add_subplot(gs[4, 0])
    ax_leg = fig.add_subplot(gs[4, 1])
    plot_multiple_sin_fits(ax, ax_leg, fit_result, label=labels[4])

    # Panel (f) — Sin fit parameters table
    ax = fig.add_subplot(gs[5, 0])
    plot_sin_fit_table(ax, fit_result, label=labels[5])

    # Panel (g) — Final fit overlay
    ax = fig.add_subplot(gs[6, 0])
    ax_cb = fig.add_subplot(gs[6, 1])
    plot_heatmap_with_final_fit(ax, ax_cb, fit_result, label=labels[6])

    fig.suptitle(
        f'Sinusoid Fitting \n{date.strftime("%Y-%m-%d")}',
        fontsize=25,
        fontweight='bold'
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    date_str = date.strftime('%Y%m%d')
    png_fname = f'{date_str}_sinfit.png'
    png_fpath = os.path.join(output_dir, png_fname)
    
    print(f'   Saving: {png_fpath}')
    fig.savefig(png_fpath, bbox_inches='tight', dpi=150)
    plt.close()
    
    return png_fpath

def stack_plot_sinfit_v2(
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
    labels = list(letters[:5])  # ['a', 'b', 'c', 'd', 'e']

    # Create figure with GridSpec for colorbar column
    fig = plt.figure(figsize=(20, 25))
    gs = gridspec.GridSpec(5, 2, width_ratios=[20, 0.5], height_ratios=[1]*5)

    # Panel (a) — Coefficient of Variance Selection
    ax = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    plot_heatmap_with_variance(ax, ax_cb, fit_result, label=labels[0])

    # Panel (b) — Polynomial Fit
    ax = fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_subplot(gs[1, 1])
    plot_heatmap_with_polynomial(ax, ax_cb, fit_result, label=labels[1])

    # Panel (c) — Detrended edge
    ax = fig.add_subplot(gs[2, 0])
    plot_detrended_before_bandpass(ax, fit_result, label=labels[2])

    # Panel (d) — Multiple sin fits
    ax = fig.add_subplot(gs[3, 0])
    ax_leg = fig.add_subplot(gs[3, 1])
    plot_multiple_sin_fits(ax, ax_leg, fit_result, label=labels[3])

    # Panel (e) — Final fit overlay
    ax = fig.add_subplot(gs[4, 0])
    ax_cb = fig.add_subplot(gs[4, 1])
    plot_heatmap_with_final_fit(ax, ax_cb, fit_result, label=labels[4])

    fig.suptitle(
        f'Sinusoid Fitting \n{date.strftime("%Y-%m-%d")}',
        fontsize=25,
        fontweight='bold'
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    date_str = date.strftime('%Y%m%d')
    png_fname = f'{date_str}_sinfit_v2.png'
    png_fpath = os.path.join(output_dir, png_fname)
    
    print(f'   Saving: {png_fpath}')
    fig.savefig(png_fpath, bbox_inches='tight', dpi=150)
    plt.close()
    
    return png_fpath

def stack_plot_sinfit_v3(
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
        plot_heatmap_with_final_fit,
        plot_selected_sin_fit
    )
    
    setup_mpl_style()

    date = fit_result.meta['date']
    labels = list(letters[:5])  # ['a', 'b', 'c', 'd', 'e']

    # Create figure with GridSpec for colorbar column
    fig = plt.figure(figsize=(19, 25))
    gs = gridspec.GridSpec(5, 2, width_ratios=[20, 0.5], height_ratios=[1]*5)

    # Panel (a) — Coefficient of Variance Selection
    ax = fig.add_subplot(gs[0, 0])
    ax_cb = fig.add_subplot(gs[0, 1])
    plot_heatmap_with_variance(ax, ax_cb, fit_result, label=labels[0])

    # Panel (b) — Polynomial Fit
    ax = fig.add_subplot(gs[1, 0])
    ax_cb = fig.add_subplot(gs[1, 1])
    plot_heatmap_with_polynomial(ax, ax_cb, fit_result, label=labels[1])

    # Panel (c) — Detrended edge
    ax = fig.add_subplot(gs[2, 0])
    plot_detrended_before_bandpass(ax, fit_result, label=labels[2])

    # Panel (d) — Selected sin fit
    ax = fig.add_subplot(gs[3, 0])
    plot_selected_sin_fit(ax, fit_result, label=labels[3])

    # Panel (e) — Final fit overlay
    ax = fig.add_subplot(gs[4, 0])
    ax_cb = fig.add_subplot(gs[4, 1])
    plot_heatmap_with_final_fit(ax, ax_cb, fit_result, label=labels[4])

    fig.suptitle(
        f'Sinusoid Fitting \n{date.strftime("%Y-%m-%d")}',
        fontsize=25,
        fontweight='bold'
    )
    
    fig.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    date_str = date.strftime('%Y%m%d')
    png_fname = f'{date_str}_sinfit_v3.png'
    png_fpath = os.path.join(output_dir, png_fname)
    
    print(f'   Saving: {png_fpath}')
    fig.savefig(png_fpath, bbox_inches='tight', dpi=150)
    plt.close()
    
    return png_fpath