#!/usr/bin/env python3
"""
dfs_thesis.py — Thesis-quality individual panel plots.

Every pipeline panel is saved as a standalone PNG. All panels use an
identical figure size (_FW × _FH inches) so that the main axes area
(the actual plot, not counting colorbars or margins) is always exactly
_AX_W × _AX_H inches — a hard geometric guarantee enforced by explicit
add_axes() positioning rather than GridSpec.

Two public functions are provided, differing only in how much text they strip:

    thesis_plot_all_panels           → strip_titles  (panel letter removed)
    thesis_plot_all_panels_no_labels → strip_all     (panel letter + all axis
                                                       and colorbar labels removed)

Output structure:
    {parent of output_dir}/thesis/strip_titles/{panel_name}/{YYYYMMDD}_{panel_name}.png
    {parent of output_dir}/thesis/strip_all/{panel_name}/{YYYYMMDD}_{panel_name}.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from scripts.data_structures import FitOutput
from scripts.plot_subplots import (
    setup_mpl_style, fmt_xaxis,
    plot_spot_location_heatmap,
    plot_raw_histogram,
    plot_gaussian_filtered,
    plot_rescaled_contours,
    plot_edge_overlay,
    plot_quantile_lines,
    plot_heatmap_with_variance,
    plot_heatmap_with_polynomial,
    plot_detrended_before_bandpass,
    plot_bandpass_filtered,
    plot_sin_fit_table,
    plot_heatmap_with_final_fit,
    plot_selected_sin_fit,
)

# ---------------------------------------------------------------------------
# Fixed figure / axes geometry
# ---------------------------------------------------------------------------
# All values in inches. Every panel — with or without colorbar — uses
# _FW × _FH so all output PNGs are pixel-identical in total size.
# The data axes always occupy [_AX_L, _AX_B, _AX_W, _AX_H].
#
# Margin accounting at fontsize-18 bold:
#   Left  (_AX_L=1.30): y-tick labels ~0.35 + y-label rotated ~0.30 + pads ~0.15 = 0.80
#   CB gap(_CB_GAP=1.10): twin-axis ticks ~0.35 + twin label ~0.30 + pads ~0.20 = 0.85
#   Right (0.95):  CB tick labels ~0.25 + CB label ~0.35 + pad ~0.20 = 0.80
#   Top  (_TOP=1.20): panel title ~0.35 + gap ~0.10 + suptitle ~0.30 + pad ~0.10 = 0.85
#   Bottom(_AX_B=0.80): x-tick labels ~0.30 + x-label ~0.25 + pads ~0.10 = 0.65

_FW  = 14.60
_FH  =  6.50

_AX_L =  1.30   # left margin
_AX_B =  0.80   # bottom margin
_AX_W = 11.00   # main axes width  ← FIXED PANEL SIZE
_AX_H =  4.50   # main axes height ← FIXED PANEL SIZE

_CB_GAP = 1.10
_CB_L   = _AX_L + _AX_W + _CB_GAP   # = 13.40 in
_CB_W   =  0.25

_ax_l = _AX_L / _FW
_ax_b = _AX_B / _FH
_ax_w = _AX_W / _FW
_ax_h = _AX_H / _FH
_cb_l = _CB_L / _FW
_cb_w = _CB_W / _FW
_cb_h = _ax_h

_SUP_Y = (_AX_B + _AX_H + (_FH - _AX_B - _AX_H) * 0.55) / _FH


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def _fig_cb():
    fig   = plt.figure(figsize=(_FW, _FH))
    ax    = fig.add_axes([_ax_l, _ax_b, _ax_w, _ax_h])
    ax_cb = fig.add_axes([_cb_l, _ax_b, _cb_w, _cb_h])
    return fig, ax, ax_cb


def _fig_plain():
    fig = plt.figure(figsize=(_FW, _FH))
    ax  = fig.add_axes([_ax_l, _ax_b, _ax_w, _ax_h])
    return fig, ax


def _save(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f'   Saving: {path}')
    fig.savefig(path, dpi=150, bbox_inches=None)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Strip helpers
# ---------------------------------------------------------------------------

def _strip_label(ax):
    """Remove the '(x) ' panel-letter prefix from every title location."""
    for loc in ('left', 'center', 'right'):
        t = ax.get_title(loc=loc)
        if t and t.startswith('(') and ') ' in t:
            ax.set_title(t.split(') ', 1)[1], loc=loc)


def _strip_all_labels(ax, fig, ax_cb=None):
    """Remove every decoration — leaves only the plotted data.

    Clears titles, axis labels, ticks, tick labels, and legends on the main
    axes, colorbar axes, and any extra axes (e.g. the CV twin axis on panel 07).
    The suptitle is also suppressed via the calling code (no suptitle call in
    strip_all mode). The figure spines are kept so the panel boundary is visible.
    """
    def _blank_axes(a):
        for loc in ('left', 'center', 'right'):
            a.set_title('', loc=loc)
        a.set_xlabel('')
        a.set_ylabel('')
        a.set_xticks([])
        a.set_yticks([])
        leg = a.get_legend()
        if leg is not None:
            leg.remove()

    _blank_axes(ax)

    for a in fig.axes:
        if a is not ax and a is not ax_cb:
            _blank_axes(a)

    if ax_cb is not None:
        ax_cb.set_yticks([])
        ax_cb.set_ylabel('')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def thesis_plot_all_panels(
    fit_result: FitOutput,
    df,
    *,
    output_dir: str,
    cb_pad: float,
    ylim: tuple,
    **_,
):
    """Strip panel letter from titles only.
    Saves to: {parent}/thesis/strip_titles/{panel_name}/{YYYYMMDD}_{panel_name}.png
    """
    return _thesis_plot(fit_result, df,
                        output_dir=output_dir, ylim=ylim,
                        strip_mode='strip_titles')


def thesis_plot_all_panels_no_labels(
    fit_result: FitOutput,
    df,
    *,
    output_dir: str,
    cb_pad: float,
    ylim: tuple,
    **_,
):
    """Strip panel letter from titles AND all axis / colorbar labels.
    Saves to: {parent}/thesis/strip_all/{panel_name}/{YYYYMMDD}_{panel_name}.png
    """
    return _thesis_plot(fit_result, df,
                        output_dir=output_dir, ylim=ylim,
                        strip_mode='strip_all')


# ---------------------------------------------------------------------------
# Shared implementation
# ---------------------------------------------------------------------------

def _thesis_plot(fit_result, df, *, output_dir, ylim, strip_mode):
    """
    Render all 14 panels, applying the requested strip mode, and save to:
        {parent of output_dir}/thesis/{strip_mode}/{panel_name}/{date}_{panel_name}.png
    """
    import cartopy.crs as ccrs

    setup_mpl_style()

    date       = fit_result.meta['date']
    date_str   = date.strftime('%Y%m%d')
    date_title = date.strftime('%Y-%m-%d')

    base_dir = os.path.dirname(output_dir) or 'output'

    def _p(name):
        return os.path.join(base_dir, 'thesis', strip_mode, name,
                            f'{date_str}_{name}.png')

    def _strip(ax, fig, ax_cb=None):
        if strip_mode == 'strip_all':
            _strip_all_labels(ax, fig, ax_cb)
        else:
            _strip_label(ax)

    saved = []

    # 01 — geographic spot density
    fig   = plt.figure(figsize=(_FW, _FH))
    ax    = fig.add_axes([_ax_l, _ax_b, _ax_w, _ax_h],
                         projection=ccrs.PlateCarree())
    ax.set_aspect('auto')
    ax_cb = fig.add_axes([_cb_l, _ax_b, _cb_w, _cb_h])
    plot_spot_location_heatmap(ax, ax_cb, df, date, label='a')
    _strip(ax, fig, ax_cb)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('01_spot_location')))

    # 02 — raw histogram
    fig, ax, ax_cb = _fig_cb()
    plot_raw_histogram(ax, ax_cb, fit_result, ylim, label='b')
    _strip(ax, fig, ax_cb)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('02_raw_histogram')))

    # 03 — Gaussian filtered
    fig, ax, ax_cb = _fig_cb()
    plot_gaussian_filtered(ax, ax_cb, fit_result, ylim, label='c')
    _strip(ax, fig, ax_cb)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('03_gaussian_filtered')))

    # 04 — 8-bit rescaled + structural contours
    fig, ax, ax_cb = _fig_cb()
    plot_rescaled_contours(ax, ax_cb, fit_result, ylim, label='d')
    _strip(ax, fig, ax_cb)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('04_rescaled_contours')))

    # 05 — greyscale heatmap + detected edge
    fig, ax, ax_cb = _fig_cb()
    plot_edge_overlay(ax, ax_cb, fit_result, ylim, label='e')
    _strip(ax, fig, ax_cb)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('05_edge_overlay')))

    # 06 — quantile threshold lines + LOWESS + detected edge
    fig, ax, ax_cb = _fig_cb()
    plot_quantile_lines(ax, ax_cb, fit_result, ylim, label='e')
    _strip(ax, fig, ax_cb)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('06_quantile_lines')))

    # 07 — heatmap + edge + coefficient of variance (has twin axis)
    fig, ax, ax_cb = _fig_cb()
    plot_heatmap_with_variance(ax, ax_cb, fit_result, label='a', ylim=ylim)
    _strip(ax, fig, ax_cb)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('07_variance_selection')))

    # 08 — heatmap + polynomial fit overlay
    fig, ax, ax_cb = _fig_cb()
    plot_heatmap_with_polynomial(ax, ax_cb, fit_result, label='b', ylim=ylim)
    _strip(ax, fig, ax_cb)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('08_polynomial_fit')))

    # 09 — edge detrended (before bandpass)
    fig, ax = _fig_plain()
    plot_detrended_before_bandpass(ax, fit_result, label='c')
    _strip(ax, fig)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('09_detrended_no_bp')))

    # 10 — 1-4.5 hr bandpass filtered edge
    fig, ax = _fig_plain()
    plot_bandpass_filtered(ax, fit_result, label='d')
    _strip(ax, fig)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('10_bandpass_filtered')))

    # 11 — all sinusoid candidates (legend inside main axis)
    fig, ax = _fig_plain()
    _plot_sin_fits_inline(ax, fit_result, label='e')
    _strip(ax, fig)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('11_multiple_sin_fits')))

    # 12 — sin fit parameter table
    fig, ax = _fig_plain()
    plot_sin_fit_table(ax, fit_result, label='f')
    _strip(ax, fig)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('12_sin_fit_table')))

    # 13 — heatmap + polynomial + sin fit
    fig, ax, ax_cb = _fig_cb()
    plot_heatmap_with_final_fit(ax, ax_cb, fit_result, label='g', ylim=ylim)
    _strip(ax, fig, ax_cb)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('13_final_fit')))

    # 14 — selected (best R²) sinusoid only
    fig, ax = _fig_plain()
    plot_selected_sin_fit(ax, fit_result, label='d')
    _strip(ax, fig)
    if strip_mode != 'strip_all':
        fig.suptitle(date_title, fontweight='bold', y=_SUP_Y)
    saved.append(_save(fig, _p('14_selected_sin_fit')))

    return saved


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _plot_sin_fits_inline(ax, fit_result: FitOutput, *, label='e'):
    """
    Variant of plot_multiple_sin_fits with the legend inside the main axis.
    Avoids needing a separate legend axis so the uniform layout is preserved.
    """
    from datetime import datetime as _dt
    from scripts.sinusoid_fitting import sinusoid
    import matplotlib.cm as cm

    fit_times    = fit_result.fit_times
    data_detrend = fit_result.data_detrend
    all_sin_fits = fit_result.all_sin_fits
    date         = fit_result.meta['date']
    xlim         = fit_result.meta['time_limits']['xlim']
    fitWinLim    = fit_result.meta.get('fitWinLim', (None, None))

    if len(fit_times) > 0:
        ax.plot(fit_times, data_detrend, color='C0', label='Bandpass Filtered')

        if all_sin_fits:
            palette  = cm.get_cmap('tab10', max(len(all_sin_fits), 1))
            midnight = _dt(date.year, date.month, date.day)
            tt_sec   = np.array([
                (t - np.datetime64(midnight, 'ns')).astype('timedelta64[s]').astype(float)
                for t in fit_times
            ])
            for i, p in enumerate(all_sin_fits):
                vals = sinusoid(tt_sec,
                                T_hr=p['T_hr'], amplitude_km=p['amplitude_km'],
                                phase_hr=p['phase_hr'], offset_km=p['offset_km'],
                                slope_kmph=p['slope_kmph'])
                if i == 0:
                    ax.plot(fit_times, vals, color='red', lw=3, ls='--',
                            label=f"Sin Fit {i + 1} (SELECTED)")
                else:
                    ax.plot(fit_times, vals, color=palette(i), lw=1.5,
                            ls='--', alpha=0.6, label=f"Sin Fit {i + 1}")

        if fitWinLim[0] is not None:
            for wl in fitWinLim:
                ax.axvline(wl, color='lime', ls='--', lw=2)

    fmt_xaxis(ax, xlim)
    ax.set_title(f"({label}) Sinusoidal Fit to Bandpass Filtered Edge", loc='left')
    ax.set_ylabel('Range [km]')
    ax.legend(loc='upper right', fontsize='x-small', ncols=2)
