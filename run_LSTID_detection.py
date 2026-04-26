#!/usr/bin/env python3

import os
import math
import time
from datetime import datetime
from datetime import timedelta
import argparse, sys, json
import polars as pl
import pyarrow.parquet as pq
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import logging
import io
import pandas as pd
import warnings

from scipy.ndimage import gaussian_filter
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
import statsmodels.api as sm

# Internal modules
from scripts.regions import REGIONS
from scripts.utils import split_datetime_range_by_day
from scripts.utils_freq import *
from scripts.json_loader import *
from scripts.hdf5_loader import HDF5PolarsLoader
from scripts.heatmap_preprocess import preprocess_heatmap
from scripts.edge_detect import detect_edge
from scripts.sinusoid_fitting import sin_fit
from scripts.plot_lstid_paper import *

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)


def daily_summary(fit_result, df, date, *,
                  t_load=None, from_cache=False,
                  t_preprocess=None, t_edge=None, t_fit=None):
    """
    Build rows for the daily sinfit CSV summary.

    Returns one row per sinfit attempt (all_sin_fits), sorted by R² descending.
    If no stable window was found, returns a single NaN row for the date.

    Timing columns (on selected row only):
      t_load_cold_sec   — load time when reading from HDF5
      t_load_cached_sec — load time when reading from cache
      t_preprocess_sec  — preprocess_heatmap wall time
      t_edge_sec        — detect_edge wall time
      t_fit_sec         — sin_fit wall time

    Parameters
    ----------
    fit_result   : FitOutput
    df           : Polars DataFrame (raw spot data for the day)
    date         : datetime — the processing date
    t_load       : float — load + histogram wall time
    from_cache   : bool  — whether the dataframe/heatmap came from cache
    t_preprocess : float — preprocess wall time
    t_edge       : float — edge detect wall time
    t_fit        : float — sin_fit wall time

    Returns
    -------
    list[dict]
    """
    n_spots    = df.height
    fitWin_0, fitWin_1 = fit_result.meta.get('fitWinLim', (None, None))

    if fitWin_0 is not None and fitWin_1 is not None:
        fitStart    = pd.Timestamp(fitWin_0).strftime('%Y-%m-%d %H:%M')
        fitEnd      = pd.Timestamp(fitWin_1).strftime('%Y-%m-%d %H:%M')
        duration_hr = (pd.Timestamp(fitWin_1) - pd.Timestamp(fitWin_0)).total_seconds() / 3600
    else:
        fitStart = fitEnd = duration_hr = np.nan

    t_load_cold   = t_load if not from_cache else np.nan
    t_load_cached = t_load if from_cache     else np.nan

    all_sin_fits = fit_result.all_sin_fits

    if not all_sin_fits:
        return [{
            'date':               date.strftime('%Y-%m-%d'),
            'selected':           np.nan,
            'T_hr':               np.nan,
            'T_hr_guess':         np.nan,
            'amplitude_km':       np.nan,
            'phase_hr':           np.nan,
            'offset_km':          np.nan,
            'slope_kmph':         np.nan,
            'r2':                 np.nan,
            'fitStart':           fitStart,
            'fitEnd':             fitEnd,
            'duration_hr':        duration_hr,
            'min_combined_fit':   np.nan,
            't_load_cold_sec':    t_load_cold,
            't_load_cached_sec':  t_load_cached,
            't_preprocess_sec':   t_preprocess,
            't_edge_sec':         t_edge,
            't_fit_sec':          t_fit,
            'n_spots':            n_spots,
        }]

    has_combined = (len(fit_result.sin_fit) > 0 and len(fit_result.poly_fit) > 0)
    min_combined = float(np.min(fit_result.sin_fit + fit_result.poly_fit)) if has_combined else np.nan

    rows = []
    for i, fit in enumerate(all_sin_fits):
        selected = (i == 0)
        rows.append({
            'date':               date.strftime('%Y-%m-%d'),
            'selected':           selected,
            'T_hr':               fit.get('T_hr'),
            'T_hr_guess':         fit.get('T_hr_guess'),
            'amplitude_km':       fit.get('amplitude_km'),
            'phase_hr':           fit.get('phase_hr'),
            'offset_km':          fit.get('offset_km'),
            'slope_kmph':         fit.get('slope_kmph'),
            'r2':                 fit.get('r2'),
            'fitStart':           fitStart,
            'fitEnd':             fitEnd,
            'duration_hr':        duration_hr,
            'min_combined_fit':   min_combined if selected else np.nan,
            't_load_cold_sec':    t_load_cold  if selected else np.nan,
            't_load_cached_sec':  t_load_cached if selected else np.nan,
            't_preprocess_sec':   t_preprocess  if selected else np.nan,
            't_edge_sec':         t_edge        if selected else np.nan,
            't_fit_sec':          t_fit         if selected else np.nan,
            'n_spots':            n_spots,
        })
    return rows

if __name__ == "__main__":

    cfg, _ = load_config()

    req_base  = ["data_dir","cache_dir","region_name","freq","distance_range","use_cache"]
    req_pre   = ["expected_shape","expected_size","x_trim","y_trim","min_dev","sigma","occurrence_n","i_max"]
    req_edg   = ["qs","lower_cutoff","select_min","exact_thresh","axis","lowess_window_size","max_abs_dev"]
    req_fit   = ["bandpass","lstid_T_hr_lim","roll_win","stab_thresh","margin_minutes"]  
    req_plot  = ["ylim","cb_pad","output_dir"]
    
    base_cfg = cfg
    pre_cfg  = cfg.get("preprocess", {})
    edge_cfg = cfg.get("edge_detection", {})
    fit_cfg  = cfg.get("fitting", {})
    plot_cfg = cfg.get("plotting", {}) 
    
    missing = []
    missing += [k for k in req_base if k not in base_cfg]
    missing += [f"preprocess.{k}" for k in req_pre if k not in pre_cfg]
    missing += [f"edge_detection.{k}" for k in req_edg if k not in edge_cfg]
    missing += [f"fitting.{k}" for k in req_fit if k not in fit_cfg]  
    missing += [f"plotting.{k}" for k in req_plot if k not in plot_cfg]

    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    base_params = dict(
        data_dir=cfg["data_dir"],
        cache_dir=cfg["cache_dir"],
        region_bounds=REGIONS[cfg["region_name"]],
        freq_range=FREQ[cfg["freq"]],
        distance_range=cfg["distance_range"],
        use_cache=cfg["use_cache"],
    )

    pre_params = dict(
        expected_shape=tuple(pre_cfg["expected_shape"]),
        expected_size=int(pre_cfg["expected_size"]),
        min_dev=float(pre_cfg["min_dev"]),
        x_trim=float(pre_cfg["x_trim"]),
        y_trim=float(pre_cfg["y_trim"]),
        sigma=float(pre_cfg["sigma"]),
        occurrence_n=int(pre_cfg["occurrence_n"]),
        i_max=int(pre_cfg["i_max"]),
    )

    edge_params = dict(
        qs=edge_cfg["qs"],
        lower_cutoff=int(edge_cfg["lower_cutoff"]),
        select_min=edge_cfg["select_min"],
        exact_thresh=edge_cfg["exact_thresh"],
        axis=int(edge_cfg["axis"]),
        lowess_window_size=int(edge_cfg["lowess_window_size"]),
        max_abs_dev=int(edge_cfg["max_abs_dev"]),
    )

    fit_params = dict(
        bandpass=bool(fit_cfg["bandpass"]),
        lstid_T_hr_lim=tuple(fit_cfg["lstid_T_hr_lim"]),
        roll_win=int(fit_cfg["roll_win"]),
        stab_thresh=float(fit_cfg["stab_thresh"]),
        margin_minutes=int(fit_cfg["margin_minutes"]),
    )

    plot_params = dict(
        ylim=tuple(plot_cfg["ylim"]),
        cb_pad=float(plot_cfg["cb_pad"]),
        output_dir=plot_cfg["output_dir"],
    )

    # --- CSV summary path (constructed once from overall date range + filter params) ---
    _dr         = cfg["distance_range"]
    _min_dist   = _dr["min_dist"]
    _max_dist   = _dr["max_dist"]
    _sDate_str  = cfg["sDate"].strftime('%Y%m%d')
    _eDate_str  = cfg["eDate"].strftime('%Y%m%d')
    _region_str = cfg["region_name"].replace(' ', '_')
    _csv_dir    = os.path.join("output", "summary_csv")
    os.makedirs(_csv_dir, exist_ok=True)
    csv_path = os.path.join(
        _csv_dir,
        f"{_sDate_str}-{_eDate_str}_{_region_str}_{cfg['freq']}MHz"
        f"_{_min_dist}-{_max_dist}km_sinfit.csv"
    )

    for s_dt, e_dt, date_str in split_datetime_range_by_day(cfg["sDate"], cfg["eDate"]):

        ### Supress NaN column warning (expected behavior)
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                        message='All-NaN slice encountered')
        cfg, _ = load_config()
        ###

        day_params = dict(base_params, sDate=s_dt, eDate=e_dt)

        loader = HDF5PolarsLoader(**day_params)
        from_cache = (loader.use_cache
                      and loader.cache_path_df.exists()
                      and loader.cache_path_hist.exists())

        _t0 = time.perf_counter()
        df = loader.get_dataframe()
        hist2d, meta = loader.gen_histogram()
        t_load = round(time.perf_counter() - _t0, 2)
        log.info(f"  → Load {'(cache)' if from_cache else '(cold)'}: {t_load}s")

        log.info("Stage 1/3: Preprocessing heatmap...")
        _t0 = time.perf_counter()
        preprocess_result = preprocess_heatmap(hist2d, meta, **pre_params)
        t_preprocess = round(time.perf_counter() - _t0, 2)
        log.info(f"Stage 1/3: Preprocessing heatmap Complete ({t_preprocess}s)")

        log.info("Stage 2/3: Edge Detection...")
        _t0 = time.perf_counter()
        edge_result = detect_edge(s_dt, preprocess_result, **edge_params)
        t_edge = round(time.perf_counter() - _t0, 2)
        log.info(f"Stage 2/3: Edge Detection Complete ({t_edge}s)")

        log.info("Stage 3/3: Sinusoidal Fitting...")
        _t0 = time.perf_counter()
        fit_result = sin_fit(edge_result, **fit_params)
        t_fit = round(time.perf_counter() - _t0, 2)
        log.info(f"Stage 3/3: Sinusoidal Fitting Complete ({t_fit}s)")

        if fit_result.sin_params:
            log.info("  → Fit passed")
        else:
            log.warning("  → Fit failed - no stable region found")

        # --- CSV summary ---
        log.info("Saving CSV summary...")
        rows = daily_summary(
            fit_result, df, s_dt,
            t_load=t_load, from_cache=from_cache,
            t_preprocess=t_preprocess, t_edge=t_edge, t_fit=t_fit,
        )
        new_df = pd.DataFrame(rows)
        date_str_csv = s_dt.strftime('%Y-%m-%d')
        if os.path.exists(csv_path):
            existing = pd.read_csv(csv_path)
            existing_day = existing[existing['date'] == date_str_csv]
            existing_other = existing[existing['date'] != date_str_csv]
            # Preserve whichever load-time column wasn't captured this run
            if not existing_day.empty:
                prev = existing_day.iloc[0]
                if from_cache:
                    cold_prev = prev.get('t_load_cold_sec', np.nan)
                    new_df.loc[new_df['selected'] == True, 't_load_cold_sec'] = cold_prev
                else:
                    cached_prev = prev.get('t_load_cached_sec', np.nan)
                    new_df.loc[new_df['selected'] == True, 't_load_cached_sec'] = cached_prev
            combined = pd.concat([existing_other, new_df], ignore_index=True)
            combined.to_csv(csv_path, index=False)
        else:
            new_df.to_csv(csv_path, index=False)
        log.info(f"CSV summary saved: {csv_path} ({len(rows)} row(s))")

        log.info("Creating stack plots...")
        stack_plot_preprocess(fit_result, df, **plot_params)
        stack_plot_sinfit_v1(fit_result, **plot_params)
        stack_plot_sinfit_v2(fit_result, **plot_params)
        stack_plot_sinfit_v3(fit_result, **plot_params)
        
        log.info(f"Processing complete for {date_str}\n")
    
    log.info("Pipeline complete!")

