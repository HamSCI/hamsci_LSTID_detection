#!/usr/bin/env python3

import os
import math
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

    for s_dt, e_dt, date_str in split_datetime_range_by_day(cfg["sDate"], cfg["eDate"]):

        ### Supress NaN column warning (expected behavior)
        warnings.filterwarnings('ignore', category=RuntimeWarning, 
                        message='All-NaN slice encountered')
        cfg, _ = load_config()
        ###

        day_params = dict(base_params, sDate=s_dt, eDate=e_dt)

        loader = HDF5PolarsLoader(**day_params)
        df = loader.get_dataframe()
        hist2d, meta = loader.gen_histogram()
        
        log.info("Stage 1/3: Preprocessing heatmap...")
        preprocess_result = preprocess_heatmap(hist2d, meta, **pre_params)
        log.info("Stage 1/3: Preprocessing heatmap Complete")
        log.info("Stage 2/3: Edge Detection...")
        edge_result = detect_edge(s_dt, preprocess_result, **edge_params)
        log.info("Stage 2/3: Edge Detection Complete")
        log.info("Stage 3/3: Sinusoidal Fitting...")
        fit_result = sin_fit(edge_result, **fit_params)
        log.info("Stage 3/3: Sinusoidal Fitting Complete")
    
        if fit_result.sin_params:
            log.info(f"  → R² = {fit_result.sin_params.get('r2', 0):.4f}")
            log.info(f"  → Period = {fit_result.sin_params.get('T_hr', 0):.2f} hours")
            log.info(f"  → Amplitude = {fit_result.sin_params.get('amplitude_km', 0):.2f} km")
        else:
            log.warning("  → Fit failed - no stable region found")
        
        log.info("Creating stack plots...")
        stack_plot_preprocess(fit_result, df, **plot_params)
        stack_plot_sinfit_v1(fit_result, **plot_params)
        stack_plot_sinfit_v2(fit_result, **plot_params)
        stack_plot_sinfit_v3(fit_result, **plot_params)
        
        log.info(f"Processing complete for {date_str}\n")
    
    log.info("Pipeline complete!")

