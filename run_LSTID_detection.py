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
from scripts.edge_detect import thresholding

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

if __name__ == "__main__":

    cfg, _ = load_config()

    req_base = ["data_dir","cache_dir","region_name","freq","distance_range","use_cache"]
    req_pre  = ["expected_shape","expected_size","x_trim","y_trim","min_dev","sigma","occurrence_n","i_max"]
    req_edg  = ["qs","lower_cutoff","select_min","exact_thresh","axis"]

    base_cfg = cfg
    pre_cfg  = cfg.get("preprocess", {})
    edge_cfg = cfg.get("edge_detection", {})

    missing = []
    missing += [k for k in req_base if k not in base_cfg]
    missing += [f"preprocess.{k}" for k in req_pre if k not in pre_cfg]
    missing += [f"edge_detection.{k}" for k in req_edg if k not in edge_cfg]

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

    for s_dt, e_dt, date_str in split_datetime_range_by_day(cfg["sDate"], cfg["eDate"]):
        day_params = dict(base_params, sDate=s_dt, eDate=e_dt)

        loader = HDF5PolarsLoader(**day_params)
        df = loader.get_dataframe()
        hist2d, meta = loader.gen_histogram()
        print(hist2d.shape)

        arr, arr_times, ranges_km, Ts_sec, Ts_td = preprocess_heatmap(hist2d, meta, **pre_params)
        daily_result = thresholding(arr, arr_times, ranges_km, Ts_sec, Ts_td, **edge_params)

        print(daily_result)


