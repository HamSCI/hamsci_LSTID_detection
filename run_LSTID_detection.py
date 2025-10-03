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
from scripts.edge_detect import edge_detection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

if __name__ == "__main__":


    cfg, _ = load_config()
    required = ["data_dir","cache_dir","region_name","freq","distance_range","use_cache"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required config keys: {missing}")

    base_params = dict(
        data_dir=cfg["data_dir"],
        cache_dir=cfg["cache_dir"],
        region_name=cfg["region_name"],
        freq_range=FREQ[cfg["freq"]],
        distance_range=cfg["distance_range"],
        use_cache=cfg["use_cache"],
        expected_shape=tuple(cfg["expected_shape"]),
        expected_size=int(cfg["expected_size"]),
        min_dev=float(cfg["min_dev"]),
        x_trim=float(cfg["x_trim"]),
        y_trim=float(cfg["y_trim"]),
        sigma=float(cfg["sigma"]),
        occurrence_n=int(cfg["occurrence_n"]),
        i_max=int(cfg["i_max"]),
        qs=cfg["qs"],
    )

    for s_dt, e_dt, date_str in split_datetime_range_by_day(cfg["sDate"], cfg["eDate"]):

        day_params = dict(base_params, sDate=s_dt, eDate=e_dt)
        loader = HDF5PolarsLoader(**day_params)
        df      = loader.get_dataframe()
        hist2d, meta = loader.gen_histogram()

        arr, arr_times, ranges_km, Ts_sec, Ts_td = preprocess_heatmap(
            hist2d, meta, **day_params
        )

        daily_result = edge_detection(arr, arr_times, ranges_km, Ts_sec, Ts_td, **day_params)

        print(dalily_result)


