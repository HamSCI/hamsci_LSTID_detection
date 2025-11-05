#!/usr/bin/env python3

import numpy as np
import polars as pl
from typing import Dict, List, Any, Tuple
from datetime import datetime, timedelta
import datetime as dt
from numpy.polynomial import polynomial as poly
from scipy.optimize import curve_fit

def sinusoid(tt_sec,T_hr,amplitude_km,phase_hr,offset_km,slope_kmph):
    """
    Sinusoid function that will be fit to data.
    """
    phase_rad       = (2.*np.pi) * (phase_hr / T_hr) 
    freq            = 1./(datetime.timedelta(hours=T_hr).total_seconds())
    result          = np.abs(amplitude_km) * np.sin( (2*np.pi*tt_sec*freq ) + phase_rad ) + (slope_kmph/3600.)*tt_sec + offset_km
    return result

def bandpass_filter(
    data,
    lowcut=0.00005556, 
    highcut=0.0001852, 
    fs=0.0166666666666667, 
    order=4):
    """
    Defaults:
    1 hour period = 0.000277777778 Hz
    5 hour period   = 0.00005556 Hz
    Sampling Freq   = 0.0166666666666667 Hz (our data is in 1 min resolution)
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = filtfilt(b, a, data)
    return filtered

def islandinfo(y, trigger_val, stopind_inclusive=True):
    """
    From https://stackoverflow.com/questions/50151417/numpy-find-indices-of-groups-with-same-value
    """
    # Setup "sentients" on either sides to make sure we have setup
    # "ramps" to catch the start and stop for the edge islands
    # (left-most and right-most islands) respectively
    y_ext = np.r_[False,y==trigger_val, False]

    # Get indices of shifts, which represent the start and stop indices
    idx = np.flatnonzero(y_ext[:-1] != y_ext[1:])

    # Lengths of islands if needed
    lens = idx[1::2] - idx[:-1:2]

    # Using a stepsize of 2 would get us start and stop indices for each island
    return list(zip(idx[:-1:2], idx[1::2]-int(stopind_inclusive))), lens

def sin_fit(df: pl.DataFrame, meta: dict):
    x_0, x_1 = meta["xlim"]
    win_0, win_1 = meta["winlim"]

    # seconds since midnight
    t0 = dt.datetime(x_0.year, x_0.month, x_0.day)
    t0_s = t0.timestamp()

    df = (
        df.sort("Time")
          .with_columns(
              (pl.col("Time").dt.epoch("s") - pl.lit(t0_s)).cast(pl.Float64).alias("tt_sec")
          )
    )

    # 15-minute rolling via time-based windows
    rolled = (
        df.group_by_dynamic(index_column="Time", every="1m", period="15m", closed="right")
          .agg([
              pl.col("sg_edge").std().alias("xx_n"),
              pl.col("sg_edge").mean().alias("xx_d"),
          ])
          .sort("Time")
    )

    df = (
        df.join_asof(rolled, on="Time", strategy="backward",
                     tolerance=pl.duration(minutes=1))
          .with_columns((pl.col("xx_n") / pl.col("xx_d")).alias("stability"))
    )

    print(df)
    return df