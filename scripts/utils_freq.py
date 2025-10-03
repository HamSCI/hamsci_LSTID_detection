#!/usr/bin/env python3

import polars as pl
from datetime import datetime

FREQ = {
    1: {
        'min_freq': 0_000_000,
        'max_freq': 2_000_000,
        'label': '1'
    },
    3: {
        'min_freq': 2_000_000,
        'max_freq': 4_000_000,
        'label': '3'
    },
    7: {
        'min_freq': 6_000_000,
        'max_freq': 8_000_000,
        'label': '7'
    },
    14: {
        'min_freq': 13_000_000,
        'max_freq': 15_000_000,
        'label': '14'
    },
    21: {
        'min_freq': 20_000_000,
        'max_freq': 22_000_000,
        'label': '21'
    },
    28: {
        'min_freq': 27_000_000,
        'max_freq': 29_000_000,
        'label': '28'
    },
    'All Freq (1,3,7,14,21,28)': {
        'min_freq': 0,
        'max_freq': 40_000_000,
        'label': 'All'
    }
}


def filter_pl_freq(df: pl.DataFrame, freq_key: int) -> pl.DataFrame:
    freq = FREQ[freq_key]
    return df.filter(pl.col('freq').is_between(freq['min_freq'], freq['max_freq']))
    
   