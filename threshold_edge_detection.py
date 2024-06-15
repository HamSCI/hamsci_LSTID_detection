# import math
# import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np
import statsmodels.api as sm
# import pandas as pd

from scipy.interpolate import CubicSpline
# from scipy.ndimage import gaussian_filter
from scipy import signal
# from utils import DateIter
# from IPython.display import clear_output

def occurrence_max(arr, n, equal=False):
    ## change this to be two sided
    hist, bins = np.histogram(arr, bins=np.arange(np.min(arr), np.max(arr) + 2))
    bins = bins[1:]

    if equal:
        bin_mask = np.where(hist >= n)
    else:
        hist, bins = hist[::-1], bins[::-1]
        hist = np.cumsum(hist)
        bin_mask = hist >= n

    max_value = np.max(bins[bin_mask])
    return max_value

def rescale_to_int(arr, occurrence_n=100, i_max=30):
    assert i_max < 255, i_max

    arr = arr - np.amin(arr)
    max_val = occurrence_max(arr.round().astype(np.uint16), occurrence_n)
    factor = i_max / max_val
    arr = arr * factor
    arr = arr.round().astype(np.uint8)
    return arr

def stack_all_thresholds(arr, select_min=True, exact_thresh=False, axis=0, **rescale_kwargs):
    arr = rescale_to_int(arr, **rescale_kwargs)

    thresholds = np.unique(arr)
    thresh_edges = list()
    for threshold in thresholds:
        if exact_thresh:
            thresh_mask = arr <= threshold
        else:
            thresh_mask = arr != threshold
        
        idx_fn = np.argmin if select_min else np.argmax
        thresh_edge = idx_fn(thresh_mask.astype(np.uint8), axis=axis, keepdims=True)
            
        assert max(thresh_edge.shape) == max(arr.shape), f'{thresh_edge.shape} | {arr.shape}'
        
        thresh_edges.append(thresh_edge)
    thresh_edge_arr = np.concatenate(thresh_edges, axis=axis)
    return thresh_edge_arr

def lowess_smooth(arr, window_size=10, x=None):
    if x is None:
        x = np.linspace(0, len(arr), len(arr))
    frac = window_size/len(arr)
    z = sm.nonparametric.lowess(arr, x, frac=frac, return_sorted=False)    
    return z

def butter_smooth(arr, tc_limits=(300, 60), btype='bandpass'):
    wn = tuple(map(lambda x : 1 / (x * 60), tc_limits))
    b, a = signal.butter(2, wn, 'bandpass', fs=fs)

    z = signal.filtfilt(b, a, arr)
    return z

def smooth_remove_abs_deviation(arr, smooth_fn, max_abs_dev=20):
    x = np.arange(0, arr.shape[0], 1)
    z = smooth_fn(arr)
    assert len(x) == len(arr)
    assert len(z) == len(x)
    dev_mask = np.abs(arr - z) < max_abs_dev
    interp = CubicSpline(x[dev_mask], z[dev_mask])
    z = interp(x)
    return z

def select_min_deviation(arrs, smooth_fn, max_abs_dev=20):
    min_arr = None
    min_dev = np.inf
    for arr in arrs:
#         z = smooth_fn(arr)
        z = smooth_remove_abs_deviation(arr, smooth_fn, max_abs_dev=max_abs_dev)
        dev = np.std(arr - z)
        if min_arr is None or dev < min_dev:
            min_arr = (arr, z)
            min_dev = dev
    return min_arr

def measure_thresholds(arr, qs=.8, lower_cutoff=10, **threshold_kwargs):
    thresh_edge_arr = stack_all_thresholds(arr, **threshold_kwargs)
    
    thresh_edge_arr = thresh_edge_arr.astype(np.float32)
    thresh_edge_arr[thresh_edge_arr < lower_cutoff] = np.nan   
    
    if isinstance(qs, float):
        qs = [qs]

    med_lines = [np.nanquantile(thresh_edge_arr, q, axis=0) for q in qs]
    min_line, minz_line = select_min_deviation(med_lines, lowess_smooth)
    
    return med_lines, min_line, minz_line

def non_diag_corr(df):
    arr = df.to_numpy()
    np.fill_diagonal(arr, np.nan)
    mean = np.nanmean(arr)
    return mean

