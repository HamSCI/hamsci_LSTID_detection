#!/usr/bin/env python
# coding: utf-8
import os
import shutil
import warnings
import pickle
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import xarray as xr
import joblib
import math
import datetime
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import signal
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from data_loading import create_xarr, mad, create_label_df
from utils import DateIter

# Load in Mary Lou West's Manual LSTID Analysis
import lstid_ham
lstid_mlw   = lstid_ham.LSTID_HAM()
df_mlw      = lstid_mlw.df.copy()
df_mlw      = df_mlw.set_index('date')
old_keys    = list(df_mlw.keys())
new_keys    = {x:'MLW_'+x for x in old_keys}
df_mlw      = df_mlw.rename(columns=new_keys)


plt.rcParams['font.size']           = 18
plt.rcParams['font.weight']         = 'bold'
plt.rcParams['axes.titleweight']    = 'bold'
plt.rcParams['axes.labelweight']    = 'bold'
plt.rcParams['axes.xmargin']        = 0
#plt.rcParams['axes.grid']           = True
#plt.rcParams['grid.linestyle']      = ':'
cb_pad = 0.04

parent_dir     = 'data_files'
data_out_path  = 'processed_data/full_data.joblib'

def fmt_xaxis(ax,xlim=None,label=True):
    ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H%M"))
    ax.set_xlabel('Time [UTC]')
    ax.set_xlim(xlim)

def scale_km(edge,ranges):
    """
    Scale detected edge array indices to kilometers.
    edge:   Edge in array indices.
    ranges: Ground range vector in km of histogram array.
    """
    ranges  = np.array(ranges) 
    edge_km = (edge / len(ranges) * ranges.ptp()) + ranges.min()

    return edge_km

def adjust_axes(ax_0,ax_1):
    """
    Force geospace environment axes to line up with histogram
    axes even though it doesn't have a color bar.
    """
    ax_0_pos    = list(ax_0.get_position().bounds)
    ax_1_pos    = list(ax_1.get_position().bounds)
    ax_0_pos[2] = ax_1_pos[2]
    ax_0.set_position(ax_0_pos)

def my_sin(tt_sec, freq, amplitude, phase, offset):
    # create the function we want to fit
    result = amplitude * np.sin( (2*np.pi*tt_sec*freq )+ phase ) + offset
    return result

def draw_mlw(
    date,
    x_trim=.08333,
    y_trim=.08,
    sigma=4.2, # 3.8 was good # Gaussian filter kernel
    output_dir='output'):

    arr = date_iter.get_date(date,raise_missing=False)
    if arr is None:
        warnings.warn(f'Date {date} has no input')
        return
        
    xl_trim, xrt_trim   = x_trim if isinstance(x_trim, (tuple, list)) else (x_trim, x_trim)
    yl_trim, yr_trim    = x_trim if isinstance(y_trim, (tuple, list)) else (y_trim, y_trim)
    xrt, xl             = math.floor(xl_trim * arr.shape[0]), math.floor(xrt_trim * arr.shape[0])
    yr, yl              = math.floor(yl_trim * arr.shape[1]), math.floor(yr_trim * arr.shape[1])
    arr                 = arr[xrt:-xl, yr:-yl]

    ranges_km   = arr.coords['height']
    arr_times   = [date + x for x in pd.to_timedelta(arr.coords['time'])]
    arr         = np.nan_to_num(arr, nan=0)
    arr         = gaussian_filter(arr.T, sigma=(sigma, sigma))

    # Plotting Code ################################################################ 
    # X-Limits for plotting
    x_0     = date + datetime.timedelta(hours=12)
    x_1     = date + datetime.timedelta(hours=24)
    xlim    = (x_0, x_1)

    nCols   = 1
    nRows   = 1
    axInx   = 0
    figsize = (18,nRows*5)

    fig     = plt.figure(figsize=figsize)
    axs     = []

    # Plot Heatmap #########################
    axInx   = axInx + 1
    ax      = fig.add_subplot(nRows,nCols,axInx)
    axs.append(ax)

    ax.set_title(f'| {date} |')
    mpbl = ax.pcolormesh(arr_times,ranges_km,arr,cmap='plasma')
    plt.colorbar(mpbl,label='Radio Spots',aspect=10,pad=cb_pad)

    if date in df_mlw.index:
        mlw = df_mlw.loc[date,:]

    if not np.isnan(mlw['MLW_period_hr']):
        tt_sec = np.array([(x-date).total_seconds() for x in arr_times])
        p0 = {}
        p0['tt_sec']    = tt_sec
        p0['freq']      = 1./(mlw['MLW_period_hr'] * 3600.)
        p0['amplitude'] = mlw['MLW_range_range']/2.
        p0['phase']     = 0.
        p0['offset']    = mlw['MLW_low_range_km'] + mlw['MLW_range_range']/2.

        mlw_sin = my_sin(**p0)
        mlw_sin = pd.Series(mlw_sin,index=arr_times)
        
        mlw_sTime   = date + datetime.timedelta(hours=mlw['MLW_start_time'])
        mlw_eTime   = date + datetime.timedelta(hours=mlw['MLW_end_time'])

        tf  = np.logical_and(mlw_sin.index >= mlw_sTime, mlw_sin.index < mlw_eTime)
        mlw_sin[~tf] = np.nan

        lbl = []
        lbl.append('$T$ = {:0.1f} hr'.format(mlw['MLW_period_hr']))
        lbl.append('$A$ = {:0.0f} km'.format(mlw['MLW_range_range']))
        lbl.append('$R_l$ = {:0.0f} km'.format(mlw['MLW_low_range_km']))
        lbl.append('$R_h$ = {:0.0f} km'.format(mlw['MLW_high_range_km']))
        lbl.append('$\phi$ = {:0.0f}'.format(p0['phase']))
        lbl = ' '.join(lbl)
        ax.plot(mlw_sin.index,mlw_sin,color='white',lw=2,ls='--',label=lbl)
        ax.legend(loc='lower right') 

    fmt_xaxis(ax,xlim)
    ax.set_ylabel('Range [km]')
    ax.set_ylim(500,2000)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    date_str    = date.strftime('%Y%m%d')
    png_fname   = f'{date_str}_mlwCurve.png'
    png_fpath   = os.path.join(output_dir,png_fname)
    print('   Saving: {!s}'.format(png_fpath))
    fig.savefig(png_fpath,bbox_inches='tight')
    plt.close()
    return

if __name__ == '__main__':
    output_dir  = os.path.join('output','draw_mlw')

    sDate   = datetime.datetime(2018,11,1)
    eDate   = datetime.datetime(2019,4,30)

#    sDate   = datetime.datetime(2018,11,9)
#    eDate   = datetime.datetime(2018,11,9)

#    sDate   = datetime.datetime(2018,11,5)
#    eDate   = datetime.datetime(2018,11,5)

    # NO PARAMETERS BELOW THIS LINE ################################################
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    tic = datetime.datetime.now()
    dates   = [sDate]
    while dates[-1] < eDate:
        dates.append(dates[-1]+datetime.timedelta(days=1))

    # Load in CSV Histograms ###############
    if not os.path.exists(data_out_path):
        full_xarr = create_xarr(
            parent_dir=parent_dir,
            expected_shape=(720, 300),
            dtype=(np.uint16, np.float32),
            apply_fn=mad,
            plot=False,
        )
        joblib.dump(full_xarr, data_out_path)

    date_iter = DateIter(data_out_path) #, label_df=label_out_path)

    ########################################
    for dinx,date in enumerate(dates):
        draw_mlw(date,output_dir=output_dir)

    toc = datetime.datetime.now()
    print('Processing and plotting time: {!s}'.format(toc-tic))

import ipdb; ipdb.set_trace()
