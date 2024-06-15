#!/usr/bin/env python
# coding: utf-8
import os
import warnings
import numpy as np
import pandas as pd
import joblib
import math
import datetime
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter
from data_loading import create_xarr, mad, create_label_df
from utils import DateIter
from threshold_edge_detection import lowess_smooth, measure_thresholds

plt.rcParams['font.size']           = 18
plt.rcParams['font.weight']         = 'bold'
plt.rcParams['axes.titleweight']    = 'bold'
plt.rcParams['axes.labelweight']    = 'bold'
plt.rcParams['axes.xmargin']        = 0
#plt.rcParams['axes.grid']           = True
#plt.rcParams['grid.linestyle']      = ':'

parent_dir     = 'data_files'
label_csv_path = 'official_labels.csv'
data_out_path  = 'processed_data/full_data.joblib'
label_out_path = 'labels/labels.joblib'


# In[3]:

tic = datetime.datetime.now()
if not os.path.exists(data_out_path):
    full_xarr = create_xarr(
        parent_dir=parent_dir,
        expected_shape=(720, 300),
        dtype=(np.uint16, np.float32),
        apply_fn=mad,
        plot=False,
    )
    joblib.dump(full_xarr, data_out_path)

if not os.path.exists(label_csv_path):
    label_df = create_label_df(
        csv_path=label_csv_path,
    )
    joblib.dump(label_df, label_out_path)

date_iter = DateIter(data_out_path) #, label_df=label_out_path)
toc = datetime.datetime.now()
print('Loading time: {!s}'.format(toc-tic))

def save_wrap(save_dir, fmt='%Y-%m-%d', ext='.png', **kwargs):
    os.makedirs(save_dir, exist_ok=True)
    def wrapped(date):
        date_str = pd.to_datetime(date).strftime(fmt)
        file_path = os.path.join(save_dir, date_str + ext)
        plt.savefig(file_path,bbox_inches='tight',**kwargs)
        return
    return wrapped

def scale_km(edge,ranges):
    """
    Scale detected edge array indices to kilometers.
    edge:   Edge in array indices.
    ranges: Ground range vector in km of histogram array.
    """
    ranges  = np.array(ranges) 
    edge_km = (edge / len(ranges) * ranges.ptp()) + ranges.min()

    return edge_km

def run_edge_detect(
    dates,
    x_trim=.08333,
    y_trim=.08,
    sigma=4.2, # 3.8 was good # Gaussian filter kernel
    qs=[.4, .5, .6],
    occurence_n = 60,
    i_max=30,
    plot=True,
    clear_every=100,
    plt_save_path=None,
    csv_save_path=None,
    thresh=None,
):

    processed_dates = list()
    if plt_save_path is not None:
        save_plt = save_wrap(plt_save_path)
    else:
        save_plt = None
        
    final_edge_list = list()
    if dates == 'all':
        date_gen = date_iter.iter_all()
    else:
        date_gen = date_iter.iter_dates(dates, raise_missing=False)

    for i, (date, arr) in enumerate(date_gen):
        if arr is None:
            warnings.warn(f'Date {date} has no input')
            continue
            
        xl_trim, xr_trim = x_trim if isinstance(x_trim, (tuple, list)) else (x_trim, x_trim)
        yl_trim, yr_trim = x_trim if isinstance(y_trim, (tuple, list)) else (y_trim, y_trim)
        xr, xl = math.floor(xl_trim * arr.shape[0]), math.floor(xr_trim * arr.shape[0])
        yr, yl = math.floor(yl_trim * arr.shape[1]), math.floor(yr_trim * arr.shape[1])

        arr = arr[xr:-xl, yr:-yl]

        ranges_km   = arr.coords['height']
        times       = [date + x for x in pd.to_timedelta(arr.coords['time'])]

        arr_xr  = arr
        arr     = np.nan_to_num(arr, nan=0)

        arr = gaussian_filter(arr.T, sigma=(sigma, sigma))  # [::-1,:]
        med_lines, min_line, minz_line = measure_thresholds(
            arr,
            qs=qs, 
            occurrence_n=occurence_n, 
            i_max=i_max
        )

        med_lines   = [scale_km(x,ranges_km) for x in med_lines]
        min_line    = scale_km(min_line,ranges_km)
        minz_line   = scale_km(minz_line,ranges_km)

        data = pd.DataFrame(
            np.array(med_lines).T,
            index=times,
            columns=qs,
        ).reset_index(
            names='Time',
        )

        if thresh is None:
            edge_line = pd.DataFrame(
                min_line, 
                index=times,
                columns=['Height'],
            ).reset_index(
                names='Time'
            )
        elif isinstance(thresh, dict):
            edge_line = (
                data[['Time', thresh[date]]]
                .rename(columns={thresh[date] : 'Height'})
            )
        elif isinstance(thresh, float):
            edge_line = (
                data[['Time', thresh]]
                .rename(columns={thresh : 'Height'})
            )
        else:
            raise ValueError(f'Threshold {thresh} of type {type(thresh)} is invalid')

        final_edge_list.append(
            pd.Series(min_line.squeeze(), index=times, name=date)
        )

        if plot:
            fig     = plt.figure(figsize=(16,8))
            ax      = fig.add_subplot(1,1,1)

            ax.set_title(f'| {date} |')

            ax.pcolormesh(times,ranges_km,arr,cmap='plasma')
            fig.autofmt_xdate()

            for col in data.columns:
                if col == 'Time':
                    continue
                lbl = '{!s}'.format(col)
                ax.plot(times,data[col],label=lbl)

            ax.plot(times,min_line,lw=2,label='Final Edge')

            ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=1))
            ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H%M"))

            ax.set_xlabel('Time [UTC]')
            ax.set_ylabel('Range [km]')

            x_0 = date + datetime.timedelta(hours=12)
            x_1 = date + datetime.timedelta(hours=24)
            
            ax.set_xlim(x_0,x_1)
            ax.set_ylim(0,3000)

            ax.legend(loc='lower right',fontsize='small',ncols=4)

            if save_plt is not None:
                save_plt(date)
            
            processed_dates.append(date)
            plt.close()

    final_edge_df = pd.concat(final_edge_list, axis=1)
    if csv_save_path:
        final_edge_df.to_csv(csv_save_path)
    
    return {'date':date,'arr':arr,'arr_xr':arr_xr,'final_edge_df':final_edge_df,'data':data,'edge_line':edge_line}

tic = datetime.datetime.now()
result = run_edge_detect(
    [('2018-11-1','2018-11-1')], 
    csv_save_path=None,
    plot=True,
    plt_save_path='output'
)
toc = datetime.datetime.now()

print('Processing and plotting time: {!s}'.format(toc-tic))

import ipdb; ipdb.set_trace()
