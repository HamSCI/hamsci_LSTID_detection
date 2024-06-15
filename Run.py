#!/usr/bin/env python
# coding: utf-8
import os
import warnings
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import joblib
import math
import datetime
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy import signal
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

def fmt_fxaxis(ax,flim=None):
    """
    Format the frequency x-axis of a spectrum plot.
    """

    if flim is None:
        T_lim_1 = datetime.timedelta(minutes=45)
        flim    = (None,1./T_lim_1.total_seconds())

    ax.set_xlim(flim)
    xtks    = ax.get_xticks()
    xtls    = []
    for etn,xtk in enumerate(xtks):
        if xtk == 0:
            T_lbl   = 'Inf'
            f_lbl   = '{:g}'.format(xtk)
        elif etn == len(xtks)-1:
            T_lbl   = 'T [min]'
            f_lbl   = 'f [mHz]'
        else:
            T_sec   = 1./xtk
            T_lbl   = '{:0.0f}'.format(T_sec/60.)
            f_lbl   = '{:g}'.format(xtk*1e3)
        
        xtl = '{!s}\n{!s}'.format(T_lbl,f_lbl)
        xtls.append(xtl)

    ax.set_xticks(xtks)
    ax.set_xticklabels(xtls)

def plot_filter_response(sos,fs,Wn=None,
                         db_lim=(-40,1),flim=None,figsize=(18,8),
                         worN=4096,plot_phase=False,
						plt_fname='filter.png'):
    """
    Plots the magnitude and phase response of a filter.
    
    sos:    second-order sections ('sos') array
    fs:     sample rate
    Wn:     cutoff frequency(ies)
    db_lim: ylimits of magnitude response plot
    flim:   frequency limits of plots
    """
    if Wn is not None:
        # Make sure Wn is an iterable.
        Wn = np.array(Wn)
        if Wn.shape == ():
            Wn.shape = (1,)
    
    f, h    = signal.sosfreqz(sos, worN=worN, fs=fs)
    
    fig     = plt.figure(figsize=figsize)
    ax      = fig.add_subplot(211)
    plt.plot(f, 20 * np.log10(abs(h)))
    # plt.xscale('log')
    plt.title('Filter Frequency Response')
#    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.grid(which='both', axis='both')
    if Wn is not None:
        for cf in Wn:
            plt.axvline(cf, color='green') # cutoff frequency
    plt.ylim(db_lim)

    fmt_fxaxis(ax)

    # plt.ylim(-6,0)
    if plot_phase:
        plt.subplot(212)
        plt.plot(f, np.unwrap(np.angle(h)))
        # plt.xscale('log')
        plt.title('Filter Phase Response')
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Phase [rad]')
        plt.grid(which='both', axis='both')
        if Wn is not None:
            for cf in Wn:
                plt.axvline(cf, color='green') # cutoff frequency
        plt.xlim(flim)

    plt.tight_layout()
    plt.savefig(plt_fname,bbox_inches='tight')


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

def fmt_xaxis(ax,xlim=None,label=True):
    ax.xaxis.set_major_locator(mpl.dates.HourLocator(interval=1))
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H%M"))
    ax.set_xlabel('Time [UTC]')
    ax.set_xlim(xlim)


def psd_series(series):
    """
    Calculate the one-sided power spectral density for a pandas series.
    """
    Ts_ns       = float(np.mean(np.diff(series.index)))
    Ts          = datetime.timedelta(seconds=(Ts_ns*1e-9))
    psd         = np.abs(np.fft.fftshift(np.fft.fft(series)*Ts.total_seconds()*2))**2
    ff          = np.fft.fftshift(np.fft.fftfreq(len(series),Ts.total_seconds()))

    tf          = ff >= 0
    psd         = psd[tf]
    ff          = ff[tf]
    psd_series  = pd.Series(psd,index=ff,name=series.name)
    return psd_series

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
        arr_times   = [date + x for x in pd.to_timedelta(arr.coords['time'])]
        Ts          = np.mean(np.diff(arr_times))

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
            index=arr_times,
            columns=qs,
        ).reset_index(
            names='Time',
        )

        if thresh is None:
            edge_line = pd.DataFrame(
                min_line, 
                index=arr_times,
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

        edge_0  = pd.Series(min_line.squeeze(), index=arr_times, name=date)
        final_edge_list.append(edge_0)

        # X-Limits for plotting
        x_0     = date + datetime.timedelta(hours=12)
        x_1     = date + datetime.timedelta(hours=24)
        xlim    = (x_0, x_1)

        # Window Limits for FFT analysis.
        win_0   = date + datetime.timedelta(hours=14)
        win_1   = date + datetime.timedelta(hours=22)
        winlim  = (win_0, win_1)

        # Select data in analysis window.
        tf      = np.logical_and(edge_0.index >= win_0, edge_0.index < win_1)
        edge_1  = edge_0[tf]

        # Detrend and Hanning Window Signal
        xx      = np.arange(len(edge_1))
        coefs   = poly.polyfit(xx, edge_1, 1)
        ffit    = poly.polyval(xx, coefs)

        hann    = np.hanning(len(edge_1))
        edge_2  = (edge_1 - ffit) * hann

        # Zero-pad and ensure signal is regularly sampled.
        times_xlim  = [xlim[0]]
        while times_xlim[-1] < xlim[1]:
            times_xlim.append(times_xlim[-1] + Ts)

        x_interp    = [x.value for x in times_xlim]
        xp_interp   = [x.value for x in edge_2.index]
        interp      = np.interp(x_interp,xp_interp,edge_2.values)
        edge_3      = pd.Series(interp,index=times_xlim,name=date)
        
        edge_3_psd  = psd_series(edge_3)

        # Design and apply band-pass filter.
        btype   = 'band'
        bp_T0   = datetime.timedelta(hours=1)
        bp_T1   = datetime.timedelta(hours=3)
        bp_dt   = datetime.timedelta(minutes=15)

        # Band Pass Edge Periods
        wp_td   = [bp_T1, bp_T0]
        # Band Stop Edge Periods
        ws_td   = [bp_T1-bp_dt, bp_T0+bp_dt]

        gpass =  3 # The maximum loss in the passband (dB).
        gstop = 40 # The minimum attenuation in the stopband (dB).

        fs      = 1./Ts.total_seconds()
        ws      = [1./x.total_seconds() for x in ws_td]
        wp      = [1./x.total_seconds() for x in wp_td]
        N_filt, Wn = signal.buttord(wp, ws, gpass, gstop, fs=fs)
        sos     = signal.butter(N_filt, Wn, btype, fs=fs, output='sos')

        filter_fpath = os.path.join(plt_save_path,'filter.png')
        plot_filter_response(sos,fs,Wn,plt_fname=filter_fpath)

        edge_4      = edge_3.copy()
        edge_4[:]   = signal.sosfiltfilt(sos,edge_3)
        edge_4_psd  = psd_series(edge_4)

        if plot:
            nCols   = 1
            nRows   = 4
            axInx   = 0
            figsize = (16,nRows*6)

            fig     = plt.figure(figsize=figsize)
            # Plot Heatmap #########################
            axInx   = axInx + 1
            ax      = fig.add_subplot(nRows,nCols,axInx)

            ax.set_title(f'| {date} |')
            ax.pcolormesh(arr_times,ranges_km,arr,cmap='plasma')

            for col in data.columns:
                if col == 'Time':
                    continue
                lbl = '{!s}'.format(col)
                ax.plot(arr_times,data[col],label=lbl)

            ax.plot(arr_times,min_line,lw=2,label='Final Edge')

            for wl in winlim:
                ax.axvline(wl,color='0.8',ls='--',lw=2)

            ax.legend(loc='lower right',fontsize='small',ncols=4)
            fmt_xaxis(ax,xlim)

            ax.set_ylabel('Range [km]')
            ax.set_ylim(0,3000)

            # Plot Processed Edge
            axInx   = axInx + 1
            ax      = fig.add_subplot(nRows,nCols,axInx)

            xx          = edge_4.index
            ed4_line    = ax.plot(xx,edge_4,label='Filtered')

            xx          = edge_3.index
            ed3_line    = ax.plot(xx,edge_3,label='Zero-Padded')

            xx          = edge_2.index
            ed2_line    = ax.plot(xx,edge_2,label='Hanning Window Detrended')

            ax.set_ylabel('Range [km]')
            
            ax.legend(loc='lower right',fontsize='small')

            fmt_xaxis(ax,xlim)

            # Plot spectra
            axInx   = axInx + 1
            ax      = fig.add_subplot(nRows,nCols,axInx)
            xx      = edge_3_psd.index
            color   = ed3_line[0].get_color()
            ax.plot(xx,edge_3_psd,label='Unfiltered',color=color)
            ax.set_title('Unfiltered Spectra')
            fmt_fxaxis(ax)

            axInx   = axInx + 1
            ax      = fig.add_subplot(nRows,nCols,axInx)
            xx      = edge_4_psd.index
            color   = ed4_line[0].get_color()
            ax.plot(xx,edge_4_psd,label='Filtered',color=color)
            ax.set_title('Filtered Spectra')
            fmt_fxaxis(ax)


            fig.tight_layout()
            if save_plt is not None:
                save_plt(date)
            
            processed_dates.append(date)
            plt.close()

    final_edge_df = pd.concat(final_edge_list, axis=1)
    if csv_save_path:
        final_edge_df.to_csv(csv_save_path)
    
    return

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