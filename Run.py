#!/usr/bin/env python
# coding: utf-8
import os
import warnings
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

def fmt_fyaxis(ax,flim=None):
    """
    Format the frequency y-axis of a spectrum plot.
    """

    if flim is None:
        T_lim_1 = datetime.timedelta(minutes=45)
        flim    = (0,1./T_lim_1.total_seconds())

    ax.set_ylim(flim)
    ytks    = ax.get_yticks()
    ytls    = []
    for etn,ytk in enumerate(ytks):
        if ytk == 0:
            T_lbl   = 'Inf'
            f_lbl   = '{:g}'.format(ytk)
        else:
            T_sec   = 1./ytk
            T_lbl   = '{:0.0f}'.format(T_sec/60.)
            f_lbl   = '{:g}'.format(ytk*1e3)
        
        ytls.append(T_lbl)

    ax.set_yticks(ytks)
    ax.set_yticklabels(ytls)

    ax.set_ylabel('Period [min]')

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

    psd         = 10*np.log10(psd)

    tf          = ff >= 0
    psd         = psd[tf]
    ff          = ff[tf]
    psd_series  = pd.Series(psd,index=ff,name=series.name)
    return psd_series

def adjust_axes(ax_0,ax_1):
    """
    Force geospace environment axes to line up with histogram
    axes even though it doesn't have a color bar.
    """
    ax_0_pos    = list(ax_0.get_position().bounds)
    ax_1_pos    = list(ax_1.get_position().bounds)
    ax_0_pos[2] = ax_1_pos[2]
    ax_0.set_position(ax_0_pos)

def curve_combo_plot(result_dct,cb_pad=0.04,
                     plot_specgrams=False,output_dir='output'):
    """
    Make a curve combo stackplot that includes:
        1. Heatmap of Ham Radio Spots
        2. Raw Detected Edge
        3. Filtered, Windowed Edge
        4. Spectra of Edges

    Input:
        result_dct: Dictionary of results produced by run_edge_detect().
        result_dct should have the following structure:
            result  = {}
            result['spotArr']           = spotArr
            result['000_detectedEdge']  = edge_0
            result['001_windowLimits']  = edge_1
            result['002_hanningDetrend']= edge_2
            result['003_zeroPad']       = edge_3
            result['003_zeroPad_PSDdB'] = edge_3_psd
            result['004_filtered']      = edge_4
            result['004_filtered_psd']  = edge_4_psd
            result['metaData']  = meta  = {}
            meta['date']        = date
            meta['x_trim']      = x_trim
            meta['y_trim']      = y_trim
            meta['sigma']       = sigma
            meta['qs']          = qs
            meta['occurence_n'] = occurence_n
            meta['i_max']       = i_max

    """
    md          = result_dct.get('metaData')
    date        = md.get('date')
    xlim        = md.get('xlim')
    winlim      = md.get('winlim')

    arr         = result_dct.get('spotArr')
    med_lines   = result_dct.get('med_lines')
    edge_0      = result_dct.get('000_detectedEdge')
    edge_2      = result_dct.get('002_hanningDetrend')
    edge_3      = result_dct.get('003_zeroPad')
    edge_3_psd  = result_dct.get('003_zeroPad_PSDdB')
    edge_4      = result_dct.get('004_filtered')
    edge_4_psd  = result_dct.get('004_filtered_psd')

    ranges_km   = arr.coords['ranges_km']
    arr_times   = [pd.Timestamp(x) for x in arr.coords['datetimes'].values]
    Ts          = np.mean(np.diff(arr_times)) # Sampling Period

    nCols   = 1
    nRows   = 4
    if plot_specgrams:
        nRows += 2

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

    for col in med_lines.columns:
        if col == 'Time':
            continue
        lbl = '{!s}'.format(col)
        ax.plot(arr_times,med_lines[col],label=lbl)

    ax.plot(arr_times,edge_0,lw=2,label='Final Edge')

    for wl in winlim:
        ax.axvline(wl,color='0.8',ls='--',lw=2)

    ax.legend(loc='lower right',fontsize='small',ncols=4)
    fmt_xaxis(ax,xlim)

    ax.set_ylabel('Range [km]')
    ax.set_ylim(0,3000)

    # Plot Processed Edge
    axInx   = axInx + 1
    ax      = fig.add_subplot(nRows,nCols,axInx)
    axs.append(ax)

    xx          = edge_4.index
    ed4_line    = ax.plot(xx,edge_4,label='Filtered')

    xx          = edge_3.index
    ed3_line    = ax.plot(xx,edge_3,label='Zero-Padded')

    xx          = edge_2.index
    ed2_line    = ax.plot(xx,edge_2,label='Hanning Window Detrended')

    ax.set_ylabel('Range [km]')
    
    ax.legend(loc='lower right',fontsize='small')

    fmt_xaxis(ax,xlim)

    # Plot Unfiltered Spectra
    if plot_specgrams:
        axInx   = axInx + 1
        ax      = fig.add_subplot(nRows,nCols,axInx)
        axs.append(ax)
#            f, t, Sxx = ss.spectrogram(smooth_arr_1, fs, nperseg = 128,noverlap= 64, window=('tukey',0.1) )
#            f_2, t_2, Sxx_2 = ss.spectrogram(smooth_arr_1, fs, nperseg = 512,noverlap= 1, window=('tukey',0.1) )

        nperseg   = 512
        noverlap  = int(0.75*nperseg) # 75% Overlap of Windows

        f, t, Sxx = signal.spectrogram(edge_3, fs,window='hann',
                            nperseg=nperseg,noverlap=noverlap)
        mpbl      = ax.pcolormesh(t, f, 10*np.log10(Sxx))
        plt.colorbar(mpbl,label='PSD [dB]',aspect=10,pad=cb_pad)
        fmt_xaxis(ax,xlim)
        fmt_fyaxis(ax)

    axInx   = axInx + 1
    ax      = fig.add_subplot(nRows,nCols,axInx)
    axs.append(ax)
    xx      = edge_3_psd.index
    color   = ed3_line[0].get_color()
    ax.plot(xx,edge_3_psd,label='Unfiltered',color=color)
    ax.set_title('Unfiltered Spectra')
    fmt_fxaxis(ax)

    # Plot Filtered Spectra
    if plot_specgrams:
        axInx   = axInx + 1
        ax      = fig.add_subplot(nRows,nCols,axInx)
        axs.append(ax)
        nperseg   = 512
        noverlap  = int(0.75*nperseg) # 75% Overlap of Windows
        f, t, Sxx = signal.spectrogram(edge_4, fs,window='hann',
                            nperseg=nperseg,noverlap=noverlap)
        mpbl      = ax.pcolormesh(t, f, 10*np.log10(Sxx))
        plt.colorbar(mpbl,label='PSD [dB]',aspect=10,pad=cb_pad)
        fmt_xaxis(ax,xlim)
        fmt_fyaxis(ax)

    axInx   = axInx + 1
    ax      = fig.add_subplot(nRows,nCols,axInx)
    axs.append(ax)
    xx      = edge_4_psd.index
    color   = ed4_line[0].get_color()
    ax.plot(xx,edge_4_psd,label='Filtered',color=color)
    ax.set_title('Filtered Spectra')
    fmt_fxaxis(ax)

    fig.tight_layout()

    # Account for colorbars and line up all axes.
    for ax_inx, ax in enumerate(axs):
        if ax_inx == 0:
            continue
        adjust_axes(ax,axs[0])

    date_str    = date.strftime('%Y%m%d')
    png_fname   = f'{date_str}_curveCombo.png'
    png_fpath   = os.path.join(output_dir,png_fname)
    fig.savefig(png_fpath,bbox_inches='tight')
    plt.close()

    return


def run_edge_detect(
    date,
    x_trim=.08333,
    y_trim=.08,
    sigma=4.2, # 3.8 was good # Gaussian filter kernel
    qs=[.4, .5, .6],
    occurence_n = 60,
    i_max=30,
    thresh=None,
    plot_filter_path=None):

    arr = date_iter.get_date(date)

    if arr is None:
        warnings.warn(f'Date {date} has no input')
        return
        
    xl_trim, xrt_trim   = x_trim if isinstance(x_trim, (tuple, list)) else (x_trim, x_trim)
    yl_trim, yr_trim    = x_trim if isinstance(y_trim, (tuple, list)) else (y_trim, y_trim)
    xrt, xl = math.floor(xl_trim * arr.shape[0]), math.floor(xrt_trim * arr.shape[0])
    yr, yl  = math.floor(yl_trim * arr.shape[1]), math.floor(yr_trim * arr.shape[1])

    arr = arr[xrt:-xl, yr:-yl]

    ranges_km   = arr.coords['height']
    arr_times   = [date + x for x in pd.to_timedelta(arr.coords['time'])]
    Ts          = np.mean(np.diff(arr_times)) # Sampling Period

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

    med_lines   = pd.DataFrame(
        np.array(med_lines).T,
        index=arr_times,
        columns=qs,
    ).reset_index(names='Time')

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
            med_lines[['Time', thresh[date]]]
            .rename(columns={thresh[date] : 'Height'})
        )
    elif isinstance(thresh, float):
        edge_line = (
            med_lines[['Time', thresh]]
            .rename(columns={thresh : 'Height'})
        )
    else:
        raise ValueError(f'Threshold {thresh} of type {type(thresh)} is invalid')

    edge_0  = pd.Series(min_line.squeeze(), index=arr_times, name=date)

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

    x_interp    = [pd.Timestamp(x).value for x in times_xlim]
    xp_interp   = [pd.Timestamp(x).value for x in edge_2.index]
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
    
    if plot_filter_path:
        plot_filter_response(sos,fs,Wn,plt_fname=plot_filter_path)

    edge_4      = edge_3.copy()
    edge_4[:]   = signal.sosfiltfilt(sos,edge_3)
    edge_4_psd  = psd_series(edge_4)

    daDct               = {}
    daDct['data']       = arr
    daDct['coords']     = coords = {}
    coords['ranges_km'] = ranges_km.values
    coords['datetimes'] = arr_times
    spotArr             = xr.DataArray(**daDct)

    # Set things up for data file.
    result  = {}
    result['spotArr']           = spotArr
    result['med_lines']         = med_lines
    result['000_detectedEdge']  = edge_0
    result['001_windowLimits']  = edge_1
    result['002_hanningDetrend']= edge_2
    result['003_zeroPad']       = edge_3
    result['003_zeroPad_PSDdB'] = edge_3_psd
    result['004_filtered']      = edge_4
    result['004_filtered_psd']  = edge_4_psd
    result['metaData']  = meta  = {}
    meta['date']        = date
    meta['x_trim']      = x_trim
    meta['y_trim']      = y_trim
    meta['sigma']       = sigma
    meta['qs']          = qs
    meta['occurence_n'] = occurence_n
    meta['i_max']       = i_max
    meta['xlim']        = xlim
    meta['winlim']      = winlim

    return result

tic = datetime.datetime.now()
date                = datetime.datetime(2018,11,1)
plot_filter_path    = os.path.join('output','filter.png')
result              = run_edge_detect(date,plot_filter_path=plot_filter_path)
curve_combo_plot(result)
toc = datetime.datetime.now()

print('Processing and plotting time: {!s}'.format(toc-tic))

import ipdb; ipdb.set_trace()
