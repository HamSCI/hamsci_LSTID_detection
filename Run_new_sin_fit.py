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
from threshold_edge_detection import lowess_smooth, measure_thresholds

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

def run_edge_detect(
    date,
    x_trim=.08333,
    y_trim=.08,
    sigma=4.2, # 3.8 was good # Gaussian filter kernel
    qs=[.4, .5, .6],
    occurence_n = 60,
    i_max=30,
    thresh=None,
    plot_filter_path=None,
    cache_dir='cache'):
    """
    """

    date_str    = date.strftime('%Y%m%d')
    pkl_fname   = f'{date_str}_edgeDetect.pkl'
    pkl_fpath   = os.path.join(cache_dir,pkl_fname)

    if os.path.exists(pkl_fpath):
        print('   LOADING: {!s}'.format(pkl_fpath))
        with open(pkl_fpath,'rb') as fl:
            result = pickle.load(fl)
    else:
        arr = date_iter.get_date(date,raise_missing=False)

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
        edge_0  = edge_0.interpolate()
        edge_0  = edge_0.fillna(0.)

        # X-Limits for plotting
        x_0     = date + datetime.timedelta(hours=12)
        x_1     = date + datetime.timedelta(hours=24)
        xlim    = (x_0, x_1)

        # Window Limits for FFT analysis.
        win_0   = date + datetime.timedelta(hours=13)
        win_1   = date + datetime.timedelta(hours=23)
        winlim  = (win_0, win_1)

        # Select data in analysis window.
        tf      = np.logical_and(edge_0.index >= win_0, edge_0.index < win_1)
        edge_1  = edge_0[tf]

        times_interp  = [x_0]
        while times_interp[-1] < x_1:
            times_interp.append(times_interp[-1] + Ts)

        x_interp    = [pd.Timestamp(x).value for x in times_interp]
        xp_interp   = [pd.Timestamp(x).value for x in edge_1.index]
        interp      = np.interp(x_interp,xp_interp,edge_1.values)
        edge_1      = pd.Series(interp,index=times_interp,name=date)
        
        sg_edge     = edge_1.copy()
        
        sg_win      = datetime.timedelta(hours=4)
        sg_win_N    = int(sg_win.total_seconds()/Ts.total_seconds())
        sg_edge[:]  = signal.savgol_filter(edge_1,sg_win_N,4)

        tf = np.logical_and(sg_edge.index >= winlim[0], sg_edge.index < winlim[1])
#        sg_edge[tf]  = sg_edge[tf]*np.hanning(np.sum(tf))
        sg_edge[~tf] = 0

        # Sinusoid Fitting
        tt_sec = np.array([x.total_seconds() for x in (sg_edge.index - sg_edge.index.min())])
        data   = sg_edge.values

        # Window Limits for FFT analysis.
        fitWin_0   = date + datetime.timedelta(hours=15)
        fitWin_1   = date + datetime.timedelta(hours=22,minutes=30)
        fitWinLim  = (fitWin_0, fitWin_1)

        tf          = np.logical_and(sg_edge.index >= fitWin_0, sg_edge.index < fitWin_1)
        fit_times   = sg_edge.index[tf].copy()
        tt_sec      = tt_sec[tf]
        data        = data[tf]

        guess_freq      = 1./datetime.timedelta(hours=3).total_seconds()
        guess_amplitude = np.ptp(sg_edge)
        guess_phase     = 0
        guess_offset    = np.mean(sg_edge)

        p0=[guess_freq, guess_amplitude, guess_phase, guess_offset]

        # create the function we want to fit
        def my_sin(tt_sec, freq, amplitude, phase, offset):
            return np.sin( (2*np.pi*tt_sec*freq )+ phase ) * amplitude + offset

        # now do the fit
        sinFit = curve_fit(my_sin, tt_sec, data, p0=p0)
        sinFit_T_hr     = (1./sinFit[0][0]) / 3600.
        sinFit_amp      = sinFit[0][1]
        sinFit_phase    = sinFit[0][2]
        sinFit_offset   = sinFit[0][3]

        # we'll use this to plot our first estimate. This might already be good enough for you
        data_first_guess = my_sin(tt_sec, *p0)

        # recreate the fitted curve using the optimized parameters
        sin_fit = my_sin(tt_sec, *sinFit[0])
        sin_fit = pd.Series(sin_fit,index=fit_times)

        # Package SpotArray into XArray
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
        result['003_sgEdge']        = sg_edge
        result['sin_fit']           = sin_fit
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
        result['sinFit_T_hr']   = sinFit_T_hr
        result['sinFit_amp']    = sinFit_amp 
        result['sinFit_phase']  = sinFit_phase
        result['sinFit_offset'] = sinFit_offset

        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        with open(pkl_fpath,'wb') as fl:
            print('   PICKLING: {!s}'.format(pkl_fpath))
            pickle.dump(result,fl)

    return result

def curve_combo_plot(result_dct,cb_pad=0.04,
                     output_dir=os.path.join('output','daily_plots')):
                     
    """
    Make a curve combo stackplot that includes:
        1. Heatmap of Ham Radio Spots
        2. Raw Detected Edge
        3. Filtered, Windowed Edge
        4. Spectra of Edges

    Input:
        result_dct: Dictionary of results produced by run_edge_detect().
    """
    md          = result_dct.get('metaData')
    date        = md.get('date')
    xlim        = md.get('xlim')
    winlim      = md.get('winlim')

    arr         = result_dct.get('spotArr')
    med_lines   = result_dct.get('med_lines')
    edge_0      = result_dct.get('000_detectedEdge')
    edge_1      = result_dct.get('001_windowLimits')
    sg_edge     = result_dct.get('003_sgEdge')
    sin_fit     = result_dct.get('sin_fit')

    ranges_km   = arr.coords['ranges_km']
    arr_times   = [pd.Timestamp(x) for x in arr.coords['datetimes'].values]
    Ts          = np.mean(np.diff(arr_times)) # Sampling Period

    nCols   = 1
    nRows   = 1

    axInx   = 0
    figsize = (18,nRows*7)

    fig     = plt.figure(figsize=figsize)
    axs     = []

    # Plot Heatmap #########################
    axInx   = axInx + 1
    ax      = fig.add_subplot(nRows,nCols,axInx)
    axs.append(ax)

    ax.set_title(f'| {date} |')
    mpbl = ax.pcolormesh(arr_times,ranges_km,arr,cmap='plasma')
    plt.colorbar(mpbl,aspect=10,pad=cb_pad)

    ed0_line    = ax.plot(arr_times,edge_0,lw=2,label='Detected Edge')
    sgf_line    = ax.plot(sg_edge.index,sg_edge,lw=2,label='SG Filtered Edge')
    ax.plot(sin_fit.index,sin_fit,label='Sin Fit',color='white',lw=3,ls='--')

    for wl in winlim:
        ax.axvline(wl,color='0.8',ls='--',lw=2)

    ax.legend(loc='lower right',fontsize='x-small',ncols=4)
    fmt_xaxis(ax,xlim)

    ax.set_ylabel('Range [km]')
    ax.set_ylim(250,2750)

#    # Print TID Info
#    axInx   = axInx + 1
#    ax      = fig.add_subplot(nRows,nCols,axInx)
#    ax.grid(False)
#    for xtl in ax.get_xticklabels():
#        xtl.set_visible(False)
#    for ytl in ax.get_yticklabels():
#        ytl.set_visible(False)
#    axs.append(ax)
#
#    if date in df_mlw.index:
#        mlw = df_mlw.loc[date,:]
#    else:
#        mlw = {}
#    #ipdb> mlw
#    #MLW_start_time         13.0
#    #MLW_end_time           23.0
#    #MLW_low_range_km      900.0
#    #MLW_high_range_km    1500.0
#    #MLW_tid_hours          10.0
#    #MLW_range_range       600.0
#    #MLW_cycles              4.0
#    #MLW_period_hr           2.5
#    #MLW_comment            nice
#    #Name: 2018-11-09 00:00:00, dtype: object
#
##        result['sinFit_T_hr']   = sinFit_T_hr
##        result['sinFit_amp']    = sinFit_amp 
##        result['sinFit_phase']  = sinFit_phase
##        result['sinFit_offset'] = sinFit_offset
#
#    txt = []
#    txt.append('               MLW          sinFit       FFT')
#    txt.append('T [hr]:        {:5.1f}      {:5.1f}      {:5.1f}'.format(mlw.get('MLW_period_hr',np.nan),result['sinFit_T_hr'],result['004_filtered_Tmax_hr']))
#    txt.append('Range_Range:   {:5.1f}'.format(mlw.get('MLW_range_range',np.nan)))
#    txt.append('MLW TID Hours: {:5.1f}'.format(mlw.get('MLW_tid_hours',np.nan)))
#    txt.append('MLW Comment:   {!s}'.format(mlw.get('MLW_comment')))
#
#    fontdict = {'weight':'normal','family':'monospace'}
#    ax.text(0.05,0.9,'\n'.join(txt),fontdict=fontdict,va='top')

    fig.tight_layout()

    # Account for colorbars and line up all axes.
    for ax_inx, ax in enumerate(axs):
        if ax_inx == 0:
            continue
        adjust_axes(ax,axs[0])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    date_str    = date.strftime('%Y%m%d')
    png_fname   = f'{date_str}_curveCombo.png'
    png_fpath   = os.path.join(output_dir,png_fname)
    print('   Saving: {!s}'.format(png_fpath))
    fig.savefig(png_fpath,bbox_inches='tight')
    plt.close()
    return

def plot_season_analysis(all_results,output_dir='output'):
    """
    Plot the LSTID analysis for the entire season.
    """

    sDate   = min(all_results.keys())
    eDate   = max(all_results.keys())

    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = sDate.strftime('%Y%m%d')
    png_fname   = '{!s}-{!s}_seasonAnalysis.png'.format(sDate_str,eDate_str)
    png_fpath   = os.path.join(output_dir,png_fname)

    # Create parameter dataframe.
    params = []
    params.append('004_filtered_Tmax_hr')
    params.append('004_filtered_PSDdBmax')
    params.append('004_filtered_intSpect')

    df_lst = []
    df_inx = []
    for date,results in all_results.items():
        if results is None:
            continue

        tmp = {}
        for param in params:
            tmp[param] = results[param]

        df_lst.append(tmp)
        df_inx.append(date)

    df = pd.DataFrame(df_lst,index=df_inx)
    # Plotting #############################
    nCols   = 3
    nRows   = 4

    axInx   = 0
    figsize = (25,nRows*5)

    gs      = mpl.gridspec.GridSpec(nrows=nRows,ncols=nCols)
    fig     = plt.figure(figsize=figsize)

    ax  = fig.add_subplot(gs[0,:2])

    ckey = '004_filtered_intSpect'

    cmap = mpl.cm.cool
    vmin = df['004_filtered_intSpect'].min() 
    vmax = df['004_filtered_intSpect'].max() 
    norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
    pos = list(ax.get_position().bounds)
    pos[0] = 0.675
    pos[1] = pos[1] + pos[3]/2.
    pos[2] = 0.025
#    rect : tuple (left, bottom, width, height)
    cax = fig.add_axes(pos)
    cbl  = mpl.colorbar.ColorbarBase(cax,cmap=cmap,norm=norm)
    cbl.set_label(ckey)
    for date,results in all_results.items():
        if results is None:
            continue
        psd = results.get('004_filtered_psd')
        color   = cmap(norm(results[ckey]))
        ax.plot(psd.index,psd,color=color)
    fmt_fxaxis(ax) 

    # Combine FFT and MLW analysis dataframes.
    dfc = pd.concat([df,df_mlw],axis=1)

    # Compare parameters - List of (df, lstid_mlw) keys to compare.
    cmps = []
    cmps.append( ('004_filtered_Tmax_hr',   'MLW_period_hr') )
    cmps.append( ('004_filtered_PSDdBmax',  'MLW_tid_hours') )
    cmps.append( ('004_filtered_intSpect',  'MLW_tid_hours') )

    for pinx,(key_0,key_1) in enumerate(cmps):
        rinx    = pinx + 1
        ax0     = fig.add_subplot(gs[rinx,:2])

        p0  = dfc[key_0]
        p1  = dfc[key_1]

#        ax0.plot(p0.index,p0,marker='.')
        hndls   = []
        hndl    = ax0.bar(p0.index,p0,width=1,color='blue',align='edge',label='FFT')
        hndls.append(hndl)
        ax0.set_ylabel(key_0)
        ax0.set_xlim(sDate,eDate)

        ax0r    = ax0.twinx()
#        ax0r.plot(p1.index,p1,marker='.')
        hndl    = ax0r.bar(p1.index,p1,width=1,color='green',align='edge',label='MLW',alpha=0.5)
        hndls.append(hndl)
        ax0r.set_ylabel(key_1)

        ax0r.legend(handles=hndls,loc='lower right')

        ax1   = fig.add_subplot(gs[rinx,2])
        ax1.scatter(p0,p1)
        ax1.set_xlabel(key_0)
        ax1.set_ylabel(key_1)

    fig.tight_layout()

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print('   Saving: {!s}'.format(png_fpath))
    fig.savefig(png_fpath,bbox_inches='tight')

if __name__ == '__main__':
    output_dir  = 'output'
    cache_dir   = 'cache'
    clear_cache = True

#    sDate   = datetime.datetime(2018,11,1)
#    eDate   = datetime.datetime(2019,4,30)

    sDate   = datetime.datetime(2018,11,9)
    eDate   = datetime.datetime(2018,11,9)

#    sDate   = datetime.datetime(2018,11,5)
#    eDate   = datetime.datetime(2018,11,5)

    # NO PARAMETERS BELOW THIS LINE ################################################
    if clear_cache and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    if clear_cache and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    tic = datetime.datetime.now()
    dates   = [sDate]
    while dates[-1] < eDate:
        dates.append(dates[-1]+datetime.timedelta(days=1))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Edge Detection ###############################################################
    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = sDate.strftime('%Y%m%d')
    pkl_fname   = '{!s}-{!s}_allResults.pkl'.format(sDate_str,eDate_str)
    pkl_fpath   = os.path.join(cache_dir,pkl_fname)
    if os.path.exists(pkl_fpath):
        with open(pkl_fpath,'rb') as fl:
            print('LOADING: {!s}'.format(pkl_fpath))
            all_results = pickle.load(fl)
    else:    
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

        all_results = {}
        for dinx,date in enumerate(dates):
            print(date)
            if dinx == 0:
                plot_filter_path    = os.path.join(output_dir,'filter.png')
            else:
                plot_filter_path    = None
            result              = run_edge_detect(date,plot_filter_path=plot_filter_path,cache_dir=cache_dir)
            all_results[date] = result
            if result is None: # Missing Data Case
                continue
            curve_combo_plot(result)

        import ipdb; ipdb.set_trace()
        with open(pkl_fpath,'wb') as fl:
            print('PICKLING: {!s}'.format(pkl_fpath))
            pickle.dump(all_results,fl)

    toc = datetime.datetime.now()

    print('Processing and plotting time: {!s}'.format(toc-tic))
    plot_season_analysis(all_results,output_dir=output_dir)

import ipdb; ipdb.set_trace()