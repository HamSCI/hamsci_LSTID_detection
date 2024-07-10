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

def mpl_style():
    plt.rcParams['font.size']           = 18
    plt.rcParams['font.weight']         = 'bold'
    plt.rcParams['axes.titleweight']    = 'bold'
    plt.rcParams['axes.labelweight']    = 'bold'
    plt.rcParams['axes.xmargin']        = 0
mpl_style()

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

def sinusoid(tt_sec,T_hr,amplitude_km,phase_hr,offset_km,slope_kmph):
    """
    Sinusoid function that will be fit to data.
    """
    phase_rad       = (2.*np.pi) * (phase_hr / T_hr) 
    freq            = 1./(datetime.timedelta(hours=T_hr).total_seconds())
    result          = np.abs(amplitude_km) * np.sin( (2*np.pi*tt_sec*freq ) + phase_rad ) + (slope_kmph/3600.)*tt_sec + offset_km
    return result

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
        
#        sg_win      = datetime.timedelta(hours=4)
#        sg_win_N    = int(sg_win.total_seconds()/Ts.total_seconds())
#        sg_edge[:]  = signal.savgol_filter(edge_1,sg_win_N,4)

        tf = np.logical_and(sg_edge.index >= winlim[0], sg_edge.index < winlim[1])
#        sg_edge[tf]  = sg_edge[tf]*np.hanning(np.sum(tf))
        sg_edge[~tf] = 0

        # Curve Fit Data ############################################################### 

        # Convert Datetime Objects to Relative Seconds and pull out data
        # for fitting.
        t0      = datetime.datetime(date.year,date.month,date.day)
        tt_sec  = np.array([x.total_seconds() for x in (sg_edge.index - t0)])
        data    = sg_edge.values

        # Calculate the rolling Coefficient of Variation and use as a stability parameter
        # to determine the start and end time of good edge detection.
        roll_win    = 15 # 15 minute rolling window
        xx_n = edge_1.rolling(roll_win).std()
        xx_d = edge_1.rolling(roll_win).mean()
        stability   = xx_n/xx_d # Coefficient of Varation

        stab_thresh = 0.05 # Require Coefficient of Variation to be less than 0.05
        tf  = stability < stab_thresh

        # Find 'islands' (aka continuous time windows) that meet the stability criteria
        islands, island_lengths  = islandinfo(tf,1)

        # Get the longest continuous time window meeting the stability criteria.
        isl_inx = np.argmax(island_lengths)
        island  = islands[isl_inx]
        sInx    = island[0]
        eInx    = island[1]

        fitWin_0    = edge_1.index[sInx]
        fitWin_1    = edge_1.index[eInx]
        
        # We know that the edges are very likely to have problems,
        # even if they meet the stability criteria. So, we require
        # the fit boundaries to be at minimum 30 minutes after after
        # and before the start and end times.
        margin = datetime.timedelta(minutes=30)
        if fitWin_0 < (win_0 + margin):
            fitWin_0 = win_0 + margin

        if fitWin_1 > (win_1 - margin):
            fitWin_1 = win_1 - margin

        # Select the data and times to be used for curve fitting.
        fitWinLim   = (fitWin_0, fitWin_1)
        tf          = np.logical_and(sg_edge.index >= fitWin_0, sg_edge.index < fitWin_1)
        fit_times   = sg_edge.index[tf].copy()
        tt_sec      = tt_sec[tf]
        data        = data[tf]

        # now do the fit
        try:
            # Curve Fit 2nd Deg Polynomial #########  
            coefs, [ss_res, rank, singular_values, rcond] = poly.polyfit(tt_sec, data, 2, full = True)
            ss_res_poly_fit = ss_res[0]
            poly_fit = poly.polyval(tt_sec, coefs)
            poly_fit = pd.Series(poly_fit,index=fit_times)

            p0_poly_fit = {}
            for cinx, coef in enumerate(coefs):
                p0_poly_fit[f'c_{cinx}'] = coef

            ss_tot_poly_fit      = np.sum( (data - np.mean(data))**2 )
            r_sqrd_poly_fit      = 1 - (ss_res_poly_fit / ss_tot_poly_fit)
            p0_poly_fit['r2']    = r_sqrd_poly_fit

            # Detrend Data Using 2nd Degree Polynomial
            data_detrend         = data - poly_fit

            # Curve Fit Sinusoid ################### 
            guess = {}
            guess['T_hr']           = 3.
            guess['amplitude_km']   = np.ptp(data_detrend)/2.
            guess['phase_hr']       = 0.
            guess['offset_km']      = np.mean(data_detrend)
            guess['slope_kmph']     = 0.

            sinFit,pcov,infodict,mesg,ier = curve_fit(sinusoid, tt_sec, data_detrend, p0=list(guess.values()),full_output=True)

            p0_sin_fit = {}
            p0_sin_fit['T_hr']           = sinFit[0]
            p0_sin_fit['amplitude_km']   = sinFit[1]
            p0_sin_fit['phase_hr']       = sinFit[2]
            p0_sin_fit['offset_km']      = sinFit[3]
            p0_sin_fit['slope_kmph']     = sinFit[4]

            sin_fit = sinusoid(tt_sec, **p0_sin_fit)
            sin_fit = pd.Series(sin_fit,index=fit_times)

            # Calculate r2 for Sinusoid Fit
            ss_res_sin_fit              = np.sum( (data_detrend - sin_fit)**2)
            ss_tot_sin_fit              = np.sum( (data_detrend - np.mean(data_detrend))**2 )
            r_sqrd_sin_fit              = 1 - (ss_res_sin_fit / ss_tot_sin_fit)
            p0_sin_fit['r2']            = r_sqrd_sin_fit

        except:
            sin_fit     = pd.Series(np.zeros(len(fit_times))*np.nan,index=fit_times)
            p0_sin_fit  = {}

            poly_fit    = sin_fit.copy()
            p0_poly_fit = {}

            data_detrend = sin_fit.copy()

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
        result['p0_sin_fit']        = p0_sin_fit
        result['poly_fit']          = poly_fit
        result['p0_poly_fit']       = p0_poly_fit
        result['stability']         = stability
        result['data_detrend']      = data_detrend
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
        meta['fitWinLim']   = fitWinLim

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
    fitWinLim   = md.get('fitWinLim')

    arr             = result_dct.get('spotArr')
    med_lines       = result_dct.get('med_lines')
    edge_0          = result_dct.get('000_detectedEdge')
    edge_1          = result_dct.get('001_windowLimits')
    sg_edge         = result_dct.get('003_sgEdge')
    sin_fit         = result_dct.get('sin_fit')
    poly_fit        = result_dct.get('poly_fit')
    p0_sin_fit      = result_dct.get('p0_sin_fit')
    p0_poly_fit     = result_dct.get('p0_poly_fit')
    stability       = result_dct.get('stability')
    data_detrend    = result_dct.get('data_detrend')

    ranges_km   = arr.coords['ranges_km']
    arr_times   = [pd.Timestamp(x) for x in arr.coords['datetimes'].values]
    Ts          = np.mean(np.diff(arr_times)) # Sampling Period

    nCols   = 1
    nRows   = 3

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

    if p0_sin_fit != {}:
        ax.plot(sin_fit.index,sin_fit+poly_fit,label='Sin Fit',color='white',lw=3,ls='--')

    ax2 = ax.twinx()
    ax2.plot(stability.index,stability,lw=2,color='0.5')
    ax2.grid(False)

    for wl in winlim:
        ax.axvline(wl,color='0.8',ls='--',lw=2)

    for wl in fitWinLim:
        ax.axvline(wl,color='lime',ls='--',lw=2)

    ax.legend(loc='lower right',fontsize='x-small',ncols=4)
    fmt_xaxis(ax,xlim)

    ax.set_ylabel('Range [km]')
#    ax.set_ylim(250,2750)
    ax.set_ylim(1000,2000)

    # Plot Detrended and fit data. #########
    axInx   = axInx + 1
    ax      = fig.add_subplot(nRows,nCols,axInx)
    axs.append(ax)

    ax.plot(data_detrend.index,data_detrend,label='Detrended Edge')
    ax.plot(sin_fit.index,sin_fit,label='Sin Fit',color='red',lw=3,ls='--')

    for wl in fitWinLim:
        ax.axvline(wl,color='lime',ls='--',lw=2)
    

    ax.set_ylabel('Range [km]')
    fmt_xaxis(ax,xlim)
    ax.legend(loc='lower right',fontsize='x-small',ncols=4)

    # Print TID Info
    axInx   = axInx + 1
    ax      = fig.add_subplot(nRows,nCols,axInx)
    ax.grid(False)
    for xtl in ax.get_xticklabels():
        xtl.set_visible(False)
    for ytl in ax.get_yticklabels():
        ytl.set_visible(False)
    axs.append(ax)

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

    fontdict = {'weight':'normal','family':'monospace'}
    
    txt = []
    txt.append('2nd Deg Poly Fit Parameters')
    txt.append('(Used for Detrending)')
    for key, val in p0_poly_fit.items():
        if key == 'r2':
            txt.append('{!s}: {:0.2f}'.format(key,val))
        else:
            txt.append('{!s}: {:0.1f}'.format(key,val))
    ax.text(0.05,0.9,'\n'.join(txt),fontdict=fontdict,va='top')

    txt = []
    txt.append('Sinusoid Fit Parameters')
    for key, val in p0_sin_fit.items():
        if key == 'r2':
            txt.append('{!s}: {:0.2f}'.format(key,val))
        else:
            txt.append('{!s}: {:0.1f}'.format(key,val))
    ax.text(0.5,0.9,'\n'.join(txt),fontdict=fontdict,va='top')

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

def plot_season_analysis(all_results,output_dir='output',compare_ds = 'NAF'):
    """
    Plot the LSTID analysis for the entire season.
    """

    sDate   = min(all_results.keys())
    eDate   = max(all_results.keys())

    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = sDate.strftime('%Y%m%d')
    png_fname   = '{!s}-{!s}_{!s}_seasonAnalysis.png'.format(sDate_str,eDate_str,compare_ds)
    png_fpath   = os.path.join(output_dir,png_fname)

    # Create parameter dataframe.
    params = []
    params.append('T_hr')
    params.append('amplitude_km')

    df_lst = []
    df_inx = []
    for date,results in all_results.items():
        if results is None:
            continue

        p0_sin_fit = results.get('p0_sin_fit')
        tmp = {}
        for param in params:
            tmp[param] = p0_sin_fit.get(param,np.nan)

        df_lst.append(tmp)
        df_inx.append(date)

    df          = pd.DataFrame(df_lst,index=df_inx)
    # Force amplitudes to be positive.
    df.loc[:,'amplitude_km']    = np.abs(df['amplitude_km'])

    csv_fname   = '{!s}-{!s}_sinFit.csv'.format(sDate_str,eDate_str)
    csv_fpath   = os.path.join(output_dir,csv_fname)
    df.to_csv(csv_fpath)
    
    # Eliminate waves with period > 5 hr.
    tf = df['T_hr'] > 5
    df.loc[tf,'T_hr']           = np.nan
    df.loc[tf,'amplitude_km']   = np.nan

    # Eliminate waves with amplitudes < 15 km.
    tf = df['amplitude_km'] < 15
    df.loc[tf,'T_hr']           = np.nan
    df.loc[tf,'amplitude_km']   = np.nan

    # Plotting #############################
    nCols   = 3
    nRows   = 3

    axInx   = 0
    figsize = (25,nRows*5)

    gs      = mpl.gridspec.GridSpec(nrows=nRows,ncols=nCols)
    fig     = plt.figure(figsize=figsize)

    if compare_ds == 'MLW':
        # Load in Mary Lou West's Manual LSTID Analysis
        import lstid_ham
        mpl_style()

        lstid_mlw   = lstid_ham.LSTID_HAM()
        df_mlw      = lstid_mlw.df.copy()
        df_mlw      = df_mlw.set_index('date')
        old_keys    = list(df_mlw.keys())
        new_keys    = {x:'MLW_'+x for x in old_keys}
        df_mlw      = df_mlw.rename(columns=new_keys)

        # Combine FFT and MLW analysis dataframes.
        dfc = pd.concat([df,df_mlw],axis=1)

        # Eliminate MLW_range_range = 0.
        tf = dfc['MLW_range_range'] <= 0
        dfc.loc[tf,'MLW_range_range']   = np.nan

        # Eliminate MLW_tid_hours = 0.
        tf = dfc['MLW_tid_hours'] <= 0
        dfc.loc[tf,'MLW_tid_hours']   = np.nan

        # Compare parameters - List of (df, lstid_mlw) keys to compare.
        cmps = []
        cmps.append( ('T_hr',          'MLW_period_hr') )
        cmps.append( ('amplitude_km',  'MLW_range_range') )
        cmps.append( ('amplitude_km',  'MLW_tid_hours') )
    elif compare_ds == 'NAF':
        import lstidFitDb
        ldb     = lstidFitDb.LSTIDFitDb()
        df_naf  = ldb.get_data_frame()

        old_keys    = list(df_naf.keys())
        new_keys    = {x:'NAF_'+x for x in old_keys}
        df_naf      = df_naf.rename(columns=new_keys)

        # Combine FFT and MLW analysis dataframes.
        dfc = pd.concat([df,df_naf],axis=1)

        # Compare parameters - List of (df, lstid_mlw) keys to compare.
        cmps = []
        cmps.append( ('T_hr',          'NAF_T_hr') )
        cmps.append( ('amplitude_km',  'NAF_amplitude_km') )
        cmps.append( ('amplitude_km',  'NAF_dur_hr') )

    for pinx,(key_0,key_1) in enumerate(cmps):
        rinx    = pinx
        ax0     = fig.add_subplot(gs[rinx,:2])

        df_hist = dfc[[key_0,key_1]]
        p0  = dfc[key_0]
        p1  = dfc[key_1]
        tf  = np.logical_and(np.isfinite(p0.values.astype(float)),np.isfinite(p1.values.astype(float)))
        pct_overlap = 100 * (np.count_nonzero(tf) / len(dfc))

        hndls   = []
        hndl    = ax0.bar(p0.index,p0,width=1,color='blue',align='edge',label='Sine Fit',alpha=0.5)
        hndls.append(hndl)
        ax0.set_ylabel(key_0)
        ax0.set_xlim(sDate,eDate)

        ax0r    = ax0.twinx()
        hndl    = ax0r.bar(p1.index,p1,width=1,color='green',align='edge',label=compare_ds,alpha=0.5)
        hndls.append(hndl)
        ax0r.set_ylabel(key_1)
        ax0r.legend(handles=hndls,loc='lower right')

        scat_data   = dfc[[key_0,key_1]].dropna().sort_values(key_0)
        p00         = scat_data[key_0].values.astype(float)
        p11         = scat_data[key_1].values.astype(float)
        ax1         = fig.add_subplot(gs[rinx,2])
        ax1.scatter(p00,p11)

        # Curve Fit Line Polynomial #########  
        coefs, [ss_res, rank, singular_values, rcond] = poly.polyfit(p00, p11, 1, full = True)
        ss_res_line_fit = ss_res[0]
        line_fit = poly.polyval(p00, coefs)

        ss_tot_line_fit      = np.sum( (p1 - np.mean(p1))**2 )
        r_sqrd_line_fit      = 1 - (ss_res_line_fit / ss_tot_line_fit)
        txt = []
        txt.append('$N$ Shown = {!s}'.format(len(p00)))
        txt.append('$N$ Dropped = {!s}'.format(len(dfc) - len(p00)))
        txt.append('$r^2$ = {:0.2f}'.format(r_sqrd_line_fit))
        ax1.plot(p00,line_fit,ls='--',label='\n'.join(txt))

        ax1.legend(loc='lower right',fontsize='x-small')
        ax1.set_xlabel(key_0)
        ax1.set_ylabel(key_1)

    fig.tight_layout()

    txt = []
    txt.append('LSTID Automatic Sinusoid Fit Compared with {!s} Manual Fit'.format(compare_ds))
    txt.append('{:0.0f}% Overlap'.format(pct_overlap))
    fig.text(0.5,1.0,'\n'.join(txt),ha='center',fontdict={'weight':'bold','size':'x-large'})

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print('   Saving: {!s}'.format(png_fpath))
    fig.savefig(png_fpath,bbox_inches='tight')

if __name__ == '__main__':
    output_dir  = 'output'
    cache_dir   = 'cache'
    clear_cache = False

    sDate   = datetime.datetime(2018,11,1)
    eDate   = datetime.datetime(2019,4,30)

#    sDate   = datetime.datetime(2018,11,9)
#    eDate   = datetime.datetime(2018,11,9)

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

        with open(pkl_fpath,'wb') as fl:
            print('PICKLING: {!s}'.format(pkl_fpath))
            pickle.dump(all_results,fl)

    toc = datetime.datetime.now()

    print('Processing and plotting time: {!s}'.format(toc-tic))
    for compare_ds in ['MLW','NAF']:
        plot_season_analysis(all_results,output_dir=output_dir,compare_ds=compare_ds)

import ipdb; ipdb.set_trace()
