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
mpl.use('Agg')
import matplotlib.pyplot as plt
from operator import itemgetter
import multiprocessing

import string
letters = string.ascii_lowercase

from raw_spot_processor import RawSpotProcessor
from ham_spot_plot import HamSpotPlot 

from scipy import signal
from scipy.signal import stft
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from data_loading import create_xarr, mad, create_label_df
from utils import DateIter
from threshold_edge_detection import lowess_smooth, measure_thresholds

import Figure_3_stackplot as fig3
import sme_plot
import merra2CipsAirsTimeSeries
sme     = sme_plot.SME_PLOT()

parent_dir      = 'data_files'
data_out_path   = 'processed_data/full_data.joblib'
lstid_T_hr_lim  = (1, 4.5)

def mpl_style():
    plt.rcParams['font.size']           = 18
    plt.rcParams['font.weight']         = 'bold'
    plt.rcParams['axes.titleweight']    = 'bold'
    plt.rcParams['axes.labelweight']    = 'bold'
    plt.rcParams['axes.xmargin']        = 0
    plt.rcParams['axes.titlesize']      = 'x-large'
mpl_style()

class SuperDARNMSTID(object):
    def __init__(self,param='meanSubIntSpect_by_rtiCnt',radars=None,
                 season = '20181101_20190501',
                output_dir='output',mstid_data_dir=None,
                write_csvs=True,calculate_reduced=False,cache_file=None):

        if mstid_data_dir is None:
            mstid_data_dir = os.path.join('data','mongo_out','mstid_GSMR_fitexfilter_rtiThresh-0.25','guc')

        if radars is None:
            radars  = []
            # 'High Latitude Radars'
            radars.append('pgr')
            radars.append('sas')
            radars.append('kap')
            radars.append('gbr')
            # 'Mid Latitude Radars'
            radars.append('cvw')
            radars.append('cve')
            radars.append('fhw')
            radars.append('fhe')
            radars.append('bks')
            radars.append('wal')

        # Load SuperDARN MSTID Index Data
        if (cache_file is None) or (not os.path.exists(cache_file)):
            po = fig3.ParameterObject(param,radars=radars,seasons=[season],
                    output_dir=output_dir,default_data_dir=mstid_data_dir,
                    calculate_reduced=calculate_reduced)

        if cache_file is not None:
            if not os.path.exists(cache_file):
                with open(cache_file,'wb') as pkl:
                    pickle.dump(po,pkl)
            else:
                print('Using Cached SD MSTID File: {!s}'.format(cache_file))
                with open(cache_file,'rb') as pkl:
                    po = pickle.load(pkl)

        self.season         = season
        self.param          = param
        self.radars         = radars
        self.mstid_data_dir = mstid_data_dir  
        self.fig3           = fig3
        self.po             = po # ParameterObject

    def plot_ax(self,ax,sDate=None,eDate=None,xlabels=True):
        """
        Plot SuperDARN MSTID Index onto an axis.
        """
        if sDate is None:
            spl = self.season.split('_')[0]
            sDate   = datetime.datetime.strptime(spl,'%Y%m%d')

        if eDate is None:
            spl = self.season.split('_')[1]
            eDate   = datetime.datetime.strptime(spl,'%Y%m%d')

        data_df = self.po.data[self.season]['df']
        ax_info = self.fig3.plot_mstid_values(data_df,ax,
                    radars=self.radars,param=self.param,xlabels=xlabels,
                    sDate=sDate,eDate=eDate,radarBox_fontsize='x-large',
                  radar_ylabels=False)


def load_df_mlw():
    """
    Load Mary Lou West's manual classification results into a datafram.

    Apply an explicit classification rule to MLW's results:
        1. TIDs must exist for more than 0 hours
        2. LSTIDs should have periods between 1 and 5 hours.
    """
    import lstid_ham
    lstid_mlw   = lstid_ham.LSTID_HAM()
    df_mlw      = lstid_mlw.df.copy()
    df_mlw      = df_mlw.set_index('date')
    old_keys    = list(df_mlw.keys())
    new_keys    = {x:'MLW_'+x for x in old_keys}
    df_mlw      = df_mlw.rename(columns=new_keys)

    # Explicitly classify MLW's results as LSTID or not.
    crits   = []
    # TIDs must exist for more than 0 hours
    tf_tid_hrs          = df_mlw['MLW_tid_hours'].astype(float).fillna(0) > 0
    crits.append(tf_tid_hrs)

    # Additional criteria beyond duration.
    mlw_lstid_criteria  = {}
    mlw_lstid_criteria['MLW_period_hr'] = lstid_T_hr_lim    

    for key, crit in mlw_lstid_criteria.items():
        result  = np.logical_and(df_mlw[key] >= crit[0], df_mlw[key] < crit[1])
        crits.append(result)

    df_mlw['MLW_is_lstid'] = np.logical_and.reduce(crits)
    return df_mlw,mlw_lstid_criteria

#df_mlw, mlw_lstid_criteria  = load_df_mlw()

def load_df_sql(sDate=datetime.datetime(2018,11,1), 
                eDate=datetime.datetime(2019,4,30),
                data_name='NAF'):
    
    import lstidFitDb
    ldb  = lstidFitDb.LSTIDFitDb()
    
    dates = [sDate]
    while dates[-1] < eDate:
        dates.append(dates[-1]+datetime.timedelta(days=1))
    
    lst = []
    inx = []
    for date in dates:
        p0, in_DB = ldb.get_fit(date)
        if in_DB:
            lst.append(p0)
            inx.append(date)
    
    df = pd.DataFrame(lst,index=inx)
    df['dur_hr'] = (df['eTime'] - df['sTime']).apply(lambda x: x.total_seconds()/3600.)

    # Explicitly classify results as LSTID or not.
    crits   = []
    # TIDs must exist for more than 0 hours
    tf_tid_hrs          = df['dur_hr'].astype(float).fillna(0) > 0
    crits.append(tf_tid_hrs)

    # Additional criteria beyond duration.
    sql_lstid_criteria  = {}
    sql_lstid_criteria['T_hr'] = lstid_T_hr_lim    

    for key, crit in sql_lstid_criteria.items():
        result  = np.logical_and(df[key] >= crit[0], df[key] < crit[1])
        crits.append(result)

    df[f'{data_name}_is_lstid'] = np.logical_and.reduce(crits)
    return df,sql_lstid_criteria

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

def run_edge_detect(
    date,
    x_trim=.08333,
    y_trim=.08,
    sigma=4.2, # 3.8 was good # Gaussian filter kernel
    qs=[.4, .5, .6],
    occurence_n = 60,
    i_max=30,
    thresh=None,
    cache_dir='cache',
    bandpass=True):
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

            # Apply bandpass filter
            lowcut  = 1/(lstid_T_hr_lim[1]*3600) # higher period limit 
            highcut = 1/(lstid_T_hr_lim[0]*3600) # lower period limit
            fs      = 1/60
            order   = 4
            
            filtered_signal  = bandpass_filter(data=data_detrend.values, lowcut=lowcut, highcut=highcut, fs=fs, order=order)
            filtered_detrend = pd.Series(data=filtered_signal, index=data_detrend.index)

            if bandpass == True:
                data_detrend = filtered_detrend
#            data_detrend.to_csv('detrend_for_testing.csv')
#            # Get MLW's initial guess
#            if date in df_mlw.index:
#                mlw = df_mlw.loc[date,:]
#            else:
#                mlw = {}
#            guess_T_hr = mlw.get('MLW_period_hr',3.)

            T_hr_guesses = np.arange(1,4.5,0.5)
            
            all_sin_fits = []
            for T_hr_guess in T_hr_guesses:
                # Curve Fit Sinusoid ################### 
                guess = {}
                guess['T_hr']           = T_hr_guess
                guess['amplitude_km']   = np.ptp(data_detrend)/2.
                guess['phase_hr']       = 0.
                guess['offset_km']      = np.mean(data_detrend)
                guess['slope_kmph']     = 0.

                try:
                    sinFit,pcov,infodict,mesg,ier = curve_fit(sinusoid, tt_sec, data_detrend, p0=list(guess.values()),full_output=True)
                except:
                    continue

                p0_sin_fit = {}
                p0_sin_fit['T_hr']           = sinFit[0]
                p0_sin_fit['amplitude_km']   = np.abs(sinFit[1])
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
                p0_sin_fit['T_hr_guess']    = T_hr_guess

                all_sin_fits.append(p0_sin_fit)
        except:
            all_sin_fits = []

        if len(all_sin_fits) > 0:
            all_sin_fits = sorted(all_sin_fits, key=itemgetter('r2'), reverse=True)
#                for fr in all_sin_fits:
#                    print(fr['r2'],fr['T_hr'])
            # Pick the best fit sinusoid.
            p0_sin_fit                  = all_sin_fits[0]
            p0                          = p0_sin_fit.copy()
            all_sin_fits[0]['selected'] = True
            del p0['r2']
            del p0['T_hr_guess']
            sin_fit     = sinusoid(tt_sec, **p0)
            sin_fit     = pd.Series(sin_fit,index=fit_times)
        else:
            sin_fit     = pd.Series(np.zeros(len(fit_times))*np.nan,index=fit_times)
            p0_sin_fit  = {}

            poly_fit    = sin_fit.copy()
            p0_poly_fit = {}

            data_detrend = sin_fit.copy()

        # Classification
        lstid_criteria = {}
        lstid_criteria['T_hr']          = lstid_T_hr_lim    
        lstid_criteria['amplitude_km']  = (20,2000)
        lstid_criteria['r2']            = (0.35,1.1)
        
        if p0_sin_fit != {}:
            crits   = []
            for key, crit in lstid_criteria.items():
                val     = p0_sin_fit[key]
                result  = np.logical_and(val >= crit[0], val < crit[1])
                crits.append(result)
            p0_sin_fit['is_lstid']  = np.all(crits)

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
        result['all_sin_fits']      = all_sin_fits

        result['metaData']          = meta  = {}
        meta['date']                = date
        meta['x_trim']              = x_trim
        meta['y_trim']              = y_trim
        meta['sigma']               = sigma
        meta['qs']                  = qs
        meta['occurence_n']         = occurence_n
        meta['i_max']               = i_max
        meta['xlim']                = xlim
        meta['winlim']              = winlim
        meta['fitWinLim']           = fitWinLim
        meta['lstid_criteria']      = lstid_criteria

        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        with open(pkl_fpath,'wb') as fl:
            print('   PICKLING: {!s}'.format(pkl_fpath))
            pickle.dump(result,fl)

    return result

def curve_combo_plot(result_dct,cb_pad=0.125,
                     output_dir=os.path.join('output','daily_plots'),
                     auto_crit=None):
    """
    Make a curve combo stackplot that includes:
        1. Heatmap of Ham Radio Spots
        2. Raw Detected Edge
        3. Filtered, Windowed Edge
        4. Spectra of Edges

    Input:
        result_dct: Dictionary of results produced by run_edge_detect().
    """
    md              = result_dct.get('metaData')
    date            = md.get('date')
    xlim            = md.get('xlim')
    winlim          = md.get('winlim')
    fitWinLim       = md.get('fitWinLim')
    lstid_criteria  = md.get('lstid_criteria')

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
    nRows   = 4

    axInx   = 0
    figsize = (18,nRows*6)

    fig     = plt.figure(figsize=figsize)
    axs     = []

    # Plot Heatmap #########################
    for plot_fit in [False, True]:
        axInx   = axInx + 1
        ax      = fig.add_subplot(nRows,nCols,axInx)
        axs.append(ax)

        mpbl = ax.pcolormesh(arr_times,ranges_km,arr,cmap='plasma')
        plt.colorbar(mpbl,aspect=10,pad=cb_pad,label='14 MHz Ham Radio Data')
        if not plot_fit:
            ax.set_title(f'| {date} |')
        else:
            ed0_line    = ax.plot(arr_times,edge_0,lw=2,label='Detected Edge')

            if p0_sin_fit != {}:
                ax.plot(sin_fit.index,sin_fit+poly_fit,label='Sin Fit',color='white',lw=3,ls='--')

            ax2 = ax.twinx()
            ax2.plot(stability.index,stability,lw=2,color='0.5')
            ax2.grid(False)
            ax2.set_ylabel('Edge Coef. of Variation\n(Grey Line)')

            for wl in winlim:
                ax.axvline(wl,color='0.8',ls='--',lw=2)

            for wl in fitWinLim:
                ax.axvline(wl,color='lime',ls='--',lw=2)

            ax.legend(loc='upper center',fontsize='x-small',ncols=4)

        fmt_xaxis(ax,xlim)
        ax.set_ylabel('Range [km]')
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

    fontdict = {'weight':'normal','family':'monospace'}
    
    txt = []
    txt.append('2nd Deg Poly Fit')
    txt.append('(Used for Detrending)')
    for key, val in p0_poly_fit.items():
        if key == 'r2':
            txt.append('{!s}: {:0.2f}'.format(key,val))
        else:
            txt.append('{!s}: {:0.1f}'.format(key,val))
    ax.text(0.01,0.95,'\n'.join(txt),fontdict=fontdict,va='top')

    txt = []
    txt.append('Sinusoid Fit')
    for key, val in p0_sin_fit.items():
        if key == 'r2':
            txt.append('{!s}: {:0.2f}'.format(key,val))
        elif key == 'is_lstid':
            txt.append('{!s}: {!s}'.format(key,val))
        else:
            txt.append('{!s}: {:0.1f}'.format(key,val))
    ax.text(0.30,0.95,'\n'.join(txt),fontdict=fontdict,va='top')

    results = {}
    if auto_crit == True:
        txt = []
        txt.append('Automatic LSTID Classification\nCriteria from Sinusoid Fit')
        for key, val in lstid_criteria.items():
            txt.append('{!s} <= {!s} < {!s}'.format(val[0],key,val[1]))
        ax.text(0.01,0.3,'\n'.join(txt),fontdict=fontdict,va='top',bbox={'facecolor':'none','edgecolor':'black','pad':5})    
            
        results['sin_is_lstid']  = {'msg':'Auto', 'classification': p0_sin_fit.get('is_lstid')}

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

    result_dct['p0_sin_fit'] = p0_sin_fit
    return result_dct

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
    params.append('T_hr')
    params.append('amplitude_km')
    params.append('is_lstid')
    params.append('agree')
    
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

    # Set non-LSTID parameters to NaN
    csv_fname   = '{!s}-{!s}_sinFit.csv'.format(sDate_str,eDate_str)
    csv_fpath   = os.path.join(output_dir,csv_fname)
    df.to_csv(csv_fpath)

def plot_sin_fit_analysis(all_results,
                          T_hr_vmin=0,T_hr_vmax=5,T_hr_cmap='rainbow',
                          output_dir='output'):
    """
    Plot an analysis of the sin fits for the entire season.
    """
    cbar_title_fontdict     = {'weight':'bold','size':42}
    cbar_ytick_fontdict     = {'weight':'bold','size':36}
    xtick_fontdict          = {'weight': 'bold', 'size':mpl.rcParams['ytick.labelsize']}
    ytick_major_fontdict    = {'weight': 'bold', 'size':24}
    ytick_minor_fontdict    = {'weight': 'bold', 'size':24}
    title_fontdict          = {'weight': 'bold', 'size':36}
    ylabel_fontdict         = {'weight': 'bold', 'size':24}
    reduced_legend_fontdict = {'weight': 'bold', 'size':20}

    sDate   = min(all_results.keys())
    eDate   = max(all_results.keys())

    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = sDate.strftime('%Y%m%d')
    png_fname   = '{!s}-{!s}_sinFitAnalysis.png'.format(sDate_str,eDate_str)
    png_fpath   = os.path.join(output_dir,png_fname)

    # Create parameter dataframe.
    params = []
    params.append('T_hr')
    params.append('amplitude_km')
    params.append('phase_hr')
    params.append('offset_km')
    params.append('slope_kmph')
    params.append('r2')
    params.append('T_hr_guess')
    params.append('selected')
#    params.append('is_lstid')

    df_lst = []
    df_inx = []
    for date,results in all_results.items():
        if results is None:
            continue

        all_sin_fits = results.get('all_sin_fits')
        for p0_sin_fit in all_sin_fits:
            tmp = {}
            for param in params:
                if param in ['selected']:
                    tmp[param] = p0_sin_fit.get(param,False)
                else:
                    tmp[param] = p0_sin_fit.get(param,np.nan)

            df_lst.append(tmp)
            df_inx.append(date)

    df          = pd.DataFrame(df_lst,index=df_inx)
    df_sel      = df[df.selected].copy() # Data frame with fits that have been selected as good.

    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = sDate.strftime('%Y%m%d')
    csv_fname   = '{!s}-{!s}_allSinFits.csv'.format(sDate_str,eDate_str)
    csv_fpath   = os.path.join(output_dir,csv_fname)
    df.to_csv(csv_fpath)

    # Plotting #############################
    nrows   = 7
    ncols   = 1
    ax_inx  = 0
    axs     = []

    cbar_info = {} # Keep track of colorbar info in a dictionary to plot at the end after fig.tight_layout() because of issues with cbar placement.

    figsize = (30,nrows*6.5)
    fig     = plt.figure(figsize=figsize)

    # ax with SuperDARN MSTID Index ################################################
    ax_inx  += 1
    ax      = fig.add_subplot(nrows,ncols,ax_inx)
    axs.append(ax)
    sd_cache    = os.path.join(cache_dir,'sd_mstid.pkl')
    sd_mstid = SuperDARNMSTID(cache_file=sd_cache)
    sd_mstid.plot_ax(ax,sDate=sDate,eDate=eDate,xlabels=(ax_inx==nrows))
    ltr = '({!s}) '.format(letters[ax_inx-1])
    ax.set_title(ltr+'North American SuperDARN Daytime MSTID Index',loc='left')
    ax.set_title('')

    # ax with MERRA2 ###############################################################
    ax_inx  += 1
    ax      = fig.add_subplot(nrows,ncols,ax_inx)
    axs.append(ax)
    ax_0            = ax

    prmd = {}
    prmd['scale_0']         = -20
    prmd['scale_1']         = 100
    prmd['levels']          = 11
    prmd['cmap']            = 'jet'

    mca     = merra2CipsAirsTimeSeries.Merra2CipsAirsTS()
    result  = mca.plot_ax(ax,ylabel_fontdict=ylabel_fontdict,plot_cbar=False,**prmd)
    ltr = '({!s}) '.format(letters[ax_inx-1])
    ax.set_title(ltr+'MERRA-2 50\N{DEGREE SIGN} N Lat Zonal Winds + CIPS & AIRS GW Variance',loc='left')
    fig3.my_xticks(sDate,eDate,ax,fmt='%d %b',plot_axvline=False)
    ax.set_xlabel('')

    # December 25, 2018
    line_lw  = 8
    line_color = 'black'
    ax.axvline(datetime.datetime(2018,12,25),lw=line_lw,color=line_color,label='25 Dec 2018')
    ax.axvline(datetime.datetime(2019,1,2),lw=line_lw,color=line_color,label='2 Jan 2019')

    cbar_info[ax_inx] = cbd = {}
    cbd['ax']       = ax
    cbd['label']    = 'MERRA-2 Zonal Wind\n[m/s]'
    cbd['mpbl']     = result['cbar_pcoll']

    # ax with LSTID Amplitude Analysis #############################################
    prmds   = {}
    prmds['amplitude_km'] = prmd = {}
    prmd['title']   = 'Ham Radio TID Amplitude'
    prmd['label']   = 'Amplitude [km]'
    prmd['vmin']    = 10
    prmd['vmax']    = 60

    prmds['T_hr'] = prmd = {}
    prmd['title']   = 'Ham Radio TID Period'
    prmd['label']   = 'Period [hr]'
    prmd['vmin']    = 0
    prmd['vmax']    = 5

    prmds['r2'] = prmd = {}
    prmd['title']   = 'Ham Radio Fit $r^2$'
    prmd['label']   = '$r^2$'
    prmd['vmin']    = 0
    prmd['vmax']    = 1

    for param in ['amplitude_km','T_hr','r2']:
        prmd            = prmds.get(param)
        title           = prmd.get('title',param)
        label           = prmd.get('label',param)

        ax_inx  += 1
        ax              = fig.add_subplot(nrows,ncols,ax_inx)
        axs.append(ax)

        xx              = df_sel.index
        yy_raw          = df_sel[param]
        rolling_days    = 5
        title           = '{!s} ({!s} Day Rolling Mean)'.format(title,rolling_days)
        yy              = df_sel[param].rolling(rolling_days,center=True).mean()

        vmin            = prmd.get('vmin',np.nanmin(yy))
        vmax            = prmd.get('vmax',np.nanmax(yy))

        cmap            = mpl.colormaps.get_cmap(T_hr_cmap)
        norm            = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        mpbl            = mpl.cm.ScalarMappable(norm,cmap)
        color           = mpbl.to_rgba(yy)
        ax.plot(xx,yy_raw,color='0.5',label='Raw Data')
        ax.plot(xx,yy,color='blue',lw=3,label='{!s} Day Rolling Mean'.format(rolling_days))
        ax.scatter(xx,yy,marker='o',c=color)
        ax.legend(loc='upper right',ncols=2)

#        # December 25, 2018
#        line_lw  = 8
#        line_color = 'black'
#        ax.axvline(datetime.datetime(2018,12,25),lw=line_lw,color=line_color,label='25 Dec 2018')
#        ax.axvline(datetime.datetime(2019,1,2),lw=line_lw,color=line_color,label='2 Jan 2019')


        trans           = mpl.transforms.blended_transform_factory( ax.transData, ax.transAxes)
        hndl            = ax.bar(xx,1,width=1,color=color,align='edge',zorder=-1,transform=trans,alpha=0.5)



        cbar_info[ax_inx] = cbd = {}
        cbd['ax']       = ax
        cbd['label']    = label
        cbd['mpbl']     = mpbl

        ax.set_ylabel(label,fontdict=ylabel_fontdict)
        fig3.my_xticks(sDate,eDate,ax,fmt='%d %b')
        ltr = '({!s}) '.format(letters[ax_inx-1])
        ax.set_title(ltr+title, loc='left')

#        ax2 = ax.twinx()
#        tf  = np.logical_and(sme.df.index >= sDate, sme.df.index < eDate)
#        sme_df  = sme.df[tf].copy()
#
#        xx  = sme_df.index
#        yy  = sme_df['SME'].rolling(int(24*60),center=True).mean()
#        ax2.plot(xx,yy,color='0.5')
#        ax2.set_ylabel('SME [nT]')
#        ax2.grid(False)

    ################################################################################
    ax_inx  += 1
    ax      = fig.add_subplot(nrows,ncols,ax_inx)
    axs.append(ax)

    result  = sme.plot_ax(ax,xlim=(sDate,eDate),ylabel_fontdict=ylabel_fontdict)
    fig3.my_xticks(sDate,eDate,ax)
    ltr = '({!s}) '.format(letters[ax_inx-1])
    ax.set_title(ltr+'SuperMAG SME Index',loc='left')
    ax.set_title('')
    ax.set_xlabel('')

    # ax with LSTID T_hr Fitting Analysis ##########################################    
    ax_inx  += 1
    ax      = fig.add_subplot(nrows,ncols,ax_inx)
    axs.append(ax)
    ax_0    = ax


    xx      = df.index
    yy      = df.T_hr
    color   = df.T_hr_guess
    r2      = df.r2.values
    r2[r2 < 0]  = 0
    alpha   = r2
    mpbl    = ax.scatter(xx,yy,c=color,alpha=alpha,marker='o',
                         vmin=T_hr_vmin,vmax=T_hr_vmax,cmap=T_hr_cmap)

    ax.scatter(df_sel.index,df_sel.T_hr,c=df_sel.T_hr_guess,ec='black',
                         marker='o',label='Selected Fit',
                         vmin=T_hr_vmin,vmax=T_hr_vmax,cmap=T_hr_cmap)
    cbar_info[ax_inx] = cbd = {}
    cbd['ax']       = ax
    cbd['label']    = 'T_hr Guess'
    cbd['mpbl']     = mpbl

#    # Plot bars for days that meet LSTID criteria.
#    trans   = mpl.transforms.blended_transform_factory( ax.transData, ax.transAxes)
##    tf      = np.logical_and(df.selected,df.is_lstid)
##    hndl    = ax.bar(df.index[tf],1,width=1,color='0.8',align='edge',
##                     label='Meets LSTID Criteria',alpha=0.5,zorder=-1,transform=trans)
#
#    best_fit_T_hr_min    = 3.
#    tf      = np.logical_and(df.selected,df.T_hr > best_fit_T_hr_min)
#    hndl    = ax.bar(df.index[tf],1,width=1,color='0.8',align='edge',
#                     label='Best Fit T_hr > {!s}'.format(best_fit_T_hr_min),alpha=0.5,zorder=-1,transform=trans)

    ax.legend(loc='upper right')
    ax.set_ylim(0,10)
    ax.set_ylabel('T_hr Fit')
    fig3.my_xticks(sDate,eDate,ax,labels=(ax_inx==nrows))

#    # ax with Plot Information #####################################################
#    ax_inx  += 1
#    ax      = fig.add_subplot(nrows,ncols,ax_inx)
#    axs.append(ax)
#    ax.grid(False)
#    for xtl in ax.get_xticklabels():
#        xtl.set_visible(False)
#    for ytl in ax.get_yticklabels():
#        ytl.set_visible(False)
#    fontdict = {'weight':'normal','family':'monospace'}
#
##    lstid_criteria  = all_results[sDate]['metaData']['lstid_criteria']
##    txt = []
##    txt.append('Automatic LSTID Classification\nCriteria from Sinusoid Fit')
##    for key, val in lstid_criteria.items():
##        txt.append('{!s} <= {!s} < {!s}'.format(val[0],key,val[1]))
##    ax.text(0.05,0.45,'\n'.join(txt),fontdict=fontdict,va='top',bbox={'facecolor':'none','edgecolor':'black','pad':5})

    fig.tight_layout()

#    # Account for colorbars and line up all axes.
#    for ax_inx, ax in enumerate(axs):
#        adjust_axes(ax,ax_0)
    for ax_inx, cbd in cbar_info.items():
        ax_pos      = cbd['ax'].get_position()
                    # [left, bottom,       width, height]
        cbar_pos    = [1.025,  ax_pos.p0[1], 0.02,   ax_pos.height] 
        cax         = fig.add_axes(cbar_pos)
        cbar        = fig.colorbar(cbd['mpbl'],label=cbd['label'],cax=cax)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    print('   Saving: {!s}'.format(png_fpath))
    fig.savefig(png_fpath,bbox_inches='tight')

def runRawProcessing(rawProcDict):
    """
    Wrapper function to use RawSpotProcessor() with multiprocessing.
    """
    processor = RawSpotProcessor(**rawProcDict)
    processor.run_analysis()
    return processor

if __name__ == '__main__':
    raw_processing_input_dir  = 'raw_data'
    raw_processing_output_dir = parent_dir
    raw_processing_multiproc    = True
    output_dir                = 'output'
    cache_dir                 = 'cache'
    clear_cache               = True
    bandpass                  = True
    automatic_lstid           = True
    raw_data_loader           = False
#    compare_lstid             = True
#    agree_compare_lstid       = True
#    #    df_name              = 'MLW'
#    df_name                   = 'NAF'
#    #    df_name              = 'DFS'

#    sDate   = datetime.datetime(2018,11,1)
#    eDate   = datetime.datetime(2019,4,30)

    sDate   = datetime.datetime(2018,11,1)
    eDate   = datetime.datetime(2018,11,5)

    # NO PARAMETERS BELOW THIS LINE ################################################
    if clear_cache and os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    if clear_cache and os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    if clear_cache and os.path.exists('processed_data'):
        shutil.rmtree('processed_data')
    if not os.path.exists('processed_data'):
        os.mkdir('processed_data')

    # Load Raw CSV data and create 2d hist CSV files
    tic = datetime.datetime.now()
    dates   = [sDate]
    while dates[-1] < eDate:
        dates.append(dates[-1]+datetime.timedelta(days=1))

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if raw_data_loader == True:
        if clear_cache and os.path.exists(raw_processing_output_dir):
            shutil.rmtree(raw_processing_output_dir)

        if not os.path.exists(raw_processing_output_dir):
            os.mkdir(raw_processing_output_dir)

        rawProcDicts    = []
        for dinx,date in enumerate(dates):
            tmp = dict(
                start_date=date,
                end_date=date,
                input_dir=raw_processing_input_dir,
                output_dir=raw_processing_output_dir,
                region='NA', 
                freq_str='14 MHz',
                csv_gen=True,
                hist_gen=True,
                geo_gen=False,
                dask=False
            )
            rawProcDicts.append(tmp)

        if not raw_processing_multiproc:
            for rawProcDict in rawProcDicts:
                runRawProcessing(rawProcDict)
        else:
            nprocs  = multiprocessing.cpu_count()
            if nprocs >= 4:
                nprocs = nprocs - 2
            with multiprocessing.Pool(nprocs) as pool:
                pool.map(runRawProcessing,rawProcDicts)
        
    # Edge Detection ###############################################################
    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = eDate.strftime('%Y%m%d')
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
            result              = run_edge_detect(date,cache_dir=cache_dir,bandpass=bandpass)
            if result is None: # Missing Data Case
                continue
            
            result = curve_combo_plot(result,auto_crit=automatic_lstid)
            all_results[date] = result
            
        with open(pkl_fpath,'wb') as fl:
            print('PICKLING: {!s}'.format(pkl_fpath))
            pickle.dump(all_results,fl)


    plot_sin_fit_analysis(all_results,output_dir=output_dir)
    plot_season_analysis(all_results,output_dir=output_dir)

    toc = datetime.datetime.now()
    print('Processing and plotting time: {!s}'.format(toc-tic))
