import os
import warnings
import pickle
import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd
import xarray as xr
import math
import datetime
from operator import itemgetter

import string

import statsmodels.api as sm
from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

lstid_T_hr_lim  = (1, 4.5)

################################################################################
# Nick's Edge Detection Code ###################################################
################################################################################

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

################################################################################
# Nathaniel and Diego's Sin Fitting Code #######################################
################################################################################

def scale_km(edge,ranges):
    """
    Scale detected edge array indices to kilometers.
    edge:   Edge in array indices.
    ranges: Ground range vector in km of histogram array.
    """
    ranges  = np.array(ranges) 
    edge_km = (edge / len(ranges) * ranges.ptp()) + ranges.min()

    return edge_km

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
    heatmaps    = None,
    x_trim      = .08333,
    y_trim      = .08,
    sigma       = 4.2, # 3.8 was good # Gaussian filter kernel
    qs          = [.4, .5, .6],
    occurence_n = 60,
    i_max       = 30,
    cache_dir   = 'cache',
    bandpass    = True,
    **kwArgs):
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
        arr = heatmaps.get_date(date,raise_missing=False)

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
        tf = np.logical_and(sg_edge.index >= winlim[0], sg_edge.index < winlim[1])
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
