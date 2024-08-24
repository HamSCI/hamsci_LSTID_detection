#!/usr/bin/env python
# coding: utf-8
import os
import shutil
import datetime
import pickle
import multiprocessing
import numpy as np

import hamsci_LSTID_detect as LSTID

# EDIT PARAMETERS HERE #########################################################
raw_processing_input_dir = 'raw_data'
datasets                = ['PSK','RBN','WSPR']

clear_cache              = True
cache_dir                = 'cache'
heatmap_csv_dir          = os.path.join(cache_dir,'heatmaps')
edge_dir                 = os.path.join(cache_dir,'edge_detect')
output_dir               = 'output'

multiproc                = True # Use multiprocessing
nprocs                   = multiprocessing.cpu_count()
bandpass                 = True

automatic_lstid          = True     # Automatic LSTID Classification
lstid_T_hr_lim           = (1, 4.5) # LSTID Classification Period Criteria

region                   = 'NA' # 'NA' --> North America
freq_str                 = '14 MHz'
sDate                    = datetime.datetime(2018,11,1)
eDate                    = datetime.datetime(2019,4,30)

# NO PARAMETERS BELOW THIS LINE ################################################
def prep_dirs(*dirs,clear_cache=False):
    """
    Prepare output directories:
        1. If clear_cache is True, delete existing directory.
        2. Create directory if it does not exist.

    dirs:   strings of directory names
    """
    for dr in dirs:
        if clear_cache and os.path.exists(dr):
            shutil.rmtree(dr)

    for dr in dirs:
        if not os.path.exists(dr):
            os.makedirs(dr)

def get_dates(sDate,eDate):
    """
    Returns a list of each date from the sDate up to the eDate.
    """
    dates   = [sDate]
    while dates[-1] < eDate:
        dates.append(dates[-1]+datetime.timedelta(days=1))
    
    return dates

def runEdgeDetectAndPlot(edgeDetectDict):
    """
    Wrapper function for edge detection and plotting to use with
    multiprocessing.
    """
    date        = edgeDetectDict['date']
    cache_dir   = edgeDetectDict.get('cache_dir','cache')
    print('Edge Detection: {!s}'.format(date))

    date_str    = date.strftime('%Y%m%d')
    pkl_fname   = f'{date_str}_edgeDetect.pkl'
    pkl_fpath   = os.path.join(cache_dir,pkl_fname)

    if os.path.exists(pkl_fpath):
        print('   LOADING: {!s}'.format(pkl_fpath))
        with open(pkl_fpath,'rb') as fl:
            result = pickle.load(fl)
    else:
        result  = LSTID.edge_detection.run_edge_detect(**edgeDetectDict)

        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        with open(pkl_fpath,'wb') as fl:
            print('   PICKLING: {!s}'.format(pkl_fpath))
            pickle.dump(result,fl)

    if result is None: # Missing Data Case
       return 
    
    auto_crit = edgeDetectDict.get('auto_crit',False)
    result = LSTID.plotting.curve_combo_plot(result,auto_crit=auto_crit)
    return result

tic = datetime.datetime.now()

prep_dirs(cache_dir,heatmap_csv_dir,edge_dir,output_dir,clear_cache=clear_cache)
dates   = get_dates(sDate,eDate)

# Cache All Results to a Pickle File ###########################################
sDate_str   = sDate.strftime('%Y%m%d')
eDate_str   = eDate.strftime('%Y%m%d')
pkl_fname   = '{!s}-{!s}_allResults.pkl'.format(sDate_str,eDate_str)
pkl_fpath   = os.path.join(cache_dir,pkl_fname)
if os.path.exists(pkl_fpath):
    with open(pkl_fpath,'rb') as fl:
        print('LOADING: {!s}'.format(pkl_fpath))
        all_results = pickle.load(fl)
else:    
    ######################################## 
    # Load Raw CSV data and create 2d hist CSV files
    # Generate a list of dictionaries with parameters of each day to be processed.
    rawProcDicts    = [] 
    for date in dates:
        tmp = dict(
            start_date = date,
            end_date   = date,
            input_dir  = raw_processing_input_dir,
            output_dir = heatmap_csv_dir,
            region     = region,
            freq_str   = freq_str,
            datasets   = datasets,
            csv_gen    = True,
            hist_gen   = True,
            geo_gen    = False,
            dask       = False
        )
        rawProcDicts.append(tmp)

    # Process each day of Raw Spots
    if not multiproc: # NO multiprocessing
        for rawProcDict in rawProcDicts:
            LSTID.data_loading.runRawProcessing(rawProcDict)
    else: # YES multiprocessing
        with multiprocessing.Pool(nprocs) as pool:
            pool.map(LSTID.data_loading.runRawProcessing,rawProcDicts)
    
    # Load in CSV Histograms/Heatmaps ###############
    heatmaps    = LSTID.data_loading.HeatmapDateIter(heatmap_csv_dir)

    # Edge Detection, Curve Fitting, and Plotting ##########
    edgeDetectDicts = []
    for date in dates:
        tmp = {}
        tmp['date']           = date
        tmp['cache_dir']      = edge_dir
        tmp['bandpass']       = bandpass
        tmp['auto_crit']      = automatic_lstid
        tmp['heatmaps']       = heatmaps
        tmp['lstid_T_hr_lim'] = lstid_T_hr_lim
        edgeDetectDicts.append(tmp)

    if not multiproc:
        results = []
        for edgeDetectDict in edgeDetectDicts:
            result = runEdgeDetectAndPlot(edgeDetectDict)
            results.append(result)
    else:
        with multiprocessing.Pool(nprocs) as pool:
            results = pool.map(runEdgeDetectAndPlot,edgeDetectDicts)

    all_results = {}
    for date,result in zip(dates,results):
        if result is None: # No data case
            continue
        print(date)
        all_results[date] = result
        
    with open(pkl_fpath,'wb') as fl:
        print('PICKLING: {!s}'.format(pkl_fpath))
        pickle.dump(all_results,fl)

LSTID.plotting.plot_sin_fit_analysis(all_results,output_dir=output_dir)
LSTID.plotting.sin_fit_key_params_to_csv(all_results,output_dir=output_dir)

toc = datetime.datetime.now()
print('Processing and plotting time: {!s}'.format(toc-tic))
