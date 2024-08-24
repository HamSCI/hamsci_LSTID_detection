#!/usr/bin/env python
# coding: utf-8
import hamsci_LSTID_detect as LSTID
import ipdb; ipdb.set_trace()

import os
import shutil
import pickle
import numpy as np
import joblib
import datetime
import multiprocessing

from raw_spot_processor import RawSpotProcessor
from data_loading import create_xarr, mad, DateIter

def runRawProcessing(rawProcDict):
    """
    Wrapper function to use RawSpotProcessor() with multiprocessing.
    """
    processor = RawSpotProcessor(**rawProcDict)
    processor.run_analysis()
    return processor

def runEdgeDetectAndPlot(edgeDetectDict):
    """
    Wrapper function for edge detection and plotting to use with
    multiprocessing.
    """
    print('Edge Detection: {!s}'.format(edgeDetectDict['date']))

    result  = run_edge_detect(**edgeDetectDict)
    if result is None: # Missing Data Case
       return 
    
    auto_crit = edgeDetectDict.get('auto_crit',False)
    result = curve_combo_plot(result,auto_crit=auto_crit)
    return result

if __name__ == '__main__':
    parent_dir                = 'data_files'
    data_out_path             = 'processed_data/full_data.joblib'
    lstid_T_hr_lim            = (1, 4.5)
    raw_processing_input_dir  = 'raw_data'
    raw_processing_output_dir = parent_dir
    multiproc                 = True
    output_dir                = 'output'
    cache_dir                 = 'cache'
    clear_cache               = True
    bandpass                  = True
    automatic_lstid           = True
    raw_data_loader           = True

#    sDate   = datetime.datetime(2018,11,1)
#    eDate   = datetime.datetime(2019,4,30)

    sDate   = datetime.datetime(2018,11,1)
    eDate   = datetime.datetime(2018,11,5)

    # NO PARAMETERS BELOW THIS LINE ################################################

    # Determine number of cores for multiprocessing.
    # Leave a couple cores open if >= 4 cores available.
    nprocs  = multiprocessing.cpu_count()
    if nprocs >= 4:
        nprocs = nprocs - 2

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
        for date in dates:
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

        if not multiproc:
            for rawProcDict in rawProcDicts:
                runRawProcessing(rawProcDict)
        else:
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
                apply_fn=mad)
            joblib.dump(full_xarr, data_out_path)

        date_iter = DateIter(data_out_path) #, label_df=label_out_path)

        # Edge Detection, Curve Fitting, and Plotting ##########
        edgeDetectDicts = []
        for date in dates:
            tmp = {}
            tmp['date']         = date
            tmp['cache_dir']    = cache_dir
            tmp['bandpass']     = bandpass
            tmp['auto_crit']    = automatic_lstid
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


    plot_sin_fit_analysis(all_results,output_dir=output_dir)
    plot_season_analysis(all_results,output_dir=output_dir)

    toc = datetime.datetime.now()
    print('Processing and plotting time: {!s}'.format(toc-tic))
