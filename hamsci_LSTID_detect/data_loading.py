import os
import sys
import logging
import datetime

import numpy as np
import pandas as pd
import xarray as xr
import dask.dataframe as dd
from dask.diagnostics import ProgressBar

################################################################################
# Diego/Nathaniel's Code for Loadig Raw Spots and ##############################
#   and Creating Preprocessed Heatmaps #########################################
################################################################################

def runRawProcessing(rawProcDict):
    """
    Wrapper function to use RawSpotProcessor() with multiprocessing.
    """
    processor = RawSpotProcessor(**rawProcDict)
    processor.run_analysis()
    return processor

class RawSpotProcessor:
    REGION_COORDINATES = {
        'NA': {'min_lat': 20, 'max_lat': 60, 'min_lon': -160, 'max_lon': -60},
        'US': {'min_lat': 24.396308, 'max_lat': 49.384358, 'min_lon': -125.0, 'max_lon': -66.93457},
        # Add more predefined regions here
    }

    FREQUENCY_RANGES = {
        '14 MHz': {'freq_low': 14000000, 'freq_high': 15000000, 'band': 'B20'},
        '7 MHz': {'freq_low': 7000000, 'freq_high': 8000000, 'band': 'B40'},
        # Add more predefined frequency ranges here
    }

#    DATASETS = ['PSK', 'RBN', 'WSPR']
    DATASETS = ['RBN']

    def __init__(self, start_date, end_date, input_dir, output_dir, 
                 region=None, 
                 custom_coords=None, 
                 freq_str=None, 
                 custom_freq=None, 
                 config=None,
                 hist_gen=False,
                 geo_gen=False,
                 csv_gen=False,
                 dask=True):
        """
        Initializes the DataAnalyzer object with the given settings and configuration.
        """

        self.start_date = start_date
        self.end_date = end_date
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.region = region
        self.custom_coords = custom_coords
        self.freq_str = freq_str
        self.custom_freq = custom_freq
        self.hist_gen = hist_gen
        self.geo_gen = geo_gen
        self.csv_gen = csv_gen
        self.dask    = dask

        # Extract date range for file selection
        self.file_date_range = pd.date_range(start=start_date, end=end_date).strftime('%Y-%m-%d')

        # Default configuration values
        default_config = {
            'min_lat': 20,
            'max_lat': 55,
            'min_lon': -130,
            'max_lon': -60,
            'freq_low': 14000000,
            'freq_high': 15000000,
            'dist_km': 3000
        }

        # Update default config with user-provided config
        if config is None:
            config = {}
        self.config = {**default_config, **config}

        # Apply region, custom coordinates, and frequency settings
        self.apply_region_or_custom_coords()
        self.apply_frequency_settings()

        self.df = None
        self.filtered_df = None

    def find_files_for_date(self):
        """
        Finds the appropriate CSV files for the given date range. Assumes filenames are in the format 'YYYY-MM-DD_<DATASET>.csv'.
        
        Returns:
            list of str: Paths to the CSV files corresponding to the dates.
        """
        files = []
        for date in self.file_date_range:
            for dataset in self.DATASETS:
                file_path = os.path.join(self.input_dir, f'{date}_{dataset}.csv.bz2')
                if os.path.isfile(file_path):
                    files.append(file_path)
        if not files:
            raise FileNotFoundError("No CSV files found for the given date range.")
        return files

    def load_data_dask(self):
        """
        Loads the raw data from the CSV files into a Dask DataFrame. Converts the 'date' column to datetime format.
        """
        file_paths = self.find_files_for_date()
        dfs = []
        column_names = ['date', 'freq', 'dist_km', 'lat', 'long']
        dtype_dict = {'freq': 'float32', 'dist_km': 'float32', 'lat': 'float32', 'long': 'float32'}

        for file_path in file_paths:
            df = dd.read_csv(
                file_path,
                header=None,
                names=column_names,
                dtype=dtype_dict,
                usecols=[0, 11, 22, 23, 24]
            )
            dfs.append(df)
        
        # Concatenate all datasets into one DataFrame
        self.df = dd.concat(dfs)

        # Convert 'date' to datetime
        self.df['date'] = dd.to_datetime(self.df['date'], format='%Y-%m-%d %H:%M:%S')

    def load_data_pd(self):
        """
        Loads raw data from CSV files into a pandas DataFrame and converts the 'date' column to datetime format.
        The CSV files are expected to have a specific structure which is read and processed.

        Assumes that `find_files_for_date` returns a list of file paths.
        """
        file_paths = self.find_files_for_date()
        if not file_paths:
            logging.warning("No files found.")
            return

        dfs = []
        column_names = ['date', 'freq', 'dist_km', 'lat', 'long']
        dtype_dict = {'freq': 'float32', 'dist_km': 'float32', 'lat': 'float32', 'long': 'float32'}

        for file_path in file_paths:
            print('Loading csv: ' + file_path,end=' ')
            try:
                tic = datetime.datetime.now()
                df = pd.read_csv(
                    file_path,
                    header=None,
                    names=column_names,
                    dtype=dtype_dict,
                    usecols=[0, 11, 22, 23, 24]
                )
                dfs.append(df)
                toc = datetime.datetime.now()
                print('(Load time: {!s})'.format(toc-tic))
                logging.info(f"Loaded file: {file_path}")
            except Exception as e:
                logging.error(f"Error loading file {file_path}: {e}")
        
        if dfs:
            # Concatenate all datasets into one DataFrame
            self.df = pd.concat(dfs, ignore_index=True)

            # Convert 'date' to datetime
            self.df['date'] = pd.to_datetime(self.df['date'], format='%Y-%m-%d %H:%M:%S')
            logging.info("Data loaded and date column converted.")
        else:
            logging.warning("No data frames to concatenate.")
        
    def filter_data(self):
        """
        Filters the data based on the geographical region, custom coordinates, frequency range, and distance.
        """
        self.filtered_df = self.df[
            (self.df['lat'] >= self.config['min_lat']) &
            (self.df['lat'] <= self.config['max_lat']) &
            (self.df['long'] >= self.config['min_lon']) &
            (self.df['long'] <= self.config['max_lon']) &
            (self.df['freq'] >= self.config['freq_low']) &
            (self.df['freq'] <= self.config['freq_high']) &
            (self.df['dist_km'] <= self.config['dist_km'])
        ].copy()

    def compute_data_dask(self):
        """
        Computes the filtered data using Dask and prints a message indicating the progress of loading the files.
        """
        sys.stdout.write(f"Loading data from files: {', '.join(self.find_files_for_date())}\n")
        sys.stdout.flush()
        
        with ProgressBar():
            self.data = self.filtered_df.compute()
            
        self.data.set_index('date', inplace=True)

    def compute_data_pd(self):
        
        self.data = self.filtered_df
        self.data.set_index('date', inplace=True)

    def generate_histogram(self):
        """
        Generates a 2D histogram of time versus distance.
        """
        # Convert datetime to second
        self.data.loc[:,'time_numeric'] = self.data.index.astype(np.int64) // 10**9
        
        # Define bins for histogram
        distance_bins = np.arange(0, self.data['dist_km'].max(), 10)
        time_bins = pd.date_range(start=self.data.index.min(), end=self.data.index.max(), freq='min').astype(np.int64) // 10**9
        
        # Compute 2D histogram
        self.hist, self.xedges, self.yedges = np.histogram2d(
            self.data['time_numeric'], self.data['dist_km'], bins=[time_bins, distance_bins]
        )

    def save_histogram(self):
        """
        Saves the histogram data to a CSV file.
        """
        date_str = self.start_date.strftime("%Y-%m-%d")
        region_str = f'_{self.region}' if self.region else ''
        freq_str = self.freq_str.replace(" ", "_") if self.freq_str else 'T0'
        band_str = self.FREQUENCY_RANGES[self.freq_str]['band'] if self.freq_str in self.FREQUENCY_RANGES else 'T0'
        datasets_str = '_'.join(self.DATASETS)
        hist_file_name = os.path.join(self.output_dir, f'spots_{date_str}_T0_B20_RBN_WSPR_PSK__{region_str}.csv')
        hist_df = pd.DataFrame(self.hist)
        hist_df.to_csv(hist_file_name, header=False, index=False)

    def save_geo_data(self):
        """
        Saves the geographical and frequency data to a CSV file.
        """
        date_str = self.start_date.strftime("%Y-%m-%d")
        region_str = f'_{self.region}' if self.region else ''
        freq_str = self.freq_str.replace(" ", "_") if self.freq_str else ''
        datasets_str = '_'.join(self.DATASETS)
        geo_file_name = os.path.join(self.output_dir, f'hamSpot_geo_{date_str}_{datasets_str}{region_str}{freq_str}.csv')
        geo_spot = self.data[['lat', 'long', 'freq', 'dist_km']]
        geo_spot.to_csv(geo_file_name)

    def apply_region_or_custom_coords(self):
        """
        Applies region or custom coordinates to the configuration.
        """
        if self.region:
            if self.region not in self.REGION_COORDINATES:
                raise ValueError(f"Region '{self.region}' not found.")
            coords = self.REGION_COORDINATES[self.region]
            self.config.update(coords)
        if self.custom_coords:
            self.config.update(self.custom_coords)

    def apply_frequency_settings(self):
        """
        Applies frequency settings based on predefined strings or custom frequencies.
        """
        if self.freq_str:
            if self.freq_str not in self.FREQUENCY_RANGES:
                raise ValueError(f"Frequency string '{self.freq_str}' not found.")
            freq_range = self.FREQUENCY_RANGES[self.freq_str]
            self.config.update({'freq_low': freq_range['freq_low'], 'freq_high': freq_range['freq_high']})
        if self.custom_freq:
            self.config.update(self.custom_freq)

    def run_analysis(self):
        """
        Executes the complete data analysis workflow: loading, filtering, processing, generating histograms, and saving data.
        """
        if self.dask:
            self.load_data_dask()
            self.filter_data()
            self.compute_data_dask()
            self.generate_histogram()
        else:
            self.load_data_pd()
            self.filter_data()
            self.compute_data_pd()
            self.generate_histogram()
        if self.csv_gen:
            if self.hist_gen:
                self.save_histogram()
            if self.geo_gen:
                self.save_geo_data()

################################################################################
# Nick Callahan's Code for Loading Preprocessed Heatmaps #######################
################################################################################

class HeatmapDateIter():
    """
    Class to make it easier to access amateur radio timeseries
    heatmaps by date.
    """
    def __init__(self,data_dir='data_files',xarr=None,**kwargs):# apply_fn=None):
        if data_dir is not None:
            xarr = create_xarr(data_dir,**kwargs)

        self.data = xarr
#        self._apply_fn = apply_fn
        return
    
#    @property
#    def apply_fn(self):
#        return self._apply_fn
#    
#    @apply_fn.setter
#    def apply_fn(self, x):
#        self._apply_fn = x
#        return
    
    def get_date(self, date, raise_missing=True):
        date = pd.to_datetime(date)
        try:
            xarr = self.data.sel(date=date)
        except KeyError as ke:
            if raise_missing:
                raise ke
            elif self.label_df is not None:
                return None, None
            else:
                return
        
        if self.apply_fn is not None:
            xarr = self.apply_fn(xarr)

        if self.label_df is not None:
            try:
                label = self.label_df.loc[date,['xmin','xmax']]
            except KeyError:
                label = None
            return xarr, label
        else:
            return xarr
    
    def iter_dates(self, dates, skip_missing=False, **get_kwargs):
        for date in dates:
            if isinstance(date, tuple):
                start_date, end_date = date
                dates = pd.date_range(start=start_date, end=end_date)
                for date, arr in self.iter_dates(dates, skip_missing=skip_missing, **get_kwargs):
                    yield date, arr
            else:
                if skip_missing:
                    try:
                        yield date, self.get_date(date, **get_kwargs)
                    except KeyError:
                        continue
                else:
                    yield date, self.get_date(date, **get_kwargs)
                
    def iter_all(self):
        return self.iter_dates(self.data.indexes['date'])
    
def pad_axis(arr, expected_size, dtype=np.uint8, axis=0):
    shape_mismatch = expected_size - arr.shape[axis]
    left_pad = shape_mismatch // 2
    
    if shape_mismatch > 0:
        right_pad = shape_mismatch - left_pad
        axis_pad = (left_pad, right_pad)
        full_pad = [(0, 0) if i != axis else axis_pad for i in range(arr.ndim)]
        arr = np.pad(arr, tuple(full_pad), mode='constant', constant_values=0)
    elif shape_mismatch < 0:
        left_pad = -shape_mismatch // 2
        right_pad = -shape_mismatch - left_pad
        arr = arr[:,left_pad:-right_pad]
        
    assert arr.dtype == dtype, dtype
    assert arr.shape[axis] == expected_size, f'{arr.shape[axis]} Mismatches Expected {axis} Dimension of {expected_size}'
    return arr
    
def pad_img(img, expected_shape=(1440, 300), dtype=np.uint8):
    """
    Raw input data has inconsistent size, very close but not precisely
    the intended (1440, 300) size. This pads the image to make
    it exactly (1440, 300) but does so evenly on both sides, if required.
    """
    assert len(expected_shape) == img.ndim
    for i in range(img.ndim):
        img = pad_axis(img, expected_shape[i], axis=i, dtype=dtype)
    return img

def cut_half(img, expected_size=1440):
    """ Simple preprocessing for image, could add additional adjustments here """
    if expected_size:
        assert img.shape[0] == expected_size, f'Mismatch with width, dim 0 of {img.shape} != {expected_size}'
        assert not expected_size % 2, 'Width must be even'
    img = img[expected_size // 2:,:]
    return img

def mad(t, min_dev=.05):
    median = np.median(t, axis=(0, 1), keepdims=True)
    abs_devs = np.abs(t - median)
    mad = abs_devs / max(np.median(abs_devs, axis=(0, 1), keepdims=True), min_dev)
    assert t.shape == mad.shape, f'{t.shape} | {mad.shape}'
    return mad

def create_xarr(
    parent_dir='raw_data/', 
    filter_fn=None, 
    max_iter=None, 
    read_pandas=True, 
    expected_shape=(720, 300),
    dtype=(np.uint16, np.float32),
    height_start=0, 
    apply_fn=mad,
    split_idx=1,
):
    in_dtype, out_dtype = dtype if len(dtype) == 2 else (dtype, dtype)
    img_list = list()
    file_list = sorted(os.listdir(parent_dir))
    if filter_fn is None:
        filter_fn = lambda x : x.endswith('.csv')
        
    for i, file in enumerate(filter(filter_fn, file_list)):
        full_path = os.path.join(parent_dir, file)
        if max_iter is not None and i >= max_iter: break
        print(i, end='\r')
        split_file = file.split('_')
        try:
            date = pd.to_datetime(split_file[split_idx].replace('.csv',''))
        except pd.errors.ParserError:
            raise ValueError(f'Split returned invalid date')

        if read_pandas:
            img = pd.read_csv(full_path)
            assert np.all(img >= 0)
            assert np.all(img <= np.iinfo(in_dtype).max)
            img = img.to_numpy(dtype=in_dtype)
        else:
            img = np.genfromtxt(full_path, delimiter=',').astype(in_dtype)

        img = pad_img(img, expected_shape=(expected_shape[0] * 2, expected_shape[1]), dtype=in_dtype) # standardize width
        img = cut_half(img, expected_size=expected_shape[0] * 2) # trim to 12 hours of daytime
        assert img.shape == expected_shape, img.shape
        if apply_fn is not None:
            img = apply_fn(img)

        img_list.append((date.to_pydatetime(), img))
        
    dates, imgs = zip(*img_list)
    times = pd.timedelta_range(start='12:00:00', end='23:59:00', freq='1min')
    heights = np.arange(height_start, 10 * expected_shape[1], 10)

    img_arr = np.stack(imgs, axis=0, dtype=out_dtype)
    assert img_arr.shape[1] == expected_shape[0], f'{img_arr.shape} | {expected_shape}'
    assert img_arr.shape[2] == expected_shape[1], f'{img_arr.shape} | {expected_shape}'
        
    full_xarr = xr.DataArray(
        img_arr,
        coords={
            'date' : list(dates),
            'time' : times,
            'height' : heights,
        },
        dims=['date','time','height'],
    )
    return full_xarr
