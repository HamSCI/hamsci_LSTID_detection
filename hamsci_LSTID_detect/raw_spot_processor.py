#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import logging
import os
import datetime
import sys

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

    DATASETS = ['PSK', 'RBN', 'WSPR']
#    DATASETS = ['RBN']

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

# Example usage
if __name__ == "__main__":
    start_date = datetime.datetime(2018,11, 1)
    end_date = datetime.datetime(2019, 5, 1)
    config = {
        'dist_km': 3000,
    }
    # custom_coords = {
    #     'min_lat': 30,
    #     'max_lat': 40,
    #     'min_lon': -120,
    #     'max_lon': -100,
    # }
    # custom_freq = {
    #     'freq_low': 8000000,
    #     'freq_high': 9000000,
    # }
    processor = RawSpotProcessor(
        start_date=start_date,
        end_date=end_date,
        input_dir='raw_data',
        output_dir='data_files',
        region='NA',  # Optional: Use predefined region coordinates
        freq_str='14 MHz',  # Optional: Use predefined frequency range
        config=config,
        csv_gen=True,
        hist_gen=True,
        geo_gen=False
    )
    processor.run_analysis()

    #spots_{date_str}_T0_B20_RBN_WSPR_PSK__{region_str}.csv
