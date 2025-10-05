#!/usr/bin/env python3

import shutil
import dask.dataframe as dd
import polars as pl
import pandas as pd
import numpy as np
from datetime import datetime
import json, pyarrow as pa
import pyarrow.parquet as pq
import logging
from pathlib import Path
from dask.diagnostics import ProgressBar
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_datetime_range_by_day(start_dt: datetime, end_dt: datetime) -> list[tuple[datetime, datetime, str]]:
    """Split a datetime range into daily chunks with start and end datetimes."""
    result = []

    current_start = start_dt
    while current_start.date() < end_dt.date():
        # End of the current day
        current_end = datetime.combine(current_start.date(), datetime.max.time()).replace(microsecond=0)
        date_str = current_start.strftime('%Y-%m-%d')
        result.append((current_start, current_end, date_str))

        # Move to next day
        current_start = datetime.combine(current_start.date() + timedelta(days=1), datetime.min.time())

    # Add final day segment
    date_str = current_start.strftime('%Y-%m-%d')
    result.append((current_start, end_dt, date_str))
    return result

class HDF5PolarsLoader:
    def __init__(self, 
                 *,
                 data_dir: str, 
                 sDate: datetime, 
                 eDate: datetime, 
                 cache_dir: str = "cache", 
                 use_cache: bool = True, 
                 region_bounds: dict | None = None, 
                 freq_range: dict = None, 
                 distance_range: dict = None, 
                 chunk_size: int = 100000, 
                 **_):
        """
        :param data_dir: Directory where the HDF5 files are stored.
        :param date_str: The date in the format 'YYYY-MM-DD' to construct the file name.
        :param cache_dir: Directory to store cached files.
        :param use_cache: Whether to load from cache if available.
        :param region_bounds: Optional geo filter with {'lat_lim': (min,max), 'lon_lim': (min,max)}.
        :param freq_range: Frequency range filter for data, containing 'min_freq' and 'max_freq'.
        :param chunk_size: Chunk size for reading the HDF5 file.
        """
        
        self.data_dir       = Path(data_dir)
        self.sDate          = sDate
        self.eDate          = eDate
        self.cache_dir      = Path(cache_dir)
        self.use_cache      = use_cache
        self.region_bounds  = region_bounds
        self.freq_range     = freq_range
        self.distance_range = distance_range
        self.chunk_size     = chunk_size
        
        # --- basic path/date sanity ---
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        if self.sDate > self.eDate:
            raise ValueError(f"sDate {self.sDate} is after eDate {self.eDate}")

        # --- filter dict shapes (light validation) ---
        if self.region_bounds is not None:
            if not {"lat_lim", "lon_lim"} <= self.region_bounds.keys():
                raise ValueError(f"Invalid region_bounds (need lat_lim/lon_lim): {self.region_bounds}")

        if self.freq_range is not None:
            if not {"min_freq", "max_freq"} <= self.freq_range.keys():
                raise ValueError(f"Invalid freq_range (need min_freq/max_freq): {self.freq_range}")

        if self.distance_range is not None:
            if not {"min_dist", "max_dist"} <= self.distance_range.keys():
                raise ValueError(f"Invalid distance_range (need min_dist/max_dist): {self.distance_range}")

        # Construct dynamic cache path
        if self.region_bounds:
            lat_min, lat_max = self.region_bounds['lat_lim']
            lon_min, lon_max = self.region_bounds['lon_lim']
            region_str = f"lat{lat_min}_{lat_max}_lon{lon_min}_{lon_max}"
        else:
            region_str = "full_region"
        
        if self.freq_range:
            # 'label' is optional; fall back to numeric Hz span
            label = self.freq_range.get("label")
            if label is None:
                label = f"{self.freq_range['min_freq']}-{self.freq_range['max_freq']}Hz"
            freq_str = f"{label}"
        else:
            freq_str = "full_freq_range"
    
        if self.distance_range:
            min_dist = self.distance_range['min_dist']
            max_dist = self.distance_range['max_dist']
            distance_str = f"{min_dist}_{max_dist}km"
        else:
            distance_str = "full_distance_range"
    
        # cache directories
        self.cache_dir_df = self.cache_dir / "dataframes"
        self.cache_dir_hist = self.cache_dir / "heatmaps"
        self.cache_dir_df.mkdir(parents=True, exist_ok=True)
        self.cache_dir_hist.mkdir(parents=True, exist_ok=True)

        # cache file paths
        self.cache_path_df   = self.cache_dir_df   / f"{sDate}_{eDate}_{region_str}_{freq_str}_{distance_str}.parquet"
        self.cache_path_hist = self.cache_dir_hist / f"{sDate}_{eDate}_{region_str}_{freq_str}_{distance_str}.parquet"

        self.df = None
        self.hist    = None
        self.xedges  = None
        self.yedges  = None

        self.log = logging.getLogger(__name__)

    def get_file_path(self, date_str):
        """Construct the file path based on the date string."""
        file_name = f"rsd{date_str}.01.hdf5"
        return self.data_dir / file_name

    def load_data(self):
        """Split, load, and concat hdf5 files in required range of dates."""
        datetime_split = split_datetime_range_by_day(self.sDate, self.eDate)
        all_dfs = []
        for sDate, eDate, date_str in datetime_split:
            file_name = f"rsd{date_str}.01.hdf5"
            file_path = self.data_dir / file_name
            self.log.info(f"Loading data from {sDate} - {eDate}...") 
            self.log.info(f"Loading data from HDF5 file {file_path}...")        

            try:
                # Load only the required columns from the HDF5 file
                dask_df = dd.read_hdf(file_path, key="Data/Table Layout", chunksize=self.chunk_size)
            except (FileNotFoundError, OSError) as e:
                self.log.warning(f"Could not load file: {file_path} - Skipping. ({e})")
                continue
    
            required_columns = [
                'year', 'month', 'day', 'hour', 'min', 'sec',
                'pthlen', 'rxlat', 'rxlon', 'txlat', 'txlon',
                'tfreq', 'latcen', 'loncen', 'ssrc'
            ]
            
            dask_df = dask_df[required_columns]
            
            # Downcast columns to lower precision where appropriate
            dask_df = dask_df.astype({
                'rxlat': 'float32',  # Downcast latitude to float32
                'rxlon': 'float32',  # Downcast longitude to float32
                'txlat': 'float32',  # Downcast latitude to float32
                'txlon': 'float32',  # Downcast longitude to float32
                'latcen': 'float32',
                'loncen': 'float32',
                'ssrc': 'category',
                'tfreq': 'float32',  # Downcast frequency to float32
                'pthlen': 'int32',    # Downcast distance to int32 (as you're only interested in distances under 8000 km)
                'year': 'int16',
                'month': 'int8',
                'day': 'int8',
                'hour': 'int8',
                'min': 'int8',
                'sec': 'int8'
            })
            
            # Apply filters to the data (if any) before proceeding with the rest of the steps
            self.log.info("Applying filters...")

            if sDate and eDate:
                self.log.info(f" → datetime filter: {sDate} to {eDate}")
                dask_df = dask_df.map_partitions(self.apply_datetime_filter, sDate, eDate)

            if self.region_bounds:
                lat_lim, lon_lim = self.region_bounds['lat_lim'], self.region_bounds['lon_lim']
                self.log.info(f" → region filter: lat: ({lat_lim[0]}, {lat_lim[1]}), lon: ({lon_lim[0]}, {lon_lim[1]})")
                dask_df = dask_df.map_partitions(self.apply_region_filter)

            if self.freq_range:
                self.log.info(f" → frequency filter: {self.freq_range['min_freq']}-{self.freq_range['max_freq']} Hz")
                dask_df = dask_df.map_partitions(self.apply_freq_filter)

            if self.distance_range:
                self.log.info(f" → distance filter: {self.distance_range['min_dist']}-{self.distance_range['max_dist']} km")
                dask_df = dask_df.map_partitions(self.apply_distance_filter)
            
            # Continue processing and converting to pandas (or directly to Polars)
            with ProgressBar():
                df = dask_df.compute()
                all_dfs.append(df)
                
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
        else:
            final_df = pd.DataFrame()

        self.log.info(f"Loaded and merged data from {self.sDate} - {self.eDate}...") 
        if not all_dfs:
            raise FileNotFoundError(
                f"No HDF5 files found/readable for {self.sDate} → {self.eDate} in {self.data_dir}"
    )
        return final_df

    def process_data(self):
        """Load the dataset from cache or process the HDF5 file using Dask and convert to polars df."""
        if self.use_cache and self.cache_path_df.exists():
            self.log.info(f"Loading data from cache for {self.sDate} - {self.eDate}...")
            self.df = pl.read_parquet(self.cache_path_df)
            return self.df

        df = self.load_data()
        if df.empty:
            raise RuntimeError(f"No data loaded for {self.sDate} → {self.eDate}.")
        
        # Renaming and processing the dataframe as before
#        df['occurred'] = pd.to_datetime(df['year'] + '-' + df['month'] + '-' + df['day'] + ' ' + df['hour'] + ':' + df['min'] + ':' + df['sec'])
#        df.drop(['year', 'month', 'day', 'hour', 'min', 'sec'], axis=1, inplace=True)
        df['occurred'] = pd.to_datetime({
            'year': df['year'],
            'month': df['month'],
            'day': df['day'],
            'hour': df['hour'],
            'minute': df['min'],
            'second': df['sec'],
        })
        df.drop(['year', 'month', 'day', 'hour', 'min', 'sec'], axis=1, inplace=True)
        
        df = df.rename(columns={"occurred": "date",
                                "pthlen": "dist_Km", 
                                "rxlat": "rx_lat", 
                                "rxlon": "rx_long", 
                                "txlat": "tx_lat", 
                                "txlon": "tx_long",
                                "tfreq": "freq",
                                "ssrc": "source",
                                "latcen": "mid_lat",
                                "loncen": "mid_long"})

        #move to polars df
        df['band'] = df['freq'].apply(self.get_band).astype('int8')
        
        df = df[['date', 'freq', 'band', 'dist_Km', 'source', 'mid_lat', 'mid_long', 'rx_lat', 'tx_lat', 'rx_long', 'tx_long']]
        
        # Convert to Polars
        df_polars = pl.from_pandas(df)
        df_polars = df_polars.with_columns(
            (pl.col("freq") / 1_000_000).round(0).cast(pl.Int32).alias("freq_MHz")
        )
        self.df = df_polars

        self.df.write_parquet(self.cache_path_df, compression='snappy')
        self.log.info(f"Cached data to {self.cache_path_df}...")

        return self.df

    def apply_datetime_filter(self, df, sDate, eDate):
        """Apply the datetime filter to the Dask DataFrame based on start and end datetime."""
        if sDate and eDate:
            # Split start datetime
            sy, smo, sd, sh, smin, ssec = (self.sDate.year, self.sDate.month, self.sDate.day,
                                           self.sDate.hour, self.sDate.minute, self.sDate.second)
            # Split end datetime
            ey, emo, ed, eh, emin, esec = (self.eDate.year, self.eDate.month, self.eDate.day,
                                           self.eDate.hour, self.eDate.minute, self.eDate.second)
    
            # Apply the datetime filter
            df = df[
                (
                    (df['year'] > sy) |
                    ((df['year'] == sy) & (df['month'] > smo)) |
                    ((df['year'] == sy) & (df['month'] == smo) & (df['day'] > sd)) |
                    ((df['year'] == sy) & (df['month'] == smo) & (df['day'] == sd) & (df['hour'] > sh)) |
                    ((df['year'] == sy) & (df['month'] == smo) & (df['day'] == sd) & (df['hour'] == sh) & (df['min'] > smin)) |
                    ((df['year'] == sy) & (df['month'] == smo) & (df['day'] == sd) & (df['hour'] == sh) & (df['min'] == smin) & (df['sec'] >= ssec))
                ) &
                (
                    (df['year'] < ey) |
                    ((df['year'] == ey) & (df['month'] < emo)) |
                    ((df['year'] == ey) & (df['month'] == emo) & (df['day'] < ed)) |
                    ((df['year'] == ey) & (df['month'] == emo) & (df['day'] == ed) & (df['hour'] < eh)) |
                    ((df['year'] == ey) & (df['month'] == emo) & (df['day'] == ed) & (df['hour'] == eh) & (df['min'] < emin)) |
                    ((df['year'] == ey) & (df['month'] == emo) & (df['day'] == ed) & (df['hour'] == eh) & (df['min'] == emin) & (df['sec'] <= esec))
                )]
        return df
    
    def apply_region_filter(self, df):
        """Apply the (optional) region_bounds to midpoint lat/lon columns."""
        if self.region_bounds:
            lat_lim, lon_lim = self.region_bounds['lat_lim'], self.region_bounds['lon_lim']
            df = df[(df['latcen'] >= lat_lim[0]) & (df['latcen'] < lat_lim[1])]
            df = df[(df['loncen'] >= lon_lim[0]) & (df['loncen'] < lon_lim[1])]
        return df

    def apply_freq_filter(self, df):
        """Apply the frequency filter to the Dask DataFrame using the 'tfreq' column."""
        if self.freq_range:
            min_freq, max_freq = self.freq_range['min_freq'], self.freq_range['max_freq']
            df = df[(df['tfreq'] >= min_freq) & (df['tfreq'] <= max_freq)]
        return df

    def apply_distance_filter(self, df):
        """Apply the distance filter to the Dask DataFrame using the 'pthlen' column."""
        if self.distance_range:
            min_dist, max_dist = self.distance_range['min_dist'], self.distance_range['max_dist']
            df = df[(df['pthlen'] >= min_dist) & (df['pthlen'] <= max_dist)]
        return df

    def get_band(self, frequency):
        """Assign a frequency band based on the value."""
        if 137 <= frequency < 2000000:          # 160 meters band (0.137 - 2 MHz)
            return 160
        elif 2000 <= frequency < 4000000:       # 80 meters band (2 - 4 MHz)
            return 80
        elif 4000 <= frequency < 7000000:       # 40 meters band (4 - 7 MHz)
            return 40
        elif 7000 <= frequency < 14000000:      # 20 meters band (7 - 14 MHz)
            return 20
        elif 14000 <= frequency < 21000000:     # 15 meters band (14 - 21 MHz)
            return 15
        elif 21000 <= frequency < 30000000:     # 10 meters band (21 - 30 MHz)
            return 10
        else:
            return 0  
    
    def gen_histogram(self):
        """
        Generates (or loads) a 2D histogram of time vs distance.
        Returns: (hist2d: np.ndarray, meta: dict)
        """
        if self.df is None or (hasattr(self.df, "height") and self.df.height == 0):
            raise RuntimeError("No dataframe loaded; call get_dataframe() before gen_histogram().")
        # ---- fast path: load from cache ----
        if self.use_cache and self.cache_path_hist.exists():
            try:
                self.log.info(f"Loading heatmap from cache for {self.sDate} - {self.eDate}...")
                # read only footer for metadata (does NOT load the table)
                schema = pq.read_schema(self.cache_path_hist)
                meta = {}
                if schema.metadata and b"heatmap_meta" in schema.metadata:
                    meta = json.loads(schema.metadata[b"heatmap_meta"].decode())

                # load the table once for the matrix
                df = pl.read_parquet(self.cache_path_hist)
                cols = sorted([c for c in df.columns if c != "time"], key=lambda x: float(x))
                self.hist = df.select(cols).to_numpy()  # (T, H)
                self.meta = meta
                return self.hist, self.meta
            except Exception as e:
                self.log.warning(f"Failed to read cached heatmap '{self.cache_path_hist}': {e}. Recomputing...")

        # ---- compute and write cache ----
        self.df = self.df.with_columns(
            pl.col("date").dt.truncate("1m").dt.epoch("s").alias("time_numeric")
        )

        if self.distance_range:
            min_dist = int(self.distance_range["min_dist"])
            max_dist = int(self.distance_range["max_dist"])
            distance_bins = np.arange(min_dist, max_dist, 10)  
        else:
            distance_bins = np.arange(0, int(self.df.select(pl.max("dist_Km")).item()), 10)
        start_edge     = int(pd.Timestamp(self.sDate).floor("min").timestamp())
        end_edge_excl  = int(pd.Timestamp(self.eDate).floor("min").timestamp()) + 60
        time_bins = np.arange(start_edge, end_edge_excl, 60)

        self.hist, self.xedges, self.yedges = np.histogram2d(
            self.df.get_column("time_numeric").to_numpy(),
            self.df.get_column("dist_Km").to_numpy(),
            bins=[time_bins, distance_bins]
        )

        hist_df = pl.DataFrame(self.hist, schema=[str(edge) for edge in self.yedges[:-1]]).with_columns(
            pl.Series("time", self.xedges[:-1])
        )

        meta = {
            "version": "v1",
            "sDate": self.sDate.isoformat(),
            "eDate": self.eDate.isoformat(),
            "time_bin_seconds": 60,
            "distance_bin_km": 10,
            "n_time": int(self.hist.shape[0]),
            "n_height": int(self.hist.shape[1]),
            "xedge_start": float(self.xedges[0]),
            "xedge_end": float(self.xedges[-1]),
            "yedge_start_km": float(self.yedges[0]),
            "yedge_end_km": float(self.yedges[-1]),
        }

        tbl = hist_df.to_arrow().replace_schema_metadata({b"heatmap_meta": json.dumps(meta).encode()})
        pq.write_table(tbl, str(self.cache_path_hist), compression="snappy")
        self.log.info(f"Cached heatmap (w/metadata) to {self.cache_path_hist}...")

        self.meta = meta
        return self.hist, self.meta

    def get_dataframe(self):
        """Return the loaded Polars DataFrame."""
        if self.df is None:
            self.process_data()
        return self.df

    def clear_cache(self):
        """Delete all files in the cache directory."""
        if self.cache_dir.exists() and self.cache_dir.is_dir():
            for cache_file in self.cache_dir.iterdir():
                if cache_file.is_file() and cache_file.name.startswith(f"{self.sDate}_{self.eDate}_"):
                    cache_file.unlink()
                    self.log.info(f"Cache file removed: {cache_file}")
        else:
            self.log.info(f"Cache directory not found: {self.cache_dir}")


if __name__ == "__main__":
    # --- simple demo run for a single day ---
    import logging
    from datetime import datetime

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # Date window (inclusive start, inclusive end by second)
    sDate = datetime(2019, 12, 1, 0, 0, 0)
    eDate = datetime(2019, 12, 1, 23, 59, 59)

    # Optional region filter — comment out to use full footprint
    region_bounds = {
        "lat_lim": [24.5, 49.5],   # Continental US example
        "lon_lim": [-125.0, -66.5]
    }
    # region_bounds = None  # ← uncomment this line to disable geographic filtering

    # Frequency & distance filters (explicit numbers the loader understands)
    # Example: “7 MHz band-ish” (6–8 MHz) in Hz:
    freq_range = {
        "min_freq": 6_000_000,
        "max_freq": 8_000_000,
        "label": "7"  # optional; used only for cache naming/logs
    }

    distance_range = {
        "min_dist": 0,     # km
        "max_dist": 3000   # km
    }

    # Paths & cache settings
    data_dir = "data/madrigal"
    cache_dir = "cache"
    use_cache = True

    # Build the loader
    loader = HDF5PolarsLoader(
        data_dir=data_dir,
        sDate=sDate,
        eDate=eDate,
        cache_dir=cache_dir,
        use_cache=use_cache,
        region_bounds=region_bounds,   # None => no geographic filter
        freq_range=freq_range,         # None => no frequency filter
        distance_range=distance_range, # None => no distance filter
        chunk_size=100_000,
    )

    # (Optional) Clear any prior cache files for this window/config
    # loader.clear_cache()

    # Load data and build histogram
    df = loader.get_dataframe()
    hist2d, meta = loader.gen_histogram()

    # Report
    print(f"\nFinished loader demo for {sDate} → {eDate}")
    print(f"Rows: {df.height if isinstance(df, pl.DataFrame) else len(df)}")
    print(f"Histogram shape: {hist2d.shape}")
    print("Meta:", {k: meta[k] for k in ("time_bin_seconds", "distance_bin_km", "n_time", "n_height")})

