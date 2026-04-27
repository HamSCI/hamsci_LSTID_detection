#!/usr/bin/env python3

import h5py
import json
import logging
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime, timedelta

from scripts.base_loader import BaseSpotLoader
from scripts.utils import split_datetime_range_by_day

REQUIRED_COLUMNS = [
    'year', 'month', 'day', 'hour', 'min', 'sec',
    'pthlen', 'rxlat', 'rxlon', 'txlat', 'txlon',
    'tfreq', 'latcen', 'loncen', 'ssrc',
]

DEFAULT_CHUNK_SIZE = 500_000


class HDF5PolarsLoader(BaseSpotLoader):

    def __init__(self, *,
                 data_dir: str,
                 sDate: datetime,
                 eDate: datetime,
                 cache_dir: str = "cache",
                 use_cache: bool = True,
                 region_bounds: dict | None = None,
                 freq_range: dict | None = None,
                 distance_range: dict | None = None,
                 chunk_size: int = DEFAULT_CHUNK_SIZE,
                 **_):
        self.data_dir        = Path(data_dir)
        self.sDate           = sDate
        self.eDate           = eDate
        self.cache_dir       = Path(cache_dir)
        self.use_cache       = use_cache
        self.region_bounds   = region_bounds
        self.freq_range      = freq_range
        self.distance_range  = distance_range
        self.chunk_size = chunk_size

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if self.sDate > self.eDate:
            raise ValueError(f"sDate {self.sDate} is after eDate {self.eDate}")
        if self.region_bounds is not None:
            if not {"lat_lim", "lon_lim"} <= self.region_bounds.keys():
                raise ValueError(f"region_bounds needs lat_lim/lon_lim: {self.region_bounds}")
        if self.freq_range is not None:
            if not {"min_freq", "max_freq"} <= self.freq_range.keys():
                raise ValueError(f"freq_range needs min_freq/max_freq: {self.freq_range}")
        if self.distance_range is not None:
            if not {"min_dist", "max_dist"} <= self.distance_range.keys():
                raise ValueError(f"distance_range needs min_dist/max_dist: {self.distance_range}")

        # Cache path naming — same convention as before for backward compatibility
        if self.region_bounds:
            lat_min, lat_max = self.region_bounds['lat_lim']
            lon_min, lon_max = self.region_bounds['lon_lim']
            region_str = f"lat{lat_min}_{lat_max}_lon{lon_min}_{lon_max}"
        else:
            region_str = "full_region"

        if self.freq_range:
            label = self.freq_range.get("label")
            freq_str = label if label else f"{self.freq_range['min_freq']}-{self.freq_range['max_freq']}Hz"
        else:
            freq_str = "full_freq_range"

        if self.distance_range:
            distance_str = f"{self.distance_range['min_dist']}_{self.distance_range['max_dist']}km"
        else:
            distance_str = "full_distance_range"

        self.cache_dir_df   = self.cache_dir / "dataframes"
        self.cache_dir_hist = self.cache_dir / "heatmaps"
        self.cache_dir_df.mkdir(parents=True, exist_ok=True)
        self.cache_dir_hist.mkdir(parents=True, exist_ok=True)

        self.cache_path_df   = self.cache_dir_df   / f"{sDate}_{eDate}_{region_str}_{freq_str}_{distance_str}.parquet"
        self.cache_path_hist = self.cache_dir_hist / f"{sDate}_{eDate}_{region_str}_{freq_str}_{distance_str}.parquet"

        self.df   = None
        self.hist = None
        self.meta = None

        self.log = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Internal loading helpers
    # ------------------------------------------------------------------

    def _read_chunk(self, dataset, start: int) -> pl.DataFrame:
        """Read one raw chunk from an h5py dataset and return as Polars DataFrame."""
        raw = dataset[start:start + self.chunk_size]
        data = {}
        for col in REQUIRED_COLUMNS:
            arr = raw[col]
            # Decode fixed-length byte strings (e.g. ssrc stored as b'rbn')
            if arr.dtype.kind == 'S':
                arr = np.char.decode(arr, 'utf-8')
            data[col] = arr
        return pl.DataFrame(data)

    def _apply_filters(self, df: pl.DataFrame, sDate: datetime, eDate: datetime) -> pl.DataFrame:
        """Apply all configured filters to a Polars chunk using raw HDF5 column names."""
        masks = []

        # Datetime: keep rows matching the target day (HDF5 files are per-day;
        # this handles any stray edge rows near midnight)
        masks.append(
            (pl.col('year').cast(pl.Int32) == sDate.year) &
            (pl.col('month').cast(pl.Int32) == sDate.month) &
            (pl.col('day').cast(pl.Int32) == sDate.day)
        )

        if self.region_bounds:
            lat_min, lat_max = self.region_bounds['lat_lim']
            lon_min, lon_max = self.region_bounds['lon_lim']
            masks.append(
                (pl.col('latcen') >= lat_min) & (pl.col('latcen') < lat_max) &
                (pl.col('loncen') >= lon_min) & (pl.col('loncen') < lon_max)
            )

        if self.freq_range:
            masks.append(
                (pl.col('tfreq') >= self.freq_range['min_freq']) &
                (pl.col('tfreq') <= self.freq_range['max_freq'])
            )

        if self.distance_range:
            masks.append(
                (pl.col('pthlen') >= self.distance_range['min_dist']) &
                (pl.col('pthlen') <= self.distance_range['max_dist'])
            )

        if not masks:
            return df

        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m
        return df.filter(combined)

    def _load_one_file(self, file_path: Path, sDate: datetime, eDate: datetime) -> pl.DataFrame:
        """Chunked h5py load + filter for a single HDF5 file. Returns filtered Polars DataFrame."""
        filtered_chunks = []
        with h5py.File(file_path, 'r') as f:
            ds = f['Data/Table Layout']
            n_rows = ds.shape[0]
            for start in range(0, n_rows, self.chunk_size):
                chunk = self._read_chunk(ds, start)
                chunk = self._apply_filters(chunk, sDate, eDate)
                if chunk.height > 0:
                    filtered_chunks.append(chunk)

        if not filtered_chunks:
            return pl.DataFrame()
        return pl.concat(filtered_chunks)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load_data(self) -> pl.DataFrame:
        """Load and filter all HDF5 files covering sDate–eDate. Returns raw filtered DataFrame."""
        day_splits = split_datetime_range_by_day(self.sDate, self.eDate)
        all_dfs = []

        for s_dt, e_dt, date_str in day_splits:
            file_path = self.data_dir / f"rsd{date_str}.01.hdf5"
            if not file_path.exists():
                self.log.warning(f"File not found: {file_path} — skipping")
                continue
            self.log.info(f"Loading {file_path}...")
            df = self._load_one_file(file_path, s_dt, e_dt)
            if df.height > 0:
                all_dfs.append(df)

        if not all_dfs:
            raise FileNotFoundError(
                f"No HDF5 files found/readable for {self.sDate} → {self.eDate} in {self.data_dir}"
            )
        return pl.concat(all_dfs)

    def process_data(self) -> pl.DataFrame:
        """Load from DataFrame cache if available, otherwise load from HDF5 and build pipeline schema."""
        if self.use_cache and self.cache_path_df.exists():
            self.log.info(f"Loading DataFrame from cache for {self.sDate}...")
            self.df = pl.read_parquet(self.cache_path_df)
            return self.df

        raw = self.load_data()

        # Build datetime column from raw integer components
        df = raw.with_columns(
            pl.datetime(
                pl.col('year').cast(pl.Int32),
                pl.col('month').cast(pl.Int32),
                pl.col('day').cast(pl.Int32),
                hour=pl.col('hour').cast(pl.Int32),
                minute=pl.col('min').cast(pl.Int32),
                second=pl.col('sec').cast(pl.Int32),
            ).alias('date')
        ).drop(['year', 'month', 'day', 'hour', 'min', 'sec'])

        # Rename to pipeline schema
        df = df.rename({
            'pthlen':  'dist_Km',
            'rxlat':   'rx_lat',
            'rxlon':   'rx_long',
            'txlat':   'tx_lat',
            'txlon':   'tx_long',
            'tfreq':   'freq',
            'ssrc':    'source',
            'latcen':  'mid_lat',
            'loncen':  'mid_long',
        })

        # Derive band and freq_MHz
        df = df.with_columns([
            pl.when(pl.col('freq') < 2_000_000).then(pl.lit(160))
              .when(pl.col('freq') < 4_000_000).then(pl.lit(80))
              .when(pl.col('freq') < 7_000_000).then(pl.lit(40))
              .when(pl.col('freq') < 14_000_000).then(pl.lit(20))
              .when(pl.col('freq') < 21_000_000).then(pl.lit(15))
              .when(pl.col('freq') < 30_000_000).then(pl.lit(10))
              .otherwise(pl.lit(0)).cast(pl.Int8).alias('band'),
            (pl.col('freq') / 1_000_000).round(0).cast(pl.Int32).alias('freq_MHz'),
        ])

        df = df.select([
            'date', 'freq', 'band', 'dist_Km', 'source',
            'mid_lat', 'mid_long', 'rx_lat', 'tx_lat', 'rx_long', 'tx_long', 'freq_MHz',
        ])

        self.df = df

        if self.use_cache:
            self.df.write_parquet(self.cache_path_df, compression='snappy')
            self.log.info(f"Cached DataFrame to {self.cache_path_df}")

        return self.df

    def get_dataframe(self) -> pl.DataFrame:
        if self.df is None:
            self.process_data()
        return self.df

    def gen_histogram(self):
        """
        Generate (or load) a 2D histogram of time vs distance.
        Returns: (hist2d: np.ndarray, meta: dict)
        meta always includes 'n_spots' (total filtered spot count).
        If histogram cache exists, returns immediately without requiring a loaded DataFrame.
        """
        if self.df is None or self.df.height == 0:
            raise RuntimeError("No dataframe loaded; call get_dataframe() before gen_histogram().")

        # Fast path: load from histogram cache
        if self.use_cache and self.cache_path_hist.exists():
            try:
                self.log.info(f"Loading heatmap from cache for {self.sDate}...")
                schema = pq.read_schema(self.cache_path_hist)
                meta = {}
                if schema.metadata and b"heatmap_meta" in schema.metadata:
                    meta = json.loads(schema.metadata[b"heatmap_meta"].decode())

                df = pl.read_parquet(self.cache_path_hist)
                cols = sorted([c for c in df.columns if c != "time"], key=lambda x: float(x))
                self.hist = df.select(cols).to_numpy()
                self.meta = meta
                return self.hist, self.meta
            except Exception as e:
                self.log.warning(f"Failed to read cached heatmap '{self.cache_path_hist}': {e}. Recomputing...")

        self.df = self.df.with_columns(
            pl.col("date").dt.truncate("1m").dt.epoch("s").alias("time_numeric")
        )

        if self.distance_range:
            min_dist = int(self.distance_range["min_dist"])
            max_dist = int(self.distance_range["max_dist"])
            distance_bins = np.arange(min_dist, max_dist, 10)
        else:
            distance_bins = np.arange(0, int(self.df.select(pl.max("dist_Km")).item()), 10)

        start_edge    = int(pd.Timestamp(self.sDate).floor("min").timestamp())
        end_edge_excl = int(pd.Timestamp(self.eDate).floor("min").timestamp()) + 60
        time_bins = np.arange(start_edge, end_edge_excl, 60)

        self.hist, self.xedges, self.yedges = np.histogram2d(
            self.df.get_column("time_numeric").to_numpy(),
            self.df.get_column("dist_Km").to_numpy(),
            bins=[time_bins, distance_bins]
        )

        hist_df = pl.DataFrame(
            self.hist, schema=[str(edge) for edge in self.yedges[:-1]]
        ).with_columns(pl.Series("time", self.xedges[:-1]))

        meta = {
            "version":          "v1",
            "sDate":            self.sDate.isoformat(),
            "eDate":            self.eDate.isoformat(),
            "time_bin_seconds": 60,
            "distance_bin_km":  10,
            "n_spots":          int(self.df.height),
            "n_time":           int(self.hist.shape[0]),
            "n_height":         int(self.hist.shape[1]),
            "xedge_start":      float(self.xedges[0]),
            "xedge_end":        float(self.xedges[-1]),
            "yedge_start_km":   float(self.yedges[0]),
            "yedge_end_km":     float(self.yedges[-1]),
        }

        tbl = hist_df.to_arrow().replace_schema_metadata(
            {b"heatmap_meta": json.dumps(meta).encode()}
        )
        pq.write_table(tbl, str(self.cache_path_hist), compression="snappy")
        self.log.info(f"Cached heatmap (w/metadata) to {self.cache_path_hist}")

        self.meta = meta
        return self.hist, self.meta

    def clear_cache(self):
        """Delete cache files for this loader's date/filter configuration."""
        for path in (self.cache_path_df, self.cache_path_hist):
            if path.exists():
                path.unlink()
                self.log.info(f"Removed cache file: {path}")
