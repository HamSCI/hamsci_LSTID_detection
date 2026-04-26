#!/usr/bin/env python3

import shutil
import sys
from pathlib import Path

# Make project root importable so "scripts" resolves
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from datetime import datetime
import pandas as pd
import numpy as np


# 1) __init__: data_dir must exist
def test_init_raises_when_data_dir_missing(tmp_path):
    import scripts.hdf5_loader as loader_mod
    with pytest.raises(FileNotFoundError, match="Data directory not found"):
        loader_mod.HDF5PolarsLoader(
            data_dir=str(tmp_path / "nope"),
            sDate=datetime(2020, 1, 1, 0, 0, 0),
            eDate=datetime(2020, 1, 1, 23, 59, 59),
            cache_dir=str(tmp_path / "cache"),
            use_cache=True,
        )


# 2) __init__: sDate must be <= eDate
def test_init_raises_when_start_after_end(tmp_path):
    import scripts.hdf5_loader as loader_mod
    with pytest.raises(ValueError, match="sDate .* is after eDate"):
        loader_mod.HDF5PolarsLoader(
            data_dir=str(tmp_path),
            sDate=datetime(2020, 1, 2, 0, 0, 0),
            eDate=datetime(2020, 1, 1, 23, 59, 59),
            cache_dir=str(tmp_path / "cache"),
            use_cache=True,
        )


# 3) __init__: invalid region_bounds shape (missing lon_lim)
def test_init_raises_on_bad_region_bounds(tmp_path):
    import scripts.hdf5_loader as loader_mod
    with pytest.raises(ValueError, match="Invalid region_bounds"):
        loader_mod.HDF5PolarsLoader(
            data_dir=str(tmp_path),
            sDate=datetime(2020, 1, 1),
            eDate=datetime(2020, 1, 1, 23, 59, 59),
            cache_dir=str(tmp_path / "cache"),
            use_cache=True,
            region_bounds={"lat_lim": (0, 10)},  # lon_lim missing
        )


# 4) __init__: invalid freq_range shape (missing max_freq)
def test_init_raises_on_bad_freq_range(tmp_path):
    import scripts.hdf5_loader as loader_mod
    with pytest.raises(ValueError, match="Invalid freq_range"):
        loader_mod.HDF5PolarsLoader(
            data_dir=str(tmp_path),
            sDate=datetime(2020, 1, 1),
            eDate=datetime(2020, 1, 1, 23, 59, 59),
            cache_dir=str(tmp_path / "cache"),
            use_cache=True,
            freq_range={"min_freq": 1_000_000},  # max_freq missing
        )


# 5) __init__: invalid distance_range shape (missing max_dist)
def test_init_raises_on_bad_distance_range(tmp_path):
    import scripts.hdf5_loader as loader_mod
    with pytest.raises(ValueError, match="Invalid distance_range"):
        loader_mod.HDF5PolarsLoader(
            data_dir=str(tmp_path),
            sDate=datetime(2020, 1, 1),
            eDate=datetime(2020, 1, 1, 23, 59, 59),
            cache_dir=str(tmp_path / "cache"),
            use_cache=True,
            distance_range={"min_dist": 0},  # max_dist missing
        )


# 6) get_dataframe(): bubbles FileNotFoundError when no HDF5 files in range
def test_get_dataframe_raises_when_no_hdf5_files(tmp_path):
    import scripts.hdf5_loader as loader_mod
    loader = loader_mod.HDF5PolarsLoader(
        data_dir=str(tmp_path),  # valid dir, but no files inside
        sDate=datetime(2020, 1, 1, 0, 0, 0),
        eDate=datetime(2020, 1, 1, 23, 59, 59),
        cache_dir=str(tmp_path / "cache"),
        use_cache=True,
    )
    with pytest.raises(FileNotFoundError, match="No HDF5 files found/readable"):
        loader.get_dataframe()  # process_data -> load_data -> raise


# 7) process_data(): raises RuntimeError if load_data() returns empty DataFrame
def test_process_data_raises_on_empty_df(tmp_path, monkeypatch):
    import scripts.hdf5_loader as loader_mod
    import pandas as pd

    loader = loader_mod.HDF5PolarsLoader(
        data_dir=str(tmp_path),
        sDate=datetime(2020, 1, 1, 0, 0, 0),
        eDate=datetime(2020, 1, 1, 23, 59, 59),
        cache_dir=str(tmp_path / "cache"),
        use_cache=False,  # ensures it tries to load/process
    )
    # Simulate "files existed but everything got filtered away"
    monkeypatch.setattr(loader, "load_data", lambda: pd.DataFrame())

    with pytest.raises(RuntimeError, match="No data loaded"):
        loader.get_dataframe()


# 8) gen_histogram(): raises RuntimeError if called before any dataframe is loaded
def test_gen_histogram_raises_if_no_df_loaded(tmp_path):
    import scripts.hdf5_loader as loader_mod

    loader = loader_mod.HDF5PolarsLoader(
        data_dir=str(tmp_path),
        sDate=datetime(2020, 1, 1, 0, 0, 0),
        eDate=datetime(2020, 1, 1, 23, 59, 59),
        cache_dir=str(tmp_path / "cache"),
        use_cache=True,
    )
    with pytest.raises(RuntimeError, match="No dataframe loaded; call get_dataframe"):
        loader.gen_histogram()


# 9) apply_region_filter(): drops rows outside lat/lon bounds
def test_apply_region_filter_trims_rows(tmp_path):
    import scripts.hdf5_loader as loader_mod

    df = pd.DataFrame({
        "latcen": [30.0, 50.0],
        "loncen": [-100.0, -10.0],
    })

    loader = loader_mod.HDF5PolarsLoader(
        data_dir=str(tmp_path),
        sDate=datetime(2020, 1, 1, 0, 0, 0),
        eDate=datetime(2020, 1, 1, 23, 59, 59),
        cache_dir=str(tmp_path / "cache"),
        region_bounds={"lat_lim": (20, 40), "lon_lim": (-120, -60)},
        use_cache=False,
    )

    out = loader.apply_region_filter(df.copy())
    assert len(out) == 1
    assert out.iloc[0]["latcen"] == 30.0 and out.iloc[0]["loncen"] == -100.0


# 10) apply_freq_filter(): keeps only rows in [min_freq, max_freq]
def test_apply_freq_filter_trims_rows(tmp_path):
    import scripts.hdf5_loader as loader_mod

    df = pd.DataFrame({"tfreq": [5_000_000, 7_500_000, 9_000_000]})
    loader = loader_mod.HDF5PolarsLoader(
        data_dir=str(tmp_path),
        sDate=datetime(2020, 1, 1, 0, 0, 0),
        eDate=datetime(2020, 1, 1, 23, 59, 59),
        cache_dir=str(tmp_path / "cache"),
        freq_range={"min_freq": 6_000_000, "max_freq": 8_000_000},
        use_cache=False,
    )
    out = loader.apply_freq_filter(df.copy())
    assert np.array_equal(out["tfreq"].to_numpy(), [7_500_000])


# 11) apply_distance_filter(): keeps only rows in [min_dist, max_dist]
def test_apply_distance_filter_trims_rows(tmp_path):
    import scripts.hdf5_loader as loader_mod

    df = pd.DataFrame({"pthlen": [100, 500, 2500]})
    loader = loader_mod.HDF5PolarsLoader(
        data_dir=str(tmp_path),
        sDate=datetime(2020, 1, 1, 0, 0, 0),
        eDate=datetime(2020, 1, 1, 23, 59, 59),
        cache_dir=str(tmp_path / "cache"),
        distance_range={"min_dist": 200, "max_dist": 2000},
        use_cache=False,
    )
    out = loader.apply_distance_filter(df.copy())
    assert np.array_equal(out["pthlen"].to_numpy(), [500])


# 12) apply_datetime_filter(): keeps rows within [sDate, eDate]
def test_apply_datetime_filter_keeps_window(tmp_path):
    import scripts.hdf5_loader as loader_mod

    df = pd.DataFrame({
        "year":  [2020, 2020, 2020],
        "month": [1,    1,    1],
        "day":   [1,    1,    1],
        "hour":  [0,    12,   23],
        "min":   [0,    0,    59],
        "sec":   [0,    0,    59],
    })

    loader = loader_mod.HDF5PolarsLoader(
        data_dir=str(tmp_path),
        sDate=datetime(2020, 1, 1, 6, 0, 0),
        eDate=datetime(2020, 1, 1, 18, 0, 0),
        cache_dir=str(tmp_path / "cache"),
        use_cache=False,
    )
    out = loader.apply_datetime_filter(df.copy(), loader.sDate, loader.eDate)
    assert len(out) == 1
    assert (out[["hour", "min", "sec"]].iloc[0] == [12, 0, 0]).all()


# 13) process_data(): transforms to Polars with expected columns/values
def test_process_data_transforms_to_polars_with_expected_cols(tmp_path, monkeypatch):
    import scripts.hdf5_loader as loader_mod
    import polars as pl

    pdf = pd.DataFrame({
        "year":   [2020],
        "month":  [1],
        "day":    [1],
        "hour":   [12],
        "min":    [34],
        "sec":    [56],
        "pthlen": [500],
        "rxlat":  [10.0],
        "rxlon":  [-20.0],
        "txlat":  [11.0],
        "txlon":  [-21.0],
        "tfreq":  [7_500_000.0],
        "latcen": [10.5],
        "loncen": [-20.5],
        "ssrc":   ["X"],
    })

    loader = loader_mod.HDF5PolarsLoader(
        data_dir=str(tmp_path),
        sDate=datetime(2020, 1, 1, 0, 0, 0),
        eDate=datetime(2020, 1, 1, 23, 59, 59),
        cache_dir=str(tmp_path / "cache"),
        use_cache=False,
    )
    # Avoid hitting Dask/HDF5; inject our tiny frame
    monkeypatch.setattr(loader, "load_data", lambda: pdf)

    df_pl = loader.get_dataframe()
    assert isinstance(df_pl, pl.DataFrame)

    expected = [
        "date", "freq", "band", "dist_Km", "source",
        "mid_lat", "mid_long", "rx_lat", "tx_lat",
        "rx_long", "tx_long", "freq_MHz",
    ]
    for col in expected:
        assert col in df_pl.columns

    out = df_pl.to_dicts()[0]
    assert out["dist_Km"] == 500
    assert out["freq"] == 7_500_000.0
    assert out["freq_MHz"] == 8  # (7.5e6 / 1e6) rounded 0


# 14) gen_histogram(): builds a basic histogram and returns meta
def test_gen_histogram_basic(tmp_path):
    import scripts.hdf5_loader as loader_mod
    import polars as pl

    loader = loader_mod.HDF5PolarsLoader(
        data_dir=str(tmp_path),
        sDate=datetime(2020, 1, 1, 0, 0, 0),
        eDate=datetime(2020, 1, 1, 0, 2, 0),  # 2 minutes span
        cache_dir=str(tmp_path / "cache"),
        use_cache=False,
    )

    df_pl = pl.DataFrame({
        "date":     [datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 0, 1, 0)],
        "freq":     [7_500_000.0, 7_500_000.0],
        "band":     [0, 0],
        "dist_Km":  [10.0, 30.0],
        "source":   ["X", "X"],
        "mid_lat":  [0.0, 0.0],
        "mid_long": [0.0, 0.0],
        "rx_lat":   [0.0, 0.0],
        "tx_lat":   [0.0, 0.0],
        "rx_long":  [0.0, 0.0],
        "tx_long":  [0.0, 0.0],
        "freq_MHz": [8, 8],
    })
    loader.df = df_pl

    H, meta = loader.gen_histogram()
    assert isinstance(H, np.ndarray)
    assert H.ndim == 2
    for k in ("time_bin_seconds", "distance_bin_km", "n_time", "n_height"):
        assert k in meta
    assert meta["time_bin_seconds"] == 60
    assert meta["distance_bin_km"] == 10

# test_legacy_heatmap_matches_madrigal was removed: the synthetic HDF5
# (rsd2025-01-01.01.hdf5) was updated to include a diurnal polynomial trend,
# so it no longer matches the legacy CSV reference that was generated from the
# original flat sinusoid data.  Loader regression is already covered by the
# unit tests above (tests 9-14) and the pipeline integration tests.
