#!/usr/bin/env python3

import sys
from pathlib import Path

# Add project root to sys.path (so "scripts" is importable)
sys.path.append(str(Path(__file__).resolve().parents[1]))

import pytest
from datetime import datetime

def test_invalid_region(monkeypatch):
    import scripts.regions as regions_mod
    import scripts.hdf5_loader as loader_mod
    monkeypatch.setattr(regions_mod, "REGIONS", {})  # no valid regions

    with pytest.raises(ValueError):
        loader_mod.HDF5PolarsLoader(
            data_dir=".",
            sDate=datetime(2020, 1, 1),
            eDate=datetime(2020, 1, 2),
            region_name="NotARegion",
        )

def test_valid_region(monkeypatch):
    import scripts.hdf5_loader as loader_mod

    monkeypatch.setattr(
        loader_mod, "REGIONS",
        {"TestRegion": {"lat_lim": (0, 10), "lon_lim": (0, 20)}},
    )

    loader = loader_mod.HDF5PolarsLoader(
        data_dir=".",
        sDate=datetime(2020, 1, 1),
        eDate=datetime(2020, 1, 2),
        region_name="TestRegion",
    )
    assert loader.region == {"lat_lim": (0, 10), "lon_lim": (0, 20)}