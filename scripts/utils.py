#!/usr/bin/env python3

import polars as pl
from datetime import datetime, timedelta, timezone, date
from pysolar.solar import get_altitude
from typing import List, Dict

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

def calculate_source_percent(df: pl.DataFrame) -> dict:
    """Calculate the percentage of each data source in the given Polars DataFrame."""
    if df.is_empty():
        return {}

    # Convert 'source' column to string and count occurrences
    source_counts = df["source"].cast(str).value_counts().to_pandas()

    # Convert to dictionary
    source_counts_dict = dict(zip(source_counts["source"], source_counts["count"]))

    total_count = sum(source_counts_dict.values())

    # Compute percentages
    return {
        key: (value / total_count) * 100 if total_count > 0 else 0
        for key, value in source_counts_dict.items()
    }

def compute_sza(dt: datetime, lat: float, lon: float) -> float:
    """Solar zenith angle in degrees (90 - solar altitude)."""
    dt = dt.replace(tzinfo=timezone.utc)
    return 90 - get_altitude(lat, lon, dt)

def get_sunset_datetime(
    day: date,
    lat: float,
    lon: float,
    altitude_km: int = 300) -> datetime:
    """
    Return sunset datetime (when SZA > 100°) for a given day and location.
    """
    start_dt = datetime(day.year, day.month, day.day, 10, 0, 0, tzinfo=timezone.utc)
    end_dt = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)

    dt = start_dt
    step = timedelta(minutes=1)

    while dt <= end_dt:
        sza = compute_sza(dt, lat, lon)
        if sza > 100:
            return dt
        dt += step

    raise ValueError(f"Sunset (SZA > 100°) not found on {day} at {lat}, {lon}")

def get_sunset_time_steps(
    day: date,
    region: Dict,
    altitude_km: int = 300) -> List[datetime]:
    """
    Get 7 time steps: from 2 hours before to 5 hours after sunset (1-hour intervals).
    """
    lat_center = sum(region['lat_lim']) / 2
    lon_center = sum(region['lon_lim']) / 2

    sunset = get_sunset_datetime(day, lat_center, lon_center, altitude_km)
    
    return [round_to_nearest_hour(sunset + timedelta(hours=h)) for h in range(-2, 6)]
    #return [sunset + timedelta(hours=h) for h in range(-2, 6)]

def round_to_nearest_hour(dt: datetime) -> datetime:
    if dt.minute >= 30:
        dt += timedelta(hours=1)
    return dt.replace(minute=0, second=0, microsecond=0)

