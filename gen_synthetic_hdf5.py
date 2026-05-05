#!/usr/bin/env python3
"""
gen_synthetic_hdf5.py — Generate synthetic Madrigal-format HDF5 files for ML training.

Each output file mimics a real rsd{YYYY-MM-DD}.01.hdf5 file with synthetic HF spot
records. Files use dates starting 2030-01-01 so they never collide with real data.
Running the file through the normal pipeline (HDF5PolarsLoader → preprocess_heatmap
→ detect_edge → sin_fit) produces a realistic 2D heatmap either with or without an
embedded LSTID signal.

Physical model
--------------
For every minute t in the propagation window:

  edge(t)  = E0 + c1*(t_hr - t_ref) + c2*(t_hr - t_ref)²      [diurnal baseline]
           + A * sin(2π * t_hr / T + φ)                          [LSTID, if label=1]

  N(t)     ~ Poisson( N_peak * diurnal_envelope(t) )            [spot count]

  pthlen is drawn from a three-component mixture:
    below edge  (fraction 1-p_above):  Exponential(scale_below) below edge [sparse leakage]
    main band   (fraction p_band):     Uniform(edge, edge + band_km)       [propagation zone]
    tail        (fraction 1-p_band):   Exponential(scale_tail_km) above edge + band_km

  The main band models the broad flat distribution of HF spots across all
  TX-RX pairs above the skip zone.  The exponential tail captures the small
  fraction of very long-range paths.

Geographic and frequency metadata are fixed per file within CONUS and 14 MHz
so every file passes the default pipeline filters.

Usage
-----
  python gen_synthetic_hdf5.py --n_days 500 --output_dir data/synthetic_hdf5
  python gen_synthetic_hdf5.py --n_days 200 --lstid_frac 0.6 --seed 42

Output
------
  {output_dir}/rsd2030-01-01.01.hdf5   (and subsequent dates)
  {output_dir}/manifest.csv            (one row per file, all generation params)

The manifest schema is shared with build_ml_manifest.py so synthetic and real
data can be combined into a single training dataset.
"""

import argparse
import csv
import os
from dataclasses import dataclass, asdict, field
from datetime import date, timedelta
from typing import Optional

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants matching the pipeline config
# ---------------------------------------------------------------------------
N_TIME   = 1440          # minutes per day
DT_MIN   = 1.0
T_HRS    = np.arange(N_TIME) / 60.0   # 0.0 … 23.983 hours

PTHLEN_MIN_KM = 0.0
PTHLEN_MAX_KM = 3000.0

# CONUS geographic bounds (must pass the default pipeline filter)
LAT_LIM = (24.5, 49.5)
LON_LIM = (-125.0, -66.5)

FREQ_HZ = 14_000_000.0   # 14 MHz band


# ---------------------------------------------------------------------------
# Generation parameters
# ---------------------------------------------------------------------------

@dataclass
class DayConfig:
    """All parameters that fully define one synthetic day."""

    date_str  : str   = '2030-01-01'
    label     : int   = 0      # 0 = quiet day,  1 = LSTID present
    seed      : int   = 0

    # Propagation window (spots only exist inside this window)
    prop_start_hr : float = 12.5
    prop_end_hr   : float = 23.5

    # Diurnal edge baseline  edge(t) = E0 + c1*(t-tref) + c2*(t-tref)²
    edge_E0_km    : float = 1200.0
    edge_tref_hr  : float = 17.0
    edge_c1_kmph  : float = 0.0
    edge_c2_kmph2 : float = 6.0

    # LSTID sinusoid (values generated regardless; only used when label=1)
    lstid_T_hr        : float = 2.0
    lstid_amplitude_km: float = 80.0
    lstid_phase_rad   : float = 0.0

    # Spot distribution
    n_spots_peak   : float = 300.0   # Poisson mean at diurnal peak
    band_km        : float = 1200.0  # width of uniform propagation band above edge
    scale_tail_km  : float = 300.0   # exponential tail beyond band_km
    p_band         : float = 0.80    # fraction of above-edge spots in the main band
    scale_below_km : float = 35.0    # leakage below edge
    p_above        : float = 0.93    # fraction of spots above edge (high: real data is sparse below)

    # Background noise (uniform pthlen)
    background_frac: float = 0.03

    # Diurnal count envelope — raised cosine (Hann window)
    # Exactly zero at prop_start and prop_end, peaks at diurnal_peak_hr.
    # No sigma needed: shape is fully determined by the three times.
    diurnal_peak_hr  : float = 18.0


def randomize_config(
    rng      : np.random.Generator,
    date_str : str,
    lstid    : bool,
) -> DayConfig:
    """Draw a physically realistic DayConfig at random."""

    # 14 MHz / CONUS: propagation strictly during local daytime.
    # Start varies slightly with season / ionospheric conditions;
    # end is always 24.0 UTC (hard physical barrier for CONUS).
    prop_start = float(rng.uniform(12.0, 13.5))
    prop_end   = 24.0

    t_ref  = float(rng.uniform(14.0, 20.0))
    E0     = float(rng.uniform(900.0, 1600.0))
    c1     = float(rng.uniform(-10.0, 10.0))
    c2     = float(rng.uniform(2.0, 18.0))

    T_hr  = float(rng.uniform(1.0, 4.5))
    amp   = float(rng.uniform(30.0, 200.0))
    phase = float(rng.uniform(0.0, 2 * np.pi))

    n_peak       = float(rng.uniform(100.0, 600.0))
    band_km      = float(rng.uniform(800.0, 2000.0))
    scale_tail   = float(rng.uniform(150.0, 500.0))
    p_band       = float(rng.uniform(0.70, 0.90))
    scale_below  = float(rng.uniform(15.0, 60.0))
    p_above      = float(rng.uniform(0.88, 0.97))
    bg_frac      = float(rng.uniform(0.01, 0.05))

    # Peak can shift: earlier = eastern-US-dominated day, later = west-coast-dominated
    d_peak  = float(rng.uniform(15.5, 20.5))

    return DayConfig(
        date_str          = date_str,
        label             = int(lstid),
        seed              = int(rng.integers(0, 2**31)),
        prop_start_hr     = prop_start,
        prop_end_hr       = prop_end,
        edge_E0_km        = E0,
        edge_tref_hr      = t_ref,
        edge_c1_kmph      = c1,
        edge_c2_kmph2     = c2,
        lstid_T_hr        = T_hr,
        lstid_amplitude_km= amp,
        lstid_phase_rad   = phase,
        n_spots_peak      = n_peak,
        band_km           = band_km,
        scale_tail_km     = scale_tail,
        p_band            = p_band,
        scale_below_km    = scale_below,
        p_above           = p_above,
        background_frac   = bg_frac,
        diurnal_peak_hr   = d_peak,
    )


# ---------------------------------------------------------------------------
# Spot generation
# ---------------------------------------------------------------------------

def _edge_km(cfg: DayConfig) -> np.ndarray:
    """Edge position in km for every minute of the day (shape N_TIME)."""
    dt = T_HRS - cfg.edge_tref_hr
    base = cfg.edge_E0_km + cfg.edge_c1_kmph * dt + cfg.edge_c2_kmph2 * dt ** 2
    if cfg.label == 1:
        base = base + cfg.lstid_amplitude_km * np.sin(
            2 * np.pi * T_HRS / cfg.lstid_T_hr + cfg.lstid_phase_rad
        )
    return np.clip(base, PTHLEN_MIN_KM, PTHLEN_MAX_KM)


def _count_per_minute(cfg: DayConfig, rng: np.random.Generator) -> np.ndarray:
    """
    Poisson spot count per minute (shape N_TIME).

    Envelope: raised cosine (Hann window) split at diurnal_peak_hr.
    - Exactly zero at prop_start_hr and prop_end_hr.
    - Rises smoothly from 0 → 1 over [prop_start_hr, diurnal_peak_hr].
    - Falls smoothly from 1 → 0 over [diurnal_peak_hr, prop_end_hr].
    - Hard zero outside the propagation window.

    The asymmetric split means peak position can shift freely within the
    window without any Gaussian tails leaking outside.
    """
    env  = np.zeros(N_TIME)

    left  = (T_HRS >= cfg.prop_start_hr) & (T_HRS <= cfg.diurnal_peak_hr)
    right = (T_HRS >  cfg.diurnal_peak_hr) & (T_HRS < cfg.prop_end_hr)

    span_left  = cfg.diurnal_peak_hr - cfg.prop_start_hr
    span_right = cfg.prop_end_hr     - cfg.diurnal_peak_hr

    if span_left  > 0:
        t_norm = (T_HRS[left]  - cfg.prop_start_hr) / span_left   # 0 → 1
        env[left]  = 0.5 * (1 - np.cos(np.pi * t_norm))

    if span_right > 0:
        t_norm = (cfg.prop_end_hr - T_HRS[right]) / span_right     # 1 → 0
        env[right] = 0.5 * (1 - np.cos(np.pi * t_norm))

    return rng.poisson(cfg.n_spots_peak * env).astype(np.int32)


def generate_spots(cfg: DayConfig) -> np.ndarray:
    """
    Generate a structured numpy array of synthetic spot records for one day.

    Returns a compound numpy array with fields matching REQUIRED_COLUMNS plus
    the auxiliary fields expected by the Madrigal HDF5 format.
    """
    rng   = np.random.default_rng(cfg.seed)
    edge  = _edge_km(cfg)
    n_per = _count_per_minute(cfg, rng)
    total = int(n_per.sum())

    # --- time index per spot ---
    t_idx = np.repeat(np.arange(N_TIME), n_per)   # minute index (0–1439)
    hour  = t_idx // 60
    minute= t_idx % 60

    # --- pthlen: asymmetric Laplace around edge ---
    edge_at_t = edge[t_idx]

    n_bg     = rng.binomial(total, cfg.background_frac)
    n_signal = total - n_bg

    # signal spots — three-component mixture
    roll      = rng.random(n_signal)
    is_above  = roll < cfg.p_above
    is_below  = ~is_above

    # below-edge leakage: small exponential
    below_offsets = -rng.exponential(cfg.scale_below_km, n_signal)

    # above-edge: split into main band (uniform) and tail (exponential)
    p_main = cfg.p_above * cfg.p_band
    in_band = roll < p_main          # subset of above-edge that falls in main band
    in_tail = is_above & ~in_band    # above-edge but beyond band

    band_offsets = rng.uniform(0.0, cfg.band_km, n_signal)           # uniform main band
    tail_offsets = cfg.band_km + rng.exponential(cfg.scale_tail_km, n_signal)  # tail beyond band

    offsets = np.where(in_band, band_offsets,
              np.where(in_tail, tail_offsets, below_offsets))

    pthlen_signal = np.clip(edge_at_t[:n_signal] + offsets, PTHLEN_MIN_KM, PTHLEN_MAX_KM)

    # background spots (uniform pthlen)
    pthlen_bg = rng.uniform(PTHLEN_MIN_KM, PTHLEN_MAX_KM, n_bg)

    pthlen_all = np.concatenate([pthlen_signal, pthlen_bg]).astype(np.float32)

    # shuffle so background isn't all at the end
    order = rng.permutation(total)
    t_idx_all  = np.concatenate([t_idx[:n_signal], t_idx[:n_bg]])[order]
    pthlen_all = pthlen_all[order]
    hour_all   = (t_idx_all // 60).astype(np.int8)
    min_all    = (t_idx_all % 60).astype(np.int8)

    # --- geographic fields: random within CONUS, consistent per-spot ---
    # latcen and loncen must pass the geographic filter (lat_lim, lon_lim)
    latcen = rng.uniform(LAT_LIM[0] + 1, LAT_LIM[1] - 1, total).astype(np.float32)
    loncen = rng.uniform(LON_LIM[0] + 1, LON_LIM[1] - 1, total).astype(np.float32)

    # rx/tx are slightly offset from the midpoint
    half_deg_lat = (pthlen_all / 111.0 / 2.0).astype(np.float32)
    half_deg_lon = (pthlen_all / (111.0 * np.cos(np.radians(latcen))) / 2.0).astype(np.float32)
    rxlat = (latcen - half_deg_lat).astype(np.float32)
    txlat = (latcen + half_deg_lat).astype(np.float32)
    rxlon = (loncen - half_deg_lon).astype(np.float32)
    txlon = (loncen + half_deg_lon).astype(np.float32)

    # --- frequency: 14 MHz ± small scatter ---
    tfreq = (FREQ_HZ + rng.uniform(-50_000, 50_000, total)).astype(np.float32)

    # --- source: mix of rbn, psk, wsp ---
    src_choices = [b'rbn', b'psk', b'wsp']
    src_idx  = rng.integers(0, 3, total)
    ssrc_arr = np.array([src_choices[i] for i in src_idx], dtype='S3')

    d = date.fromisoformat(cfg.date_str)

    dt_out = np.dtype([
        ('index',  np.int64),
        ('year',   np.int16),
        ('month',  np.int8),
        ('day',    np.int8),
        ('hour',   np.int8),
        ('min',    np.int8),
        ('sec',    np.int8),
        ('pthlen', np.float32),
        ('rxlat',  np.float32),
        ('rxlon',  np.float32),
        ('txlat',  np.float32),
        ('txlon',  np.float32),
        ('tfreq',  np.float32),
        ('latcen', np.float32),
        ('loncen', np.float32),
        ('ssrc',   'S3'),
    ])

    records = np.zeros(total, dtype=dt_out)
    records['index']  = np.arange(total, dtype=np.int64)
    records['year']   = np.int16(d.year)
    records['month']  = np.int8(d.month)
    records['day']    = np.int8(d.day)
    records['hour']   = hour_all
    records['min']    = min_all
    records['sec']    = np.int8(0)
    records['pthlen'] = pthlen_all
    records['rxlat']  = rxlat
    records['rxlon']  = rxlon
    records['txlat']  = txlat
    records['txlon']  = txlon
    records['tfreq']  = tfreq
    records['latcen'] = latcen
    records['loncen'] = loncen
    records['ssrc']   = ssrc_arr

    return records


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_hdf5(records: np.ndarray, path: str) -> None:
    """Write spot records to a PyTables-format HDF5 file readable by HDF5PolarsLoader."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, 'w') as f:
        grp = f.require_group('Data/Table Layout')
        grp.create_dataset('table', data=records, compression='gzip',
                           compression_opts=4)


# ---------------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    # shared with real data
    'date', 'hdf5_file', 'source', 'label',
    'T_hr', 'amplitude_km', 'phase_rad', 'r2',
    'fitStart', 'fitEnd',
    # synthetic-only (NaN in real manifest)
    'seed',
    'prop_start_hr', 'prop_end_hr',
    'edge_E0_km', 'edge_tref_hr', 'edge_c1_kmph', 'edge_c2_kmph2',
    'n_spots_peak', 'band_km', 'scale_tail_km', 'p_band',
    'scale_below_km', 'p_above',
    'background_frac', 'diurnal_peak_hr',
]


def _manifest_row(cfg: DayConfig, hdf5_rel_path: str) -> dict:
    d = asdict(cfg)
    row = {
        'date'             : cfg.date_str,
        'hdf5_file'        : hdf5_rel_path,
        'source'           : 'synthetic',
        'label'            : cfg.label,
        'T_hr'             : cfg.lstid_T_hr   if cfg.label == 1 else '',
        'amplitude_km'     : cfg.lstid_amplitude_km if cfg.label == 1 else '',
        'phase_rad'        : cfg.lstid_phase_rad    if cfg.label == 1 else '',
        'r2'               : '',    # not applicable for synthetic ground truth
        'fitStart'         : '',
        'fitEnd'           : '',
        'seed'             : cfg.seed,
        'prop_start_hr'    : cfg.prop_start_hr,
        'prop_end_hr'      : cfg.prop_end_hr,
        'edge_E0_km'       : cfg.edge_E0_km,
        'edge_tref_hr'     : cfg.edge_tref_hr,
        'edge_c1_kmph'     : cfg.edge_c1_kmph,
        'edge_c2_kmph2'    : cfg.edge_c2_kmph2,
        'n_spots_peak'     : cfg.n_spots_peak,
        'band_km'          : cfg.band_km,
        'scale_tail_km'    : cfg.scale_tail_km,
        'p_band'           : cfg.p_band,
        'scale_below_km'   : cfg.scale_below_km,
        'p_above'          : cfg.p_above,
        'background_frac'  : cfg.background_frac,
        'diurnal_peak_hr'  : cfg.diurnal_peak_hr,
    }
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate synthetic Madrigal-format HDF5 files for ML training.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--n_days',     type=int,   default=200,
                        help='Number of synthetic days to generate')
    parser.add_argument('--lstid_frac', type=float, default=0.5,
                        help='Fraction of days that contain an LSTID signal')
    parser.add_argument('--output_dir', type=str,   default='data/synthetic_hdf5',
                        help='Directory to write HDF5 files and manifest')
    parser.add_argument('--start_date', type=str,   default='2030-01-01',
                        help='First synthetic date (YYYY-MM-DD); increments daily')
    parser.add_argument('--seed',       type=int,   default=0,
                        help='Master RNG seed for full reproducibility')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Assign labels
    n_lstid  = int(round(args.n_days * args.lstid_frac))
    labels   = [1] * n_lstid + [0] * (args.n_days - n_lstid)
    rng.shuffle(labels)

    current_date = date.fromisoformat(args.start_date)
    manifest_path = os.path.join(args.output_dir, 'manifest.csv')

    print(f'Generating {args.n_days} days  '
          f'({n_lstid} LSTID / {args.n_days - n_lstid} quiet)')
    print(f'Output: {args.output_dir}')
    print(f'Dates:  {args.start_date} → '
          f'{(current_date + timedelta(days=args.n_days - 1)).isoformat()}')
    print()

    with open(manifest_path, 'w', newline='') as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        for i in range(args.n_days):
            date_str  = current_date.isoformat()
            lstid     = bool(labels[i])
            cfg       = randomize_config(rng, date_str=date_str, lstid=lstid)

            fname     = f'rsd{date_str}.01.hdf5'
            fpath     = os.path.join(args.output_dir, fname)
            records   = generate_spots(cfg)
            write_hdf5(records, fpath)

            writer.writerow(_manifest_row(cfg, fname))

            if (i + 1) % 50 == 0 or i == 0:
                n_spots = len(records)
                print(f'  [{i+1:>4}/{args.n_days}]  {date_str}  '
                      f'label={cfg.label}  '
                      f'spots={n_spots:,}  '
                      f'E0={cfg.edge_E0_km:.0f} km  '
                      f'T={cfg.lstid_T_hr:.2f} hr  '
                      f'A={cfg.lstid_amplitude_km:.0f} km')

            current_date += timedelta(days=1)

    print(f'\nManifest written: {manifest_path}')
    print(f'\nNext step: run build_ml_manifest.py to combine with real data.')


if __name__ == '__main__':
    main()
