#!/usr/bin/env python3
"""
gen_synthetic_hdf5.py — Generate synthetic Madrigal-format HDF5 files for ML training.

Produces rsd{YYYY-MM-DD}.01.hdf5 files (dates 2030+) identical in structure to real
Madrigal data. Run them through the normal pipeline unchanged.

Physical model
--------------
  edge(t) = E0 + c1*(t-tref) + c2*(t-tref)²  +  A·sin(2π·t/T + φ)

  A = 0       → truly flat day, no wave
  A = 1–100   → sinusoid always embedded; small A looks quiet, large A looks like LSTID

  Spot count: raised-cosine (Hann) envelope, zero before ~12 UTC, zero at 24 UTC.
  pthlen:     uniform main band above edge + exponential tail + sparse leakage below.
  Noise days: background_frac scaled up by noise_multiplier (swamps the wave).
  Data gaps:  random 1–3 hr time blocks zeroed out (simulates real outages).

Manifest
--------
  amplitude_km, T_hr    — regression targets (no binary label)
  curriculum_phase      — 1 = clear (|A|<phase1_max or |A|>phase1_min), 2 = ambiguous
  noise_multiplier      — >1 means noisy day
  n_gaps                — number of data gaps applied

Usage
-----
  python gen_synthetic_hdf5.py --n_days 500 --output_dir data/synthetic_hdf5
  python gen_synthetic_hdf5.py --n_days 200 --seed 42 --phase1_min_amp 45
"""

import argparse
import csv
import os
from dataclasses import dataclass, asdict
from datetime import date, timedelta

import h5py
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_TIME        = 1440
T_HRS         = np.arange(N_TIME) / 60.0
PTHLEN_MIN_KM = 0.0
PTHLEN_MAX_KM = 3000.0
LAT_LIM       = (24.5, 49.5)
LON_LIM       = (-125.0, -66.5)
FREQ_HZ       = 14_000_000.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class DayConfig:
    date_str          : str   = '2030-01-01'
    seed              : int   = 0
    prop_start_hr     : float = 12.5
    prop_end_hr       : float = 24.0
    edge_E0_km        : float = 1200.0
    edge_tref_hr      : float = 17.0
    edge_c1_kmph      : float = 0.0
    edge_c2_kmph2     : float = 6.0
    lstid_T_hr        : float = 2.0
    lstid_amplitude_km: float = 50.0
    lstid_phase_rad   : float = 0.0
    n_spots_peak      : float = 300.0
    band_km           : float = 1200.0
    scale_tail_km     : float = 300.0
    p_band            : float = 0.80
    scale_below_km    : float = 35.0
    p_above           : float = 0.93
    background_frac   : float = 0.03
    noise_multiplier  : float = 1.0
    n_gaps            : int   = 0
    diurnal_peak_hr   : float = 18.0


def randomize_config(rng: np.random.Generator, date_str: str) -> DayConfig:

    prop_start = float(rng.uniform(12.0, 13.5))
    E0         = float(rng.uniform(900.0, 1600.0))
    t_ref      = float(rng.uniform(14.0, 20.0))
    c1         = float(rng.uniform(-10.0, 10.0))
    c2         = float(rng.uniform(2.0, 18.0))

    T_hr  = float(rng.uniform(1.0, 4.5))
    phase = float(rng.uniform(0.0, 2 * np.pi))
    # 20% truly flat (A=0), 80% uniform 0–100 km
    amp   = 0.0 if rng.random() < 0.20 else float(rng.uniform(0.0, 100.0))

    n_peak      = float(rng.uniform(100.0, 600.0))
    band_km     = float(rng.uniform(800.0, 2000.0))
    scale_tail  = float(rng.uniform(150.0, 500.0))
    p_band      = float(rng.uniform(0.70, 0.90))
    scale_below = float(rng.uniform(15.0, 60.0))
    p_above     = float(rng.uniform(0.88, 0.97))
    bg_frac     = float(rng.uniform(0.01, 0.05))

    # 20% high-noise days
    noise_mult = float(rng.uniform(2.0, 5.0)) if rng.random() < 0.20 else 1.0

    # 70% no gaps, 20% one gap, 10% two gaps
    u      = rng.random()
    n_gaps = 0 if u < 0.70 else (1 if u < 0.90 else 2)

    d_peak = float(rng.uniform(15.5, 20.5))

    return DayConfig(
        date_str          = date_str,
        seed              = int(rng.integers(0, 2**31)),
        prop_start_hr     = prop_start,
        prop_end_hr       = 24.0,
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
        noise_multiplier  = noise_mult,
        n_gaps            = n_gaps,
        diurnal_peak_hr   = d_peak,
    )


# ---------------------------------------------------------------------------
# Spot generation
# ---------------------------------------------------------------------------

def _edge_km(cfg: DayConfig) -> np.ndarray:
    dt = T_HRS - cfg.edge_tref_hr
    return np.clip(
        cfg.edge_E0_km + cfg.edge_c1_kmph * dt + cfg.edge_c2_kmph2 * dt ** 2
        + cfg.lstid_amplitude_km * np.sin(
            2 * np.pi * T_HRS / cfg.lstid_T_hr + cfg.lstid_phase_rad),
        PTHLEN_MIN_KM, PTHLEN_MAX_KM,
    )


def _count_per_minute(cfg: DayConfig, rng: np.random.Generator) -> np.ndarray:
    env   = np.zeros(N_TIME)
    left  = (T_HRS >= cfg.prop_start_hr) & (T_HRS <= cfg.diurnal_peak_hr)
    right = (T_HRS >  cfg.diurnal_peak_hr) & (T_HRS < cfg.prop_end_hr)
    sl    = cfg.diurnal_peak_hr - cfg.prop_start_hr
    sr    = cfg.prop_end_hr     - cfg.diurnal_peak_hr
    if sl > 0:
        env[left]  = 0.5 * (1 - np.cos(np.pi * (T_HRS[left]  - cfg.prop_start_hr) / sl))
    if sr > 0:
        env[right] = 0.5 * (1 - np.cos(np.pi * (cfg.prop_end_hr - T_HRS[right]) / sr))

    counts = rng.poisson(cfg.n_spots_peak * env).astype(np.int32)

    # Data gaps: zero out n_gaps random 1–3 hr blocks
    for _ in range(cfg.n_gaps):
        start = int(rng.uniform(cfg.prop_start_hr * 60, cfg.prop_end_hr * 60 - 60))
        dur   = int(rng.uniform(60, 180))
        counts[start : min(start + dur, N_TIME)] = 0

    return counts


def generate_spots(cfg: DayConfig) -> np.ndarray:
    rng   = np.random.default_rng(cfg.seed)
    edge  = _edge_km(cfg)
    n_per = _count_per_minute(cfg, rng)
    total = int(n_per.sum())

    t_idx     = np.repeat(np.arange(N_TIME), n_per)
    edge_at_t = edge[t_idx]

    # noise_multiplier inflates background fraction
    eff_bg   = min(cfg.background_frac * cfg.noise_multiplier, 0.60)
    n_bg     = rng.binomial(total, eff_bg)
    n_signal = total - n_bg

    roll   = rng.random(n_signal)
    p_main = cfg.p_above * cfg.p_band
    in_band = roll < p_main
    in_tail = (roll >= p_main) & (roll < cfg.p_above)

    offsets = np.where(
        in_band,  rng.uniform(0.0, cfg.band_km, n_signal),
        np.where(
        in_tail,  cfg.band_km + rng.exponential(cfg.scale_tail_km, n_signal),
                 -rng.exponential(cfg.scale_below_km, n_signal),
    ))

    pthlen_all = np.concatenate([
        np.clip(edge_at_t[:n_signal] + offsets, PTHLEN_MIN_KM, PTHLEN_MAX_KM),
        rng.uniform(PTHLEN_MIN_KM, PTHLEN_MAX_KM, n_bg),
    ]).astype(np.float32)

    order      = rng.permutation(total)
    t_idx_all  = np.concatenate([t_idx[:n_signal], t_idx[:n_bg]])[order]
    pthlen_all = pthlen_all[order]

    latcen = rng.uniform(LAT_LIM[0] + 1, LAT_LIM[1] - 1, total).astype(np.float32)
    loncen = rng.uniform(LON_LIM[0] + 1, LON_LIM[1] - 1, total).astype(np.float32)
    hdlat  = (pthlen_all / 222.0).astype(np.float32)
    hdlon  = (pthlen_all / (222.0 * np.cos(np.radians(latcen)))).astype(np.float32)

    d      = date.fromisoformat(cfg.date_str)
    src    = [b'rbn', b'psk', b'wsp']

    dt_out = np.dtype([
        ('index', np.int64), ('year', np.int16), ('month', np.int8), ('day', np.int8),
        ('hour', np.int8), ('min', np.int8), ('sec', np.int8),
        ('pthlen', np.float32), ('rxlat', np.float32), ('rxlon', np.float32),
        ('txlat', np.float32), ('txlon', np.float32), ('tfreq', np.float32),
        ('latcen', np.float32), ('loncen', np.float32), ('ssrc', 'S3'),
    ])

    rec = np.zeros(total, dtype=dt_out)
    rec['index']  = np.arange(total, dtype=np.int64)
    rec['year']   = np.int16(d.year)
    rec['month']  = np.int8(d.month)
    rec['day']    = np.int8(d.day)
    rec['hour']   = (t_idx_all // 60).astype(np.int8)
    rec['min']    = (t_idx_all % 60).astype(np.int8)
    rec['sec']    = np.int8(0)
    rec['pthlen'] = pthlen_all
    rec['rxlat']  = latcen - hdlat
    rec['rxlon']  = loncen - hdlon
    rec['txlat']  = latcen + hdlat
    rec['txlon']  = loncen + hdlon
    rec['tfreq']  = (FREQ_HZ + rng.uniform(-50_000, 50_000, total)).astype(np.float32)
    rec['latcen'] = latcen
    rec['loncen'] = loncen
    rec['ssrc']   = np.array([src[i] for i in rng.integers(0, 3, total)], dtype='S3')
    return rec


# ---------------------------------------------------------------------------
# HDF5 writer
# ---------------------------------------------------------------------------

def write_hdf5(records: np.ndarray, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with h5py.File(path, 'w') as f:
        f.require_group('Data/Table Layout').create_dataset(
            'table', data=records, compression='gzip', compression_opts=4)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

MANIFEST_FIELDS = [
    'date', 'hdf5_file', 'source', 'curriculum_phase',
    'T_hr', 'amplitude_km',
    'seed', 'noise_multiplier', 'n_gaps',
    'prop_start_hr', 'prop_end_hr',
    'edge_E0_km', 'edge_tref_hr', 'edge_c1_kmph', 'edge_c2_kmph2',
    'n_spots_peak', 'band_km', 'scale_tail_km', 'p_band',
    'scale_below_km', 'p_above', 'background_frac', 'diurnal_peak_hr',
]


def _phase(amp: float, noise_mult: float, max_clear: float, min_clear: float,
           max_noise: float = 1.5) -> int:
    """
    Phase 1 (clear): low noise AND (clearly quiet OR clearly LSTID).
    Phase 2 (ambiguous): noisy days OR borderline amplitude.
    Noisy days are always phase 2 regardless of amplitude — noise creates
    spurious apparent wave patterns that make the example hard, not easy.
    """
    if noise_mult > max_noise:
        return 2
    return 1 if (amp < max_clear or amp > min_clear) else 2


def _manifest_row(cfg: DayConfig, fname: str, phase: int) -> dict:
    return {
        'date'             : cfg.date_str,
        'hdf5_file'        : fname,
        'source'           : 'synthetic',
        'curriculum_phase' : phase,
        'T_hr'             : cfg.lstid_T_hr,
        'amplitude_km'     : cfg.lstid_amplitude_km,
        'seed'             : cfg.seed,
        'noise_multiplier' : cfg.noise_multiplier,
        'n_gaps'           : cfg.n_gaps,
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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--n_days',         type=int,   default=200)
    parser.add_argument('--output_dir',     type=str,   default='data/synthetic_hdf5')
    parser.add_argument('--start_date',     type=str,   default='2030-01-01')
    parser.add_argument('--seed',           type=int,   default=0)
    parser.add_argument('--phase1_max_amp', type=float, default=5.0,
                        help='A below this → phase 1 (clear quiet)')
    parser.add_argument('--phase1_min_amp', type=float, default=45.0,
                        help='A above this → phase 1 (clear LSTID)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rng          = np.random.default_rng(args.seed)
    current_date = date.fromisoformat(args.start_date)
    manifest_path = os.path.join(args.output_dir, 'manifest.csv')

    print(f'Generating {args.n_days} days → {args.output_dir}')
    print(f'Dates:   {args.start_date} → '
          f'{(current_date + timedelta(days=args.n_days-1)).isoformat()}')
    print(f'Phase 1: A < {args.phase1_max_amp} or A > {args.phase1_min_amp} km')
    print(f'Phase 2: {args.phase1_max_amp} ≤ A ≤ {args.phase1_min_amp} km')
    print()

    p1 = p2 = 0

    with open(manifest_path, 'w', newline='') as mf:
        writer = csv.DictWriter(mf, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()

        for i in range(args.n_days):
            date_str = current_date.isoformat()
            cfg      = randomize_config(rng, date_str)
            fname    = f'rsd{date_str}.01.hdf5'

            write_hdf5(generate_spots(cfg), os.path.join(args.output_dir, fname))

            phase = _phase(cfg.lstid_amplitude_km, cfg.noise_multiplier,
                           args.phase1_max_amp, args.phase1_min_amp)
            writer.writerow(_manifest_row(cfg, fname, phase))

            if phase == 1: p1 += 1
            else:          p2 += 1

            if (i + 1) % 50 == 0 or i == 0:
                print(f'  [{i+1:>4}/{args.n_days}]  {date_str}  '
                      f'A={cfg.lstid_amplitude_km:5.1f} km  phase={phase}  '
                      f'noise={cfg.noise_multiplier:.1f}x  gaps={cfg.n_gaps}')

            current_date += timedelta(days=1)

    print(f'\nPhase 1 (clear): {p1}   Phase 2 (ambiguous): {p2}')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
