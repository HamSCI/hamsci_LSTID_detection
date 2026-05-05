#!/usr/bin/env python3
"""
build_ml_manifest.py — Combine synthetic and real data into a unified ML manifest.

Reads the synthetic manifest produced by gen_synthetic_hdf5.py and any number of
pipeline sinfit CSVs from output/summary_csv/, applies a configurable label threshold
to convert sinfit results into binary LSTID labels, and writes a single manifest CSV
ready for ML training.

Label derivation for real data
-------------------------------
A day is labelled LSTID=1 when the pipeline's best selected fit satisfies ALL of:
  r2          >= --r2_thresh       (default 0.3)
  T_hr        in [--T_min, --T_max]  (default 1.0–4.5 hr)
  amplitude_km >= --amp_min        (default 30 km)
  duration_hr >= --dur_min         (default 2 hr)

Days where the pipeline found no stable window (all NaN) are labelled LSTID=0.
Days that are ambiguous (fit exists but fails threshold) are labelled LSTID=0.
Set --keep_ambiguous to exclude those from the manifest instead.

Manifest schema (same for synthetic and real rows)
---------------------------------------------------
  date           YYYY-MM-DD
  hdf5_file      path to HDF5 relative to its data_dir
  source         'synthetic' | 'real'
  label          0 or 1
  T_hr           fitted/true period (empty = quiet)
  amplitude_km   fitted/true amplitude (empty = quiet)
  phase_rad      fitted phase in radians (empty = quiet)
  r2             sinfit R²  (empty for synthetic — ground truth is exact)
  fitStart       fit window start timestamp (empty if quiet)
  fitEnd         fit window end timestamp (empty if quiet)
  seed           RNG seed (synthetic only)
  [synthetic generation parameters…]   NaN for real rows

Usage
-----
  # Combine everything
  python build_ml_manifest.py \\
      --synthetic_manifest  data/synthetic_hdf5/manifest.csv \\
      --real_csv_dir        output/summary_csv \\
      --real_hdf5_dir       data/madrigal \\
      --output              data/ml_manifest.csv

  # Real data only
  python build_ml_manifest.py \\
      --real_csv_dir output/summary_csv \\
      --real_hdf5_dir data/madrigal \\
      --output data/real_manifest.csv

  # Synthetic only
  python build_ml_manifest.py \\
      --synthetic_manifest data/synthetic_hdf5/manifest.csv \\
      --output data/synthetic_manifest_only.csv
"""

import argparse
import csv
import glob
import os
from typing import Optional

import pandas as pd

# Manifest field order — matches gen_synthetic_hdf5.py MANIFEST_FIELDS
MANIFEST_FIELDS = [
    'date', 'hdf5_file', 'source', 'label',
    'T_hr', 'amplitude_km', 'phase_rad', 'r2',
    'fitStart', 'fitEnd',
    'seed',
    'prop_start_hr', 'prop_end_hr',
    'edge_E0_km', 'edge_tref_hr', 'edge_c1_kmph', 'edge_c2_kmph2',
    'n_spots_peak', 'band_km', 'scale_tail_km', 'p_band',
    'scale_below_km', 'p_above',
    'background_frac', 'diurnal_peak_hr',
]

SYNTHETIC_ONLY_FIELDS = MANIFEST_FIELDS[10:]   # seed and beyond


# ---------------------------------------------------------------------------
# Real-data processing
# ---------------------------------------------------------------------------

def load_sinfit_csvs(csv_dir: str) -> pd.DataFrame:
    """Load and concatenate all sinfit CSVs from the pipeline output directory."""
    pattern = os.path.join(csv_dir, '*.csv')
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No CSVs found in {csv_dir}')

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            # Normalise column names across older/newer CSV formats
            df.columns = df.columns.str.strip()
            # Keep only the best-fit row per date
            if 'selected' in df.columns:
                df = df[df['selected'].astype(str).str.lower().isin(['true', '1'])]
            elif 'date' in df.columns:
                # Older format: first row per date is the selected one
                df = df.groupby('date', sort=False).first().reset_index()
            frames.append(df)
        except Exception as e:
            print(f'  Warning: could not read {p}: {e}')

    if not frames:
        raise RuntimeError('All sinfit CSVs failed to load.')

    combined = pd.concat(frames, ignore_index=True)
    # De-duplicate: one row per date (latest run wins)
    combined = combined.drop_duplicates(subset='date', keep='last')
    return combined.reset_index(drop=True)


def derive_labels(
    df          : pd.DataFrame,
    r2_thresh   : float,
    T_min       : float,
    T_max       : float,
    amp_min     : float,
    dur_min     : float,
    keep_ambig  : bool,
) -> pd.DataFrame:
    """
    Assign label=1 (LSTID) or label=0 (quiet) to each real-data day.

    Ambiguous days (fit exists but fails threshold, suggesting a marginal
    detection) are either kept as label=0 or dropped depending on keep_ambig.
    """
    df = df.copy()

    has_fit = df['r2'].notna()

    lstid_mask = (
        has_fit
        & (df['r2']          >= r2_thresh)
        & (df['T_hr']        >= T_min)
        & (df['T_hr']        <= T_max)
        & (df['amplitude_km'] >= amp_min)
    )

    if 'duration_hr' in df.columns:
        lstid_mask = lstid_mask & (df['duration_hr'] >= dur_min)

    df['label'] = 0
    df.loc[lstid_mask, 'label'] = 1

    if not keep_ambig:
        # Drop days where a fit was found but failed threshold (ambiguous)
        ambig = has_fit & ~lstid_mask
        df = df[~ambig].reset_index(drop=True)

    return df


def real_rows(
    df          : pd.DataFrame,
    hdf5_dir    : str,
    skip_missing: bool,
) -> list[dict]:
    """Convert the real sinfit DataFrame to manifest row dicts."""
    rows = []
    for _, r in df.iterrows():
        date_str = str(r['date'])
        fname    = f'rsd{date_str}.01.hdf5'
        fpath    = os.path.join(hdf5_dir, fname)

        if skip_missing and not os.path.exists(fpath):
            continue

        label = int(r['label'])
        row = {f: '' for f in MANIFEST_FIELDS}   # fill with empty by default
        row['date']       = date_str
        row['hdf5_file']  = fname
        row['source']     = 'real'
        row['label']      = label
        row['r2']         = r.get('r2', '')
        row['fitStart']   = r.get('fitStart', '')
        row['fitEnd']     = r.get('fitEnd', '')

        if label == 1:
            row['T_hr']        = r.get('T_hr', '')
            row['amplitude_km']= r.get('amplitude_km', '')
            # Convert phase from hours to radians if needed
            phase_hr = r.get('phase_hr', None)
            if pd.notna(phase_hr) and phase_hr != '':
                T_hr = r.get('T_hr', None)
                if pd.notna(T_hr) and T_hr and float(T_hr) > 0:
                    import math
                    phase_rad = (float(phase_hr) / float(T_hr)) * 2 * math.pi
                    row['phase_rad'] = phase_rad
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build a unified ML training manifest from synthetic and real data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--synthetic_manifest', type=str, default=None,
                        help='Path to manifest.csv from gen_synthetic_hdf5.py')
    parser.add_argument('--real_csv_dir',  type=str, default=None,
                        help='Directory of pipeline sinfit CSVs (output/summary_csv/)')
    parser.add_argument('--real_hdf5_dir', type=str, default='data/madrigal',
                        help='Directory containing real rsd*.hdf5 files')
    parser.add_argument('--output', type=str, default='data/ml_manifest.csv',
                        help='Output manifest CSV path')
    # Label threshold parameters
    parser.add_argument('--r2_thresh',      type=float, default=0.30,
                        help='Minimum R² for LSTID label')
    parser.add_argument('--T_min',          type=float, default=1.0,
                        help='Minimum LSTID period (hr)')
    parser.add_argument('--T_max',          type=float, default=4.5,
                        help='Maximum LSTID period (hr)')
    parser.add_argument('--amp_min',        type=float, default=30.0,
                        help='Minimum LSTID amplitude (km)')
    parser.add_argument('--dur_min',        type=float, default=2.0,
                        help='Minimum fit window duration (hr)')
    parser.add_argument('--keep_ambiguous', action='store_true',
                        help='Keep ambiguous days (fit exists but fails threshold) as label=0')
    parser.add_argument('--skip_missing_hdf5', action='store_true',
                        help='Skip real-data days whose HDF5 file is not found')
    args = parser.parse_args()

    if args.synthetic_manifest is None and args.real_csv_dir is None:
        parser.error('Provide at least one of --synthetic_manifest or --real_csv_dir')

    all_rows = []

    # --- Synthetic ---
    if args.synthetic_manifest:
        print(f'Loading synthetic manifest: {args.synthetic_manifest}')
        synth_df = pd.read_csv(args.synthetic_manifest)
        for _, r in synth_df.iterrows():
            row = {f: r.get(f, '') for f in MANIFEST_FIELDS}
            row['source'] = 'synthetic'
            all_rows.append(row)
        n_lstid  = int((synth_df['label'] == 1).sum())
        n_quiet  = int((synth_df['label'] == 0).sum())
        print(f'  {len(synth_df)} synthetic days  ({n_lstid} LSTID / {n_quiet} quiet)')

    # --- Real ---
    if args.real_csv_dir:
        print(f'Loading real sinfit CSVs from: {args.real_csv_dir}')
        sinfit_df = load_sinfit_csvs(args.real_csv_dir)
        sinfit_df = derive_labels(
            sinfit_df,
            r2_thresh  = args.r2_thresh,
            T_min      = args.T_min,
            T_max      = args.T_max,
            amp_min    = args.amp_min,
            dur_min    = args.dur_min,
            keep_ambig = args.keep_ambiguous,
        )
        real = real_rows(sinfit_df, args.real_hdf5_dir,
                         skip_missing=args.skip_missing_hdf5)
        n_lstid = sum(1 for r in real if r['label'] == 1)
        n_quiet = sum(1 for r in real if r['label'] == 0)
        print(f'  {len(real)} real days  ({n_lstid} LSTID / {n_quiet} quiet)')
        print(f'  Label thresholds: r2>={args.r2_thresh}, '
              f'T=[{args.T_min},{args.T_max}]hr, A>={args.amp_min}km, '
              f'dur>={args.dur_min}hr')
        all_rows.extend(real)

    # --- Write ---
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    total    = len(all_rows)
    n_lstid  = sum(1 for r in all_rows if str(r['label']) == '1')
    n_quiet  = sum(1 for r in all_rows if str(r['label']) == '0')
    n_synth  = sum(1 for r in all_rows if r['source'] == 'synthetic')
    n_real   = sum(1 for r in all_rows if r['source'] == 'real')

    print(f'\nManifest written: {args.output}')
    print(f'  Total: {total}  ({n_lstid} LSTID / {n_quiet} quiet)')
    print(f'  Sources: {n_synth} synthetic / {n_real} real')


if __name__ == '__main__':
    main()
