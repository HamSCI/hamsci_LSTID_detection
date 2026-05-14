#!/usr/bin/env python3
"""
build_ml_manifest.py — Combine synthetic and real data into a unified ML manifest.

Regression targets are T_hr and amplitude_km.  No binary labels — classification
of LSTID vs quiet happens downstream after the model predicts amplitude.

Manifest schema
---------------
  date           YYYY-MM-DD
  hdf5_file      path to HDF5 file (relative to its data_dir)
  source         'synthetic' | 'real'
  T_hr           sinusoid period (hr)
  amplitude_km   sinusoid amplitude (km)  ← primary regression target
  r2             sinfit R²  (real data only, empty for synthetic)
  [synthetic generation parameters — empty for real rows]

Usage
-----
  # Combine synthetic + real
  python build_ml_manifest.py \\
      --synthetic_manifest data/synthetic_hdf5/manifest.csv \\
      --real_csv_dir       output/summary_csv \\
      --real_hdf5_dir      data/madrigal \\
      --output             data/ml_manifest.csv

  # Synthetic only
  python build_ml_manifest.py \\
      --synthetic_manifest data/synthetic_hdf5/manifest.csv \\
      --output             data/ml_manifest.csv

  # Real only
  python build_ml_manifest.py \\
      --real_csv_dir   output/summary_csv \\
      --real_hdf5_dir  data/madrigal \\
      --output         data/ml_manifest.csv
"""

import argparse
import csv
import glob
import os

import pandas as pd

MANIFEST_FIELDS = [
    'date', 'hdf5_file', 'source',
    'T_hr', 'amplitude_km', 'r2',
    # synthetic generation parameters (empty for real rows)
    'seed',
    'prop_start_hr', 'prop_end_hr',
    'edge_E0_km', 'edge_tref_hr', 'edge_c1_kmph', 'edge_c2_kmph2',
    'n_spots_peak', 'band_km', 'scale_tail_km', 'p_band',
    'scale_below_km', 'p_above', 'background_frac', 'diurnal_peak_hr',
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_synthetic(manifest_path: str) -> list[dict]:
    df = pd.read_csv(manifest_path)
    rows = []
    for _, r in df.iterrows():
        row = {f: r.get(f, '') for f in MANIFEST_FIELDS}
        row['source'] = 'synthetic'
        rows.append(row)
    return rows


def load_real(csv_dir: str, hdf5_dir: str, skip_missing: bool) -> list[dict]:
    """
    Load pipeline sinfit CSVs and extract T_hr, amplitude_km, r2 per day.
    Only the selected (best) fit row per date is used.
    """
    pattern = os.path.join(csv_dir, '*.csv')
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f'No CSVs found in {csv_dir}')

    frames = []
    for p in paths:
        try:
            df = pd.read_csv(p)
            df.columns = df.columns.str.strip()
            if 'selected' in df.columns:
                df = df[df['selected'].astype(str).str.lower().isin(['true', '1'])]
            frames.append(df)
        except Exception as e:
            print(f'  Warning: could not read {p}: {e}')

    if not frames:
        raise RuntimeError('All sinfit CSVs failed to load.')

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset='date', keep='last').reset_index(drop=True)

    rows = []
    for _, r in combined.iterrows():
        date_str = str(r['date'])
        fname    = f'rsd{date_str}.01.hdf5'

        if skip_missing and not os.path.exists(os.path.join(hdf5_dir, fname)):
            continue

        row = {f: '' for f in MANIFEST_FIELDS}
        row['date']         = date_str
        row['hdf5_file']    = fname
        row['source']       = 'real'
        row['T_hr']         = r.get('T_hr', '')
        row['amplitude_km'] = r.get('amplitude_km', '')
        row['r2']           = r.get('r2', '')
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Build unified ML regression manifest from synthetic and real data.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--synthetic_manifest', type=str, default=None,
                        help='manifest.csv from gen_synthetic_hdf5.py')
    parser.add_argument('--real_csv_dir',       type=str, default=None,
                        help='Directory of pipeline sinfit CSVs (output/summary_csv/)')
    parser.add_argument('--real_hdf5_dir',      type=str, default='data/madrigal',
                        help='Directory containing real rsd*.hdf5 files')
    parser.add_argument('--output',             type=str, default='data/ml_manifest.csv',
                        help='Output manifest CSV path')
    parser.add_argument('--skip_missing_hdf5',  action='store_true',
                        help='Skip real-data days whose HDF5 file is not found')
    args = parser.parse_args()

    if args.synthetic_manifest is None and args.real_csv_dir is None:
        parser.error('Provide at least one of --synthetic_manifest or --real_csv_dir')

    all_rows = []

    if args.synthetic_manifest:
        rows = load_synthetic(args.synthetic_manifest)
        all_rows.extend(rows)
        print(f'Synthetic: {len(rows)} days  '
              f'(A range {min(float(r["amplitude_km"]) for r in rows):.0f}–'
              f'{max(float(r["amplitude_km"]) for r in rows):.0f} km)')

    if args.real_csv_dir:
        rows = load_real(args.real_csv_dir, args.real_hdf5_dir, args.skip_missing_hdf5)
        all_rows.extend(rows)
        amps = [float(r['amplitude_km']) for r in rows if r['amplitude_km'] != '']
        print(f'Real:      {len(rows)} days  '
              f'(A range {min(amps):.0f}–{max(amps):.0f} km  '
              f'mean {sum(amps)/len(amps):.0f} km)' if amps else f'Real: {len(rows)} days')

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f'\nManifest: {args.output}  ({len(all_rows)} total rows)')


if __name__ == '__main__':
    main()
