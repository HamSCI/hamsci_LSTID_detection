# hamsci_LSTID_detection
[![DOI](https://zenodo.org/badge/847098909.svg)](https://zenodo.org/doi/10.5281/zenodo.13630866)

Automated detection of **Large Scale Traveling Ionospheric Disturbances (LSTIDs)** from amateur radio spot data (RBN, PSKReporter, WSPRNet) stored in Madrigal HDF5 format. The pipeline loads daily HDF5 files, builds 2D spot-density histograms, detects the HF skip-distance edge, and fits a sinusoid to characterise any LSTID present.

Developed by the HamSCI NASA Space Weather Operations to Research (SWO2R) Team:

- Nathaniel Frissell W2NAF
- Nicholas Callahan
- Diego Sanchez KD2RLM
- Bill Engelke AB4EJ
- Mary Lou West KC2NMC

---

## Requirements

Python 3.11+ on Linux (tested on Ubuntu 22.04 / WSL2).

```
polars
pyarrow
numpy
scipy
statsmodels
matplotlib
pandas
h5py
cartopy
pysolar
pillow
dask
```

Install all dependencies and the package in editable mode:

```bash
pip install -e .
```

---

## Data

Input data is Madrigal daily HDF5 files named `rsd{YYYY-MM-DD}.01.hdf5`.  
Nov 2018 – Apr 2019 data is available at **https://doi.org/10.5281/zenodo.10673981**.

Place files in the directory specified by `data_dir` in your config.

Expected HDF5 columns: `year month day hour min sec pthlen rxlat rxlon txlat txlon tfreq latcen loncen ssrc`

---

## Quick Start

```bash
# 1. Clone and install
git clone <repo>
cd hamsci_LSTID_detection
pip install -e .

# 2. Edit config.json (set data_dir, date range, region, frequency)
# 3. Run
python run_LSTID_detection.py -p config.json
```

---

## Configuration

All algorithm parameters are in a JSON config file. Key fields:

| Key | Description |
|-----|-------------|
| `data_dir` | Path to Madrigal HDF5 files |
| `cache_dir` | Parquet cache directory |
| `region_name` | Geographic region (see `scripts/regions.py`) |
| `freq` | Frequency band in MHz (e.g. `14`) or `"All Freq"` |
| `distance_range` | `{"min_dist": 0, "max_dist": 3000}` in km |
| `use_cache` | `true` to reuse prior runs |
| `chunk_size` | HDF5 rows per read chunk (default `500000`) |
| `n_workers` | Parallel worker processes (default: CPU count) |
| `preprocess` | MAD normalisation, Gaussian filter parameters |
| `edge_detection` | Quantile thresholds, LOWESS window, outlier cutoff |
| `fitting` | Bandpass limits, stability threshold, fit window margin |
| `plotting` | `ylim`, `cb_pad`, `output_dir` |

---

## Pipeline Architecture

```
HDF5 Files
    └─► HDF5PolarsLoader          stage 1: load & filter spots, build 2D histogram
            └─► preprocess_heatmap    stage 2: MAD normalise, Gaussian filter, uint8 rescale
                    └─► detect_edge       stage 3: quantile thresholding, LOWESS, edge selection
                            └─► sin_fit       stage 4: stability windowing, polynomial detrend,
                                              bandpass, sinusoidal curve fit
                                    └─► plots + CSV summary
```

Processing is **multiprocessed**: each day runs as an independent worker via `ProcessPoolExecutor`.

### Stage 1 — Data Loading (`scripts/hdf5_loader.py`)
`HDF5PolarsLoader` reads Madrigal HDF5 files in chunks (`chunk_size` rows at a time), applies geographic / frequency / distance filters with Polars, and bins spots into a 2D histogram (10 km × 1 min). Both the filtered DataFrame and histogram are cached as Parquet under `cache/dataframes/` and `cache/heatmaps/`.

### Stage 2 — Preprocessing (`scripts/heatmap_preprocess.py`)
Pads/crops to expected shape, trims to daylight hours, applies MAD normalisation column-by-column, 2D Gaussian filter (σ = 4.2), and rescales to uint8.

### Stage 3 — Edge Detection (`scripts/edge_detect.py`)
Builds quantile-based threshold stacks along the distance axis, smooths with LOWESS, removes outliers by absolute deviation, and selects the most stable edge.

### Stage 4 — Sinusoidal Fitting (`scripts/sinusoid_fitting.py`)
1. Rolling CV stability metric
2. Largest contiguous stable region with safety margins
3. 2nd-degree polynomial detrend
4. Optional 1–4.5 hr Butterworth bandpass
5. Multi-guess `curve_fit` across period candidates (1–4 hr), best R² selected

---

## Output

```
output/
├── daily_plots/
│   ├── {YYYYMMDD}_preprocessing.png        5-panel preprocessing stackplot
│   ├── {YYYYMMDD}_preprocessing_v2.png     same but with quantile threshold lines
│   ├── {YYYYMMDD}_sinfit.png               7-panel sinfit diagnostic
│   ├── {YYYYMMDD}_sinfit_v2.png            5-panel sinfit
│   ├── {YYYYMMDD}_sinfit_v3.png            5-panel sinfit (selected fit only)
│   └── panels/{YYYYMMDD}/                  13 standalone panel PNGs
├── thesis/
│   ├── full/{panel_name}/                  14 panels — all labels and titles intact
│   ├── strip_titles/{panel_name}/          14 panels — panel letter (a)(b)… removed
│   └── strip_all/{panel_name}/             14 panels — data only, no text decorations
└── summary_csv/
    └── {daterange}_{region}_{freq}_sinfit.csv   one row per fit attempt per day
```

### Summary CSV columns
`date · selected · T_hr · amplitude_km · phase_hr · offset_km · slope_kmph · r2 · fitStart · fitEnd · duration_hr · n_spots · t_load_cold_sec · t_load_cached_sec · t_preprocess_sec · t_edge_sec · t_fit_sec`

---

## Thesis-Quality Plots (`scripts/dfs_thesis.py`)

Three functions producing identical-size panels (11.0 × 4.5 in axes, fixed `bbox_inches=None`) across all 14 panel types:

| Function | Folder | Description |
|----------|--------|-------------|
| `thesis_plot_all_panels_full` | `thesis/full/` | Everything kept — panel letters, titles, all labels and ticks |
| `thesis_plot_all_panels` | `thesis/strip_titles/` | Panel letter `(a)(b)…` removed, all labels kept |
| `thesis_plot_all_panels_no_labels` | `thesis/strip_all/` | Data only — no titles, labels, ticks, or colorbar text |

All three are called automatically by the pipeline for every processed day.

---

## Synthetic Data Generation

### Generate synthetic HDF5 files

```bash
python gen_synthetic_hdf5.py --n_days 500 --lstid_frac 0.5 --output_dir data/synthetic_hdf5
```

Produces Madrigal-format HDF5 files (dates 2030-01-01 onwards) that pass through the pipeline unchanged. The physical model:

- **Edge position**: diurnal quadratic baseline + optional LSTID sinusoid
- **Spot distribution**: uniform across a propagation band (800–2000 km) above the edge + exponential tail; sparse leakage below
- **Diurnal count envelope**: raised-cosine (Hann) window, exactly zero before ~12 UTC and at 24 UTC, peak variable 15.5–20.5 UTC — matching real 14 MHz / CONUS daylight propagation
- **All parameters randomised** per day within physically realistic bounds

Writes `manifest.csv` with all ground-truth generation parameters and labels.

### Build unified ML training manifest

```bash
python build_ml_manifest.py \
    --synthetic_manifest data/synthetic_hdf5/manifest.csv \
    --real_csv_dir       output/summary_csv \
    --real_hdf5_dir      data/madrigal \
    --output             data/ml_manifest.csv \
    --skip_missing_hdf5
```

Combines synthetic ground-truth labels with real-data pseudo-labels derived from the pipeline's sinfit results (`r² ≥ 0.3`, `T ∈ [1, 4.5]` hr, `amplitude ≥ 30` km). Produces a single manifest with consistent schema for both sources.

---

## Tests

```bash
python -m pytest tests/ -v                          # all tests
python -m pytest tests/test_hdf5_loader.py -v       # loader unit tests
python -m pytest tests/test_pipeline_integration.py # end-to-end with synthetic HDF5
```

Test suite covers: loader validation, filter logic, DataFrame schema, histogram generation, preprocessing output, edge detection, all sinfit sub-functions, and a two-day integration test (LSTID day vs. quiet day).

---

## Acknowledgments

This work was supported by NASA Grants 80NSSC21K1772, 80NSSC23K0848 and United States National Science Foundation (NSF) Grant AGS-2045755.
