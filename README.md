# hamsci_LSTID_detection

Automated detection of **Large Scale Traveling Ionospheric Disturbances (LSTIDs)** from amateur radio spot data (RBN, PSKReporter, WSPRNet) stored in Madrigal HDF5 format.

Developed by the HamSCI NASA Space Weather Operations to Research (SWO2R) Team:

- Nathaniel Frissell W2NAF
- Nicholas Callahan
- Diego Sanchez KD2RLM
- Bill Engelke AB4EJ
- Mary Lou West KC2NMC

# Requirements

Python 3.11+ on Linux (tested on Ubuntu 22.04 / WSL2).

```
cartopy==0.24.1
dask==2024.8.2
h5py==3.11.0
matplotlib==3.10.8
numpy==1.26.4
pandas==2.2.2
pillow==12.1.1
polars==1.35.1
pyarrow==16.1.0
pysolar==0.13
scipy==1.13.1
statsmodels==0.14.2
```

# Instructions

1. Clone the repository and install:
   ```bash
   pip install -e .
   ```
2. Place Madrigal HDF5 data files into the directory specified by `data_dir` in your config.
   - Files must be named `rsd{YYYY-MM-DD}.01.hdf5`.
3. Create a config file in the `config/` directory. Each config file defines a complete experiment (date range, region, frequency band, data paths, and all algorithm parameters). Use `config/config_test.json` as a starting template.
4. Run the pipeline, passing your config:
   ```bash
   python run_LSTID_detection.py -p config/config_test.json
   ```

The `config/` directory is intended to hold one JSON file per experiment or parameter set, making runs fully reproducible and easy to track. Key config options: `data_dir`, `cache_dir`, `region_name` (see `scripts/regions.py`), `freq` (MHz), `distance_range`, `use_cache`, `n_workers`.

# Algorithm Description

## 1. Data Loading (`scripts/hdf5_loader.py`)
Reads Madrigal HDF5 files in chunks, applies geographic, frequency, and distance filters using Polars, and bins spots into a 2D histogram (10 km × 1 min bins). Results are cached as Parquet files.

## 2. Preprocessing (`scripts/heatmap_preprocess.py`)
Trims to daylight hours, applies column-wise MAD normalisation, a 2D Gaussian filter (σ = 4.2), and rescales to uint8.

## 3. Edge Detection (`scripts/edge_detect.py`)
Builds quantile threshold stacks (q = 0.4, 0.5, 0.6), smooths with LOWESS, removes outliers by absolute deviation, and selects the most stable edge.

## 4. Sinusoidal Fitting (`scripts/sinusoid_fitting.py`)
1. Computes a 15-minute rolling coefficient of variation (CV) on the detected edge.
2. Selects the largest contiguous stable region (CV < 0.05) between 1300–2300 UTC.
3. Fits a 2nd-degree polynomial to detrend the edge.
4. Applies an optional 1–4.5 hr Butterworth bandpass filter.
5. Runs `curve_fit` for period guesses T = [1, 1.5, 2, 2.5, 3, 3.5, 4] hr; best R² selected.

Processing is multiprocessed — each day runs as an independent worker via `ProcessPoolExecutor`.

# Plotting

Plotting is handled by modules in `scripts/`. The general structure has two layers:

- **Panel functions** (`plot_subplots.py`) — individual plot panels that draw onto axes passed in explicitly. These are the building blocks.
- **Composite plot functions** (`plot_lstid_paper.py`, `dfs_thesis.py`) — assemble one or more panels into a full figure and save it to disk. Each function takes `fit_result`, optionally `df`, and `**plot_params` (unpacked from the `plotting` block in your config).

To enable or add a plot, edit the plotting block inside `process_one_day()` in `run_LSTID_detection.py`, immediately after the `sin_fit()` call:

```python
fit_result = sin_fit(edge_result, **fit_params)

# ← add or uncomment plot calls here
plot_all_panels_individual(fit_result, df, **plot_params)
thesis_plot_all_panels_full(fit_result, df, **plot_params)
```

Any composite function imported from the plotting modules can be dropped in at this point. The `**plot_params` dict supplies `output_dir`, `ylim`, and `cb_pad` from the config automatically.

# Synthetic Data Generation

Generates Madrigal-format HDF5 files compatible with the pipeline for ML training. Dates use 2030+ to avoid collision with real data.

```bash
# Generate synthetic HDF5 files
python synthetic_data_gen/gen_synthetic_hdf5.py --n_days 200 --output_dir data/synthetic_hdf5

# Build unified ML training manifest (synthetic + real pipeline output)
python synthetic_data_gen/build_ml_manifest.py \
    --synthetic_manifest data/synthetic_hdf5/manifest.csv \
    --real_csv_dir output/summary_csv \
    --real_hdf5_dir data/madrigal \
    --output data/ml_manifest.csv --skip_missing_hdf5
```

Each synthetic day embeds a sinusoidal LSTID of randomised amplitude (0–100 km) and period (1–4.5 hr) on a diurnal quadratic baseline. 20% of days are flat (A = 0), 20% are high-noise, and 30% contain random data gaps. The manifest records `T_hr` and `amplitude_km` as regression targets alongside a `curriculum_phase` label (1 = clear, 2 = ambiguous). Use `--phase1_max_amp` and `--phase1_min_amp` to control the phase boundary.

# Tests

```bash
python -m pytest tests/ -v
```

# Acknowledgments

This work was supported by NASA Grants 80NSSC21K1772, 80NSSC23K0848 and United States National Science Foundation (NSF) Grant AGS-2045755.
