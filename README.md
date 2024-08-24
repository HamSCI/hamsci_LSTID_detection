# hamsci_LSTID_detection
Code for automatically detecting Large Scale Traveling Ionospheric Disturbances (LSTIDs) from ham radio spot data.

Developed by the HamSCI NASA SWO2R Team with major contributions by:
* Nathaniel Frissell W2NAF
* Nicholas Callahan
* Diego Sanchez KD2RLM
* Bill Engelke AB4EJ
* Mary Lou West KC2NMC

To facilitate climatological studies of amateur radio-observed LSTIDs, a new, fully-automated detection and measurement algorithm was developed for this study. Briefly, amateur radio data is first gridded into 10 km x 1 min bins. After re-scaling, smoothing, and thresholding, a column-wise percentile is used to identify the bottom edge first-hop skip distance. A 15 min rolling coefficient of variation $CV = \sigma/\mu$ is computed on the raw detected edge for use as a quality parameter. The largest contiguous time period between 1330 and 2230 UTC where $CV < 0.5$ is selected. The raw detected edge within this time period is detrended using a least-squares best-fit second degree polynomial; a $1 < T < 5$ hr bandpass filter is then applied.

# Requirements
This code was tested on an x86 Ubuntu 22.04 LTS Linux machine with python v3.11.9 and the following libraries:
```
dask==2024.5.0
matplotlib==3.8.4
numpy==2.1.0
pandas==2.2.2
scipy==1.14.1
statsmodels==0.14.2
xarray==2023.6.0
```

# Instructions
1. Clone Github Repository
2. `pip install -e .`
3. Place raw spot data files into `raw_data` directory.
    1. Raw spot data should be bzip2 compressed daily files.
    2. Names should be in the form of: `2018-11-01_PSK.csv.bz2`, `2018-11-01_RBN.csv.bz2`, and `2018-11-01_WSPR.csv.bz2`, etc.
    3. Data files for 1 November 2018 - 30 April 2019 are availaible from https://doi.org/10.5281/zenodo.10673982.
4. Edit parameters in the top of `run_LSTID_detection.py`.
5. Run `./run_LSTID_detection.py`

# Notes
Using multiprocessing on a 64-thread machine with 512 GB RAM, this code takes about 12 minutes to process the 1 November 2018 - 30 April 2019 data from https://doi.org/10.5281/zenodo.10673982.

# Full Algorithm Description
## 1. Data Loading and Gridding
Data Loading and Gridding is handled by `hamsci_LSTID_detect.data_loading.RawSpotProcessor()`.
1. For each day, RBN, PSK, and WSPRNet spot data is combined into a single data frame.
2. Data is filtered based on frequency, TX-RX midpoint location, and TX-RX ground range. For Frissell et al. (2024, GRL), the following filters are used, which corresponds to 14 MHz signals over North America:
    1. 20˚ < lat < 60˚
    2. -160˚ < lon < -60˚
    3. 14 MHz < f < 15 MHz
    4. 0 km < R_gc < 3000 km
3. Filtered data is gridded into 10 km range by 1 minute bins.

## 2. Gridded Array Re-scaling
Gridded array re-scaling is handled by `hamsci_LSTID_detect.data_loading.create_xarr()`.
1. Data array is trimmed so that only daylight hours in North America are used (1200-2359 UTC).
2. A scaled version $M_{ad}$ of the gridded array $A$ is computed by `hamsci_LSTID_detect.data_loading.mad()` as follows:
$$M_{ad} = \frac{|A-\mbox{Med}(A)|}{\mbox{max}(\mbox{Med}(A),0.05)}$$

## 3. Skip Distance Edge-Detection
Skip distance edge-detection is handled by `hamsci_LSTID_detect.edge_detection.run_edge_detect()`.
https://github.com/w2naf/lstid_detection_hamSpots/blob/hamsci_LSTID_detect/hamsci_LSTID_detect/edge_detection.py#L172-L208C1
1. The x- and y- dimensions of the gridded array are trimmed by 8%.
2. A `scipy.ndimage.gaussian_filter()` with $\sigma=(4.2, 4.2)$ is applied to the gridded array.
3.  

## 4. Sine Fitting
A theoretical sinusoid is fit to the detected edge by `hamsci_LSTID_detect.edge_detection.run_edge_detect()`.
1. A 15 minute rolling coefficient of variation $CV = \sigma/\mu$ is computed on the raw detected edge for use as a quality parameter.
2. The largest contiguous time period between 1330 and 2230 UTC where $CV < 0.5$ is selected as "good".
3. A 2nd-degree polynomial is fit to the good period.
4. The raw detected edge within this time period is detrended using a least-squares best-fit second degree polynomial.
5. A $1 < T < 5$ hr bandpass filter is applied to the detrended edge.
6. This filtered, detrended result is curve-fit to $$A\sin(2\pi ft+\phi) + mt +b$$ to determine the LSTID amplitude $A$ and period $T=1/f$.
     1. `scipy.optimize.curve_fit()` is used as the curve fitter.
     2. The curve fit routine is run for each of the follwing intital period guesses: $T_{hr} = [1,1.5,2,2.5,3,3.5,4].
     3. Other parameter initial guesses are as follows:
        - `guess['amplitude_km']   = np.ptp(data_detrend)/2.`
        - `guess['phase_hr']       = 0.`
        - `guess['offset_km']      = np.mean(data_detrend)`
        - `guess['slope_kmph']     = 0.`
     4. The fit with the highest $r^2$ value is selected as the best fit.
  
# Acknowledgments
This work was supported by NASA Grants 80NSSC21K1772, 80NSSC23K0848 and United States National Science Foundation (NSF) Grant AGS-2045755.
