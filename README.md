# lstid_detection_hamSpots
Code for automatically detecting LSTIDs from ham radio spot data

Developed by the HamSCI NASA SWO2R Team with major contributions by:
 * Bill Engelke AB4EJ
 * Nicholas Callahan
 * Mary Lou West KC2NMC
 * Nathaniel Frissell W2NAF

To facilitate climatological studies of amateur radio-observed LSTIDs, a new, fully-automated detection and measurement algorithm was developed for this study. Briefly, amateur radio data is first gridded into 10 km x 1 min bins. After re-scaling, smoothing, and thresholding, a column-wise percentile is used to identify the bottom edge first-hop skip distance. A 15 min rolling coefficient of variation $CV = \sigma/\mu$ is computed on the raw detected edge for use as a quality parameter. The largest contiguous time period between 1330 and 2230 UTC where $CV < 0.5$ is selected. The raw detected edge within this time period is detrended using a least-squares best-fit second degree polynomial; a $1 < T < 5$ hr bandpass filter is then applied.
