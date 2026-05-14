from abc import ABC, abstractmethod


class BaseSpotLoader(ABC):
    @abstractmethod
    def get_dataframe(self):
        """Return a Polars DataFrame with columns:
        date, freq, band, dist_Km, source,
        mid_lat, mid_long, rx_lat, tx_lat, rx_long, tx_long, freq_MHz
        """
        ...

    @abstractmethod
    def gen_histogram(self):
        """Return (hist2d: np.ndarray, meta: dict).

        meta must contain:
            time_bin_seconds, distance_bin_km,
            xedge_start, yedge_start_km, n_time, n_height
        """
        ...
