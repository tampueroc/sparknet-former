import torch
import numpy as np
import rioxarray
import pandas as pd

class LandscapeNormalize:
    """
    Loads a geotiff using rioxarray, replaces no-data values, and normalizes each channel.
    By default, it will skip normalizing strictly binary channels.
    """

    def __init__(self, no_data_value=-9999.0, skip_binary=True):
        """
        Args:
            no_data_value (float): Values to treat as "no data" and replace.
            skip_binary (bool): If True, channels that are strictly {0,1} or {-1,1} won't be normalized.
        """
        self.no_data_value = no_data_value
        self.skip_binary = skip_binary

        # These will be set after reading data
        self.landscape_data = None
        self.landscape_max = None
        self.landscape_min = None

    def __call__(self, landscape_path):
        """
        Loads and normalizes the geotiff file.

        Args:
            landscape_path (str): Path to the input geotiff.

        Returns:
            xarray.DataArray: The normalized landscape data (channels first).
        """
        with rioxarray.open_rasterio(landscape_path) as src:
            # Replace no-data with -1
            data = src.where(src != self.no_data_value, -1)
            # Compute min/max per channel
            self.landscape_max = data.max(dim=["x", "y"]).values
            self.landscape_min = data.min(dim=["x", "y"]).values

            # Normalize each channel
            for i in range(len(self.landscape_max)):
                channel = data[i, :, :].values
                unique_vals = np.unique(channel)
                if self.skip_binary and (
                    np.array_equal(unique_vals, [0, 1]) or
                    np.array_equal(unique_vals, [-1, 1])
                ):
                    # Skip normalizing strictly binary channels
                    continue
                denom = (self.landscape_max[i] - self.landscape_min[i])
                if denom != 0:
                    channel = (channel - self.landscape_min[i]) / denom
                data[i, :, :] = channel

            self.landscape_data = data
            return data


class WeatherNormalize:
    """
    Stores global min/max for wind speed (WS) and wind direction (WD) across all CSVs,
    then normalizes wind data on demand.
    """
    def __init__(self):
        self.max_ws = float('-inf')
        self.min_ws = float('inf')
        self.max_wd = float('-inf')
        self.min_wd = float('inf')

        # Dictionary to hold raw weather data for each file
        self.weathers = {}

    def fit_transform(self, weather_folder, weather_history_path):
        """
        Loads all CSVs in `weather_folder`, reads `weather_history.csv`, and finds global min/max.
        Args:
            weather_folder (str): path to folder with CSV files (e.g., WS and WD).
            weather_history_path (str): CSV file that references these weather files.
        Returns:
            (dict, pd.DataFrame): (loaded weather dataframes, weather history),
                                  with global min/max stored internally.
        """
        # Read weather history
        weather_history = pd.read_csv(weather_history_path, header=None)

        # Load each CSV in the folder
        import os
        files = os.listdir(weather_folder)
        for fname in files:
            fullpath = os.path.join(weather_folder, fname)
            df = pd.read_csv(fullpath)
            self.weathers[fname] = df

            # Update global min/max
            self.max_ws = max(self.max_ws, df['WS'].max())
            self.min_ws = min(self.min_ws, df['WS'].min())
            self.max_wd = max(self.max_wd, df['WD'].max())
            self.min_wd = min(self.min_wd, df['WD'].min())

        return self.weathers, weather_history

    def __call__(self, wind_speed, wind_direction):
        """
        Normalize a single pair of (WS, WD).
        """
        ws_norm = (wind_speed - self.min_ws) / (self.max_ws - self.min_ws)
        wd_norm = (wind_direction - self.min_wd) / (self.max_wd - self.min_wd)
        return torch.tensor([ws_norm, wd_norm], dtype=torch.float32)

