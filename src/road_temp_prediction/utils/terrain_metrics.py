#!/usr/bin/env python3
"""
terrain_metrics.py
Compute elevation, slope, aspect, distance-to-ridge/valley and lat/lon
for a set of UTM stations using individual GeoTIFF DEM tiles.
"""

# ---- imports
import pathlib
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from scipy.ndimage import gaussian_filter, distance_transform_edt
from skimage.morphology import local_maxima, local_minima
from pyproj import Transformer
import ast

# ---- user params
TIF_LIST_FILE = "selected_tif_metadata.txt" #paths to DEM tiles
STATIONS_FILE = "../data/selected_kriging_points_utm.csv"
CRS_UTM = "EPSG:25832"    # <- change to your UTM zone
OUT_CSV = "station_metrics_kriging_points.csv"

# Ridge / valley extraction tweaks
SMOOTH_SIGMA = 1.0    # Gaussian blur in pixels
RIDGE_VALLEY_DT = False    # Turn off if you do not need distances

# ---- helpers
def slope_aspect(dem, dx, dy):
    """Return slope (deg) and aspect (deg clockwise from north)."""
    dz_dy, dz_dx = np.gradient(dem, dy, dx)  # note order!
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect = np.degrees(np.arctan2(-dz_dx, dz_dy))  # GIS convention
    aspect = (aspect + 360) % 360
    return slope, aspect

def ridges_valleys(dem):
    """Boolean masks for ridge and valley cells."""
    smooth = gaussian_filter(dem, sigma=SMOOTH_SIGMA)
    return local_maxima(smooth), local_minima(smooth)

def read_dsm_files(filename='list_of_tif.txt'):
    with open(filename, 'r') as f:
        content = f.read()
        # Remove the variable assignment part if present
        if content.startswith('dsm_files = '):
            content = content[len('dsm_files = '):]
        # Safely evaluate the Python literal
        return ast.literal_eval(content)

def process_station(station_data, tif_path):
    """Process a single station using one TIF file."""
    with rasterio.open(tif_path) as src:
        # Check if station is within bounds of this TIF
        x, y = station_data['easting'], station_data['northing']

        # Convert coordinates to pixel coordinates
        row, col = src.index(x, y)

        # Check if point is within raster bounds
        if (0 <= row < src.height and 0 <= col < src.width):
            # Read a small window around the point (for gradient calculation)
            window_size = 3  # Adjust as needed for slope calculation
            window = Window(
                col - window_size // 2,
                row - window_size // 2,
                window_size,
                window_size
            )

            # Ensure window is within bounds
            window = window.intersection(Window(0, 0, src.width, src.height))

            # Read the data
            dem_window = src.read(1, window=window)

            # Get the elevation at the exact point
            elevation = src.read(1, window=Window(col, row, 1, 1))[0][0]

            # Calculate slope and aspect
            res_x, res_y = src.res
            slope, aspect = slope_aspect(dem_window, res_x, res_y)

            # Get central pixel values (where our point is)
            center_idx = window_size // 2
            slope_val = slope[center_idx, center_idx]
            aspect_val = aspect[center_idx, center_idx]

            # Calculate ridge/valley distances if needed
            dist2ridge, dist2valley = None, None
            if RIDGE_VALLEY_DT:
                ridge_mask, valley_mask = ridges_valleys(dem_window)
                dist2ridge = distance_transform_edt(~ridge_mask)[center_idx, center_idx] * res_x
                dist2valley = distance_transform_edt(~valley_mask)[center_idx, center_idx] * res_x

            return {
                'elev_m': elevation,
                'slope_deg': slope_val,
                'aspect_deg': aspect_val,
                'dist2ridge_m': dist2ridge,
                'dist2valley_m': dist2valley,
                'found': True
            }

    return {'found': False}

def main():
    # Read station data
    stations = pd.read_csv(STATIONS_FILE, sep="|",
                          names=["easting", "northing", "station_id", "d1", "d2"])

    # Read TIF file list
    dsm_files = read_dsm_files()
    tif_paths = [file_info['name'] for file_info in dsm_files]

    # Initialize results columns
    stations['elev_m'] = np.nan
    stations['slope_deg'] = np.nan
    stations['aspect_deg'] = np.nan
    if RIDGE_VALLEY_DT:
        stations['dist2ridge_m'] = np.nan
        stations['dist2valley_m'] = np.nan

    # Process each station
    for idx, station in stations.iterrows():
        station_processed = False

        # Try each TIF file until we find the one containing our station
        for tif_path in tif_paths:
            result = process_station(station, tif_path)

            if result['found']:
                stations.at[idx, 'elev_m'] = result['elev_m']
                stations.at[idx, 'slope_deg'] = result['slope_deg']
                stations.at[idx, 'aspect_deg'] = result['aspect_deg']
                if RIDGE_VALLEY_DT:
                    stations.at[idx, 'dist2ridge_m'] = result['dist2ridge_m']
                    stations.at[idx, 'dist2valley_m'] = result['dist2valley_m']
                station_processed = True
                break

        if not station_processed:
            print(f"Warning: Station {station['station_id']} not found in any TIF file")

    # Convert UTM to lat/lon
    tran = Transformer.from_crs(CRS_UTM, "EPSG:4326", always_xy=True)
    lons, lats = tran.transform(stations["easting"].values, stations["northing"].values)
    stations["lon"], stations["lat"] = lons, lats

    # Export results
    output_columns = ["station_id", "easting", "northing",
                     "lat", "lon",
                     "elev_m", "slope_deg", "aspect_deg"]
    if RIDGE_VALLEY_DT:
        output_columns.extend(["dist2ridge_m", "dist2valley_m"])

    stations[output_columns].to_csv(OUT_CSV, index=False)
    print(f"\nSaved metrics for {len(stations)} stations -> {OUT_CSV}")

if __name__ == "__main__":
    main()
