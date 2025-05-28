import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import ast
import os

TIF_LIST_FILE = "../data/selected_tif_metadata.txt"
STATIONS_FILE = "../data/selected_kriging_points_utm.csv"

def read_dsm_files(filename=TIF_LIST_FILE):
    with open(filename, 'r') as f:
        content = f.read()
        if content.startswith('dsm_files = '):
            content = content[len('dsm_files = '):]
        return ast.literal_eval(content)

def find_tif_for_station(easting, northing, tif_paths):
    for tif_path in tif_paths:
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            if (bounds.left <= easting <= bounds.right) and (bounds.bottom <= northing <= bounds.top):
                return tif_path
    return None

def create_profile_line(easting, northing, length=1000, orientation='E-W', num_points=200):
    half = length / 2
    if orientation == 'E-W':
        x_start, x_end = easting - half, easting + half
        y_start, y_end = northing, northing
    else:  # N-S
        x_start, x_end = easting, easting
        y_start, y_end = northing - half, northing + half
    x_coords = np.linspace(x_start, x_end, num_points)
    y_coords = np.linspace(y_start, y_end, num_points)
    return x_coords, y_coords

def sample_elevations(tif_path, x_coords, y_coords):
    elevations = []
    with rasterio.open(tif_path) as src:
        for x, y in zip(x_coords, y_coords):
            row, col = src.index(x, y)
            if 0 <= row < src.height and 0 <= col < src.width:
                elevation = src.read(1, window=rasterio.windows.Window(col, row, 1, 1))[0][0]
                elevations.append(elevation)
            else:
                elevations.append(np.nan)
    return np.array(elevations)

def plot_profile(distances, elevations, station_id, output_file, station_index):
    plt.figure(figsize=(10, 6))
    plt.plot(distances, elevations, 'b-', linewidth=2)
    plt.fill_between(distances, np.nanmin(elevations), elevations, alpha=0.3, color='brown')
    # Mark the station location (always at the center)
    plt.scatter([distances[station_index]], [elevations[station_index]], color='red', zorder=5, label='Station')
    plt.axvline(distances[station_index], color='red', linestyle='--', alpha=0.5)
    plt.text(distances[station_index], elevations[station_index], f"  Station {int(station_id)}", color='red', va='bottom', fontsize=10)
    plt.xlabel('Distance along profile (m)')
    plt.ylabel('Elevation (m)')
    plt.title(f'Terrain Profile through Station {int(station_id)}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved profile to {output_file}")

def main():
    stations = pd.read_csv(STATIONS_FILE, sep="|", names=["easting", "northing", "station_id", "d1", "d2"])
    dsm_files = read_dsm_files()
    tif_paths = [file_info['name'] for file_info in dsm_files]
    output_dir = "terrain_profiles"
    os.makedirs(output_dir, exist_ok=True)

    for idx, station in stations.iterrows():
        easting, northing, station_id = station['easting'], station['northing'], station['station_id']
        tif_path = find_tif_for_station(easting, northing, tif_paths)
        if tif_path is None:
            print(f"No TIF file found for station {station_id}")
            continue
        x_coords, y_coords = create_profile_line(easting, northing, length=1000, orientation='N-S', num_points=200)
        distances = np.linspace(-1000, 1000, 200)  # Centered at station
        elevations = sample_elevations(tif_path, x_coords, y_coords)
        station_index = len(distances) // 2  # Station is at the center
        output_file = os.path.join(output_dir, f"terrain_profile_station_{int(station_id)}.png")
        plot_profile(distances, elevations, station_id, output_file, station_index)

if __name__ == "__main__":
    main()
