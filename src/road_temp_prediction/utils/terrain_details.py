import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import ast
import os
from matplotlib.colors import LightSource
from matplotlib import cm  # Import the colormap module

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

def plot_profile(distances, elevations, station_id, output_file, station_index,orientation):
    plt.figure(figsize=(10, 6))
    plt.plot(distances, elevations, 'b-', linewidth=2)
    plt.fill_between(distances, np.nanmin(elevations), elevations, alpha=0.3, color='brown')
    # Mark the station location (always at the center)
    plt.scatter([distances[station_index]], [elevations[station_index]], color='red', zorder=5, label='Station')
    plt.axvline(distances[station_index], color='red', linestyle='--', alpha=0.5)
    plt.text(distances[station_index], elevations[station_index], f"  Station {int(station_id)}", color='red', va='bottom', fontsize=10)
    plt.xlabel('Distance along profile (m)')
    plt.ylabel('Elevation (m)')
    plt.title(f'{orientation} terrain Profile through Station {int(station_id)}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()
    print(f"Saved profile to {output_file}")

def plot_terrain_height(tif_path, station_id, easting, northing, output_file, buffer_distance=2000):
    """
    Plot the terrain height around a station using the same TIF file used for the profile.

    Args:
        tif_path: Path to the TIF file
        station_id: ID of the station
        easting: Easting coordinate of the station
        northing: Northing coordinate of the station
        output_file: Path to save the output plot
        buffer_distance: Distance in meters around the station to plot
    """
    with rasterio.open(tif_path) as src:
        # Calculate the window to read based on buffer distance
        station_row, station_col = src.index(easting, northing)

        # Convert buffer distance from meters to pixels
        pixel_size = src.res[0]  # Assuming square pixels
        buffer_pixels = int(buffer_distance / pixel_size)

        # Define window bounds ensuring they're within the raster
        row_start = max(0, station_row - buffer_pixels)
        row_end = min(src.height, station_row + buffer_pixels)
        col_start = max(0, station_col - buffer_pixels)
        col_end = min(src.width, station_col + buffer_pixels)

        # Read the data
        window = rasterio.windows.Window(col_start, row_start, col_end - col_start, row_end - row_start)
        elevation = src.read(1, window=window)

        # Get the spatial coordinates for the window
        window_transform = rasterio.windows.transform(window, src.transform)
        x_min, y_max = window_transform * (0, 0)
        x_max, y_min = window_transform * (window.width, window.height)

        # Create the figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create a shaded relief
        ls = LightSource(azdeg=315, altdeg=45)
        # Use the colormap object instead of the string
        shaded_relief = ls.shade(elevation, cmap=cm.terrain, vert_exag=3, blend_mode='soft')

        # Plot the terrain
        im = ax.imshow(elevation, extent=[x_min, x_max, y_min, y_max], cmap='terrain', alpha=0.7)
        ax.imshow(shaded_relief, extent=[x_min, x_max, y_min, y_max], alpha=0.7)

        # Add a colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Elevation (m)')

        # Mark the station location
        ax.scatter([easting], [northing], color='red', s=100, marker='*', label=f'Station {int(station_id)}')

        # Add N-S and E-W profile lines
        ax.axhline(y=northing, color='blue', linestyle='--', alpha=0.5, label='E-W Profile')
        ax.axvline(x=easting, color='green', linestyle='--', alpha=0.5, label='N-S Profile')

        # Set the title and labels
        ax.set_title(f'Terrain Height around Station {int(station_id)}')
        ax.set_xlabel('Easting (m)')
        ax.set_ylabel('Northing (m)')

        # Add a legend
        ax.legend()

        # Add a north arrow
        ax.text(x_max - 200, y_min + 200, 'N', fontsize=12, ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.5, boxstyle='circle'))
        ax.arrow(x_max - 200, y_min + 150, 0, 100, head_width=50, head_length=50,
                fc='black', ec='black')

        # Add a scale bar
        scale_bar_length = 500  # meters
        ax.plot([x_min + 200, x_min + 200 + scale_bar_length], [y_min + 200, y_min + 200], 'k-', lw=2)
        ax.text(x_min + 200 + scale_bar_length/2, y_min + 250, f'{scale_bar_length} m',
                ha='center', va='center', bbox=dict(facecolor='white', alpha=0.5))

        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"Saved terrain height plot to {output_file}")

def main(orientation):
    stations = pd.read_csv(STATIONS_FILE, sep="|", names=["easting", "northing", "station_id", "d1", "d2"])
    dsm_files = read_dsm_files()
    tif_paths = [file_info['name'] for file_info in dsm_files]

    # Create output directories
    profile_dir = f"../../../results/figures/terrain_profiles_{orientation}"
    terrain_dir = "../../../results/figures/terrain_heights"
    os.makedirs(profile_dir, exist_ok=True)
    os.makedirs(terrain_dir, exist_ok=True)

    for idx, station in stations.iterrows():
        easting, northing, station_id = station['easting'], station['northing'], station['station_id']
        tif_path = find_tif_for_station(easting, northing, tif_paths)

        if tif_path is None:
            print(f"No TIF file found for station {station_id}")
            continue

        print(f"Processing station {station_id} using TIF file: {tif_path}")

        # Create and save the profile plot
        x_coords, y_coords = create_profile_line(easting, northing, length=1000, orientation=orientation, num_points=200)
        distances = np.linspace(-500, 500, 200)  # Centered at station
        elevations = sample_elevations(tif_path, x_coords, y_coords)
        station_index = len(distances) // 2  # Station is at the center
        profile_output = os.path.join(profile_dir, f"terrain_profile_station_{int(station_id)}.png")
        plot_profile(distances, elevations, station_id, profile_output, station_index,orientation)

        # Create and save the terrain height plot
        terrain_output = os.path.join(terrain_dir, f"terrain_height_station_{int(station_id)}.png")
        plot_terrain_height(tif_path, station_id, easting, northing, terrain_output)

# For testing with a small sample
def test_with_sample():
    # Create a sample station
    sample_station = pd.DataFrame({
        'easting': [550000],
        'northing': [6200000],
        'station_id': [9999]
    })

    # Use a sample TIF file
    sample_tif = "path/to/sample.tif"

    # Create output directories
    profile_dir = "terrain_profiles"
    terrain_dir = "terrain_heights"
    os.makedirs(profile_dir, exist_ok=True)
    os.makedirs(terrain_dir, exist_ok=True)

    # Process the sample station
    easting, northing, station_id = sample_station.iloc[0]

    # Create and save the profile plot
    x_coords, y_coords = create_profile_line(easting, northing, length=1000, orientation='N-S', num_points=200)
    distances = np.linspace(-500, 500, 200)
    elevations = np.random.normal(100, 10, 200)  # Simulated elevations
    station_index = len(distances) // 2
    profile_output = os.path.join(profile_dir, f"terrain_profile_station_{int(station_id)}.png")
    plot_profile(distances, elevations, station_id, profile_output, station_index,orientation)

    # Note: We can't test the terrain height plot without a real TIF file

if __name__ == "__main__":
    # Uncomment to run the main function with real data
    orientation="N-S"
    main(orientation)

    # Uncomment to run the test with sample data
    # test_with_sample()

