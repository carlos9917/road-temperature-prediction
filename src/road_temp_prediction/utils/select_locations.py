"""
Script to select locations from a map using
the coordinates of the weather stations.
The selected points are dumped to a csv file
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from matplotlib.widgets import Button
import io
import urllib.request
from PIL import Image
import sqlite3
import os
from datetime import datetime

def load_and_merge_data_optimized(variables, year, DB):
    """
    Load data from SQLite databases and merge into a single dataframe.
    Uses all available data without day limits.
    """
    try:
        dataframes = []

        for variable in variables:
            db_path = os.path.join(DB, f'OBSTABLE_{variable}_{year}.sqlite')
            if not os.path.exists(db_path):
                print(f"Warning: Database file not found: {db_path}")
                continue

            conn = sqlite3.connect(db_path)
            # Optimize SQLite performance
            conn.execute('PRAGMA synchronous = OFF')
            conn.execute('PRAGMA journal_mode = MEMORY')
            query = f"SELECT valid_dttm, SID, lat, lon, {variable} FROM SYNOP"

            try:
                # For demonstration, we'll use a small sample
                df = pd.read_sql_query(query, conn)
                dataframes.append(df)
            except sqlite3.Error as e:
                print(f"SQLite error when reading {variable}: {e}")
            finally:
                conn.close()

        if not dataframes:
            raise ValueError("No data loaded from database")

        # For demonstration, we'll create a sample dataframe
        if not dataframes:
            # Create sample data for demonstration
            print("Creating sample data for demonstration")
            sample_data = {
                'valid_dttm': [1673406000] * 10,  # Jan 11, 2023
                'SID': [f'S{i}' for i in range(10)],
                'lat': np.random.uniform(55.0, 58.0, 10),  # Sample coordinates for Denmark
                'lon': np.random.uniform(8.0, 13.0, 10),
                'TROAD': np.random.uniform(-5, 5, 10)
            }
            return pd.DataFrame(sample_data)

        # Merge all dataframes
        full_df = pd.concat(dataframes, ignore_index=True)
        merged_df = full_df.groupby(['valid_dttm', 'SID', 'lat', 'lon']).first().reset_index()
        return merged_df
    except (sqlite3.Error, ValueError) as e:
        print(f"Error loading data: {str(e)}")
        # Create sample data for demonstration
        print("Creating sample data for demonstration")
        sample_data = {
            'valid_dttm': [1673406000] * 10,  # Jan 11, 2023
            'SID': [f'S{i}' for i in range(10)],
            'lat': np.random.uniform(55.0, 58.0, 10),  # Sample coordinates for Denmark
            'lon': np.random.uniform(8.0, 13.0, 10),
            'TROAD': np.random.uniform(-5, 5, 10)
        }
        return pd.DataFrame(sample_data)


# Load the data
DB = "../data/OBSTABLE"  # This path won't be used in demo mode
variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
year = 2023

print("Loading data (or creating sample data for demonstration)...")
df = load_and_merge_data_optimized(variables, year, DB)


## select only the ones I want

# data dumped from the terrain geometry analysis
station_gis_metrics = pd.read_csv('../data/station_metrics.csv')

#drop these columns from the gis data, since they are already included in df
columns_to_drop = ['lat', 'lon']
cleaned = station_gis_metrics.drop(columns=columns_to_drop)

merged = df.merge(
    cleaned,
    left_on='SID',
    right_on='station_id',
    how='inner'
)

del df
df = merged.copy()
del df
df = merged.copy()
# Convert timestamp to datetime
df["dates"] = pd.to_datetime(df["valid_dttm"], unit="s")
date_chosen = datetime(2023, 1, 11, 2) # choosing a particular date here

# Filter for a specific date if needed
# In a real scenario, you would use your actual date filtering
try:
    df_filtered = df[df["dates"] == date_chosen]
    if len(df_filtered) == 0:
        raise ValueError("No data for the specified date")
    df = df_filtered
except:
    print(f"No data found for {date_chosen}")
    exit(1)

print(f"Data loaded with {len(df)} stations")
print("Opening interactive map. Click on the map to select points.")


def get_map_image(bbox, zoom=8):
    """
    Get a map image from OpenStreetMap for the given bounding box.
    
    Args:
        bbox: (min_lon, min_lat, max_lon, max_lat)
        zoom: zoom level (1-19)
    
    Returns:
        PIL Image object
    """
    min_lon, min_lat, max_lon, max_lat = bbox
    
    # Calculate OSM tile coordinates
    def deg2num(lat_deg, lon_deg, zoom):
        lat_rad = np.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - np.log(np.tan(lat_rad) + (1 / np.cos(lat_rad))) / np.pi) / 2.0 * n)
        return (xtile, ytile)
    
    # Get tile coordinates for the corners
    x1, y1 = deg2num(max_lat, min_lon, zoom)
    x2, y2 = deg2num(min_lat, max_lon, zoom)
    
    # Ensure we don't request too many tiles
    if abs(x2 - x1) > 5 or abs(y2 - y1) > 5:
        print("Warning: Large area requested, limiting to 5x5 tiles")
        x2 = min(x1 + 5, x2)
        y2 = min(y1 + 5, y2)
    
    # Download and stitch tiles
    width, height = (x2 - x1 + 1) * 256, (y2 - y1 + 1) * 256
    map_img = Image.new('RGB', (width, height))
    
    for x in range(x1, x2 + 1):
        for y in range(y1, y2 + 1):
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            headers = {'User-Agent': 'Mozilla/5.0 (compatible; MapPlotting/1.0)'}
            req = urllib.request.Request(url, headers=headers)
            
            try:
                with urllib.request.urlopen(req) as response:
                    tile_img = Image.open(io.BytesIO(response.read()))
                    map_img.paste(tile_img, ((x - x1) * 256, (y - y1) * 256))
            except Exception as e:
                print(f"Error downloading tile {x},{y}: {e}")
                # Create a gray tile as placeholder
                tile_img = Image.new('RGB', (256, 256), (200, 200, 200))
                map_img.paste(tile_img, ((x - x1) * 256, (y - y1) * 256))
    
    return map_img

def interactive_map_with_clicks(df, value_col='TROAD'):
    """
    Create an interactive matplotlib map where users can click to get coordinates.
    Now with a map background.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get the bounding box with some padding
    min_lon, max_lon = df['lon'].min() - 0.5, df['lon'].max() + 0.5
    min_lat, max_lat = df['lat'].min() - 0.5, df['lat'].max() + 0.5
    bbox = (min_lon, min_lat, max_lon, max_lat)
    
    # Get the map image
    print("Downloading map tiles...")
    map_img = get_map_image(bbox)
    
    # Display the map
    ax.imshow(map_img, extent=[min_lon, max_lon, min_lat, max_lat], aspect='auto', alpha=0.7)
    
    # Plot the stations with their values
    scatter = ax.scatter(df['lon'], df['lat'], c=df[value_col], cmap='coolwarm', 
                         s=50, alpha=0.8, edgecolors='k')
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label(value_col)
    
    # Add station labels
    for i, row in df.iterrows():
        ax.annotate(str(row['SID']), (row['lon'], row['lat']), 
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Click on the map to select points\n'
                f'Existing stations shown with {value_col} values')
    
    # Create a list to store selected points
    selected_points = []
    
    # Create a text object to display coordinates
    coord_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, 
                        bbox=dict(facecolor='white', alpha=0.8))
    
    # Create a scatter plot for selected points (initially empty)
    selected_scatter = ax.scatter([], [], c='red', s=100, marker='x')

    # Initialize counter variable
    counter = 0
    
    # Function to update the selected points scatter plot
    def update_selected_scatter():
        if selected_points:
            selected_df = pd.DataFrame(selected_points)
            selected_scatter.set_offsets(selected_df[['lon', 'lat']].values)
        else:
            selected_scatter.set_offsets(np.empty((0, 2)))
        fig.canvas.draw_idle()
    
    # Function to handle clicks
    def onclick(event):
        nonlocal counter
        if event.inaxes == ax:
            lon, lat = event.xdata, event.ydata
            counter += 1
            dummy_station_name = "9999"+str(counter).zfill(2)
            selected_points.append({'SID': dummy_station_name, 'name': 'dummy station', 'lon': lon, 'lat': lat})
            
            # Update the text display
            points_str = '\n'.join([f"Point {i+1}: Lat={p['lat']:.6f}, Lon={p['lon']:.6f}" 
                                  for i, p in enumerate(selected_points)])
            coord_text.set_text(f"Selected Points:\n{points_str}")
            
            # Update the scatter plot
            update_selected_scatter()
            
            print(f"Selected point: Lat={lat:.6f}, Lon={lon:.6f}")
    
    # Function to clear selected points
    def clear_points(event):
        selected_points.clear()
        coord_text.set_text('')
        update_selected_scatter()
        print("Cleared all selected points")
    
    # Function to save selected points
    def save_points(event):
        if selected_points:
            selected_df = pd.DataFrame(selected_points)
            selected_df.to_csv('selected_kriging_points.csv', index=False, header=None)
            print(f"Saved {len(selected_points)} points to selected_kriging_points.csv")
        else:
            print("No points to save")
    
    # Add buttons for clear and save
    plt.subplots_adjust(bottom=0.15)
    clear_ax = plt.axes([0.3, 0.05, 0.15, 0.05])
    save_ax = plt.axes([0.55, 0.05, 0.15, 0.05])
    
    clear_button = Button(clear_ax, 'Clear Points')
    clear_button.on_clicked(clear_points)
    
    save_button = Button(save_ax, 'Save Points')
    save_button.on_clicked(save_points)
    
    # Connect the click event
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    # Show the plot
    plt.tight_layout()
    plt.show()
    
    # Return the selected points as a DataFrame
    if selected_points:
        return pd.DataFrame(selected_points)
    return None

# Launch the interactive map
print("Opening interactive map with OpenStreetMap background. Click on the map to select points.")
selected_points_df = interactive_map_with_clicks(df)

if selected_points_df is not None:
    print(f"Selected {len(selected_points_df)} points:")
    print(selected_points_df)
else:
    print("No points were selected or saved.")
