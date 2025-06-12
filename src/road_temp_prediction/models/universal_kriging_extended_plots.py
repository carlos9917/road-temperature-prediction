import pandas as pd
import numpy as np
from pykrige.uk import UniversalKriging
from sklearn.preprocessing import StandardScaler
import sqlite3
import os
from datetime import datetime
import matplotlib.pyplot as plt
import geopandas as gpd
import contextily as ctx
from shapely.geometry import Point
from matplotlib.colors import Normalize
import matplotlib.lines as mlines
from scipy.spatial.distance import pdist, squareform

def load_and_merge_data_optimized(variables, year):
    """
    Load data from SQLite databases and merge into a single dataframe.
    Uses all available data without day limits.

    Args:
    variables: List of meteorological variables to load
    year: Year of data to load

    Returns:
    DataFrame with merged data from all variables
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
                for chunk in pd.read_sql_query(query, conn, chunksize=10000):
                    dataframes.append(chunk)
            except sqlite3.Error as e:
                print(f"SQLite error when reading {variable}: {e}")
            finally:
                conn.close()

        if not dataframes:
            raise ValueError("No data loaded from database")

        # Merge all dataframes
        full_df = pd.concat(dataframes, ignore_index=True)
        merged_df = full_df.groupby(['valid_dttm', 'SID', 'lat', 'lon']).first().reset_index()
        return merged_df
    except (sqlite3.Error, ValueError) as e:
        print(f"Error loading data: {str(e)}")
        return None

def compute_empirical_variogram(coords, values, n_lags=15, max_dist=None):
    """
    Compute empirical variogram for spatial data.
    
    Args:
        coords: Array of coordinates (n_points, 2)
        values: Array of values at each coordinate
        n_lags: Number of distance lags
        max_dist: Maximum distance to consider
    
    Returns:
        lags: Distance lags
        semivariance: Semivariance values
        counts: Number of pairs in each lag
    """
    # Calculate pairwise distances
    distances = pdist(coords)
    
    # Calculate pairwise differences in values
    n_points = len(values)
    value_diffs = []
    for i in range(n_points):
        for j in range(i+1, n_points):
            value_diffs.append((values[i] - values[j])**2)
    
    value_diffs = np.array(value_diffs)
    
    # Set maximum distance if not provided
    if max_dist is None:
        max_dist = np.max(distances) * 0.5
    
    # Create distance bins
    lag_edges = np.linspace(0, max_dist, n_lags + 1)
    lags = (lag_edges[:-1] + lag_edges[1:]) / 2
    
    # Compute semivariance for each lag
    semivariance = np.zeros(n_lags)
    counts = np.zeros(n_lags)
    
    for i in range(n_lags):
        mask = (distances >= lag_edges[i]) & (distances < lag_edges[i+1])
        if np.sum(mask) > 0:
            semivariance[i] = np.mean(value_diffs[mask]) / 2
            counts[i] = np.sum(mask)
    
    return lags, semivariance, counts

def plot_variogram_analysis(UK_model, coords, values, title="Variogram Analysis",outfig="variogram.png"):
    """
    Plot empirical and fitted variograms.
    """
    # Compute empirical variogram
    lags, semivariance, counts = compute_empirical_variogram(coords, values)
    
    # Get fitted variogram from the model
    fitted_lags = np.linspace(0, np.max(lags), 100)
    fitted_semivariance = []
    
    # Get variogram parameters from the fitted model
    variogram_params = UK_model.variogram_model_parameters
    
    for lag in fitted_lags:
        if UK_model.variogram_model == 'exponential':
            # Exponential model: γ(h) = sill * (1 - exp(-3h/range)) + nugget
            sill = variogram_params[0]
            range_param = variogram_params[1] 
            nugget = variogram_params[2]
            
            if lag == 0:
                gamma = nugget
            else:
                gamma = sill * (1 - np.exp(-3 * lag / range_param)) + nugget
                
        elif UK_model.variogram_model == 'spherical':
            # Spherical model
            sill = variogram_params[0]
            range_param = variogram_params[1]
            nugget = variogram_params[2]
            
            if lag == 0:
                gamma = nugget
            elif lag <= range_param:
                gamma = sill * (1.5 * lag / range_param - 0.5 * (lag / range_param)**3) + nugget
            else:
                gamma = sill + nugget
                
        elif UK_model.variogram_model == 'gaussian':
            # Gaussian model
            sill = variogram_params[0]
            range_param = variogram_params[1]
            nugget = variogram_params[2]
            
            if lag == 0:
                gamma = nugget
            else:
                gamma = sill * (1 - np.exp(-(lag / range_param)**2)) + nugget
                
        else:
            # Linear model or fallback
            gamma = lag * 0.1  # Simple linear approximation
            
        fitted_semivariance.append(gamma)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot variogram
    mask = counts > 0  # Only plot lags with data
    ax1.scatter(lags[mask], semivariance[mask], c='red', s=50, alpha=0.7, label='Empirical')
    ax1.plot(fitted_lags, fitted_semivariance, 'b-', linewidth=2, label=f'Fitted ({UK_model.variogram_model})')
    ax1.set_xlabel('Distance (degrees)')
    ax1.set_ylabel('Semivariance')
    ax1.set_title(f'{title} - Variogram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add variogram parameters as text
    param_text = f'Sill: {variogram_params[0]:.3f}\\nRange: {variogram_params[1]:.3f}\\nNugget: {variogram_params[2]:.3f}'
    ax1.text(0.05, 0.95, param_text, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot number of pairs
    ax2.bar(lags[mask], counts[mask], width=(lags[1]-lags[0])*0.8, alpha=0.7, color='green')
    ax2.set_xlabel('Distance (degrees)')
    ax2.set_ylabel('Number of Pairs')
    ax2.set_title('Number of Pairs per Distance Lag')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print(f"Saving analysis to {outfig}")
    fig.savefig(outfig)
    return lags, semivariance, fitted_lags, fitted_semivariance

def create_prediction_grid(bounds, resolution=0.1):
    """
    Create a regular grid for continuous mapping.
    
    Args:
        bounds: (min_lon, min_lat, max_lon, max_lat)
        resolution: Grid resolution in degrees
    
    Returns:
        grid_coords: Array of grid coordinates
        grid_shape: Shape of the grid (ny, nx)
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    
    # Create coordinate arrays
    lon_grid = np.arange(min_lon, max_lon + resolution, resolution)
    lat_grid = np.arange(min_lat, max_lat + resolution, resolution)
    
    # Create meshgrid
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Flatten for prediction
    grid_coords = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
    
    return grid_coords, lon_mesh.shape

def interpolate_covariates_for_grid(grid_coords, station_coords, station_covariates):
    """
    Interpolate covariates to grid points using nearest neighbor.
    For a more sophisticated approach, you could use spatial interpolation.
    """
    from scipy.spatial import cKDTree
    
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(station_coords)
    
    # Find nearest station for each grid point
    distances, indices = tree.query(grid_coords)
    
    # Assign covariate values from nearest stations
    grid_covariates = station_covariates[indices]
    
    return grid_covariates

def plot_continuous_temperature_map(grid_coords, predictions, grid_shape, title="Temperature Map", 
                                        original_stations=None, predicted_stations=None,outfig="map.png"):
    """
    Plot continuous temperature map with geographical basemap.
    
    Args:
        grid_coords: Array of grid coordinates
        predictions: Array of predictions at grid points
        grid_shape: Shape of the grid (ny, nx)
        title: Plot title
        original_stations: Optional GeoDataFrame of original stations
        predicted_stations: Optional GeoDataFrame of predicted stations
    """
    # Reshape predictions to grid
    temp_grid = predictions.reshape(grid_shape)
    
    # Create coordinate arrays for plotting
    min_lon, max_lon = grid_coords[:, 0].min(), grid_coords[:, 0].max()
    min_lat, max_lat = grid_coords[:, 1].min(), grid_coords[:, 1].max()
    
    # Create meshgrid for contour plotting
    lon_grid = np.linspace(min_lon, max_lon, grid_shape[1])
    lat_grid = np.linspace(min_lat, max_lat, grid_shape[0])
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Create GeoDataFrame for the grid (for projection)
    grid_gdf = gpd.GeoDataFrame(
        {'temp': predictions}, 
        geometry=[Point(x, y) for x, y in grid_coords],
        crs='EPSG:4326'
    )
    
    # Convert to Web Mercator for basemap compatibility
    grid_gdf_mercator = grid_gdf.to_crs('EPSG:3857')
    
    # Get bounds in Web Mercator
    bounds = grid_gdf_mercator.total_bounds
    
    # Plot temperature surface as contour
    # First convert coordinates to Web Mercator
    from pyproj import Transformer
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:3857', always_xy=True)
    lon_mesh_merc, lat_mesh_merc = transformer.transform(lon_mesh, lat_mesh)
    
    # Create filled contour plot
    levels = np.linspace(predictions.min(), predictions.max(), 20)
    contour = ax.contourf(lon_mesh_merc, lat_mesh_merc, temp_grid, 
                         levels=levels, cmap='coolwarm', alpha=0.7, extend='both')
    
    # Add contour lines
    contour_lines = ax.contour(lon_mesh_merc, lat_mesh_merc, temp_grid, 
                              levels=levels[::2], colors='black', alpha=0.3, linewidths=0.5)
    
    # Add station points if provided
    if original_stations is not None:
        # Convert to Web Mercator if not already
        if original_stations.crs != 'EPSG:3857':
            original_stations = original_stations.to_crs('EPSG:3857')
        original_stations.plot(ax=ax, color='black', markersize=50, marker='o', 
                              edgecolor='black', linewidth=1, alpha=0.9, zorder=5,
                              label='Original Stations')
    
    if predicted_stations is not None:
        # Convert to Web Mercator if not already
        if predicted_stations.crs != 'EPSG:3857':
            predicted_stations = predicted_stations.to_crs('EPSG:3857')
        predicted_stations.plot(ax=ax, color='blue', markersize=40, marker='^', 
                               edgecolor='black', linewidth=1, alpha=0.9, zorder=5,
                               label='Predicted Stations')
    
    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, zorder=0)
    
    # Set extent
    ax.set_xlim(bounds[0], bounds[2])
    ax.set_ylim(bounds[1], bounds[3])
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('Temperature (°C)', fontsize=12)
    
    # Add legend if stations are provided
    if original_stations is not None or predicted_stations is not None:
        ax.legend(loc='upper right', fontsize=10)
    
    # Set title and remove axis labels (since we have a basemap)
    ax.set_title(title, fontsize=16, pad=20)
    ax.set_axis_off()
    
    plt.tight_layout()
    plt.show()
    print(f"Saving map to {outfig}")
    fig.savefig(outfig)

def plot_continuous_temperature_map_simple(grid_coords, predictions, grid_shape, title="Temperature Map"):
    """
    Simple version without basemap for quick visualization.
    """
    # Reshape predictions to grid
    temp_grid = predictions.reshape(grid_shape)
    
    # Create coordinate arrays for plotting
    min_lon, max_lon = grid_coords[:, 0].min(), grid_coords[:, 0].max()
    min_lat, max_lat = grid_coords[:, 1].min(), grid_coords[:, 1].max()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot temperature surface
    im = ax.imshow(temp_grid, extent=[min_lon, max_lon, min_lat, max_lat], 
                   origin='lower', cmap='coolwarm', alpha=0.8)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Temperature (°C)', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title, fontsize=14)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Define the variables and year
DB = "/media/cap/extra_work/road_model/OBSTABLE"
variables = ['TROAD', 'T2m', 'Td2m', 'D10m', 'S10m', 'AccPcp12h']
year = 2023
date_chosen = datetime(year,11,2,0) 
date_chosen = datetime(year,8,11,15) 
date_chosen = datetime(year,2,11,0) 
date_str = datetime.strftime(date_chosen,"%Y%m%d%H")

# Load all available data
print("Loading all available data...")
df = load_and_merge_data_optimized(variables, year)

## load the dem data previously pre processed
# Replace 'stations.csv' with your actual CSV file path
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
merged["dates"] = pd.to_datetime(merged["valid_dttm"], unit="s")

# Filter for specific date
df = merged[merged["dates"]==date_chosen]

# Prepare coordinates and features
coords = df[['lon', 'lat']].values
values = df['TROAD'].values

# Select covariates
covariates = ['elev_m', 'slope_deg', 'aspect_deg']
X = df[covariates].values

# Scale covariates
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\\n=== VARIOGRAM ANALYSIS ===")
print("Fitting Universal Kriging model...")

# Universal Kriging with specified drift terms
UK = UniversalKriging(
    df['lon'], df['lat'], df['TROAD'],
    drift_terms=['specified'],
    specified_drift=[X_scaled[:, i] for i in range(X_scaled.shape[1])],  # All your covariates
    variogram_model='exponential',
    verbose=True,
    enable_plotting=False
)


# Add model temperature as external drift (when I have numerical model data)
#UK = UniversalKriging(
#    df['lon'], df['lat'], df['TROAD'],
#    drift_terms=['external_Z'],
#    external_Z=model_temperature_at_stations,  # Model values at station locations
#    variogram_model='exponential'
#)

print(f"Fitted variogram parameters:")
print(f"  Sill: {UK.variogram_model_parameters[0]:.3f}")
print(f"  Range: {UK.variogram_model_parameters[1]:.3f}")
print(f"  Nugget: {UK.variogram_model_parameters[2]:.3f}")

# Plot variogram analysis
outfig_var = f"../../../results/figures/variogram_analysis_{date_str}.png"
outfig_troad_map = f"../../../results/figures/continuous_troad_map_{date_str}.png"
outfig_troad_var = f"../../../results/figures/continuous_troad_var_{date_str}.png"
plot_variogram_analysis(UK, coords, values, "Universal Kriging",outfig_var)

print("\\n=== CONTINUOUS MAPPING ===")
print("Creating prediction grid...")

# Define bounds for continuous mapping (adjust based on your study area)
bounds = (df['lon'].min() - 0.1, df['lat'].min() - 0.1, 
          df['lon'].max() + 0.1, df['lat'].max() + 0.1)

# Create prediction grid (adjust resolution as needed)
grid_coords, grid_shape = create_prediction_grid(bounds, resolution=0.05)
print(f"Created grid with {len(grid_coords)} points ({grid_shape[0]}x{grid_shape[1]})")

# Interpolate covariates to grid points
print("Interpolating covariates to grid...")
grid_covariates = interpolate_covariates_for_grid(grid_coords, coords, X)
grid_covariates_scaled = scaler.transform(grid_covariates)

# Predict on grid
print("Predicting temperatures on grid...")
grid_predictions, grid_variance = UK.execute(
    'points', 
    grid_coords[:, 0], 
    grid_coords[:, 1],
    specified_drift_arrays=[grid_covariates_scaled[:, i] for i in range(grid_covariates_scaled.shape[1])]
)

# Convert dataframes to GeoDataFrames for mapping
def df_to_gdf(df, lon_col='lon', lat_col='lat', crs='EPSG:4326'):
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    return gpd.GeoDataFrame(df, geometry=geometry, crs=crs)

# Create GeoDataFrames for station overlay
gdf_original = df_to_gdf(df)

# Plot continuous temperature map with basemap and stations
plot_continuous_temperature_map(grid_coords, grid_predictions, grid_shape, 
                               "Continuous Temperature Map - Universal Kriging",
                               original_stations=gdf_original,outfig=outfig_troad_map)

# Plot uncertainty map
plot_continuous_temperature_map(grid_coords, grid_variance, grid_shape, 
                               "Prediction Uncertainty (Variance)",
                               original_stations=gdf_original,outfig=outfig_troad_var)

print("\\n=== DISCRETE POINT PREDICTIONS ===")
# Load the new points where we want to predict TROAD
new_points = pd.read_csv('../data/station_metrics_kriging_points.csv')
print(f"Loaded {len(new_points)} new points for prediction")

# Prepare coordinates for prediction
pred_coords = new_points[['lon', 'lat']].values

# Prepare covariates for prediction (same as used in training)
covariates = ['elev_m', 'slope_deg', 'aspect_deg']
X_pred = new_points[covariates].values

# Scale covariates using the same scaler used for training
X_pred_scaled = scaler.transform(X_pred)

# Predict TROAD at new locations using Universal Kriging
pred_troad, pred_var = UK.execute('points', pred_coords[:, 0], pred_coords[:, 1],
    specified_drift_arrays=[X_pred_scaled[:, i] for i in range(X_pred_scaled.shape[1])])


# For prediction, provide model values at prediction points (place holder for adding numerical data later)
#pred_troad, pred_var = UK.execute(
#    'points', pred_coords[:, 0], pred_coords[:, 1],
#    external_Z=model_temperature_at_pred_points
#)



# Add predictions to the dataframe
new_points['TROAD_predicted'] = pred_troad
new_points['TROAD_variance'] = pred_var

# Save results to a new CSV file
OUT="../../../results/reports/troad_predictions_universal_kriging.csv"
new_points.to_csv(OUT, index=False)
#extra save for comparison with the other model
new_points[["station_id","lat","lon","TROAD_predicted","TROAD_variance"]].to_csv(f"kriging_predictions_{date_str}.csv", index=False)
print(f"Prediction complete. Results saved to {OUT}")
print(new_points[['station_id', 'lat', 'lon', 'TROAD_predicted']].head())

#### Enhanced plotting with continuous maps and stations

# Create GeoDataFrames
gdf_predicted = df_to_gdf(new_points)

# Add a column to identify the source
gdf_original['source'] = 'Original'
gdf_predicted['source'] = 'Predicted'

# Rename the temperature column in predicted data to match original
gdf_predicted = gdf_predicted.rename(columns={'TROAD_predicted': 'TROAD'})

# Combine the datasets
gdf_combined = pd.concat([gdf_original, gdf_predicted])

# Convert to Web Mercator for basemap compatibility
gdf_combined = gdf_combined.to_crs(epsg=3857)

# Define colormap for temperature
vmin = gdf_combined['TROAD'].min()
vmax = gdf_combined['TROAD'].max()
norm = Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.coolwarm

# Create enhanced visualization
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

# Plot 1: All stations with different markers
for source, marker in [('Original', 'o'), ('Predicted', '^')]:
    subset = gdf_combined[gdf_combined['source'] == source]
    subset.plot(column='TROAD', cmap=cmap, norm=norm,
                markersize=80, marker=marker, edgecolor='black',
                linewidth=1, alpha=0.8, ax=ax1, zorder=2)

ctx.add_basemap(ax1, source=ctx.providers.CartoDB.Positron, zorder=0)
cbar1 = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax1, pad=0.01)
cbar1.set_label('Temperature (°C)', fontsize=12)

orig_legend = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                           markersize=10, markeredgecolor='black', label='Original Stations')
pred_legend = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                           markersize=10, markeredgecolor='black', label='Predicted Stations')
ax1.legend(handles=[orig_legend, pred_legend], fontsize=10, loc='lower right')
ax1.set_title('TROAD Values - All Stations', fontsize=14)
ax1.set_axis_off()

# Plot 2: Kriging Uncertainty
gdf_predicted.plot(column='TROAD_variance', cmap=plt.cm.viridis,
                  markersize=80, marker='^', edgecolor='black',
                  linewidth=1, alpha=0.8, ax=ax2, zorder=2)

ctx.add_basemap(ax2, source=ctx.providers.CartoDB.Positron, zorder=0)
vmin_var = gdf_predicted['TROAD_variance'].min()
vmax_var = gdf_predicted['TROAD_variance'].max()
norm_var = Normalize(vmin=vmin_var, vmax=vmax_var)
cbar2 = fig.colorbar(plt.cm.ScalarMappable(norm=norm_var, cmap=plt.cm.viridis), ax=ax2, pad=0.01)
cbar2.set_label('Kriging Variance', fontsize=12)
ax2.set_title('Kriging Uncertainty - Predicted Stations', fontsize=14)
ax2.set_axis_off()

# Plot 3: Simple continuous temperature map
temp_grid = grid_predictions.reshape(grid_shape)
min_lon, max_lon = grid_coords[:, 0].min(), grid_coords[:, 0].max()
min_lat, max_lat = grid_coords[:, 1].min(), grid_coords[:, 1].max()

im3 = ax3.imshow(temp_grid, extent=[min_lon, max_lon, min_lat, max_lat], 
                origin='lower', cmap='coolwarm', alpha=0.8)
cbar3 = fig.colorbar(im3, ax=ax3, pad=0.01)
cbar3.set_label('Temperature (°C)', fontsize=10)
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.set_title('Continuous Temperature Surface', fontsize=14)
ax3.grid(True, alpha=0.3)

# Plot 4: Simple continuous uncertainty map
var_grid = grid_variance.reshape(grid_shape)
im4 = ax4.imshow(var_grid, extent=[min_lon, max_lon, min_lat, max_lat], 
                origin='lower', cmap='viridis', alpha=0.8)
cbar4 = fig.colorbar(im4, ax=ax4, pad=0.01)
cbar4.set_label('Variance', fontsize=10)
ax4.set_xlabel('Longitude')
ax4.set_ylabel('Latitude')
ax4.set_title('Continuous Uncertainty Surface', fontsize=14)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

