#!/usr/bin/env python3
"""
Random Forest Road Temperature Prediction
Integrates with existing kriging data pipeline 
"""

import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class RoadTemperatureRF:
    """Random Forest model for road temperature prediction"""
    
    def __init__(self, random_state=42,obs_vars= ['TROAD', 'T2m', 'Td2m']):
        self.rf_model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.random_state = random_state
        self.feature_importance_ = None
        self.obs_vars = obs_vars
        
    def load_and_merge_data_optimized(self, variables, year, db_path="./OBSTABLE"):
        """
        Load data from SQLite databases and merge into a single dataframe.
        """
        dataframes = []

        for variable in variables:
            db_file = os.path.join(db_path, f'OBSTABLE_{variable}_{year}.sqlite')
            if not os.path.exists(db_file):
                print(f"Warning: Database file not found: {db_file}")
                continue

            conn = sqlite3.connect(db_file)
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

        # Merge all dataframes and remove duplicates
        full_df = pd.concat(dataframes, ignore_index=True)
        merged_df = full_df.groupby(['valid_dttm', 'SID', 'lat', 'lon']).first().reset_index()
        return merged_df
    
    def create_spatial_features(self, df):
        """Create spatial features for capturing spatial autocorrelation"""
        coords = df[['lon', 'lat']].values
        
        # Distance-based features (proxy for spatial autocorrelation)
        if len(coords) > 1:
            distances = cdist(coords, coords)
            # Minimum distance to nearest station (excluding self)
            np.fill_diagonal(distances, np.inf)
            df['min_distance'] = np.min(distances, axis=1)
            
            # Mean distance to 5 nearest neighbors
            if len(coords) >= 5:
                sorted_distances = np.sort(distances, axis=1)
                df['mean_distance_5nn'] = np.mean(sorted_distances[:, :5], axis=1)
            else:
                df['mean_distance_5nn'] = df['min_distance']
        else:
            df['min_distance'] = 0
            df['mean_distance_5nn'] = 0
        
        # Normalized spatial coordinates for trend modeling
        df['lon_norm'] = (df['lon'] - df['lon'].mean()) / df['lon'].std() if df['lon'].std() > 0 else 0
        df['lat_norm'] = (df['lat'] - df['lat'].mean()) / df['lat'].std() if df['lat'].std() > 0 else 0
        
        # Spatial interaction terms
        df['lon_lat_interaction'] = df['lon_norm'] * df['lat_norm']
        
        return df
    
    def create_temporal_features(self, df):
        """Create temporal features from datetime"""
        df['dates'] = pd.to_datetime(df['valid_dttm'], unit='s')
        
        # Cyclical encoding for hour and month
        df['hour'] = df['dates'].dt.hour
        df['month'] = df['dates'].dt.month
        df['day_of_year'] = df['dates'].dt.dayofyear
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        return df
    
    def create_terrain_interactions(self, df):
        """Create interaction features between terrain variables"""
        # Check if terrain features exist
        terrain_cols = ['elev_m', 'slope_deg', 'aspect_deg']
        available_terrain = [col for col in terrain_cols if col in df.columns]
        
        if 'elev_m' in df.columns and 'slope_deg' in df.columns:
            df['elev_slope_interaction'] = df['elev_m'] * df['slope_deg']
        
        if 'elev_m' in df.columns:
            df['elev_squared'] = df['elev_m'] ** 2
            
        if 'slope_deg' in df.columns:
            df['slope_squared'] = df['slope_deg'] ** 2
            
        # Aspect transformations (convert to x,y components)
        if 'aspect_deg' in df.columns:
            aspect_rad = np.deg2rad(df['aspect_deg'])
            df['aspect_sin'] = np.sin(aspect_rad)
            df['aspect_cos'] = np.cos(aspect_rad)
        
        return df
    
    def prepare_features(self, df):
        """Prepare complete feature matrix"""
        # Create all feature types
        df = self.create_spatial_features(df)
        df = self.create_temporal_features(df)
        df = self.create_terrain_interactions(df)
        
        # Define feature groups
        base_features = ['lon', 'lat']
        
        # Terrain features (from GIS processing)
        terrain_features = []
        for col in ['elev_m', 'slope_deg', 'aspect_deg', 'elev_slope_interaction', 
                   'elev_squared', 'slope_squared', 'aspect_sin', 'aspect_cos']:
            if col in df.columns:
                terrain_features.append(col)
        
        # Meteorological features (from obstable database)
        met_features = []
        for col in self.obs_vars: #['T2m', 'Td2m']: #, 'D10m', 'S10m', 'AccPcp12h']:
            if col in df.columns:
                met_features.append(col)
        
        # Spatial features
        spatial_features = ['lon_norm', 'lat_norm', 'lon_lat_interaction', 
                          'min_distance', 'mean_distance_5nn']
        
        # Temporal features
        temporal_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos', 
                           'day_sin', 'day_cos']
        
        # Combine all features
        self.feature_names = (base_features + terrain_features + met_features + 
                            spatial_features + temporal_features)
        
        # Filter to only include available columns
        available_features = [col for col in self.feature_names if col in df.columns]
        self.feature_names = available_features
        
        print(f"Using {len(self.feature_names)} features: {self.feature_names}")
        
        return df[self.feature_names]
    
    def fit(self, df, target_col='TROAD', test_size=0.2, tune_hyperparams=True):
        """Fit Random Forest model with optional hyperparameter tuning"""
        print("Preparing features...")
        
        # Prepare feature matrix
        X = self.prepare_features(df)
        y = df[target_col]
        
        # Remove rows with missing values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X_clean = X[mask]
        y_clean = y[mask]
        
        print(f"Training data: {len(X_clean)} samples, {X_clean.shape[1]} features")
        print(f"Target range: {y_clean.min():.1f}°C to {y_clean.max():.1f}°C")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_clean, test_size=test_size, random_state=self.random_state
        )
        
        if tune_hyperparams:
            print("Tuning hyperparameters...")
            # Hyperparameter grid
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            rf_base = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            self.rf_model = GridSearchCV(
                rf_base, param_grid, cv=5, 
                scoring='neg_mean_squared_error', n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
            
            print(f"Best parameters: {self.rf_model.best_params_}")
            print(f"Best CV score (RMSE): {np.sqrt(-self.rf_model.best_score_):.3f}°C")
            
            # Get feature importance
            self.feature_importance_ = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.rf_model.best_estimator_.feature_importances_
            }).sort_values('importance', ascending=False)
            
        else:
            # Use default parameters
            self.rf_model = RandomForestRegressor(
                n_estimators=200, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=self.random_state, n_jobs=-1
            )
            self.rf_model.fit(X_train, y_train)
            
            self.feature_importance_ = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.rf_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Evaluate on test set
        y_pred = self.rf_model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\\nTest Set Performance:")
        print(f"RMSE: {rmse:.3f}°C")
        print(f"MAE: {mae:.3f}°C")
        print(f"R²: {r2:.3f}")
        
        return self
    
    def predict(self, df_pred):
        """Predict temperatures at new locations"""
        if self.rf_model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Prepare features for prediction
        X_pred = self.prepare_features(df_pred)
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Make predictions
        if hasattr(self.rf_model, 'predict'):
            predictions = self.rf_model.predict(X_pred_scaled)
        else:
            predictions = self.rf_model.best_estimator_.predict(X_pred_scaled)
        
        # Estimate uncertainty using ensemble variance
        if hasattr(self.rf_model, 'best_estimator_'):
            estimator = self.rf_model.best_estimator_
        else:
            estimator = self.rf_model
            
        # Get predictions from individual trees
        tree_predictions = np.array([
            tree.predict(X_pred_scaled) for tree in estimator.estimators_
        ])
        prediction_std = np.std(tree_predictions, axis=0)
        
        return predictions, prediction_std
    
    def get_feature_importance(self, top_n=10):
        """Get top N most important features"""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        return self.feature_importance_.head(top_n)
    
    def plot_feature_importance(self, top_n=15, figsize=(10, 8)):
        """Plot feature importance"""
        if self.feature_importance_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        plt.figure(figsize=figsize)
        top_features = self.feature_importance_.head(top_n)
        
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title(f'Top {top_n} Feature Importances - Random Forest')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return top_features

def main():
    """Main execution function - example usage"""
    
    # Configuration (adjust paths as needed)
    DB_PATH = "../data/OBSTABLE"  # Path to SQLite databases
    DB_PATH = "/media/cap/extra_work/road_model/OBSTABLE"
    STATION_METRICS_PATH = "../data/station_metrics.csv"  # the  terrain metrics to use for the known stations
    PREDICTION_POINTS_PATH = "../data/station_metrics_kriging_points.csv"  # Prediction points with terrain metrics
    
    # Variables to load
    variables = ['TROAD', 'T2m', 'Td2m'] #, 'D10m', 'S10m', 'AccPcp12h']
    year = 2023
    
    # Test dates
    test_dates = [
        datetime(2023, 2, 11, 15),  # Winter date
        datetime(2023, 8, 11, 15),  # Summer date
    ]
    
    print("=== Random Forest Road Temperature Prediction ===\\n")
    
    # Initialize model
    rf_model = RoadTemperatureRF(obs_vars=variables)
    
    try:
        # Load meteorological data
        print("Loading meteorological data...")
        df = rf_model.load_and_merge_data_optimized(variables, year, DB_PATH)
        print(f"Loaded {len(df)} records")
        
        # Load terrain metrics
        if os.path.exists(STATION_METRICS_PATH):
            print("Loading terrain metrics...")
            station_metrics = pd.read_csv(STATION_METRICS_PATH)
            
            # Merge datasets
            merged_df = df.merge(
                station_metrics.drop(columns=['lat', 'lon'], errors='ignore'),
                left_on='SID', right_on='station_id', how='inner'
            )
            print(f"Merged dataset: {len(merged_df)} records")
        else:
            print(f"Terrain metrics file not found: {STATION_METRICS_PATH}")
            merged_df = df
        
        # Test on different dates
        for date_chosen in test_dates:
            print(f"\\n=== Testing on {date_chosen.strftime('%Y-%m-%d %H:%M')} ===")
            
            # Filter for specific date
            merged_df["dates"] = pd.to_datetime(merged_df["valid_dttm"], unit="s")
            df_filtered = merged_df[merged_df["dates"] == date_chosen]
            
            if len(df_filtered) < 10:
                print(f"Insufficient data for {date_chosen} ({len(df_filtered)} records)")
                continue
            
            print(f"Training on {len(df_filtered)} stations")
            
            # Fit model
            rf_model.fit(df_filtered, tune_hyperparams=True)
            
            # Show feature importance
            print("\\nTop 10 Most Important Features:")
            print(rf_model.get_feature_importance(10))
            
            # Predict at kriging points if available
            if os.path.exists(PREDICTION_POINTS_PATH):
                print("\nPredicting at unknown points...")
                prediction_points = pd.read_csv(PREDICTION_POINTS_PATH)
                
                # Add temporal features for prediction
                prediction_points['valid_dttm'] = int(date_chosen.timestamp())
                
                predictions, uncertainties = rf_model.predict(prediction_points)
                
                # Add results
                prediction_points['TROAD_predicted_RF'] = predictions
                prediction_points['TROAD_uncertainty_RF'] = uncertainties
                
                print(f"Predictions range: {predictions.min():.1f}°C to {predictions.max():.1f}°C")
                print(f"Mean uncertainty: {uncertainties.mean():.3f}°C")
                
                # Save results
                output_file = f"rf_predictions_{date_chosen.strftime('%Y%m%d_%H%M')}.csv"
                prediction_points.to_csv(output_file, index=False)
                print(f"Results saved to: {output_file}")
            
    except Exception as e:
        print(f"Error: {e}")
        print("\\nMake sure your data paths are correct and databases are accessible.")
        print("You may need to adjust the DB_PATH and file paths in the configuration section.")

if __name__ == "__main__":
    main()
