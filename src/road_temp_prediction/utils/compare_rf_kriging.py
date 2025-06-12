"""
Compare Random Forest and Kriging Predictions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import glob
import os

def load_predictions(rf_pattern="rf_predictions_*.csv", kriging_pattern="kriging_predictions_*.csv"):
    """Load prediction results from both methods"""
    
    # Find matching files
    rf_files = glob.glob(rf_pattern)
    kriging_files = glob.glob(kriging_pattern)
    
    if not rf_files:
        raise FileNotFoundError(f"No Random Forest files found matching: {rf_pattern}")
    if not kriging_files:
        raise FileNotFoundError(f"No Kriging files found matching: {kriging_pattern}")
    
    print(f"Found {len(rf_files)} RF files and {len(kriging_files)} Kriging files")
    
    # Load and combine all files
    rf_data = []
    kriging_data = []
    
    for rf_file in rf_files:
        date_str = rf_file.split('_')[-1].replace('.csv', '')
        df_rf = pd.read_csv(rf_file)
        df_rf['date'] = date_str
        df_rf['method'] = 'Random Forest'
        rf_data.append(df_rf)
    
    for kriging_file in kriging_files:
        date_str = kriging_file.split('_')[-1].replace('.csv', '')
        df_kriging = pd.read_csv(kriging_file)
        df_kriging['date'] = date_str
        df_kriging['method'] = 'Kriging'
        # Rename columns to match RF format
        df_kriging = df_kriging.rename(columns={
            'TROAD_predicted': 'TROAD_predicted_RF',
            'TROAD_variance': 'TROAD_uncertainty_RF'
        })
        kriging_data.append(df_kriging)
    
    rf_combined = pd.concat(rf_data, ignore_index=True)
    kriging_combined = pd.concat(kriging_data, ignore_index=True)
    
    return rf_combined, kriging_combined

def merge_predictions(rf_df, kriging_df):
    """Merge predictions from both methods"""
    
    # Merge on station_id and date
    merged = rf_df.merge(
        kriging_df[['station_id', 'date', 'TROAD_predicted_RF', 'TROAD_uncertainty_RF']],
        on=['station_id', 'date'],
        suffixes=('_rf', '_kriging')
    )
    
    # Rename columns for clarity
    merged = merged.rename(columns={
        'TROAD_predicted_RF_rf': 'RF_prediction',
        'TROAD_uncertainty_RF_rf': 'RF_uncertainty',
        'TROAD_predicted_RF_kriging': 'Kriging_prediction',
        'TROAD_uncertainty_RF_kriging': 'Kriging_uncertainty'
    })
    
    return merged

def create_comparison_plots(merged_df, save_plots=True):
    """Create comprehensive comparison plots"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # 1. Scatter plot: RF vs Kriging predictions
    ax1 = plt.subplot(2, 3, 1)
    
    for date in merged_df['date'].unique():
        subset = merged_df[merged_df['date'] == date]
        plt.scatter(subset['RF_prediction'], subset['Kriging_prediction'], 
                   alpha=0.7, s=50, label=f'Date: {date}')
    
    # Add 1:1 line
    min_temp = min(merged_df['RF_prediction'].min(), merged_df['Kriging_prediction'].min())
    max_temp = max(merged_df['RF_prediction'].max(), merged_df['Kriging_prediction'].max())
    plt.plot([min_temp, max_temp], [min_temp, max_temp], 'k--', alpha=0.8, label='1:1 line')
    
    # Calculate correlation
    corr, p_value = pearsonr(merged_df['RF_prediction'], merged_df['Kriging_prediction'])
    plt.text(0.05, 0.95, f'r = {corr:.3f}\\np = {p_value:.3e}', 
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.xlabel('Random Forest Prediction (°C)')
    plt.ylabel('Kriging Prediction (°C)')
    plt.title('RF vs Kriging Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Difference map
    ax2 = plt.subplot(2, 3, 2)
    merged_df['prediction_diff'] = merged_df['RF_prediction'] - merged_df['Kriging_prediction']
    
    scatter = plt.scatter(merged_df['lon'], merged_df['lat'], 
                         c=merged_df['prediction_diff'], 
                         cmap='RdBu_r', s=60, alpha=0.8)
    plt.colorbar(scatter, label='RF - Kriging (°C)')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Prediction Differences (RF - Kriging)')
    plt.grid(True, alpha=0.3)
    
    # 3. Uncertainty comparison
    ax3 = plt.subplot(2, 3, 3)
    plt.scatter(merged_df['RF_uncertainty'], merged_df['Kriging_uncertainty'], 
               alpha=0.6, s=50)
    
    min_unc = min(merged_df['RF_uncertainty'].min(), merged_df['Kriging_uncertainty'].min())
    max_unc = max(merged_df['RF_uncertainty'].max(), merged_df['Kriging_uncertainty'].max())
    plt.plot([min_unc, max_unc], [min_unc, max_unc], 'k--', alpha=0.8)
    
    plt.xlabel('RF Uncertainty')
    plt.ylabel('Kriging Uncertainty')
    plt.title('Uncertainty Comparison')
    plt.grid(True, alpha=0.3)
    
    # 4. Distribution comparison
    ax4 = plt.subplot(2, 3, 4)
    plt.hist(merged_df['RF_prediction'], bins=20, alpha=0.6, label='Random Forest', density=True)
    plt.hist(merged_df['Kriging_prediction'], bins=20, alpha=0.6, label='Kriging', density=True)
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Density')
    plt.title('Prediction Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Box plot by date
    ax5 = plt.subplot(2, 3, 5)
    
    # Reshape data for box plot
    plot_data = []
    for date in merged_df['date'].unique():
        subset = merged_df[merged_df['date'] == date]
        for method, col in [('RF', 'RF_prediction'), ('Kriging', 'Kriging_prediction')]:
            for val in subset[col]:
                plot_data.append({'Date': date, 'Method': method, 'Temperature': val})
    
    plot_df = pd.DataFrame(plot_data)
    sns.boxplot(data=plot_df, x='Date', y='Temperature', hue='Method', ax=ax5)
    plt.title('Temperature Distributions by Date')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # 6. Summary statistics table
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    # Calculate summary statistics
    stats_data = []
    for date in merged_df['date'].unique():
        subset = merged_df[merged_df['date'] == date]
        
        rf_stats = {
            'Date': date,
            'Method': 'Random Forest',
            'Mean': f"{subset['RF_prediction'].mean():.2f}",
            'Std': f"{subset['RF_prediction'].std():.2f}",
            'Min': f"{subset['RF_prediction'].min():.2f}",
            'Max': f"{subset['RF_prediction'].max():.2f}",
            'Mean Uncertainty': f"{subset['RF_uncertainty'].mean():.3f}"
        }
        
        kriging_stats = {
            'Date': date,
            'Method': 'Kriging',
            'Mean': f"{subset['Kriging_prediction'].mean():.2f}",
            'Std': f"{subset['Kriging_prediction'].std():.2f}",
            'Min': f"{subset['Kriging_prediction'].min():.2f}",
            'Max': f"{subset['Kriging_prediction'].max():.2f}",
            'Mean Uncertainty': f"{subset['Kriging_uncertainty'].mean():.3f}"
        }
        
        stats_data.extend([rf_stats, kriging_stats])
    
    stats_df = pd.DataFrame(stats_data)
    
    # Create table
    table = ax6.table(cellText=stats_df.values, colLabels=stats_df.columns,
                     cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)
    ax6.set_title('Summary Statistics', pad=20)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('rf_vs_kriging_comparison.png', dpi=300, bbox_inches='tight')
        print("Comparison plot saved as 'rf_vs_kriging_comparison.png'")
    
    plt.show()
    
    return merged_df

def print_detailed_comparison(merged_df):
    """Print detailed comparison statistics"""
    
    print("\n" + "="*60)
    print("DETAILED COMPARISON RESULTS")
    print("="*60)
    
    # Overall statistics
    print(f"\nTotal prediction points: {len(merged_df)}")
    print(f"Number of dates: {merged_df['date'].nunique()}")
    print(f"Dates analyzed: {', '.join(merged_df['date'].unique())}")
    
    # Correlation analysis
    corr, p_value = pearsonr(merged_df['RF_prediction'], merged_df['Kriging_prediction'])
    print(f"\\nOverall correlation between methods: {corr:.4f} (p = {p_value:.2e})")
    
    # Difference analysis
    diff = merged_df['RF_prediction'] - merged_df['Kriging_prediction']
    print(f"\nPrediction differences (RF - Kriging):")
    print(f"  Mean difference: {diff.mean():.3f}°C")
    print(f"  Std difference: {diff.std():.3f}°C")
    print(f"  Max absolute difference: {abs(diff).max():.3f}°C")
    print(f"  RMSE: {np.sqrt(np.mean(diff**2)):.3f}°C")
    
    # Method comparison by date
    print(f"\nComparison by date:")
    for date in merged_df['date'].unique():
        subset = merged_df[merged_df['date'] == date]
        rf_mean = subset['RF_prediction'].mean()
        kriging_mean = subset['Kriging_prediction'].mean()
        date_diff = subset['RF_prediction'] - subset['Kriging_prediction']
        
        print(f"  {date}:")
        print(f"    RF mean: {rf_mean:.2f}°C, Kriging mean: {kriging_mean:.2f}°C")
        print(f"    Mean difference: {date_diff.mean():.3f}°C")
        print(f"    RMSE: {np.sqrt(np.mean(date_diff**2)):.3f}°C")
    
    # Uncertainty comparison
    print(f"\nUncertainty comparison:")
    print(f"  RF mean uncertainty: {merged_df['RF_uncertainty'].mean():.4f}")
    print(f"  Kriging mean uncertainty: {merged_df['Kriging_uncertainty'].mean():.4f}")
    
    # Spatial analysis
    print(f"\nSpatial coverage:")
    print(f"  Longitude range: {merged_df['lon'].min():.3f} to {merged_df['lon'].max():.3f}")
    print(f"  Latitude range: {merged_df['lat'].min():.3f} to {merged_df['lat'].max():.3f}")

def main():
    """Main comparison function"""
    
    print("=== Random Forest vs Kriging Comparison ===\\n")
    rf_pred = "../models/rf_*csv"
    kr_pred = "../models/krig*csv"
    #load_predictions(rf_pattern="rf_predictions_*.csv", kriging_pattern="kriging_predictions_*.csv"):
    try:
        # Load predictions
        print("Loading prediction files...")
        rf_df, kriging_df = load_predictions(rf_pattern=rf_pred,kriging_pattern=kr_pred)
        
        # Merge predictions
        print("Merging predictions...")
        merged_df = merge_predictions(rf_df, kriging_df)
        
        print(f"Successfully merged {len(merged_df)} prediction points")
        
        # Create comparison plots
        print("Creating comparison plots...")
        merged_df = create_comparison_plots(merged_df)
        
        # Print detailed comparison
        print_detailed_comparison(merged_df)
        
        # Save merged results
        merged_df.to_csv('merged_predictions_comparison.csv', index=False)
        print("\nMerged results saved to 'merged_predictions_comparison.csv'")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure you have:")
        print("1. RF prediction files: rf_predictions_*.csv")
        print("2. Kriging prediction files: kriging_predictions_*.csv")
        print("3. Both files have matching station_id and date information")

if __name__ == "__main__":
    main()
