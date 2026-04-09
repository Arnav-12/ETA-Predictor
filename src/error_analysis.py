"""
Residual error analysis on the test set.
Produces distribution plots, actual-vs-predicted scatter, and segmented error tables.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def plot_residual_distribution(residuals, path='reports/residual_distribution.png'):
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=60, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
    plt.xlabel('Residual (Actual - Predicted)')
    plt.ylabel('Frequency')
    plt.title('Residual Distribution')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def plot_actual_vs_predicted(y_true, y_pred, path='reports/actual_vs_predicted.png'):
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.3, s=10, color='steelblue')
    lo = min(y_true.min(), y_pred.min()) - 1
    hi = max(y_true.max(), y_pred.max()) + 1
    plt.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect Prediction')
    plt.xlabel('Actual Delivery Minutes')
    plt.ylabel('Predicted Delivery Minutes')
    plt.title('Actual vs Predicted Delivery Time')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


def segmented_error_table(df, y_true, y_pred):
    """Print MAE broken down by distance bucket, weather, and hour bucket."""
    df = df.copy()
    df['abs_error'] = np.abs(y_true - y_pred)

    print("\n" + "=" * 60)
    print("SEGMENTED ERROR ANALYSIS (Mean Absolute Error)")
    print("=" * 60)

    # Distance bucket
    bins = [0, 1, 2, 3, 5, 100]
    labels = ['0-1 km', '1-2 km', '2-3 km', '3-5 km', '5+ km']
    df['dist_bucket'] = pd.cut(df['haversine_distance_km'], bins=bins, labels=labels, include_lowest=True)
    print("\n--- By Distance ---")
    print(df.groupby('dist_bucket', observed=True)['abs_error'].agg(['mean', 'count']).round(3).to_string())

    # Weather
    print("\n--- By Weather ---")
    print(df.groupby('weather_condition')['abs_error'].agg(['mean', 'count']).round(3).to_string())

    # Hour bucket
    def hour_bucket(h):
        if 6 <= h <= 9:   return 'Morning (6-9)'
        if 10 <= h <= 11: return 'Late Morning (10-11)'
        if 12 <= h <= 14: return 'Afternoon (12-14)'
        if 15 <= h <= 17: return 'Late Afternoon (15-17)'
        if 18 <= h <= 21: return 'Evening (18-21)'
        return 'Night (22-5)'

    df['hour_bucket'] = df['hour_of_day'].apply(hour_bucket)
    print("\n--- By Hour Bucket ---")
    print(df.groupby('hour_bucket')['abs_error'].agg(['mean', 'count']).round(3).to_string())
    print("=" * 60)


def main():
    print("Loading test data with predictions...")
    test_df = pd.read_pickle('data/processed/test_data_with_preds.pkl')

    y_true = test_df['actual_delivery_minutes'].values
    y_pred = test_df['predicted_delivery_minutes'].values
    residuals = y_true - y_pred

    print(f"Test set size: {len(y_true):,}")
    print(f"Residual mean: {residuals.mean():.3f}  std: {residuals.std():.3f}")

    os.makedirs('reports', exist_ok=True)
    plot_residual_distribution(residuals)
    plot_actual_vs_predicted(y_true, y_pred)
    segmented_error_table(test_df, y_true, y_pred)


if __name__ == '__main__':
    main()
