"""
Feature engineering pipeline for delivery ETA prediction.
"""

import numpy as np
import pandas as pd

try:
    import h3
    HAS_H3 = True
except ImportError:
    HAS_H3 = False


def haversine_vec(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    R = 6371.0
    lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
    lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2)**2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


def add_h3_index(df, lat_col='store_lat', lng_col='store_lng', resolution=7):
    if not HAS_H3:
        return df
    df = df.copy()
    df['h3_index'] = df.apply(
        lambda row: h3.latlng_to_cell(row[lat_col], row[lng_col], resolution), axis=1
    )
    df['h3_hash'] = df['h3_index'].apply(lambda x: hash(x) % 10000)
    return df


def add_cyclical_features(df):
    df = df.copy()
    df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    return df


def add_interaction_features(df):
    df = df.copy()
    df['distance_x_traffic'] = df['haversine_distance_km'] * df['traffic_index']
    df['peak_x_concurrent'] = df['is_peak_hour'] * df['concurrent_orders_at_store']
    return df


def one_hot_encode(df):
    df = df.copy()
    weather_dummies = pd.get_dummies(df['weather_condition'], prefix='weather', dtype=int)
    store_dummies = pd.get_dummies(df['store_type'], prefix='store', dtype=int)
    df = pd.concat([df, weather_dummies, store_dummies], axis=1)
    return df


def apply_log_transforms(df):
    df = df.copy()
    df['log_order_value'] = np.log1p(df['order_value_inr'])
    df['log_rider_exp'] = np.log1p(df['rider_experience_days'])
    return df


FEATURE_COLUMNS = [
    'haversine_distance_km',
    'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour',
    'traffic_index', 'num_items',
    'concurrent_orders_at_store',
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'distance_x_traffic', 'peak_x_concurrent',
    'log_order_value', 'log_rider_exp',
    'weather_clear', 'weather_fog', 'weather_heavy_rain', 'weather_rain',
    'store_dark_store', 'store_express', 'store_hub',
]

H3_COL = ['h3_hash']


def build_features(df):
    """Run full feature engineering pipeline. Returns X, y, enriched df, and column names."""
    df = df.copy()

    df['haversine_distance_km'] = haversine_vec(
        df['store_lat'], df['store_lng'], df['customer_lat'], df['customer_lng']
    )

    df = add_h3_index(df)
    df = add_cyclical_features(df)
    df = add_interaction_features(df)
    df = one_hot_encode(df)
    df = apply_log_transforms(df)

    cols = list(FEATURE_COLUMNS)
    if 'h3_hash' in df.columns:
        cols += H3_COL

    for c in cols:
        if c not in df.columns:
            df[c] = 0

    y = df['actual_delivery_minutes'].values
    X = df[cols].values.astype(np.float64)
    return X, y, df, cols
