"""
Generate synthetic delivery ETA dataset for Hyderabad, India.
Produces data/raw/delivery_data.csv with 50,000 rows.
"""

import uuid
import numpy as np
import pandas as pd
from math import radians, sin, cos, sqrt, atan2

np.random.seed(42)

NUM_ROWS = 50_000

# Hyderabad bounding box
LAT_MIN, LAT_MAX = 17.25, 17.55
LNG_MIN, LNG_MAX = 78.30, 78.65


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def generate_customer_coords(store_lat, store_lng, radius_km=5):
    R = 6371.0
    radius_deg = radius_km / R
    angle = np.random.uniform(0, 2 * np.pi)
    r = radius_deg * np.sqrt(np.random.uniform(0, 1))
    cust_lat = store_lat + r * np.cos(angle)
    cust_lng = store_lng + r * np.sin(angle) / np.cos(radians(store_lat))
    return cust_lat, cust_lng


def compute_delivery_minutes(row):
    """Target variable: delivery time as a function of real-world factors."""
    base = 10.0
    distance_penalty = row['haversine_distance_km'] * 4.5
    traffic_penalty = row['traffic_index'] * 12.0
    weather_map = {'clear': 0, 'fog': 3, 'rain': 5, 'heavy_rain': 10}
    weather_penalty = weather_map.get(row['weather_condition'], 0)
    peak_penalty = 6.0 if row['is_peak_hour'] else 0
    item_penalty = row['num_items'] * 0.3
    experience_bonus = -min(row['rider_experience_days'] * 0.008, 4.0)
    concurrent_penalty = row['concurrent_orders_at_store'] * 0.5
    noise = np.random.normal(0, 2.5)
    total = (base + distance_penalty + traffic_penalty + weather_penalty
             + peak_penalty + item_penalty + experience_bonus
             + concurrent_penalty + noise)
    return max(8, min(60, total))


def main():
    print(f"Generating {NUM_ROWS:,} rows...")

    start_date = pd.Timestamp('2023-01-01')
    end_date = pd.Timestamp('2024-12-31')
    timestamps = pd.to_datetime(
        np.random.randint(start_date.value // 10**9, end_date.value // 10**9, NUM_ROWS),
        unit='s',
    )

    weather_choices = ['clear', 'rain', 'heavy_rain', 'fog']
    weather_weights = [0.60, 0.20, 0.08, 0.12]

    store_type_choices = ['dark_store', 'hub', 'express']
    store_type_weights = [0.50, 0.30, 0.20]

    records = []
    for i in range(NUM_ROWS):
        ts = timestamps[i]
        store_lat = np.random.uniform(LAT_MIN, LAT_MAX)
        store_lng = np.random.uniform(LNG_MIN, LNG_MAX)
        cust_lat, cust_lng = generate_customer_coords(store_lat, store_lng)
        dist = haversine(store_lat, store_lng, cust_lat, cust_lng)

        hour = ts.hour
        dow = ts.dayofweek
        is_weekend = int(dow >= 5)
        is_peak = int(hour in [7, 8, 12, 13, 18, 19, 20])
        weather = np.random.choice(weather_choices, p=weather_weights)

        # Traffic correlated with hour and weather
        base_traffic = 0.3
        if hour in [8, 9, 18, 19]:
            base_traffic += 0.35
        elif hour in [12, 13]:
            base_traffic += 0.2
        if weather in ['rain', 'heavy_rain']:
            base_traffic += 0.15
        traffic_index = np.clip(base_traffic + np.random.normal(0, 0.1), 0, 1)

        num_items = np.random.randint(1, 21)
        order_value = max(50, num_items * np.random.uniform(40, 120) + np.random.normal(0, 50))
        store_type = np.random.choice(store_type_choices, p=store_type_weights)
        rider_exp = np.random.randint(0, 1001)
        concurrent = np.random.randint(1, 16)

        row = {
            'order_id': str(uuid.uuid4()),
            'order_timestamp': ts,
            'store_lat': round(store_lat, 6),
            'store_lng': round(store_lng, 6),
            'customer_lat': round(cust_lat, 6),
            'customer_lng': round(cust_lng, 6),
            'haversine_distance_km': round(dist, 3),
            'hour_of_day': hour,
            'day_of_week': dow,
            'is_weekend': is_weekend,
            'is_peak_hour': is_peak,
            'weather_condition': weather,
            'traffic_index': round(traffic_index, 4),
            'num_items': num_items,
            'order_value_inr': round(order_value, 2),
            'store_type': store_type,
            'rider_experience_days': rider_exp,
            'concurrent_orders_at_store': concurrent,
        }
        row['actual_delivery_minutes'] = round(compute_delivery_minutes(row), 2)
        records.append(row)

    df = pd.DataFrame(records)
    df = df.sort_values('order_timestamp').reset_index(drop=True)

    output_path = 'data/raw/delivery_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} rows to {output_path}")
    print(f"Columns: {list(df.columns)}")
    print(f"Target stats:\n{df['actual_delivery_minutes'].describe()}")


if __name__ == '__main__':
    main()
