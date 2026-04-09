# Delivery ETA Predictor

A production-style machine learning system that estimates last-mile delivery time for quick-commerce orders. Built with synthetic data modeled after real-world platforms operating in Hyderabad, India. Covers the full ML lifecycle: data generation, feature engineering, model training with hyperparameter optimization, residual analysis, drift monitoring, and REST inference.

---

## Table of Contents

- [Architecture](#architecture)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Results](#model-results)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Reproducing Results](#reproducing-results)

---

## Architecture

```
+-------------------+     +--------------------+     +---------------------+
|  Synthetic Data   |---->|  Feature           |---->|  Model Training     |
|  Generator        |     |  Engineering       |     |  (Ridge / RF / XGB) |
|  (50K rows)       |     |  Pipeline          |     |  + Optuna Tuning    |
+-------------------+     +--------------------+     +----------+----------+
                                                                |
                                                                v
+-------------------+     +--------------------+     +---------------------+
|  Drift Monitor    |<----|  Error Analysis    |<----|  Best Model         |
|  (HTML Report)    |     |  (Segmented MAE)   |     |  (Serialized .pkl)  |
+-------------------+     +--------------------+     +----------+----------+
                                                                |
                                                                v
                                                      +---------------------+
                                                      |  FastAPI Service    |
                                                      |  POST /predict      |
                                                      |  GET  /health       |
                                                      +---------------------+
```

---

## Dataset

Synthetic dataset of **50,000 delivery orders** spanning Jan 2023 -- Dec 2024 in the Hyderabad metropolitan area.

| Column                     | Type    | Description                                           |
|----------------------------|---------|-------------------------------------------------------|
| `order_id`                 | str     | UUID identifier                                       |
| `order_timestamp`          | datetime| Order placement time                                  |
| `store_lat`, `store_lng`   | float   | Store coordinates (Hyderabad bounding box)             |
| `customer_lat`, `customer_lng` | float | Customer coordinates (within 5 km of store)        |
| `haversine_distance_km`    | float   | Great-circle distance between store and customer       |
| `hour_of_day`              | int     | 0--23                                                 |
| `day_of_week`              | int     | 0 (Mon) -- 6 (Sun)                                    |
| `is_weekend`               | int     | 1 if Saturday or Sunday                               |
| `is_peak_hour`             | int     | 1 if 7--9am, 12--2pm, or 6--9pm                       |
| `weather_condition`        | str     | clear (60%), rain (20%), fog (12%), heavy_rain (8%)   |
| `traffic_index`            | float   | 0--1, correlated with hour and weather                |
| `num_items`                | int     | 1--20                                                 |
| `order_value_inr`          | float   | INR, correlated with num_items                        |
| `store_type`               | str     | dark_store (50%), hub (30%), express (20%)             |
| `rider_experience_days`    | int     | 0--1000                                               |
| `concurrent_orders_at_store` | int   | 1--15                                                 |
| `actual_delivery_minutes`  | float   | **Target variable** (8--53 min, mean ~23 min)         |

---

## Feature Engineering

Applied in `src/features.py`:

| Category                  | Features                                                       |
|---------------------------|----------------------------------------------------------------|
| Geospatial                | Haversine distance, H3 hex index (resolution 7)               |
| Cyclical encoding         | sin/cos of hour_of_day, sin/cos of day_of_week                |
| Interaction features      | distance x traffic_index, is_peak_hour x concurrent_orders    |
| One-hot encoding          | weather_condition (4 cols), store_type (3 cols)               |
| Log transforms            | log1p(order_value_inr), log1p(rider_experience_days)          |

Total: **24 engineered features** fed into all models.

---

## Model Results

Time-based split: 70% train / 15% validation / 15% test. XGBoost was tuned with Optuna (30 trials, minimizing validation RMSE).

### Validation Set

| Model              | RMSE   | MAE    | MAPE   | R2     |
|--------------------|--------|--------|--------|--------|
| Ridge Regression   | 2.5412 | 2.0183 | 9.69%  | 0.8615 |
| Random Forest      | 2.6075 | 2.0756 | 9.93%  | 0.8542 |
| **XGBoost (Optuna)** | **2.5184** | **2.0049** | **9.62%** | **0.8640** |

### Test Set (Best Model: XGBoost)

| RMSE   | MAE    | MAPE  | R2     |
|--------|--------|-------|--------|
| 2.4953 | 1.9947 | 9.46% | 0.8666 |

### Optuna Best Hyperparameters

| Parameter            | Value  |
|----------------------|--------|
| n_estimators         | 342    |
| max_depth            | 4      |
| learning_rate        | 0.045  |
| subsample            | 0.741  |
| colsample_bytree     | 0.966  |
| min_child_weight     | 6      |
| reg_alpha            | 1.429  |
| reg_lambda           | 0.169  |

---

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/delivery-eta-predictor.git
cd delivery-eta-predictor
pip install -r requirements.txt
```

### Run the Pipeline

Using Make:

```bash
make install          # Install dependencies
make generate-data    # Generate 50K synthetic orders
make train            # Train Ridge, RF, XGBoost; save best model
make evaluate         # Run error analysis + drift monitoring
make serve            # Start FastAPI inference server on port 8000
make notebook         # Launch Jupyter notebook
```

Or step by step:

```bash
pip install -r requirements.txt
python data/generate_data.py
python src/train.py
python src/error_analysis.py
python src/drift_monitor.py
```

### Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

---

## API Reference

### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-03-15T14:30:00.000000"
}
```

### `POST /predict`

Predict delivery time for a single order.

**Request Body:**

```json
{
  "store_lat": 17.385,
  "store_lng": 78.487,
  "customer_lat": 17.410,
  "customer_lng": 78.500,
  "haversine_distance_km": 3.2,
  "hour_of_day": 19,
  "day_of_week": 4,
  "is_weekend": 0,
  "is_peak_hour": 1,
  "weather_condition": "rain",
  "traffic_index": 0.72,
  "num_items": 5,
  "order_value_inr": 650.0,
  "store_type": "dark_store",
  "rider_experience_days": 180,
  "concurrent_orders_at_store": 8
}
```

**Response:**

```json
{
  "predicted_minutes": 28.45,
  "confidence_interval_lower": 25.12,
  "confidence_interval_upper": 31.78,
  "model_version": "1.0.0",
  "model_name": "XGBoost"
}
```

**cURL Example:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store_lat": 17.385,
    "store_lng": 78.487,
    "customer_lat": 17.410,
    "customer_lng": 78.500,
    "haversine_distance_km": 3.2,
    "hour_of_day": 19,
    "day_of_week": 4,
    "is_weekend": 0,
    "is_peak_hour": 1,
    "weather_condition": "rain",
    "traffic_index": 0.72,
    "num_items": 5,
    "order_value_inr": 650.0,
    "store_type": "dark_store",
    "rider_experience_days": 180,
    "concurrent_orders_at_store": 8
  }'
```

---

## Project Structure

```
delivery-eta-predictor/
|
|-- api/
|   |-- __init__.py
|   +-- main.py                 # FastAPI inference service with Pydantic validation
|
|-- data/
|   |-- generate_data.py        # Synthetic dataset generator (50K rows, Hyderabad)
|   |-- raw/
|   |   +-- delivery_data.csv   # Generated dataset (gitignored)
|   +-- processed/
|       |-- feature_names.json  # Ordered list of feature column names
|       |-- test_data.pkl       # Test split for error analysis
|       +-- test_data_with_preds.pkl  # Test data with model predictions
|
|-- models/
|   +-- best_model.pkl          # Serialized best model + metadata
|
|-- notebooks/
|   +-- EDA_and_Modelling.ipynb # Exploratory analysis and model comparison
|
|-- reports/
|   |-- feature_importance.png  # Top-20 feature importances
|   |-- residual_distribution.png
|   |-- actual_vs_predicted.png
|   +-- drift_report.html       # Data drift analysis report
|
|-- src/
|   |-- __init__.py
|   |-- features.py             # Feature engineering pipeline
|   |-- train.py                # Multi-model training with MLflow logging
|   |-- error_analysis.py       # Residual analysis and segmented MAE
|   +-- drift_monitor.py        # Production data drift simulation
|
|-- .gitignore
|-- Makefile
|-- README.md
+-- requirements.txt
```

---

## Tech Stack

| Category         | Libraries                                          |
|------------------|----------------------------------------------------|
| Modeling         | scikit-learn 1.6, XGBoost 2.1, LightGBM 4.6      |
| Tuning           | Optuna 4.2                                         |
| Experiment Tracking | MLflow 2.20                                     |
| Drift Monitoring | Evidently 0.7 (with manual fallback)               |
| Explainability   | SHAP 0.47                                          |
| API              | FastAPI 0.115, Uvicorn, Pydantic 2.10             |
| Visualization    | Matplotlib 3.10, Seaborn 0.13, Plotly 5.x         |
| Geospatial       | H3 4.2, Haversine                                  |
| Notebook         | Jupyter 1.1                                        |

---

## Reproducing Results

1. Ensure Python 3.10+ is installed.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the full pipeline: `make generate-data && make train && make evaluate`
4. All outputs land in `reports/`, `models/`, and `data/processed/`.
5. MLflow logs are written to the local `mlruns/` directory.

The dataset uses a fixed random seed (42) for reproducibility. Running `python data/generate_data.py` will always produce the same CSV.
