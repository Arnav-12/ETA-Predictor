"""
Training pipeline: Ridge, Random Forest, XGBoost with Optuna tuning.
Logs metrics to MLflow and saves the best model.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import mlflow
    import mlflow.sklearn
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import build_features


def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1, None))) * 100
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': round(rmse, 4), 'MAE': round(mae, 4), 'MAPE': round(mape, 2), 'R2': round(r2, 4)}


def train_ridge(X_train, y_train):
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    return model


def train_rf(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200, max_depth=15, n_jobs=-1, random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgb_optuna(X_train, y_train, X_val, y_val, n_trials=30):
    """XGBoost with Optuna hyperparameter search."""

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),
        }
        model = xgb.XGBRegressor(**params, random_state=42, n_jobs=-1, verbosity=0)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        return np.sqrt(mean_squared_error(y_val, preds))

    if HAS_OPTUNA:
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = study.best_params
        print(f"Optuna best params: {best_params}")
        print(f"Optuna best val RMSE: {study.best_value:.4f}")
    else:
        best_params = {
            'n_estimators': 300, 'max_depth': 6, 'learning_rate': 0.1,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'min_child_weight': 3,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
        }

    model = xgb.XGBRegressor(**best_params, random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_train, y_train)
    return model


def plot_feature_importance(model, feature_names, output_path='reports/feature_importance.png'):
    if not hasattr(model, 'feature_importances_'):
        print("No feature_importances_ attribute, skipping plot.")
        return

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(20, len(feature_names))

    plt.figure(figsize=(10, 8))
    plt.title("Feature Importance (Top 20)")
    plt.barh(range(top_n), importances[indices[:top_n]][::-1], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel('Importance')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to {output_path}")


def main():
    print("Loading data...")
    df = pd.read_csv('data/raw/delivery_data.csv', parse_dates=['order_timestamp'])
    print(f"Data shape: {df.shape}")

    print("Building features...")
    X, y, df_feat, feature_names = build_features(df)
    print(f"Feature matrix shape: {X.shape}")

    # Time-based split: 70/15/15
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    print(f"Train: {X_train.shape[0]:,}  Val: {X_val.shape[0]:,}  Test: {X_test.shape[0]:,}")

    # Persist test split for downstream analysis
    test_df = df_feat.iloc[val_end:].copy()
    os.makedirs('data/processed', exist_ok=True)
    test_df.to_pickle('data/processed/test_data.pkl')
    with open('data/processed/feature_names.json', 'w') as f:
        json.dump(feature_names, f)

    if HAS_MLFLOW:
        mlflow.set_experiment("delivery-eta-prediction")
        mlflow.start_run()

    results = {}

    # Ridge
    print("\nTraining Ridge Regression...")
    ridge = train_ridge(X_train, y_train)
    results['Ridge'] = compute_metrics(y_val, ridge.predict(X_val))
    print(f"  Val: {results['Ridge']}")
    if HAS_MLFLOW:
        mlflow.log_metrics({f"ridge_{k}": v for k, v in results['Ridge'].items()})

    # Random Forest
    print("\nTraining Random Forest (n_estimators=200)...")
    rf = train_rf(X_train, y_train)
    results['RandomForest'] = compute_metrics(y_val, rf.predict(X_val))
    print(f"  Val: {results['RandomForest']}")
    if HAS_MLFLOW:
        mlflow.log_metrics({f"rf_{k}": v for k, v in results['RandomForest'].items()})

    # XGBoost + Optuna
    print("\nTraining XGBoost with Optuna (30 trials)...")
    xgb_model = train_xgb_optuna(X_train, y_train, X_val, y_val, n_trials=30)
    results['XGBoost'] = compute_metrics(y_val, xgb_model.predict(X_val))
    print(f"  Val: {results['XGBoost']}")
    if HAS_MLFLOW:
        mlflow.log_metrics({f"xgb_{k}": v for k, v in results['XGBoost'].items()})

    # Select best
    best_name = min(results, key=lambda k: results[k]['RMSE'])
    best_model = {'Ridge': ridge, 'RandomForest': rf, 'XGBoost': xgb_model}[best_name]
    print(f"\nBest model: {best_name} (val RMSE: {results[best_name]['RMSE']})")

    test_preds = best_model.predict(X_test)
    test_metrics = compute_metrics(y_test, test_preds)
    print(f"Test metrics: {test_metrics}")

    # Save predictions for error analysis
    test_df['predicted_delivery_minutes'] = test_preds
    test_df.to_pickle('data/processed/test_data_with_preds.pkl')

    # Serialize best model
    os.makedirs('models', exist_ok=True)
    artifact = {
        'model': best_model,
        'feature_names': feature_names,
        'model_name': best_name,
        'val_metrics': results[best_name],
        'test_metrics': test_metrics,
        'train_residuals_std': float(np.std(y_train - best_model.predict(X_train))),
    }
    with open('models/best_model.pkl', 'wb') as f:
        pickle.dump(artifact, f)
    print("Saved best model to models/best_model.pkl")

    plot_feature_importance(best_model, feature_names)

    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Validation Set)")
    print("=" * 70)
    print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'MAPE':<10} {'R2':<10}")
    print("-" * 70)
    for name, m in results.items():
        tag = " <-- BEST" if name == best_name else ""
        print(f"{name:<20} {m['RMSE']:<10} {m['MAE']:<10} {m['MAPE']:<10} {m['R2']:<10}{tag}")
    print("=" * 70)

    if HAS_MLFLOW:
        mlflow.end_run()


if __name__ == '__main__':
    main()
