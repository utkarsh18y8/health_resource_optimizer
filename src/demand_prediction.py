"""
Demand Prediction Machine Learning Module for ElNino Health Resource Optimizer (EN-HRO)
Trains a regression model to estimate regional disease burden from climate and demographic variables.

Status: Prototype demand estimation model
"""

import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
OUTPUTS_FIG_DIR = os.path.join(BASE_DIR, "outputs", "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_METRICS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_FIG_DIR, exist_ok=True)

FEATURE_COLUMNS = [
    "enso_index",
    "rainfall",
    "rainfall_anomaly",
    "temperature",
    "temperature_anomaly",
    "population",
    "population_density",
    "flood_risk",
    "heatwave_risk",
]
TARGET_COLUMN = "estimated_disease_cases"


def train_demand_model(data_path: str = None) -> dict:
    """
    Trains a Random Forest demand estimation model on regional and historical variation profiles.
    Saves trained artifacts and evaluation metrics.
    """
    if data_path is None:
        data_path = os.path.join(BASE_DIR, "data", "processed", "healthcare_optimization_dataset.csv")

    base_df = pd.read_csv(data_path)

    # To provide robust statistical training for the pilot prototype,
    # generate a realistic parametric cohort of 300 regional epidemiological simulations
    # based on the 10 pilot districts under varying climate anomalies.
    np.random.seed(42)
    cohort_records = []
    for _ in range(30):
        for _, row in base_df.iterrows():
            noise_enso = row["enso_index"] + np.random.uniform(-0.6, 0.8)
            noise_temp_anom = np.clip(row["temperature_anomaly"] + np.random.uniform(-0.5, 0.7), -0.5, 4.0)
            noise_rain_anom = row["rainfall_anomaly"] + np.random.uniform(-100, 120)
            noise_temp = row["temperature"] + (noise_temp_anom - row["temperature_anomaly"])
            noise_rain = max(50.0, row["rainfall"] + (noise_rain_anom - row["rainfall_anomaly"]))
            noise_flood = np.clip(row["flood_risk"] + np.random.uniform(-0.1, 0.15), 0.05, 0.98)
            noise_heat = np.clip(row["heatwave_risk"] + np.random.uniform(-0.1, 0.15), 0.05, 0.98)

            norm_density = min(1.0, row["population_density"] / 15000.0)
            v_risk = np.clip(0.32 + 0.26 * (noise_temp_anom / 2.5) + 0.18 * norm_density + 0.14 * noise_flood, 0.1, 0.99)
            w_risk = np.clip(0.28 + 0.35 * noise_flood + 0.25 * noise_heat + 0.12 * norm_density, 0.1, 0.99)
            comp_risk = 0.55 * v_risk + 0.45 * w_risk

            rate = 180 + 360 * comp_risk + 50 * norm_density + np.random.normal(0, 10)
            cases = int(max(50, round((row["population"] / 100000.0) * rate)))

            cohort_records.append({
                "enso_index": round(float(noise_enso), 3),
                "rainfall": round(float(noise_rain), 1),
                "rainfall_anomaly": round(float(noise_rain_anom), 1),
                "temperature": round(float(noise_temp), 2),
                "temperature_anomaly": round(float(noise_temp_anom), 2),
                "population": int(row["population"]),
                "population_density": float(row["population_density"]),
                "flood_risk": round(float(noise_flood), 3),
                "heatwave_risk": round(float(noise_heat), 3),
                "estimated_disease_cases": cases
            })

    dataset = pd.DataFrame(cohort_records)

    X = dataset[FEATURE_COLUMNS]
    y = dataset[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "model_type": "Random Forest Regressor",
        "description": "Prototype demand estimation model (climate-aware disease cases)",
        "samples_trained": len(X_train),
        "samples_tested": len(X_test),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2_score": round(r2, 4),
        "feature_names": FEATURE_COLUMNS,
        "feature_importances": {
            feat: round(float(imp), 4) for feat, imp in zip(FEATURE_COLUMNS, model.feature_importances_)
        }
    }

    # Save model and columns
    model_path = os.path.join(MODELS_DIR, "demand_model.pkl")
    cols_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
    joblib.dump(model, model_path)
    joblib.dump(FEATURE_COLUMNS, cols_path)

    # Save metrics JSON
    metrics_path = os.path.join(OUTPUTS_METRICS_DIR, "demand_model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Generate and save feature importance chart
    try:
        plt.figure(figsize=(9, 5))
        sorted_idx = np.argsort(model.feature_importances_)
        plt.barh(np.array(FEATURE_COLUMNS)[sorted_idx], model.feature_importances_[sorted_idx], color="#1E88E5")
        plt.title("Prototype Demand Model — Feature Importances", fontsize=12, fontweight="bold")
        plt.xlabel("Gini Importance Score")
        plt.tight_layout()
        chart_path = os.path.join(OUTPUTS_FIG_DIR, "feature_importance.png")
        plt.savefig(chart_path, dpi=200)
        plt.close()
    except Exception as e:
        print(f"Feature chart generation error: {e}")

    print(f"Demand model trained: R2={r2:.4f}, MAE={mae:.2f}, RMSE={rmse:.2f}")
    return metrics


def load_demand_model():
    """Loads pre-trained demand model and feature list."""
    model_path = os.path.join(MODELS_DIR, "demand_model.pkl")
    cols_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
    if not os.path.exists(model_path) or not os.path.exists(cols_path):
        train_demand_model()
    model = joblib.load(model_path)
    cols = joblib.load(cols_path)
    return model, cols


def predict_demand(df_input: pd.DataFrame) -> np.ndarray:
    """Predicts disease cases using trained Random Forest model."""
    model, cols = load_demand_model()
    X = df_input[cols]
    preds = model.predict(X)
    return np.round(np.clip(preds, 0, None)).astype(int)


if __name__ == "__main__":
    train_demand_model()
