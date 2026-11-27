"""
Data Loader Module for ElNino Health Resource Optimizer (EN-HRO)
Handles loading and validating raw and processed datasets, including GeoJSON boundaries.
"""

import json
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_raw_enso() -> pd.DataFrame:
    """Loads historical/current ENSO Oceanic Niño Index dataset."""
    path = os.path.join(DATA_RAW_DIR, "enso.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"ENSO dataset missing at {path}. Run generate_demo_data.py first.")
    return pd.read_csv(path)


def load_raw_climate() -> pd.DataFrame:
    """Loads raw climate dataset."""
    path = os.path.join(DATA_RAW_DIR, "climate.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Climate dataset missing at {path}. Run generate_demo_data.py first.")
    return pd.read_csv(path)


def load_raw_disease() -> pd.DataFrame:
    """Loads raw epidemiological and demand dataset."""
    path = os.path.join(DATA_RAW_DIR, "disease_demand.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Disease demand dataset missing at {path}. Run generate_demo_data.py first.")
    return pd.read_csv(path)


def load_raw_resources() -> pd.DataFrame:
    """Loads raw healthcare baseline resources dataset."""
    path = os.path.join(DATA_RAW_DIR, "healthcare_resources.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Healthcare resources dataset missing at {path}. Run generate_demo_data.py first.")
    return pd.read_csv(path)


def load_districts_geojson() -> dict:
    """Loads district GeoJSON boundaries for map visualization."""
    path = os.path.join(DATA_RAW_DIR, "districts.geojson")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Districts GeoJSON missing at {path}. Run generate_demo_data.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_processed_data() -> pd.DataFrame:
    """Loads unified regional healthcare optimization dataset."""
    path = os.path.join(DATA_PROCESSED_DIR, "healthcare_optimization_dataset.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset missing at {path}. Run generate_demo_data.py first.")
    df = pd.read_csv(path)
    return df
