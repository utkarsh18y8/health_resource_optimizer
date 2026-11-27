"""
Preprocessing Module for ElNino Health Resource Optimizer (EN-HRO)
Validates schema, cleans anomalies, handles nulls, and standardizes data types.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

REQUIRED_COLUMNS: List[str] = [
    "district", "state", "latitude", "longitude", "population",
    "rainfall", "rainfall_anomaly", "temperature", "temperature_anomaly",
    "enso_index", "enso_state", "disease_risk", "estimated_disease_cases",
    "vaccine_demand", "medicine_demand", "bed_demand", "staff_demand",
    "available_vaccines", "available_medicines", "available_beds", "available_staff",
    "priority_score", "transport_cost"
]


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Checks if the dataset contains all mandatory columns and valid data types.
    Returns (is_valid, list_of_issues).
    """
    issues = []
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Missing required columns: {missing_cols}")

    if df.empty:
        issues.append("Dataset is empty.")

    # Check for negative resource demands
    numeric_cols = ["vaccine_demand", "medicine_demand", "bed_demand", "staff_demand", "population"]
    for col in numeric_cols:
        if col in df.columns and (df[col] < 0).any():
            issues.append(f"Negative values detected in {col}.")

    return (len(issues) == 0, issues)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw/merged data by imputing missing values, enforcing types, and bounding risks.
    """
    cleaned = df.copy()

    # Numeric conversions
    numeric_fields = [
        "latitude", "longitude", "population", "population_density",
        "rainfall", "rainfall_anomaly", "temperature", "temperature_anomaly",
        "enso_index", "flood_risk", "heatwave_risk", "vector_borne_risk",
        "waterborne_disease_risk", "disease_risk", "estimated_disease_cases",
        "vaccine_demand", "medicine_demand", "bed_demand", "staff_demand",
        "available_vaccines", "available_medicines", "available_beds", "available_staff",
        "transport_cost", "priority_score"
    ]

    for col in numeric_fields:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            # Fill NaNs with median
            if cleaned[col].isnull().any():
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # Ensure risk values are strictly bounded in [0.0, 1.0]
    risk_fields = ["flood_risk", "heatwave_risk", "vector_borne_risk", "waterborne_disease_risk", "disease_risk"]
    for r in risk_fields:
        if r in cleaned.columns:
            cleaned[r] = cleaned[r].clip(0.0, 1.0)

    # Ensure integer demands are non-negative integers
    int_fields = [
        "population", "estimated_disease_cases",
        "vaccine_demand", "medicine_demand", "bed_demand", "staff_demand",
        "available_vaccines", "available_medicines", "available_beds", "available_staff"
    ]
    for col in int_fields:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].clip(lower=0).round().astype(int)

    # Ensure string fields are clean
    for str_col in ["district", "state", "enso_state"]:
        if str_col in cleaned.columns:
            cleaned[str_col] = cleaned[str_col].astype(str).str.strip()

    return cleaned
