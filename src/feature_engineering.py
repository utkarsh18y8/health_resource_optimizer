"""
Feature Engineering Module for ElNino Health Resource Optimizer (EN-HRO)
Derives climate indicators, epidemiological vulnerability scores, and resource demand profiles.
"""

import numpy as np
import pandas as pd


def compute_climate_risk_index(df: pd.DataFrame) -> pd.Series:
    """
    Computes a composite climate risk score [0, 1] incorporating:
    - ENSO intensity (El Niño positive anomaly)
    - Normalized temperature anomaly
    - Extreme rainfall or drought deviation
    - Heatwave / flood risk indicators
    """
    enso_norm = np.clip((df["enso_index"] - 0.0) / 2.5, 0.0, 1.0)
    temp_norm = np.clip(df["temperature_anomaly"] / 3.0, 0.0, 1.0)
    rain_norm = np.clip(np.abs(df["rainfall_anomaly"]) / 400.0, 0.0, 1.0)
    flood = df.get("flood_risk", pd.Series(0.3, index=df.index))
    heat = df.get("heatwave_risk", pd.Series(0.3, index=df.index))

    climate_index = 0.30 * enso_norm + 0.25 * temp_norm + 0.20 * rain_norm + 0.15 * flood + 0.10 * heat
    return climate_index.clip(0.0, 1.0).round(3)


def compute_epidemiological_demand(
    df: pd.DataFrame,
    enso_multiplier: float = 1.0,
    rainfall_multiplier: float = 1.0,
    temp_multiplier: float = 1.0
) -> pd.DataFrame:
    """
    Transparently recalculates disease risk and resource demands under base or simulated climate conditions.
    Formula structure:
    Disease Risk = f(ENSO, Temp Anomaly, Rain Anomaly, Population Density, Flood/Heatwave Risk)
    """
    df_out = df.copy()

    # Apply adjustments
    effective_enso = df_out["enso_index"] * enso_multiplier
    effective_temp_anom = df_out["temperature_anomaly"] * temp_multiplier
    effective_rain_anom = df_out["rainfall_anomaly"] * rainfall_multiplier

    pop_density = df_out.get("population_density", df_out["population"] / 1000.0)
    norm_density = np.clip(pop_density / 15000.0, 0.05, 1.0)

    flood_risk = np.clip(
        df_out.get("flood_risk", 0.3) + 0.15 * (effective_rain_anom / 200.0), 0.05, 0.98
    )
    heat_risk = np.clip(
        df_out.get("heatwave_risk", 0.3) + 0.12 * (effective_temp_anom / 2.0), 0.05, 0.98
    )

    # Vector-borne (dengue/malaria) surge
    vector_risk = np.clip(
        0.30 + 0.28 * (effective_temp_anom / 2.5) + 0.18 * norm_density + 0.14 * flood_risk + 0.10 * (effective_enso / 2.0),
        0.05, 0.99
    )

    # Water-borne (cholera/typhoid) surge
    water_risk = np.clip(
        0.28 + 0.35 * flood_risk + 0.25 * heat_risk + 0.12 * norm_density,
        0.05, 0.99
    )

    composite_risk = (0.55 * vector_risk + 0.45 * water_risk).round(3)

    # Estimated cases per 100k
    case_rate_per_100k = 180 + 360 * composite_risk + 50 * norm_density
    est_cases = np.round((df_out["population"] / 100000.0) * case_rate_per_100k).astype(int)

    # Resource requirements
    vax_demand = np.round(est_cases * (1.3 + 0.5 * composite_risk)).astype(int)
    med_demand = np.round(est_cases * (1.1 + 0.3 * composite_risk)).astype(int)
    bed_demand = np.round(est_cases * (0.16 + 0.10 * composite_risk)).astype(int)
    staff_demand = np.round(bed_demand / 10.0).astype(int)

    # Priority score (0 - 100)
    priority = np.clip(
        composite_risk * 50.0 + (est_cases / (df_out["population"] / 100000.0)) * 0.08 + norm_density * 20.0,
        15.0, 99.0
    ).round(1)

    df_out["enso_index"] = effective_enso.round(3)
    df_out["temperature_anomaly"] = effective_temp_anom.round(2)
    df_out["rainfall_anomaly"] = effective_rain_anom.round(1)
    df_out["vector_borne_risk"] = vector_risk.round(3)
    df_out["waterborne_disease_risk"] = water_risk.round(3)
    df_out["disease_risk"] = composite_risk
    df_out["risk_level"] = np.where(composite_risk >= 0.70, "High", np.where(composite_risk >= 0.45, "Medium", "Low"))
    df_out["estimated_disease_cases"] = est_cases
    df_out["vaccine_demand"] = vax_demand
    df_out["medicine_demand"] = med_demand
    df_out["bed_demand"] = bed_demand
    df_out["staff_demand"] = staff_demand
    df_out["priority_score"] = priority

    return df_out
