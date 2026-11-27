"""
Demo Data Generator for ElNino Health Resource Optimizer (EN-HRO)
Generates scientifically framed, logically consistent synthetic datasets for 10 Maharashtra districts.

Disclaimer:
Prototype demonstration dataset — values are synthetic and must be replaced
with validated official datasets for real-world deployment.
"""

import json
import os
import numpy as np
import pandas as pd

# Set fixed seed for reproducibility
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

# 10 Representative Pilot Districts in Maharashtra
DISTRICTS = [
    {
        "district": "Mumbai",
        "state": "Maharashtra",
        "latitude": 19.0760,
        "longitude": 72.8777,
        "population": 12442373,
        "population_density": 20500,  # persons/sq km
        "distance_from_supply_center": 15,  # km (Central State Depot at Mumbai)
        "healthcare_capacity": 24000,  # base hospital beds
        "coastal": 1,
    },
    {
        "district": "Pune",
        "state": "Maharashtra",
        "latitude": 18.5204,
        "longitude": 73.8567,
        "population": 9429408,
        "population_density": 603,
        "distance_from_supply_center": 152,
        "healthcare_capacity": 16000,
        "coastal": 0,
    },
    {
        "district": "Nagpur",
        "state": "Maharashtra",
        "latitude": 21.1458,
        "longitude": 79.0882,
        "population": 4653570,
        "population_density": 470,
        "distance_from_supply_center": 820,
        "healthcare_capacity": 8500,
        "coastal": 0,
    },
    {
        "district": "Nashik",
        "state": "Maharashtra",
        "latitude": 19.9975,
        "longitude": 73.7898,
        "population": 6107187,
        "population_density": 393,
        "distance_from_supply_center": 167,
        "healthcare_capacity": 7200,
        "coastal": 0,
    },
    {
        "district": "Thane",
        "state": "Maharashtra",
        "latitude": 19.2183,
        "longitude": 72.9781,
        "population": 11060148,
        "population_density": 1157,
        "distance_from_supply_center": 35,
        "healthcare_capacity": 14000,
        "coastal": 1,
    },
    {
        "district": "Chhatrapati Sambhajinagar",
        "state": "Maharashtra",
        "latitude": 19.8762,
        "longitude": 75.3433,
        "population": 3701282,
        "population_density": 366,
        "distance_from_supply_center": 335,
        "healthcare_capacity": 5500,
        "coastal": 0,
    },
    {
        "district": "Kolhapur",
        "state": "Maharashtra",
        "latitude": 16.7050,
        "longitude": 74.2433,
        "population": 3876001,
        "population_density": 504,
        "distance_from_supply_center": 375,
        "healthcare_capacity": 5800,
        "coastal": 0,
    },
    {
        "district": "Amravati",
        "state": "Maharashtra",
        "latitude": 20.9320,
        "longitude": 77.7523,
        "population": 2888445,
        "population_density": 237,
        "distance_from_supply_center": 660,
        "healthcare_capacity": 4200,
        "coastal": 0,
    },
    {
        "district": "Solapur",
        "state": "Maharashtra",
        "latitude": 17.6599,
        "longitude": 75.9064,
        "population": 4317756,
        "population_density": 290,
        "distance_from_supply_center": 400,
        "healthcare_capacity": 5100,
        "coastal": 0,
    },
    {
        "district": "Ratnagiri",
        "state": "Maharashtra",
        "latitude": 16.9902,
        "longitude": 73.3120,
        "population": 1615262,
        "population_density": 197,
        "distance_from_supply_center": 330,
        "healthcare_capacity": 2800,
        "coastal": 1,
    },
]


def generate_enso_data():
    """Generates Oceanic Niño Index (ONI) dataset."""
    years = [2021, 2022, 2023, 2024, 2025, 2026]
    records = []
    for y in years:
        for m in range(1, 13):
            # Baseline cycle with 2024-2026 showing moderate to strong El Niño conditions
            if y == 2024:
                oni = 1.35 + 0.3 * np.sin(m / 12 * 2 * np.pi) + np.random.normal(0, 0.05)
            elif y == 2025:
                oni = 0.85 + 0.2 * np.cos(m / 12 * 2 * np.pi) + np.random.normal(0, 0.04)
            elif y == 2026:
                oni = 1.65 + 0.25 * np.sin(m / 12 * 2 * np.pi) + np.random.normal(0, 0.05)
            else:
                oni = -0.4 + 0.3 * np.sin(m / 12 * 2 * np.pi) + np.random.normal(0, 0.05)

            state = "El Niño" if oni >= 0.5 else ("La Niña" if oni <= -0.5 else "Neutral")
            records.append({
                "year": y,
                "month": m,
                "enso_index": round(float(oni), 3),
                "enso_state": state,
                "data_source_label": "Synthetic ONI demo series based on NOAA ONI anomaly patterns"
            })
    df_enso = pd.DataFrame(records)
    df_enso.to_csv(os.path.join(DATA_RAW_DIR, "enso.csv"), index=False)
    return df_enso


def generate_climate_data(current_enso_index=1.45, current_enso_state="El Niño"):
    """
    Generates district climate metrics.
    Logical framing:
    - High positive ENSO index (El Niño) induces warmer temperature anomalies across Maharashtra
      and erratic monsoon rainfall patterns (coastal excessive rain/flooding, inland rain deficits & heatwaves).
    """
    records = []
    for d in DISTRICTS:
        # Coastal districts experience higher humidity and flood surges; inland experience heatwaves
        is_coastal = d["coastal"]
        base_temp = 27.5 - 0.003 * d["distance_from_supply_center"] + (2.5 if not is_coastal else 0.5)
        # El Nino temperature anomaly is positive
        temp_anomaly = round(float(0.8 + 0.5 * current_enso_index + np.random.uniform(-0.2, 0.4)), 2)
        temp = round(base_temp + temp_anomaly, 2)

        if is_coastal:
            # Coastal flood surge during erratic monsoons
            base_rain = 2200.0
            rain_anomaly = round(float(180.0 * current_enso_index + np.random.uniform(-40, 60)), 1)
            flood_risk = round(float(min(0.95, 0.55 + 0.2 * (rain_anomaly / 200))), 3)
            heatwave_risk = round(float(min(0.80, 0.20 + 0.15 * temp_anomaly)), 3)
        else:
            # Inland rain deficit / unseasonal spurts & intense dry heat
            base_rain = 750.0
            rain_anomaly = round(float(-80.0 * current_enso_index + np.random.uniform(-30, 40)), 1)
            flood_risk = round(float(max(0.10, 0.30 + 0.1 * (rain_anomaly / 100))), 3)
            heatwave_risk = round(float(min(0.95, 0.45 + 0.25 * temp_anomaly)), 3)

        rainfall = round(max(100.0, base_rain + rain_anomaly), 1)

        records.append({
            "district": d["district"],
            "state": d["state"],
            "latitude": d["latitude"],
            "longitude": d["longitude"],
            "enso_index": current_enso_index,
            "enso_state": current_enso_state,
            "rainfall": rainfall,
            "rainfall_anomaly": rain_anomaly,
            "temperature": temp,
            "temperature_anomaly": temp_anomaly,
            "flood_risk": flood_risk,
            "heatwave_risk": heatwave_risk,
            "data_source_label": "Synthetic IMD/NOAA climate simulation for Maharashtra pilot"
        })
    df_climate = pd.DataFrame(records)
    df_climate.to_csv(os.path.join(DATA_RAW_DIR, "climate.csv"), index=False)
    return df_climate


def generate_disease_and_demand_data(df_climate):
    """
    Calculates disease risk and estimated healthcare demand based on transparent scientific formulas.
    Diseases:
    - Vector-borne: Dengue, Malaria (triggered by warm temperatures, stagnant rain pockets, urban density)
    - Water-borne: Cholera, Typhoid (triggered by flood contamination or drought-driven sanitation stress)
    """
    records = []
    for _, row in df_climate.iterrows():
        dist_meta = next(d for d in DISTRICTS if d["district"] == row["district"])
        pop = dist_meta["population"]
        pop_density = dist_meta["population_density"]
        temp_anom = row["temperature_anomaly"]
        rain_anom = row["rainfall_anomaly"]
        flood_risk = row["flood_risk"]
        heat_risk = row["heatwave_risk"]

        # Vector-borne risk (dengue/malaria): increases with temperature anomaly, humidity & high density
        norm_density = min(1.0, pop_density / 15000)
        vector_borne_risk = round(float(
            np.clip(0.35 + 0.25 * (temp_anom / 2.0) + 0.20 * norm_density + 0.15 * flood_risk + np.random.normal(0, 0.03), 0.1, 0.98)
        ), 3)

        # Water-borne risk (cholera/typhoid): increases with severe flood or severe drought (water scarcity)
        water_stress = 0.4 * flood_risk + 0.4 * heat_risk + 0.2 * norm_density
        waterborne_disease_risk = round(float(
            np.clip(0.30 + 0.40 * water_stress + np.random.normal(0, 0.03), 0.1, 0.95)
        ), 3)

        # Composite Disease Risk (weighted blend of vector and waterborne risks)
        disease_risk = round(float(0.55 * vector_borne_risk + 0.45 * waterborne_disease_risk), 3)

        # Baseline expected cases per 100,000 population scaled by disease risk
        case_rate_per_100k = 180 + 350 * disease_risk + 40 * (pop_density / 5000)
        estimated_disease_cases = int(round((pop / 100000.0) * case_rate_per_100k))

        # Demand estimations:
        # Vaccine demand (preventative target cohort: ~1.2 to 1.8x estimated case burden in risk zones)
        vaccine_demand = int(round(estimated_disease_cases * (1.3 + 0.5 * disease_risk)))
        # Medicine demand (treatments packs: ~1.05 to 1.3x estimated cases)
        medicine_demand = int(round(estimated_disease_cases * (1.1 + 0.3 * disease_risk)))
        # Hospital bed demand (acute cases requiring admission: ~15% to 25% of cases)
        bed_demand = int(round(estimated_disease_cases * (0.16 + 0.10 * disease_risk)))
        # Medical staff demand (doctors & nurses required: 1 per 8-12 acute patients)
        staff_demand = int(round(bed_demand / 10.0))

        # Priority score (scale 0-100 based on risk, population density, and disease burden)
        priority_score = round(float(
            np.clip(
                disease_risk * 50 + (estimated_disease_cases / (pop / 100000.0)) * 0.08 + norm_density * 20,
                15.0, 99.0
            )
        ), 1)

        records.append({
            "district": row["district"],
            "vector_borne_risk": vector_borne_risk,
            "waterborne_disease_risk": waterborne_disease_risk,
            "disease_risk": disease_risk,
            "estimated_disease_cases": estimated_disease_cases,
            "vaccine_demand": vaccine_demand,
            "medicine_demand": medicine_demand,
            "bed_demand": bed_demand,
            "staff_demand": staff_demand,
            "priority_score": priority_score,
            "risk_level": "High" if disease_risk >= 0.70 else ("Medium" if disease_risk >= 0.45 else "Low")
        })

    df_disease = pd.DataFrame(records)
    df_disease.to_csv(os.path.join(DATA_RAW_DIR, "disease_demand.csv"), index=False)
    return df_disease


def generate_healthcare_resources_data(df_disease):
    """
    Generates available healthcare resources and transportation costs.
    Supply center is located in Mumbai (Central Medical Supply Depot).
    """
    records = []
    for _, row in df_disease.iterrows():
        dist_meta = next(d for d in DISTRICTS if d["district"] == row["district"])
        dist_km = dist_meta["distance_from_supply_center"]
        base_cap = dist_meta["healthcare_capacity"]

        # Locally available on-hand baseline inventory before emergency allocation
        # Usually local stock covers 40% to 70% of peak surge demand
        avail_vax = int(round(row["vaccine_demand"] * np.random.uniform(0.35, 0.55)))
        avail_med = int(round(row["medicine_demand"] * np.random.uniform(0.40, 0.60)))
        avail_bed = int(round(row["bed_demand"] * np.random.uniform(0.45, 0.65)))
        avail_staff = int(round(row["staff_demand"] * np.random.uniform(0.45, 0.65)))

        # Transport cost per unit:
        # Based on distance from central depot (INR 12 per km flat freight base + volume factor)
        # Flat cost factor per unit: 15 base + 0.15 * distance_km
        transport_cost_per_unit = round(float(12.0 + 0.18 * dist_km), 2)

        records.append({
            "district": row["district"],
            "distance_from_supply_center": dist_km,
            "base_hospital_capacity": base_cap,
            "available_vaccines": avail_vax,
            "available_medicines": avail_med,
            "available_beds": avail_bed,
            "available_staff": avail_staff,
            "transport_cost_per_unit": transport_cost_per_unit,
            "data_source_label": "Synthetic Maharashtra Health Department logistics baseline"
        })

    df_resources = pd.DataFrame(records)
    df_resources.to_csv(os.path.join(DATA_RAW_DIR, "healthcare_resources.csv"), index=False)
    return df_resources


def generate_districts_geojson():
    """
    Generates a valid GeoJSON FeatureCollection with polygon boundaries for the 10 pilot districts.
    Provides realistic bounding boxes centered around actual district geographical coordinates.
    """
    features = []
    # Approximate polygon radii for each district (in degrees, ~0.25 - 0.45 deg)
    offsets = {
        "Mumbai": [(-0.15, -0.08), (-0.15, 0.08), (0.15, 0.08), (0.15, -0.08)],
        "Pune": [(-0.35, -0.40), (-0.35, 0.40), (0.35, 0.40), (0.35, -0.40)],
        "Nagpur": [(-0.30, -0.35), (-0.30, 0.35), (0.30, 0.35), (0.30, -0.35)],
        "Nashik": [(-0.32, -0.35), (-0.32, 0.35), (0.32, 0.35), (0.32, -0.35)],
        "Thane": [(-0.25, -0.22), (-0.25, 0.22), (0.25, 0.22), (0.25, -0.22)],
        "Chhatrapati Sambhajinagar": [(-0.30, -0.32), (-0.30, 0.32), (0.30, 0.32), (0.30, -0.32)],
        "Kolhapur": [(-0.28, -0.30), (-0.28, 0.30), (0.28, 0.30), (0.28, -0.30)],
        "Amravati": [(-0.32, -0.35), (-0.32, 0.35), (0.32, 0.35), (0.32, -0.35)],
        "Solapur": [(-0.35, -0.38), (-0.35, 0.38), (0.35, 0.38), (0.35, -0.38)],
        "Ratnagiri": [(-0.40, -0.18), (-0.40, 0.18), (0.40, 0.18), (0.40, -0.18)],
    }

    for d in DISTRICTS:
        name = d["district"]
        lat = d["latitude"]
        lon = d["longitude"]
        delta = offsets.get(name, [(-0.25, -0.25), (-0.25, 0.25), (0.25, 0.25), (0.25, -0.25)])

        # Create closed polygon coordinates [lon, lat]
        polygon = []
        for d_lat, d_lon in delta:
            polygon.append([round(lon + d_lon, 4), round(lat + d_lat, 4)])
        polygon.append(polygon[0])  # Close ring

        feature = {
            "type": "Feature",
            "properties": {
                "district": name,
                "state": d["state"],
                "population": d["population"],
                "population_density": d["population_density"],
                "latitude": lat,
                "longitude": lon,
            },
            "geometry": {
                "type": "Polygon",
                "coordinates": [polygon]
            }
        }
        features.append(feature)

    geojson_data = {
        "type": "FeatureCollection",
        "features": features
    }

    geojson_path = os.path.join(DATA_RAW_DIR, "districts.geojson")
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_data, f, indent=2)

    return geojson_path


def build_unified_processed_dataset(df_climate, df_disease, df_resources):
    """
    Merges all raw modules into a unified regional dataset with complete schema as required by Section 9.
    """
    # Merge on 'district'
    merged = df_climate.merge(df_disease, on="district")
    merged = merged.merge(df_resources, on="district")

    # Add metadata
    meta_df = pd.DataFrame(DISTRICTS)[["district", "population", "population_density", "healthcare_capacity"]]
    merged = merged.merge(meta_df, on="district")

    # Rename transport_cost_per_unit to transport_cost for schema compatibility
    merged["transport_cost"] = merged["transport_cost_per_unit"]

    # Reorder columns logically
    cols = [
        "district", "state", "latitude", "longitude", "population", "population_density",
        "rainfall", "rainfall_anomaly", "temperature", "temperature_anomaly",
        "enso_index", "enso_state", "flood_risk", "heatwave_risk",
        "vector_borne_risk", "waterborne_disease_risk", "disease_risk", "risk_level",
        "estimated_disease_cases",
        "vaccine_demand", "medicine_demand", "bed_demand", "staff_demand",
        "available_vaccines", "available_medicines", "available_beds", "available_staff",
        "healthcare_capacity", "distance_from_supply_center", "transport_cost", "priority_score"
    ]

    final_df = merged[cols].copy()
    processed_path = os.path.join(DATA_PROCESSED_DIR, "healthcare_optimization_dataset.csv")
    final_df.to_csv(processed_path, index=False)
    print(f"Generated unified processed dataset: {processed_path} ({len(final_df)} records)")
    return final_df


def generate_all_data():
    """Generates all raw and processed demo datasets."""
    print("Generating demo data for ElNino Health Resource Optimizer...")
    generate_enso_data()
    df_climate = generate_climate_data(current_enso_index=1.45, current_enso_state="El Niño")
    df_disease = generate_disease_and_demand_data(df_climate)
    df_resources = generate_healthcare_resources_data(df_disease)
    generate_districts_geojson()
    unified_df = build_unified_processed_dataset(df_climate, df_disease, df_resources)
    print("All demo data generated successfully.")
    return unified_df


if __name__ == "__main__":
    generate_all_data()
