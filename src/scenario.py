"""
Scenario Simulation Module for ElNino Health Resource Optimizer (EN-HRO)
Simulates what-if climate shocks and resource stockpile deficits,
re-estimating epidemiological demand and resolving ILP allocations.
"""

from typing import Dict, Any, Tuple
import pandas as pd
from src.feature_engineering import compute_epidemiological_demand
from src.optimization import run_resource_optimization


def run_scenario_simulation(
    base_df: pd.DataFrame,
    enso_multiplier: float = 1.0,
    temp_anomaly_delta: float = 0.0,
    rainfall_anomaly_multiplier: float = 1.0,
    vaccine_multiplier: float = 1.0,
    medicine_multiplier: float = 1.0,
    bed_multiplier: float = 1.0,
    staff_multiplier: float = 1.0,
    transport_budget: float = None,
    fairness_ratio: float = 0.15
) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Simulates a counterfactual or emergency scenario and compares it against baseline.
    Returns:
        (comparison_dict, baseline_alloc_df, scenario_alloc_df)
    """
    # 1. Baseline Run
    base_vax = int(round(base_df["vaccine_demand"].sum() * 0.78))
    base_med = int(round(base_df["medicine_demand"].sum() * 0.82))
    base_bed = int(round(base_df["bed_demand"].sum() * 0.75))
    base_staff = int(round(base_df["staff_demand"].sum() * 0.75))
    baseline_resources = {
        "vaccines": base_vax,
        "medicines": base_med,
        "beds": base_bed,
        "staff": base_staff
    }

    base_status, base_alloc, base_summary, _ = run_resource_optimization(
        base_df,
        total_resources=baseline_resources,
        fairness_ratio=fairness_ratio
    )

    # 2. Recalculate Climate & Epidemiological Demand for Scenario
    scenario_input_df = base_df.copy()
    scenario_input_df["temperature_anomaly"] += float(temp_anomaly_delta)

    scenario_demand_df = compute_epidemiological_demand(
        scenario_input_df,
        enso_multiplier=float(enso_multiplier),
        rainfall_multiplier=float(rainfall_anomaly_multiplier),
        temp_multiplier=1.0  # Already added delta above
    )

    # Adjusted Available Resources
    scenario_resources = {
        "vaccines": max(100, int(round(base_vax * float(vaccine_multiplier)))),
        "medicines": max(100, int(round(base_med * float(medicine_multiplier)))),
        "beds": max(50, int(round(base_bed * float(bed_multiplier)))),
        "staff": max(10, int(round(base_staff * float(staff_multiplier)))),
    }

    # 3. Solve ILP on Scenario
    scen_status, scen_alloc, scen_summary, _ = run_resource_optimization(
        scenario_demand_df,
        total_resources=scenario_resources,
        transport_budget=transport_budget,
        fairness_ratio=fairness_ratio
    )

    # 4. Comparative Metrics
    comparison = {
        "baseline": {
            "status": base_status,
            "total_demand": base_summary.get("total_demand", 0),
            "total_allocated": base_summary.get("total_allocated", 0),
            "coverage_pct": base_summary.get("overall_coverage_pct", 0.0),
            "unmet_demand": base_summary.get("total_unmet_demand", 0),
            "transport_cost": base_summary.get("total_transport_cost", 0.0),
            "utilization_pct": base_summary.get("resource_utilization_pct", 0.0),
            "high_risk_districts": int((base_demand_df["risk_level"] == "High").sum()) if "base_demand_df" in locals() else int((base_df["risk_level"] == "High").sum())
        },
        "scenario": {
            "status": scen_status,
            "total_demand": scen_summary.get("total_demand", 0),
            "total_allocated": scen_summary.get("total_allocated", 0),
            "coverage_pct": scen_summary.get("overall_coverage_pct", 0.0),
            "unmet_demand": scen_summary.get("total_unmet_demand", 0),
            "transport_cost": scen_summary.get("total_transport_cost", 0.0),
            "utilization_pct": scen_summary.get("resource_utilization_pct", 0.0),
            "high_risk_districts": int((scenario_demand_df["risk_level"] == "High").sum())
        },
        "changes": {
            "demand_change": int(scen_summary.get("total_demand", 0) - base_summary.get("total_demand", 0)),
            "demand_change_pct": round(
                (scen_summary.get("total_demand", 0) - base_summary.get("total_demand", 0)) / max(1, base_summary.get("total_demand", 1)) * 100.0, 1
            ),
            "coverage_change_pp": round(
                scen_summary.get("overall_coverage_pct", 0.0) - base_summary.get("overall_coverage_pct", 0.0), 2
            ),
            "unmet_demand_change": int(scen_summary.get("total_unmet_demand", 0) - base_summary.get("total_unmet_demand", 0)),
            "transport_cost_change": round(
                scen_summary.get("total_transport_cost", 0.0) - base_summary.get("total_transport_cost", 0.0), 2
            )
        }
    }

    return comparison, base_alloc, scen_alloc
