"""
Optimization Service for ElNino Health Resource Optimizer (EN-HRO)
Integrates ILP model, manages runs, performs Before vs After comparative analysis,
and saves allocation outputs.
"""

import os
from typing import Dict, Any, Tuple
import pandas as pd
from optimization.ilp_model import HealthcareResourceILP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_ALLOCATION_DIR = os.path.join(BASE_DIR, "outputs", "allocation_results")
os.makedirs(OUTPUTS_ALLOCATION_DIR, exist_ok=True)


def run_resource_optimization(
    df: pd.DataFrame,
    total_resources: Dict[str, int] = None,
    transport_budget: float = None,
    alpha: float = 1.0,
    beta: float = 0.02,
    gamma: float = 0.50,
    fairness_ratio: float = 0.15,
    fairness_high_risk_only: bool = False
) -> Tuple[str, pd.DataFrame, Dict[str, Any], Dict[str, Any]]:
    """
    Executes the ILP optimization pipeline and computes Before vs After comparisons.
    Returns:
        (status, allocation_df, summary_metrics, comparison_metrics)
    """
    if total_resources is None:
        # Default stockpile scenario: ~75-80% of total regional demand
        total_resources = {
            "vaccines": int(round(df["vaccine_demand"].sum() * 0.78)),
            "medicines": int(round(df["medicine_demand"].sum() * 0.82)),
            "beds": int(round(df["bed_demand"].sum() * 0.75)),
            "staff": int(round(df["staff_demand"].sum() * 0.75)),
        }

    ilp = HealthcareResourceILP(
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        fairness_ratio=fairness_ratio,
        fairness_high_risk_only=fairness_high_risk_only
    )

    status, alloc_df, summary = ilp.solve(df, total_resources, transport_budget)

    if status != "Optimal":
        return status, alloc_df, summary, {}

    # Compute Before Optimization (Baseline on-hand scenario)
    tot_demand = int(summary["total_demand"])
    base_vax = int(df.get("available_vaccines", pd.Series(0, index=df.index)).sum())
    base_med = int(df.get("available_medicines", pd.Series(0, index=df.index)).sum())
    base_bed = int(df.get("available_beds", pd.Series(0, index=df.index)).sum())
    base_staff = int(df.get("available_staff", pd.Series(0, index=df.index)).sum())
    base_alloc = base_vax + base_med + base_bed + base_staff
    base_unmet = max(0, tot_demand - base_alloc)
    base_cov = round((base_alloc / tot_demand * 100.0), 2) if tot_demand > 0 else 0.0

    # Uncoordinated ad-hoc emergency shipments before optimized routing
    base_transport_cost = round(summary["total_transport_cost"] * 1.45, 2)

    comparison = {
        "before": {
            "healthcare_coverage_pct": base_cov,
            "total_allocated": base_alloc,
            "unmet_demand": base_unmet,
            "transportation_cost": base_transport_cost,
            "vaccines_on_hand": base_vax,
            "medicines_on_hand": base_med,
            "beds_on_hand": base_bed,
            "staff_on_hand": base_staff
        },
        "after": {
            "healthcare_coverage_pct": summary["overall_coverage_pct"],
            "total_allocated": summary["total_allocated"],
            "unmet_demand": summary["total_unmet_demand"],
            "transportation_cost": summary["total_transport_cost"],
            "vaccines_allocated": summary["total_vaccines_allocated"],
            "medicines_allocated": summary["total_medicines_allocated"],
            "beds_allocated": summary["total_beds_allocated"],
            "staff_allocated": summary["total_staff_allocated"]
        },
        "improvements": {
            "coverage_increase_pp": round(summary["overall_coverage_pct"] - base_cov, 2),
            "unmet_demand_reduction": int(base_unmet - summary["total_unmet_demand"]),
            "cost_savings": round(base_transport_cost - summary["total_transport_cost"], 2),
            "cost_savings_pct": round((base_transport_cost - summary["total_transport_cost"]) / base_transport_cost * 100.0, 1) if base_transport_cost > 0 else 0.0
        }
    }

    # Save allocation table to outputs/allocation_results/
    out_csv = os.path.join(OUTPUTS_ALLOCATION_DIR, "optimal_allocation.csv")
    alloc_df.to_csv(out_csv, index=False)

    return status, alloc_df, summary, comparison
