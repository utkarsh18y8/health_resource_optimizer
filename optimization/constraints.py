"""
Constraints Module for Integer Linear Programming (ILP)
Implements mathematical constraints for healthcare and vaccine distribution.
"""

from typing import Dict, List
import pulp
import pandas as pd


def add_supply_capacity_constraints(
    model: pulp.LpProblem,
    decision_vars: Dict[str, Dict[str, pulp.LpVariable]],
    total_available: Dict[str, int]
) -> None:
    """
    Constraint 1-4: Total allocated resources cannot exceed total available stockpiles.
    sum_i(V_i) <= Total Vaccines, sum_i(M_i) <= Total Medicines, etc.
    """
    districts = list(decision_vars["vaccines"].keys())

    if "vaccines" in total_available and total_available["vaccines"] is not None:
        model += (
            pulp.lpSum([decision_vars["vaccines"][d] for d in districts]) <= int(total_available["vaccines"]),
            "Total_Vaccine_Supply_Limit"
        )

    if "medicines" in total_available and total_available["medicines"] is not None:
        model += (
            pulp.lpSum([decision_vars["medicines"][d] for d in districts]) <= int(total_available["medicines"]),
            "Total_Medicine_Supply_Limit"
        )

    if "beds" in total_available and total_available["beds"] is not None:
        model += (
            pulp.lpSum([decision_vars["beds"][d] for d in districts]) <= int(total_available["beds"]),
            "Total_Bed_Supply_Limit"
        )

    if "staff" in total_available and total_available["staff"] is not None:
        model += (
            pulp.lpSum([decision_vars["staff"][d] for d in districts]) <= int(total_available["staff"]),
            "Total_Staff_Supply_Limit"
        )


def add_demand_upper_bounds(
    model: pulp.LpProblem,
    decision_vars: Dict[str, Dict[str, pulp.LpVariable]],
    df: pd.DataFrame
) -> None:
    """
    Constraint 5: Regional demand cap.
    Allocation to region i should not exceed regional requirement (prevents wastage).
    V_i <= demand_i
    """
    for _, row in df.iterrows():
        d = row["district"]
        model += decision_vars["vaccines"][d] <= int(row["vaccine_demand"]), f"Demand_Cap_Vax_{d}"
        model += decision_vars["medicines"][d] <= int(row["medicine_demand"]), f"Demand_Cap_Med_{d}"
        model += decision_vars["beds"][d] <= int(row["bed_demand"]), f"Demand_Cap_Bed_{d}"
        model += decision_vars["staff"][d] <= int(row["staff_demand"]), f"Demand_Cap_Staff_{d}"


def add_fairness_constraints(
    model: pulp.LpProblem,
    decision_vars: Dict[str, Dict[str, pulp.LpVariable]],
    df: pd.DataFrame,
    min_coverage_ratio: float = 0.15,
    apply_to_high_risk_only: bool = False
) -> None:
    """
    Constraint 6: Fairness and equity guarantee.
    Ensures that vulnerable or all districts receive a minimum guaranteed proportion of their required demand.
    Allocation_i >= floor(min_coverage_ratio * Demand_i)
    """
    if min_coverage_ratio <= 0.0:
        return

    for _, row in df.iterrows():
        d = row["district"]
        is_high_risk = (row.get("risk_level") == "High") or (row.get("disease_risk", 0) >= 0.65)

        if apply_to_high_risk_only and not is_high_risk:
            continue

        min_vax = int(int(row["vaccine_demand"]) * min_coverage_ratio)
        min_med = int(int(row["medicine_demand"]) * min_coverage_ratio)
        min_bed = int(int(row["bed_demand"]) * min_coverage_ratio)
        min_staff = int(int(row["staff_demand"]) * min_coverage_ratio)

        if min_vax > 0:
            model += decision_vars["vaccines"][d] >= min_vax, f"Fairness_Min_Vax_{d}"
        if min_med > 0:
            model += decision_vars["medicines"][d] >= min_med, f"Fairness_Min_Med_{d}"
        if min_bed > 0:
            model += decision_vars["beds"][d] >= min_bed, f"Fairness_Min_Bed_{d}"
        if min_staff > 0:
            model += decision_vars["staff"][d] >= min_staff, f"Fairness_Min_Staff_{d}"


def add_transport_budget_constraint(
    model: pulp.LpProblem,
    decision_vars: Dict[str, Dict[str, pulp.LpVariable]],
    df: pd.DataFrame,
    transport_budget: float
) -> None:
    """
    Constraint 7: Transportation budget.
    Sum over i of (Unit Transport Cost_i * Resources_i) <= Transportation Budget
    """
    if transport_budget is None or transport_budget <= 0:
        return

    cost_terms = []
    for _, row in df.iterrows():
        d = row["district"]
        c_i = float(row.get("transport_cost", 20.0))
        # Weighted resource volume cost: vaccines & medicines (1 unit load), bed/staff deployment (3 unit load)
        dist_alloc_units = (
            decision_vars["vaccines"][d] * 0.1 +
            decision_vars["medicines"][d] * 0.1 +
            decision_vars["beds"][d] * 2.0 +
            decision_vars["staff"][d] * 3.0
        )
        cost_terms.append(c_i * dist_alloc_units)

    model += (pulp.lpSum(cost_terms) <= float(transport_budget), "Transportation_Budget_Limit")
