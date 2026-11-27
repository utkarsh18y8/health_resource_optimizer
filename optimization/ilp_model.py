"""
Integer Linear Programming (ILP) Model for Healthcare & Vaccine Allocation
Formulates and solves the multi-objective resource allocation problem using PuLP.
"""

from typing import Dict, Any, Tuple
import pulp
import numpy as np
import pandas as pd
from optimization.constraints import (
    add_supply_capacity_constraints,
    add_demand_upper_bounds,
    add_fairness_constraints,
    add_transport_budget_constraint,
)


class HealthcareResourceILP:
    """
    ILP Optimizer for distributing vaccines, medicines, beds, and medical staff
    across climate-affected districts during El Niño outbreaks.
    """

    def __init__(
        self,
        alpha: float = 1.0,     # Weight on healthcare coverage / priority benefit
        beta: float = 0.02,     # Weight on transportation cost penalty
        gamma: float = 0.50,    # Weight on unmet demand penalty in high-risk zones
        fairness_ratio: float = 0.15,  # Minimum guaranteed coverage ratio
        fairness_high_risk_only: bool = False
    ):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.fairness_ratio = float(fairness_ratio)
        self.fairness_high_risk_only = fairness_high_risk_only
        self.status = "Not Solved"
        self.objective_value = 0.0

    def solve(
        self,
        df_input: pd.DataFrame,
        total_available: Dict[str, int],
        transport_budget: float = None
    ) -> Tuple[str, pd.DataFrame, Dict[str, Any]]:
        """
        Solves the ILP model.
        Returns:
            (status_string, results_dataframe, summary_metrics_dict)
        """
        df = df_input.copy()
        districts = df["district"].tolist()

        # Check for immediate infeasibility regarding fairness vs supply
        if self.fairness_ratio > 0.0:
            for res, col in [("vaccines", "vaccine_demand"), ("medicines", "medicine_demand"),
                             ("beds", "bed_demand"), ("staff", "staff_demand")]:
                avail = total_available.get(res, 0)
                if self.fairness_high_risk_only:
                    sub = df[df["risk_level"] == "High"]
                    required_min = int(sub[col].sum() * self.fairness_ratio)
                else:
                    required_min = int(df[col].sum() * self.fairness_ratio)

                if avail < required_min:
                    diag = {
                        "status": "Infeasible",
                        "error_message": (
                            f"No feasible allocation exists under the current resource and fairness constraints. "
                            f"Guaranteed fairness threshold ({self.fairness_ratio * 100:.0f}%) requires at least {required_min:,} "
                            f"{res}, but total available stockpile is only {avail:,} (Deficit: {required_min - avail:,})."
                        ),
                        "resource_deficit": {res: required_min - avail},
                        "total_demand": int(df[col].sum()),
                        "available_resources": avail
                    }
                    return "Infeasible", pd.DataFrame(), diag

        # Initialize PuLP Problem
        prob = pulp.LpProblem("ElNino_Healthcare_Resource_Allocation", pulp.LpMaximize)

        # Decision Variables: Non-negative Integers
        vax_vars = {d: pulp.LpVariable(f"Vax_{d}", lowBound=0, cat=pulp.LpInteger) for d in districts}
        med_vars = {d: pulp.LpVariable(f"Med_{d}", lowBound=0, cat=pulp.LpInteger) for d in districts}
        bed_vars = {d: pulp.LpVariable(f"Bed_{d}", lowBound=0, cat=pulp.LpInteger) for d in districts}
        staff_vars = {d: pulp.LpVariable(f"Staff_{d}", lowBound=0, cat=pulp.LpInteger) for d in districts}

        decision_vars = {
            "vaccines": vax_vars,
            "medicines": med_vars,
            "beds": bed_vars,
            "staff": staff_vars
        }

        # Objective Function Formulation:
        # Maximize: alpha * Coverage_Benefit + gamma * Priority_Service - beta * Transport_Cost
        # Where Coverage & Priority benefit reward allocating units to high-priority and high-risk districts.
        obj_terms = []
        for _, row in df.iterrows():
            d = row["district"]
            p_score = float(row.get("priority_score", 50.0)) / 100.0
            d_risk = float(row.get("disease_risk", 0.5))
            c_trans = float(row.get("transport_cost", 25.0))

            # Priority factor scales coverage reward and shortage prevention
            # Relative weights: Vaccines (1.0), Medicines (1.0), Beds (4.0 per acute bed), Staff (8.0 per staff)
            effective_priority = (1.0 + 1.5 * p_score + 1.2 * d_risk)
            coverage_gain = (self.alpha + self.gamma) * effective_priority

            # Objective components
            obj_terms.append(coverage_gain * 1.0 * vax_vars[d])
            obj_terms.append(coverage_gain * 1.0 * med_vars[d])
            obj_terms.append(coverage_gain * 4.0 * bed_vars[d])
            obj_terms.append(coverage_gain * 8.0 * staff_vars[d])

            # Transport cost penalty
            # Freight units: small items = 0.05 load factor; bed/staff = 1.5 load factor
            trans_penalty = self.beta * c_trans * (
                0.05 * vax_vars[d] +
                0.05 * med_vars[d] +
                1.50 * bed_vars[d] +
                2.00 * staff_vars[d]
            )
            obj_terms.append(-trans_penalty)

        prob += pulp.lpSum(obj_terms), "Multi_Objective_Healthcare_Utility"

        # Apply Constraints
        add_supply_capacity_constraints(prob, decision_vars, total_available)
        add_demand_upper_bounds(prob, decision_vars, df)
        add_fairness_constraints(prob, decision_vars, df, self.fairness_ratio, self.fairness_high_risk_only)

        if transport_budget is not None and transport_budget > 0:
            add_transport_budget_constraint(prob, decision_vars, df, transport_budget)

        # Solve with robust solver selection (HiGHS preferred on ARM64/Mac/Linux, then CBC)
        solver = None
        try:
            import highspy
            solver = pulp.HiGHS(msg=False)
        except Exception:
            solver = None

        if solver is None:
            try:
                solver = pulp.PULP_CBC_CMD(msg=False)
            except Exception:
                solver = None

        try:
            solve_status = prob.solve(solver)
        except Exception as e:
            # Fallback attempt
            try:
                solve_status = prob.solve(pulp.HiGHS(msg=False))
            except Exception as e2:
                diag = {
                    "status": "SolverError",
                    "error_message": f"Solver execution error: {e2}. Please ensure highspy or cbc is available."
                }
                return "SolverError", pd.DataFrame(), diag

        status_name = pulp.LpStatus[solve_status]
        self.status = status_name

        if status_name != "Optimal":
            diag = {
                "status": status_name,
                "error_message": f"Solver ended with status: '{status_name}'. No feasible optimal plan could be computed with the given constraints."
            }
            return status_name, pd.DataFrame(), diag

        self.objective_value = float(pulp.value(prob.objective))

        # Extract Results
        results = []
        tot_trans_cost = 0.0

        for _, row in df.iterrows():
            d = row["district"]
            v_val = int(round(vax_vars[d].varValue or 0))
            m_val = int(round(med_vars[d].varValue or 0))
            b_val = int(round(bed_vars[d].varValue or 0))
            s_val = int(round(staff_vars[d].varValue or 0))

            v_dem = int(row["vaccine_demand"])
            m_dem = int(row["medicine_demand"])
            b_dem = int(row["bed_demand"])
            s_dem = int(row["staff_demand"])

            tot_dem = v_dem + m_dem + b_dem + s_dem
            tot_alloc = v_val + m_val + b_val + s_val
            unmet = max(0, tot_dem - tot_alloc)

            cov_pct = round((tot_alloc / tot_dem * 100.0), 1) if tot_dem > 0 else 100.0
            vax_cov = round((v_val / v_dem * 100.0), 1) if v_dem > 0 else 100.0
            med_cov = round((m_val / m_dem * 100.0), 1) if m_dem > 0 else 100.0
            bed_cov = round((b_val / b_dem * 100.0), 1) if b_dem > 0 else 100.0
            staff_cov = round((s_val / s_dem * 100.0), 1) if s_dem > 0 else 100.0

            c_rate = float(row.get("transport_cost", 25.0))
            # Transportation cost calculation in INR
            dist_cost = round(c_rate * (0.05 * v_val + 0.05 * m_val + 1.5 * b_val + 2.0 * s_val), 2)
            tot_trans_cost += dist_cost

            results.append({
                "district": d,
                "risk_level": row.get("risk_level", "Medium"),
                "disease_risk": row.get("disease_risk", 0.5),
                "priority_score": row.get("priority_score", 50.0),
                "total_demand": tot_dem,
                "vaccine_demand": v_dem,
                "medicine_demand": m_dem,
                "bed_demand": b_dem,
                "staff_demand": s_dem,
                "allocated_vaccines": v_val,
                "allocated_medicines": m_val,
                "allocated_beds": b_val,
                "allocated_staff": s_val,
                "total_allocated": tot_alloc,
                "unmet_demand": unmet,
                "coverage_percentage": cov_pct,
                "vaccine_coverage_pct": vax_cov,
                "medicine_coverage_pct": med_cov,
                "bed_coverage_pct": bed_cov,
                "staff_coverage_pct": staff_cov,
                "transportation_cost": dist_cost
            })

        res_df = pd.DataFrame(results)

        # Aggregate System Metrics
        sum_dem = res_df["total_demand"].sum()
        sum_alloc = res_df["total_allocated"].sum()
        sum_unmet = res_df["unmet_demand"].sum()
        overall_cov = round((sum_alloc / sum_dem * 100.0), 2) if sum_dem > 0 else 100.0

        vax_util = round((res_df["allocated_vaccines"].sum() / total_available.get("vaccines", 1) * 100.0), 1)
        med_util = round((res_df["allocated_medicines"].sum() / total_available.get("medicines", 1) * 100.0), 1)
        bed_util = round((res_df["allocated_beds"].sum() / total_available.get("beds", 1) * 100.0), 1)
        staff_util = round((res_df["allocated_staff"].sum() / total_available.get("staff", 1) * 100.0), 1)
        avg_util = round(np.mean([vax_util, med_util, bed_util, staff_util]), 1)

        summary = {
            "solver_status": status_name,
            "objective_value": round(self.objective_value, 2),
            "total_demand": int(sum_dem),
            "total_allocated": int(sum_alloc),
            "total_unmet_demand": int(sum_unmet),
            "overall_coverage_pct": overall_cov,
            "total_transport_cost": round(tot_trans_cost, 2),
            "resource_utilization_pct": avg_util,
            "vaccine_utilization_pct": min(100.0, vax_util),
            "medicine_utilization_pct": min(100.0, med_util),
            "bed_utilization_pct": min(100.0, bed_util),
            "staff_utilization_pct": min(100.0, staff_util),
            "total_vaccines_allocated": int(res_df["allocated_vaccines"].sum()),
            "total_medicines_allocated": int(res_df["allocated_medicines"].sum()),
            "total_beds_allocated": int(res_df["allocated_beds"].sum()),
            "total_staff_allocated": int(res_df["allocated_staff"].sum()),
            "available_resources": total_available
        }

        return status_name, res_df, summary
