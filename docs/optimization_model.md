# Integer Linear Programming (ILP) Optimization Model

## 1. Problem Formulation

The healthcare allocation problem is modeled as a multi-commodity Integer Linear Program (MILP). Let $\mathcal{D} = \{1, 2, \dots, N\}$ denote the set of districts ($N=10$ in the Maharashtra pilot).

### Decision Variables

For each district $i \in \mathcal{D}$:
- $V_i \in \mathbb{Z}_{\ge 0}$: Quantity of vaccines allocated to district $i$.
- $M_i \in \mathbb{Z}_{\ge 0}$: Quantity of medicine treatment courses allocated to district $i$.
- $B_i \in \mathbb{Z}_{\ge 0}$: Number of acute hospital beds allocated / staffed in district $i$.
- $S_i \in \mathbb{Z}_{\ge 0}$: Number of healthcare personnel deployed to district $i$.

---

## 2. Objective Function

The objective balances healthcare coverage maximization and disease burden reduction against transportation logistics costs:

$$\max \sum_{i \in \mathcal{D}} \left[ (\alpha + \gamma) \cdot \omega_i \cdot \left( w_V V_i + w_M M_i + w_B B_i + w_S S_i \right) - \beta \cdot c_i \cdot \left( \ell_V V_i + \ell_M M_i + \ell_B B_i + \ell_S S_i \right) \right]$$

where:
- $\alpha \ge 0$: User-configurable weight for healthcare coverage maximization.
- $\beta \ge 0$: User-configurable weight penalizing transportation costs.
- $\gamma \ge 0$: User-configurable penalty weight for unmet demand in high-risk zones.
- $\omega_i = 1.0 + 1.5 \cdot \frac{P_i}{100} + 1.2 \cdot R_{\text{disease}, i}$: Priority multiplier incorporating composite disease risk and district vulnerability score $P_i$.
- $w_V=1.0, w_M=1.0, w_B=4.0, w_S=8.0$: Clinical utility weights reflecting patient impact per resource unit.
- $c_i$: Distance-based freight cost per unit to district $i$ from the central state depot.
- $\ell_V=0.05, \ell_M=0.05, \ell_B=1.5, \ell_S=2.0$: Logistical load coefficients.

---

## 3. Constraints

### 1. Central Stockpile Capacity Constraints
The sum of all allocated resources across all districts cannot exceed the available inventory:

$$\sum_{i \in \mathcal{D}} V_i \le K_V \quad (\text{Available Vaccines})$$
$$\sum_{i \in \mathcal{D}} M_i \le K_M \quad (\text{Available Medicines})$$
$$\sum_{i \in \mathcal{D}} B_i \le K_B \quad (\text{Available Beds})$$
$$\sum_{i \in \mathcal{D}} S_i \le K_S \quad (\text{Available Staff})$$

### 2. Regional Demand Upper Bounds
To eliminate wastage, no district receives more units than its estimated requirement:

$$V_i \le \text{Demand}^V_i \quad \forall i \in \mathcal{D}$$
$$M_i \le \text{Demand}^M_i \quad \forall i \in \mathcal{D}$$
$$B_i \le \text{Demand}^B_i \quad \forall i \in \mathcal{D}$$
$$S_i \le \text{Demand}^S_i \quad \forall i \in \mathcal{D}$$

### 3. Fairness & Equity Constraints
To prevent starvation of remote or smaller districts, a minimum guaranteed coverage ratio $\tau \in [0, 1]$ is enforced:

$$V_i \ge \lfloor \tau \cdot \text{Demand}^V_i \rfloor \quad \forall i \in \mathcal{D}$$
$$M_i \ge \lfloor \tau \cdot \text{Demand}^M_i \rfloor \quad \forall i \in \mathcal{D}$$
$$B_i \ge \lfloor \tau \cdot \text{Demand}^B_i \rfloor \quad \forall i \in \mathcal{D}$$
$$S_i \ge \lfloor \tau \cdot \text{Demand}^S_i \rfloor \quad \forall i \in \mathcal{D}$$

### 4. Transportation Budget Cap (Optional)
If a municipal or disaster relief budget cap $C_{\text{budget}}$ is specified:

$$\sum_{i \in \mathcal{D}} c_i \cdot \left( \ell_V V_i + \ell_M M_i + \ell_B B_i + \ell_S S_i \right) \le C_{\text{budget}}$$

### 5. Integrality & Non-Negativity
$$V_i, M_i, B_i, S_i \in \mathbb{Z}_{\ge 0} \quad \forall i \in \mathcal{D}$$
