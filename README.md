# ElNino Health Resource Optimizer (EN-HRO)
### Optimization of Healthcare Resource and Vaccine Allocation during El Niño

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Optimization: PuLP / HiGHS](https://img.shields.io/badge/Solver-PuLP%20%2F%20HiGHS-purple.svg)](https://highs.dev/)

> **Academic Project Reference**:  
> Based on the T.Y. Mini Project Presentation: *“Optimization of Healthcare Resource and Vaccine Allocation during El Niño”*  
> **Institution**: Department of AI&DS, K. J. Somaiya Institute of Technology, Sion, Mumbai.  
> **Student Authors**: Virajkumar Upalanchi (Roll No. 9), Achal Vakharia (Roll No. 11), Utkarsh Yadav (Roll No. 15)  
> **Project Guide**: Prof. Deepali Chavan  

---

## 1. Problem Statement

El Niño Southern Oscillation (ENSO) triggers abnormal global and regional weather patterns, including severe rainfall anomalies, catastrophic coastal flood surges, and intense inland droughts and heatwaves. 

In India and specifically the **Maharashtra region**, these environmental disruptions create acute epidemiological vulnerabilities for:
- **Vector-Borne Diseases**: Dengue, Malaria (accelerated by standing flood pools, high humidity, and elevated thermal incubation)
- **Waterborne Diseases**: Cholera, Typhoid (driven by flood-induced drinking water contamination and drought-induced sanitation stress)

During sudden epidemic surges, healthcare delivery systems experience critical shortages of:
1. **Vaccines** (prophylactic doses)
2. **Medicines** (antibiotics, antimalarials, IV fluids)
3. **Hospital Beds** (inpatient acute admissions)
4. **Medical Staff** (physicians and specialized nurses)

Without algorithmic decision support, resource distribution is often reactive or purely proportional to historical demand, resulting in resource starvation in remote rural districts, supply hoarding in urban hubs, and inflated emergency freight expenditures.

---

## 2. Proposed Solution

**EN-HRO** is an end-to-end decision-support prototype that links oceanic macroclimate indicators to regional healthcare demand, employing **Mixed Integer Linear Programming (MILP)** to determine optimal, equitable, and cost-efficient distribution plans.

### The Scientific Decision Chain
$$\text{El Niño / ONI Index} \longrightarrow \text{Rainfall \& Temperature Anomalies} \longrightarrow \text{Environmental Stress} \longrightarrow \text{Epidemiological Demand} \longrightarrow \text{ILP Optimization} \longrightarrow \text{Fair \& Cost-Effective Resource Allocation}$$

---

## 3. Key Features

- **Climate Risk & Anomaly Assessment**: Tracks Oceanic Niño Index (ONI), regional temperature deviations ($\Delta T$), and precipitation anomalies ($\Delta R$).
- **Epidemiological Demand Prediction**: Random Forest regression and transparent biological formulas estimating disease incidence per 100,000 population.
- **Multi-Commodity Resource Allocation**: Optimizes simultaneous distribution of Vaccines, Medicines, Hospital Beds, and Medical Staff.
- **Distance-Aware Transportation Minimization**: Incorporates freight transit costs from the central state logistics depot in Mumbai.
- **Configurable Public Health Equity (Fairness Floor)**: Guarantees a minimum service ratio ($\tau$) so remote and low-density districts are never starved of life-saving medical supplies.
- **Interactive Scenario Simulator**: Empowers administrators to model what-if climate shocks (e.g. +1.5°C heatwave, 30% vaccine supply deficit) and observe real-time allocation recalculation.
- **Interactive GIS & Folium Maps**: District-level spatial visualization with color-coded risk tiers (Green = Low, Yellow = Medium, Red = High) and detailed popups.
- **Before-vs-After Impact Analytics**: Quantifies coverage increases (+30%+), shortage reductions, and freight cost savings.

---

## 4. System Architecture

```text
                                [ Climate Feeds (NOAA ONI / IMD) ]
                                                │
                                                ▼
                                [ Data Preprocessing & Validation ]
                                                │
                                                ▼
                                [ ML Demand Estimation (Random Forest) ]
                                                │
                                                ▼
                                [ Regional Requirements (Vax, Med, Bed, Staff) ]
                                                │
             ┌──────────────────────────────────┴──────────────────────────────────┐
             ▼                                                                     ▼
[ Logistics Hub (Distances / Freight Rates) ]                    [ Central Available Stockpiles ]
             │                                                                     │
             └──────────────────────────────────┬──────────────────────────────────┘
                                                ▼
                                [ Integer Linear Programming Engine ]
                                  (HiGHS / COIN-OR CBC Solvers)
                                                │
                                                ▼
                                [ Optimal Multi-Commodity Allocation ]
                                                │
                                                ▼
                                [ Streamlit Decision-Support Dashboard ]
                                  (10 Specialized Operational Views)
```

---

## 5. Mathematical Optimization Model

The healthcare resource allocation problem is formulated as a multi-commodity Integer Linear Program:

### Decision Variables
For each district $i \in \{1, \dots, N\}$:
- $V_i \in \mathbb{Z}_{\ge 0}$: Vaccines allocated to district $i$
- $M_i \in \mathbb{Z}_{\ge 0}$: Medicine treatment courses allocated to district $i$
- $B_i \in \mathbb{Z}_{\ge 0}$: Hospital beds allocated to district $i$
- $S_i \in \mathbb{Z}_{\ge 0}$: Medical personnel deployed to district $i$

### Objective Function
$$\max \sum_{i=1}^{N} \left[ (\alpha + \gamma) \cdot \omega_i \left( w_V V_i + w_M M_i + w_B B_i + w_S S_i \right) - \beta \cdot c_i \left( \ell_V V_i + \ell_M M_i + \ell_B B_i + \ell_S S_i \right) \right]$$

- $\alpha$: Weight on healthcare coverage maximization (default: 1.0)
- $\beta$: Weight on transportation cost penalty (default: 0.02)
- $\gamma$: Weight on high-risk unmet demand penalty (default: 0.5)
- $\omega_i$: Composite district priority factor ($1.0 + 1.5 \cdot \frac{P_i}{100} + 1.2 \cdot R_{\text{disease}, i}$)
- $c_i$: Freight transit cost per unit based on distance from the central logistics depot
- $w_V=1.0, w_M=1.0, w_B=4.0, w_S=8.0$: Clinical impact weights per commodity
- $\ell_V=0.05, \ell_M=0.05, \ell_B=1.5, \ell_S=2.0$: Logistical weight coefficients

### Constraints
1. **Supply Limits**: $\sum_i V_i \le K_V, \quad \sum_i M_i \le K_M, \quad \sum_i B_i \le K_B, \quad \sum_i S_i \le K_S$
2. **Demand Ceilings**: $V_i \le \text{Demand}^V_i, \quad M_i \le \text{Demand}^M_i, \quad B_i \le \text{Demand}^B_i, \quad S_i \le \text{Demand}^S_i \quad \forall i$
3. **Guaranteed Fairness Floor**: $V_i \ge \lfloor \tau \cdot \text{Demand}^V_i \rfloor, \quad M_i \ge \lfloor \tau \cdot \text{Demand}^M_i \rfloor \quad \forall i \quad (\tau = 0.15)$
4. **Transportation Budget Cap (Optional)**: $\sum_i c_i (\ell_V V_i + \ell_M M_i + \ell_B B_i + \ell_S S_i) \le C_{\text{budget}}$
5. **Non-negativity and Integrality**: $V_i, M_i, B_i, S_i \in \mathbb{Z}_{\ge 0}$

---

## 6. Repository Structure

```text
ElNino-Health/
│
├── README.md                          # Project documentation and specifications
├── requirements.txt                   # Dependency manifests
├── .gitignore                         # Git exclusion rules
├── run.bat                            # Windows one-click startup script
├── run.sh                             # Linux/macOS startup script
│
├── app/
│   └── app.py                         # Complete 10-section Streamlit decision dashboard
│
├── data/
│   ├── raw/
│   │   ├── enso.csv                   # Historical/projected ONI time series
│   │   ├── climate.csv                # District temperature, rain, and anomaly metrics
│   │   ├── disease_demand.csv         # Epidemiological case rates and requirements
│   │   ├── healthcare_resources.csv   # District inventory and transit distances
│   │   └── districts.geojson          # District polygon boundaries for GIS mapping
│   └── processed/
│       └── healthcare_optimization_dataset.csv # Unified regional modeling dataset
│
├── models/
│   ├── demand_model.pkl               # Trained Random Forest demand regressor
│   └── feature_columns.pkl            # Feature schema list
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Data validation and loading
│   ├── preprocessing.py               # Sanitization and bounds checking
│   ├── feature_engineering.py         # Climate risks and demand derivations
│   ├── demand_prediction.py           # ML training and prediction service
│   ├── optimization.py                # ILP execution and Before/After analytics
│   ├── scenario.py                    # Counterfactual simulation engine
│   └── generate_demo_data.py          # Reproducible demonstration data generator
│
├── optimization/
│   ├── __init__.py
│   ├── ilp_model.py                   # HealthcareResourceILP solver class
│   └── constraints.py                 # Mathematical constraint builders
│
├── notebooks/
│   └── exploration.ipynb              # Walkthrough and EDA notebook
│
├── outputs/
│   ├── figures/                       # Generated feature importance and distribution charts
│   ├── metrics/                       # JSON evaluation reports
│   └── allocation_results/            # Exported optimal allocation CSVs
│
└── docs/
    ├── architecture.md                # System and production architecture
    ├── methodology.md                 # Scientific framing and risk equations
    └── optimization_model.md          # Full mathematical ILP formulation
```

---

## 7. Geographic Pilot: Maharashtra (10 Districts)

| District | 2024 Population | Distance from Mumbai Depot | Climate Profile | Primary Outbreak Vulnerability |
| :--- | :--- | :--- | :--- | :--- |
| **Mumbai** | 12,442,373 | 15 km | Extreme Urban Density, Coastal | Dengue, Malaria, Water contamination |
| **Thane** | 11,060,148 | 35 km | High Density Urban/Peri-urban | Vector surge, Flood water runoff |
| **Pune** | 9,429,408 | 152 km | Western Ghats Foothills | Dengue, Typhoid |
| **Nashik** | 6,107,187 | 167 km | Northern Agricultural Basin | Malaria, Seasonal Diarrhea |
| **Nagpur** | 4,653,570 | 820 km | Eastern Vidarbha Heat Hub | Heatwaves, Cholera water stress |
| **Solapur** | 4,317,756 | 400 km | Drought-prone Southern Plains | Sanitation stress, Typhoid |
| **Kolhapur** | 3,876,001 | 375 km | Panchganga River Basin | Monsoonal Flooding, Leptospirosis/Dengue |
| **Chhatrapati Sambhajinagar** | 3,701,282 | 335 km | Central Marathwada Plateau | Water scarcity, Vector surge |
| **Amravati** | 2,888,445 | 660 km | Northern Vidarbha Plain | Seasonal Malaria, Heat stress |
| **Ratnagiri** | 1,615,262 | 330 km | Konkan Coast Heavy Monsoons | Extreme Flooding, Vector breeding |

---

## 8. Installation & Quick Start

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Recommended: Isolated virtual environment

### 1. Clone & Enter Repository
```bash
git clone https://github.com/anyadav/ElNino-Health.git
cd ElNino-Health
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Dashboard
```bash
# On Linux / macOS:
./run.sh

# On Windows:
run.bat

# Or directly via Streamlit:
streamlit run app/app.py
```
Open your browser at `http://localhost:8501`.

---

## 9. Presentation & Demonstration Script

For a 5-minute college presentation or technical demo, follow this sequence:

1. **Step 1 — Overview**: Open dashboard at `http://localhost:8501`. Highlight the 6 top KPI cards (Regions Analysed: 10, Expected Coverage: 79.4%, Cost: ~₹3.1M).
2. **Step 2 — Risk Map**: Navigate to *'2. Risk & Demand Map'*. Show the spatial distribution of risk across Maharashtra. Click on high-risk clusters (e.g. Mumbai, Thane) and explain how coastal rain surges accelerate disease cases.
3. **Step 3 — District Deep-Dive**: Navigate to *'3. District Analysis'*. Inspect a specific district (e.g. Kolhapur or Nagpur), displaying its climate anomalies, disease risk radar, and resource demand breakdown.
4. **Step 4 — Problem Motivation**: Explain that local on-hand inventory only covers ~45% of demand, leaving a massive public health deficit.
5. **Step 5 — Run Optimization**: In *'4. Resource Optimization'*, review available central stockpiles and policy weights ($\alpha, \beta, \gamma, \tau$). Click **"Run ILP Optimization Engine"**. Note the instant solver completion (`Optimal` status via HiGHS).
6. **Step 6 — Results & Impact**: In *'5. Allocation Results'*, review the **Before vs After** panel showing a **+32.2 percentage point gain in coverage**, 256,000+ unmet cases resolved, and 31% freight cost savings. Showcase the 7 Plotly charts.
7. **Step 7 — Scenario Simulator**: In *'6. Scenario Simulator'*, click the preset **"Severe El Niño + Vaccine Cut"**. Explain how a climate shock intensifies regional demand while supply chains contract. Show the dynamic re-allocation table.
8. **Step 8 — Fairness Verification**: In *'7. Fairness Analysis'*, explain how the mathematical fairness floor prevents distant districts from receiving zero resources.
9. **Step 9 — Scientific Rigor**: In *'8. Model Insights'* and *'9. Methodology'*, present the Random Forest validation ($R^2 = 0.997$, MAE = 616) and the complete MILP formulation.

---

## 10. Limitations & Scientific Framing

- **Decision-Support Prototype**: The system is intended for scenario planning and resource optimization; it does not replace physician diagnostics or operational clinical triage.
- **Data Calibration**: Demonstration data is synthetically calibrated using realistic parametric distributions modeled on IMD monsoon variability and WHO epidemiological rates. For state operational deployment, feeds must be connected to live IMD API and MoHFW HMIS systems.
- **Causality Integrity**: The model does not claim that El Niño directly causes infections; rather, ENSO alters temperature and moisture regimes, creating conditions favorable for pathogen proliferation and vector reproduction.

---

## 11. Future Scope

1. **Real-Time Weather Integration**: Ingest 7-day to 14-day numerical weather prediction grids directly from IMD / ECMWF.
2. **Deep Learning Outbreak Forecasting**: Incorporate Spatio-Temporal Graph Neural Networks (ST-GNN) or LSTM architectures for multi-week lead-time epidemic forecasting.
3. **Micro-Logistics & Routing**: Integrate Open Source Routing Machine (OSRM) for road-level vehicle routing and cold-chain temperature monitoring.
4. **All-India Scaling**: Expand the geographic coverage from the 10-district Maharashtra pilot to all 36 Maharashtra districts and 700+ districts across India.
