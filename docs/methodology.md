# Scientific & Epidemiological Methodology

## 1. Scientific Framing & Disclaimer

> **IMPORTANT DISCLAIMER**:
> This project is a prototype decision-support system demonstrating climate-aware healthcare demand estimation and Integer Linear Programming-based resource allocation. Prototype or synthetic data and model outputs should not be interpreted as official disease forecasts, medical advice, or government resource-allocation recommendations.

We do **NOT** assert that El Niño directly causes a disease outbreak. Rather, we model the established scientific causal chain:

$$\text{El Niño Southern Oscillation (ENSO)}$$
$$\downarrow$$
$$\text{Macroclimate Variability (SST Anomaly)}$$
$$\downarrow$$
$$\text{Regional Weather Extremes (Monsoon Disruptions, Thermal Surges)}$$
$$\downarrow$$
$$\text{Ecological Conditions (Mosquito Vector Breeding, Water Supply Stagnation/Deficit)}$$
$$\downarrow$$
$$\text{Heightened Epidemiological Vulnerability}$$
$$\downarrow$$
$$\text{Surge in Healthcare Resource Demand}$$
$$\downarrow$$
$$\text{Integer Linear Programming (ILP) Resource Allocation}$$

---

## 2. Epidemiological Risk Formulation

### A. Vector-Borne Risk (Dengue & Malaria)
Vector transmission dynamics accelerate with warmer temperatures (shortened extrinsic incubation period in *Aedes* and *Anopheles* mosquitoes) and localized pooling following flood surges or unseasonal showers:

$$R_{\text{vector}, i} = \text{clip}\left(0.30 + 0.28 \cdot \frac{\Delta T_i}{2.5} + 0.18 \cdot \frac{\rho_i}{\rho_{\max}} + 0.14 \cdot R_{\text{flood}, i} + 0.10 \cdot \frac{\text{ONI}}{2.0}, 0.05, 0.99\right)$$

where:
- $\Delta T_i$ = District temperature anomaly (°C)
- $\rho_i$ = Population density (persons / $\text{km}^2$)
- $R_{\text{flood}, i}$ = Flood surge index $[0, 1]$
- $\text{ONI}$ = Oceanic Niño Index

### B. Water-Borne Risk (Cholera & Typhoid)
Waterborne pathogens proliferate when heavy flooding contaminates drinking infrastructure, or alternatively when drought/heatwaves cause severe water shortages and reliance on untreated reservoirs:

$$R_{\text{water}, i} = \text{clip}\left(0.28 + 0.35 \cdot R_{\text{flood}, i} + 0.25 \cdot R_{\text{heat}, i} + 0.12 \cdot \frac{\rho_i}{\rho_{\max}}, 0.05, 0.99\right)$$

### C. Composite Disease Burden & Demand Estimation
The composite risk combines vector and waterborne vulnerabilities:

$$R_{\text{disease}, i} = 0.55 \cdot R_{\text{vector}, i} + 0.45 \cdot R_{\text{water}, i}$$

Expected disease case burden per 100,000 population:

$$\text{Case Rate}_i = 180 + 360 \cdot R_{\text{disease}, i} + 50 \cdot \frac{\rho_i}{\rho_{\max}}$$

$$\text{Estimated Cases}_i = \left\lfloor \frac{\text{Population}_i}{100,000} \times \text{Case Rate}_i \right\rfloor$$

### D. Resource Demand Conversion
From case burden, specific resource demands are derived:
1. **Vaccine Demand**:
   $$\text{Demand}^V_i = \left\lfloor \text{Estimated Cases}_i \times (1.3 + 0.5 \cdot R_{\text{disease}, i}) \right\rfloor$$
2. **Medicine Demand**:
   $$\text{Demand}^M_i = \left\lfloor \text{Estimated Cases}_i \times (1.1 + 0.3 \cdot R_{\text{disease}, i}) \right\rfloor$$
3. **Hospital Bed Demand** (Acute cases requiring hospitalization):
   $$\text{Demand}^B_i = \left\lfloor \text{Estimated Cases}_i \times (0.16 + 0.10 \cdot R_{\text{disease}, i}) \right\rfloor$$
4. **Medical Staff Demand** (Dedicated medical officers/nurses):
   $$\text{Demand}^S_i = \left\lfloor \frac{\text{Demand}^B_i}{10} \right\rfloor$$
