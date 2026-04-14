# HydroGraphDiff: A Generative AI–Driven Probabilistic Flood Risk & Catastrophe Modeling System (Hyderabad, Musi Basin)

HydroGraphDiff is an end-to-end probabilistic flood risk and catastrophe modeling system developed for the Hyderabad (Musi basin) region. The project integrates geospatial data, extreme value theory (EVT), graph-based spatial modeling, and generative AI (Diffusion Models) to simulate flood hazards and quantify climate-driven financial risk at ward-level resolution.

## 1. Data Used
IMERG (satellite rainfall)(2001–2024), 
IMD (gauge rainfall)(2001–2024), 
DEM (elevation, slope), 
ESA WorldCover (urbanization), 
OSM (roads, hospitals), 
WorldPop (population),
CMIP6 (MIROC6-historical, ssp245, ssp585)(2021–2050)

- **AOI (Area of Interest):** Hyderabad city and Musi river sub-basin (ward-level analysis)

## 2. Architecture
<img width="706" height="707" alt="image" src="https://github.com/user-attachments/assets/f649a0e3-a7dc-4819-a2f1-b65e5b8b642e" />

Figure.1: Final architecture of HydroGraphDiff following a standard hazard–vulnerability–risk framework.

Hazard is derived from ERA5 and IMD rainfall using quantile-mapped bias correction and EVT-based extreme modeling to estimate extreme rainfall magnitudes (return levels). 
A spatial graph $G=(V,E,W)$ is constructed, where nodes $V$ represent grid cells(wards), edges $E$ encode spatial adjacency, and weights $W$ capture distance, terrain, and drainage influence. This graph is first used in a graph diffusion step to propagate extreme rainfall across neighboring regions, producing spatially coherent hazard fields that reflect physical spillover and connectivity.
The same graph $G$ is then reused within a Graph Conditioned DDPM, where it plays two roles: (i) as a conditioning structure guiding the model to learn spatial dependencies in hazard fields, and (ii) through a graph Laplacian regularization ($L = D - W$), which enforces smoothness and physical consistency by penalizing unrealistic spatial discontinuities. The diffusion model learns the conditional distribution of hazard fields and generates multiple stochastic realizations, thereby capturing uncertainty and variability in extreme rainfall patterns.
Vulnerability is constructed from geographic features (terrain, infrastructure, population) into sensitivity, exposure, and adaptive capacity. Risk is computed as $R = H \times V$, with uncertainty, extreme probability, climate scenarios (SSP245/585), and loss exceedance curves (EP, PML, TVaR) providing a comprehensive probabilistic climate risk assessment.

## 3. Bias Correction, Extreme Value Theorem and Graph Diffusion

Figure.2: Extreme rainfall modeling and spatial enhancement pipeline
<img width="1500" height="787" alt="image" src="https://github.com/user-attachments/assets/9f59a9c5-5d77-436f-b24c-134f7b9dea89" />
<img width="1086" height="855" alt="image" src="https://github.com/user-attachments/assets/dd6e9694-6b69-45e8-bc55-ed0705342e6e" />
<img width="1500" height="675" alt="image" src="https://github.com/user-attachments/assets/14a0e4d7-01fc-4b15-9b21-13a833dd7f37" />



<p align="center">
  <b>(a)</b> Bias correction of satellite rainfall using <b>Quantile Mapping</b>, where IMERG precipitation is statistically aligned with India Meteorological Department observations via 
  <b>R<sup>BC</sup> = F<sup>-1</sup><sub>IMD</sub>(F<sub>IMERG</sub>(R))</b>, ensuring reliable extreme value estimation.

  &nbsp;&nbsp;&nbsp;

  <b>(b)</b> Return level comparison using <b>Generalized Pareto Distribution (Peaks Over Threshold)</b> and <b>Generalized Extreme Value</b> methods. The GPD-based approach is selected for downstream modeling as it provides more stable and physically consistent tail estimates for high return periods by directly modeling exceedances over threshold, which is critical for flood extremes.

  &nbsp;&nbsp;&nbsp;

  <b>(c)</b> Spatial hazard fields derived from Extreme Value Theory (GPD-based) and enhanced using <b>graph-based diffusion</b>, where a spatial graph G = (V, E, W) encodes adjacency, distance, terrain, and drainage connectivity. The diffusion process 
  H<sub>i</sub> ← αH<sub>i</sub> + (1 − α)∑<sub>j∈N(i)</sub> w<sub>ij</sub>H<sub>j</sub> propagates extreme rainfall across neighboring regions, significantly improving spatial coherence and reducing artifacts from pointwise EVT estimation. This leads to more physically consistent and reliable <b>ward-level rainfall hazard estimation</b> by capturing spatial dependencies and hydrological connectivity.
</p>

## 4. Graph Conditioned DDPM (Denoising Diffusion Probabilistic Model) 
<img width="679" height="725" alt="image" src="https://github.com/user-attachments/assets/50bd2c24-b010-4fc2-b769-e4051d4d0ecf" />

Figure.3: Final architecture of the Graph Conditioned DDPM in HydroGraphDiff. A for-
ward diffusion process progressively corrupts EVT-conditioned hazard fields into Gaussian noise.
A UNet-based neural network learns to predict noise conditioned on geospatial features (c), posi-
tional encoding, temporal embeddings (t), and spatial graph structure (G = (V, E, W )), where V
denotes nodes (grid cells or wards), E represents spatial adjacency, and W encodes edge weights
(distance, similarity, or terrain influence). A graph Laplacian regularization explicitly enforces
spatial consistency and smoothness. During reverse diffusion, the learned noise distribution is iter-
atively removed to generate stochastic, spatially coherent hazard realizations consistent with both
data and underlying spatial dependencies. 

<img width="1712" height="325" alt="forward" src="https://github.com/user-attachments/assets/25040c5c-18c0-45fe-ac45-60c8bb9e644b" />

<img width="1712" height="337" alt="reverse" src="https://github.com/user-attachments/assets/6fc7e116-6bc2-472f-91d7-4e5cce035793" />

Figure.4: Diffusion dynamics in HydroGraphDiff for RL100 extreme rainfall scenario.
Top: Forward diffusion progressively corrupts the EVT-conditioned hazard field into noise, de-
stroying spatial structure as timestep increases. Bottom: Reverse diffusion using the Conditional
Graph DDPM reconstructs spatially coherent hazard fields, guided by geospatial conditioning and
graph-based constraints. The panels show standardized hazard anomaly fields, where positive
values indicate above-average rainfall and negative values indicate below-average rainfall at each
location. 

Together, these results illustrate the core DDPM mechanism within the HydroGraphDiff framework.

In the forward process, the structured EVT-conditioned hazard field $x_0$ is progressively corrupted by Gaussian noise according to a known schedule, producing intermediate noisy states:

$$
x_t \sim \mathcal{N}(\sqrt{\bar{\alpha}_t} \, x_0,\; (1 - \bar{\alpha}_t) I)
$$

As $t$ increases, spatial structure is gradually destroyed, and the field approaches an isotropic Gaussian distribution.

The Conditional Graph DDPM is trained to learn the noise function:

$$
\epsilon_\theta(x_t, t, c, G)
$$

which implicitly models the conditional distribution of noise given:
- the corrupted field $x_t$
- diffusion timestep $t$
- geospatial conditioning variables $c$
- spatial graph structure $G$
  
The model leverages a UNet-style encoder–decoder to fuse multi-channel inputs:

$$
(x_t, c, \text{pos})
$$

while incorporating:
- temporal embeddings
- graph-aware spatial constraints via the Laplacian operator

During reverse diffusion, the model iteratively removes the estimated noise component from $x_t$, reconstructing spatial structure step-by-step.

Stochasticity is preserved through injected noise:

$$
z \sim \mathcal{N}(0, I)
$$

at each timestep, ensuring sampling from the learned data distribution rather than producing a deterministic output.

As a result, the framework generates multiple stochastic hazard realizations:

$$
x_0^{(s)}
$$

each representing a physically consistent extreme rainfall scenario conditioned on:
- terrain
- urban characteristics
- climate-driven extremes

This enables **probabilistic hazard modeling**, capturing both:
- spatial dependencies  
- uncertainty in extreme event generation

  <img width="1500" height="695" alt="image" src="https://github.com/user-attachments/assets/f4500387-bc20-4323-a16d-3fd4e932b8c7" />

Figure.5: Graph-conditioned DDPM generates diverse stochastic realizations of RL100(100 year return level) extreme rainfall, capturing spatial variability and uncertainty across the urban flood landscape.

### Hazard index calculation at ward-level

Building on the stochastic hazard generation framework, the **Graph Conditioned DDPM** generates ensembles of spatially coherent rainfall hazard fields H<sup>(k)</sup>(x, y) for each return level. These ensembles (typically ~400 scenarios per return level in our case) are then aggregated to the ward scale using area-weighted spatial integration, ensuring that sub-grid variability and spatial heterogeneity are preserved.

For each ward, the ensemble of scenario-wise hazard realizations is used to construct a probabilistic hazard representation capturing central tendency, tail behavior, and uncertainty. Specifically, a normalized hazard index is computed as:

**H = 0.5·μ′ + 0.35·Tail + 0.15·σ′**

where **μ′** is the normalized mean hazard across scenarios, **σ′** represents normalized variability (standard deviation), and the tail component captures extreme behavior using high quantiles:

**Tail = 0.6·P90′ + 0.4·P95′**

Here, **P90′** and **P95′** denote the normalized 90th and 95th percentile hazard values, respectively, ensuring that both moderate and extreme tail risks are incorporated. This formulation explicitly balances average conditions, extreme events, and uncertainty, resulting in a robust probabilistic hazard index.
  
## 5. Exposure & Vulnerability 
## Vulnerability and Risk Formulation

The ward-level vulnerability framework is constructed using physically interpretable components derived from terrain, infrastructure, and socio-economic data. All variables are normalized using min-max scaling:

X′ = (X − X_min) / (X_max − X_min)

---

### 1. Sensitivity Index (S)

Sensitivity captures the intrinsic physical susceptibility of a ward to flooding, based on terrain, drainage, and land characteristics.

Components:

- Elevation (lower → higher sensitivity)  
- Slope (flatter → higher sensitivity)  
- Distance to drainage (closer → higher sensitivity)  
- Distance to Musi river (floodplain proximity)  
- Impervious surface fraction  
- Drainage density  

Formulation:

S_elev = 1 − elev′  
S_slope = 1 − slope′  
S_drain = 1 − dist_drain′  
S_floodplain = 1 − dist_musi′  
S_imperv = impervious′  
S_drain_density = drain_density′  

Final Sensitivity Index:

**S = (S_elev + S_slope + S_drain + S_floodplain + S_imperv + S_drain_density) / 6**

---

### 2. Adaptive Capacity (AC)

Adaptive capacity represents the ability of a ward to respond to and recover from flooding, based on infrastructure and accessibility.

Components:

- Road density (higher → better capacity)  
- Distance to nearest hospital (closer → better capacity)  

Formulation:

AC_roads = road_density′  
AC_hospital = 1 − dist_hospital′  

Final Adaptive Capacity:

**AC = (AC_roads + AC_hospital) / 2**

---

### 3. Exposure (E)

Exposure represents the concentration of population at risk.

Formulation:

**E = population_density′**

---

### 4. Potential Impact

Potential impact captures the interaction between physical susceptibility and exposed population, without accounting for resilience.

**Potential Impact = S × E**

---

### 5. Structural Vulnerability (V)

Structural vulnerability incorporates adaptive capacity into the risk formulation, following the IPCC-aligned framework.

**V = S × E × (1 − AC)**

---

### Interpretation

- **Sensitivity (S):** Physical flood susceptibility (terrain + hydrology + urbanization)  
- **Exposure (E):** Population at risk  
- **Adaptive Capacity (AC):** Infrastructure and response capability  
- **Potential Impact:** Raw flood impact potential  
- **Structural Vulnerability (V):** Realistic vulnerability accounting for resilience

  This formulation ensures a spatially explicit, interpretable, and physically consistent assessment of ward-level flood vulnerability.

<img width="1286" height="1125" alt="image" src="https://github.com/user-attachments/assets/453621df-f042-42c3-8f72-c7e9addcc653" />

Figure.6: Ward-level flood vulnerability components for Hyderabad, showing spatial patterns of adaptive capacity, sensitivity, potential impact, and resulting structural vulnerability.
Insight: High vulnerability emerges where elevated sensitivity and exposure coincide with low adaptive capacity, highlighting flood risk hotspots along the Musi river corridor.

## 6. Flood Risk(Impact) Formulation at ward level

The overall flood risk is computed by integrating the probabilistic hazard index derived from the generative ensemble with structural vulnerability:

**R = H · V**

where **H** represents the probabilistic hazard index capturing mean intensity, extreme behavior (P90–P95 tail), and uncertainty across generated scenarios.

Expanding the formulation:

**R = H · S · E · (1 − AC)**

where:
- **S** = Sensitivity (physical susceptibility)  
- **E** = Exposure (population at risk)  
- **AC** = Adaptive Capacity (infrastructure and resilience)  

This probabilistic definition of risk enables a more realistic representation of climate-driven flood impacts by explicitly incorporating both **uncertainty in hazard generation** and **heterogeneity in vulnerability across wards**, leading to spatially explicit and decision-relevant risk estimates.

## Flood Risk and Climate Impact Formulation

---

### Flood Risk (Impact)

Hazard from generative ensembles:

**H<sub>w</sub> = 0.5·μ′<sub>w</sub> + 0.35·(0.6·P90′<sub>w</sub> + 0.4·P95′<sub>w</sub>) + 0.15·σ′<sub>w</sub>**

where:
- μ′<sub>w</sub> = normalized mean hazard  
- σ′<sub>w</sub> = normalized standard deviation  
- P90′<sub>w</sub>, P95′<sub>w</sub> = tail quantiles  

---

Structural vulnerability:

**V<sub>w</sub> = S<sub>w</sub> · E<sub>w</sub> · (1 − AC<sub>w</sub>)**

---

Final risk:

**R<sub>w</sub> = H<sub>w</sub> · V<sub>w</sub>**

---

Normalization:

**R′<sub>w</sub> = (R<sub>w</sub> − min(R)) / (max(R) − min(R))**

---

### Probability of Extreme Events

Threshold:

**T<sub>w</sub> = (1 / N) · Σ H<sub>w</sub><sup>(k)</sup>**

Probability:

**P<sub>w</sub> = (1 / N) · Σ I(H<sub>w</sub><sup>(k)</sup> > T<sub>w</sub>)**

where:
- H<sub>w</sub><sup>(k)</sup> = k-th scenario  
- Σ = summation over scenarios  
- I(·) = indicator function  
- N = total number of generated hazard scenarios (ensemble size, typically ~400 in this framework)
---

##  Future Climate Risk Evolution (CMIP6 — MIROC6, SSP245, SSP585)

### Climate Scenarios

- **SSP245** → intermediate emissions (~2.5°C warming)  
- **SSP585** → high emissions (~4–5°C warming)  

Using **CMIP6 MIROC6 projections**, rainfall fields are scaled:

---

###  Climate Scaling

**Δ<sub>w</sub> = Mean<sub>future</sub> / Mean<sub>historical</sub>**

**H<sub>w</sub><sup>future</sup> = H<sub>w</sub><sup>baseline</sup> · Δ<sub>w</sub>**

**R<sub>w</sub><sup>future</sup> = H<sub>w</sub><sup>future</sup> · V<sub>w</sub>**

---

###  Change in Risk

**ΔR<sub>w</sub> = R<sub>w</sub><sup>future</sup> − R<sub>w</sub><sup>baseline</sup>**

---

###  Change in Probability

**ΔP<sub>w</sub> = P<sub>w</sub><sup>future</sup> − P<sub>w</sub><sup>baseline</sup>**

---

### Climate Amplification

**A<sub>w</sub> = R<sub>w</sub><sup>future</sup> / R<sub>w</sub><sup>baseline</sup>**

---

### Decision Score

**D<sub>w</sub> = R<sub>w</sub> · P<sub>w</sub> · A<sub>w</sub>**

---

Thresholds:

- T<sub>D</sub> = 80th percentile of D  
- T<sub>R</sub> = 75th percentile of R  

---

Classification:

- **Act:** D<sub>w</sub> > T<sub>D</sub>  
- **Monitor:** D<sub>w</sub> ≤ T<sub>D</sub> and R<sub>w</sub> > T<sub>R</sub>  
- **Low Priority:** otherwise  

---

## Pipeline

**H<sup>(k)</sup>(x, y) → H<sub>w</sub> → R<sub>w</sub> → P<sub>w</sub> → A<sub>w</sub> → D<sub>w</sub>**

<img width="1500" height="861" alt="image" src="https://github.com/user-attachments/assets/0aaecafa-bba4-4ee1-9a85-90a2ad712931" />

Figure.7: Ward-level flood risk (impact) and probability of extreme flood events (likelihood) across increasing return periods (10–100 years) under observed climate (2001–2024).

Insight: Risk intensity and extreme-event likelihood both amplify with higher return levels, with hotspots consistently concentrated along the Musi river corridor.


<img width="1196" height="1125" alt="image" src="https://github.com/user-attachments/assets/00d703ae-860f-419e-954e-57be55b9e52f" />

Figure.8: Future climate-driven evolution of ward-level flood risk for the 100-year return period using CMIP6 (MIROC6) projections under SSP245 and SSP585 scenarios. The rows show flood risk (impact), change in risk (ΔRisk), change in extreme event probability (ΔProbability), climate amplification (risk ratio: Future/Baseline), and final decision zones based on the composite score D = R · P · A.

Insight: Climate change systematically amplifies flood risk and its spatial heterogeneity, with SSP585 exhibiting stronger intensification and variability compared to SSP245. While some regions show reductions (blue) due to localized shifts in rainfall patterns, most high-risk wards—especially along the Musi river corridor—experience increased risk and/or likelihood. Climate amplification maps highlight where future hazards disproportionately escalate relative to baseline conditions, and the final decision zones translate these compounded effects into actionable priorities (Act/Monitor/Low), enabling targeted, risk-informed climate adaptation planning.

## 🌍 Climate-Conditioned Catastrophe Risk Modeling (EP Curves, Loss Metrics & Spatial Impact)

This section presents a full probabilistic catastrophe modeling pipeline integrating generative hazard ensembles, exposure modeling, vulnerability functions (damage curves) and Monte Carlo simulation for probabilistic flood risk estimation under baseline and future climate scenarios (CMIP6 – SSP245, SSP585).

---

# 1. Hazard Aggregation (Grid → Ward)

Hazard fields from generative ensembles are aggregated to ward level using area-weighted integration:

H<sub>w</sub><sup>(k)</sup> = Σ<sub>i</sub> ( H<sub>i</sub><sup>(k)</sup> · w<sub>i,w</sub> )

where:

- H<sub>i</sub><sup>(k)</sup> = hazard at grid cell *i* for scenario *k*  
- w<sub>i,w</sub> = area weight of grid cell *i* within ward *w*  
- Σ<sub>i</sub> w<sub>i,w</sub> = 1  

---

#  2. Exposure Model (Asset Value)

Ward-level asset value is constructed using proxy economic indicators:

A<sub>w</sub> = (ρ<sub>w</sub> · α) + (R<sub>w</sub> · β) + (Area<sub>w</sub> · γ)

where:

- ρ<sub>w</sub> = population density  
- R<sub>w</sub> = road density  
- Area<sub>w</sub> = ward area (km²)  
- α = 2000, β = 8000, γ = 50000  (assumptions)

Scaling:

A<sub>w</sub> = A<sub>w</sub> / 10⁶  

Units: ₹ Millions  

---

#  3. Vulnerability Function (Damage Function: Hazard → Damage)

A logistic vulnerability curve converts hazard intensity into fractional damage:

### SSP245:

D<sub>w</sub> = 1 / (1 + exp(-2.5 · (H<sub>w</sub> - 1.2)))

### SSP585:

D<sub>w</sub> = 1 / (1 + exp(-3.2 · (H<sub>w</sub> - 1.1)))

where:

- H<sub>w</sub> = hazard intensity  
- D<sub>w</sub> ∈ [0,1] = damage ratio  

---

#  4. Hazard Uncertainty & Scaling

Hazard is perturbed using stochastic variability:

H<sub>w</sub> ← H<sub>w</sub> · RL<sub>scale</sub> · LogNormal(0, σ<sub>rl</sub>) · Normal(1, 0.25)

where:

- RL<sub>scale</sub> = return-level scaling (e.g., 1.0, 1.2, 1.5, 2.0)  
- σ<sub>rl</sub> = uncertainty scaling by return level  

---

#  5. Climate Amplification (SSP Forcing)

Normalized hazard:

Ĥ<sub>w</sub> = H<sub>w</sub> / max(H<sub>w</sub>)

### SSP245:

C<sub>w</sub> = 1 + 0.15 · Ĥ<sub>w</sub>

### SSP585:

C<sub>w</sub> = 1 + 0.5 · Ĥ<sub>w</sub>

---

# 6. Spatial Amplification

Urban effects are incorporated as:

S<sub>w</sub> = 1 + 0.6 · I<sub>w</sub> + 0.3 · ε

where:

- I<sub>w</sub> = impervious surface fraction  
- ε ~ N(0,1)  

---

# 7. Event Loss Formulation

Ward-level loss:

L<sub>w</sub> = D<sub>w</sub> · A<sub>w</sub> · S<sub>w</sub> · C<sub>w</sub>

Event-level stochastic scaling:

L<sub>w</sub> ← L<sub>w</sub> · LogNormal(0, 0.7)

Total event loss:

L<sub>event</sub> = Σ<sub>w</sub> L<sub>w</sub>

---

# 8. Event Frequency Model

Number of events per year:

λ ~ Gamma(2, 2.5)  
N ~ Poisson(λ)

Return-level sampling probabilities:

P(RL) = {10: 0.6, 25: 0.25, 50: 0.1, 100: 0.05}

---

# 9. Monte Carlo Simulation of Annual Loss

A Monte Carlo framework is used to simulate stochastic event sequences and aggregate annual losses over T = 10,000 synthetic years.

Each simulation samples event frequency, hazard intensity, vulnerability response, and loss uncertainty to generate a full probabilistic loss distribution.

Annual loss:

L<sub>year</sub> = Σ<sub>i=1</sub><sup>N</sup> L<sub>event,i</sub>

Simulated over:

T = 10,000 years  

---

# 10. Exceedance Probability (EP Curve)

Loss exceedance probability:

EP(x) = P(L ≥ x)

Empirical estimate:

EP(L<sub>i</sub>) = i / N

(after sorting losses in descending order)

---

# 11. Risk Metrics

### Expected Annual Loss (AAL)

AAL = (1/N) · Σ L<sub>year</sub>

---

### Probable Maximum Loss (P99)

P99 = Quantile(L<sub>year</sub>, 0.99)

---

### Tail Value at Risk (TVaR)

TVaR = E[L<sub>year</sub> | L<sub>year</sub> ≥ P99]

---

# 12. Spatial Risk Metrics

Ward-level extreme loss:

L<sub>w</sub><sup>P99</sup> = Quantile(L<sub>w</sub>, 0.99)

Expected loss:

E[L<sub>w</sub>] = mean(L<sub>w</sub>)

---

# 13. Climate Risk Divergence

Relative impact (SSP585 vs SSP245):

Δ<sub>w</sub> = (L<sub>w</sub><sup>585</sup> − L<sub>w</sub><sup>245</sup>) / (L<sub>w</sub><sup>245</sup> + ε)

Clipped for stability:

Δ<sub>w</sub> ∈ [-0.5, 0.5]

---

# 14. Interpretation

- **EP Curve:** Shift to right → increasing loss severity  
- **AAL:** captures average yearly loss  
- **P99:** captures extreme catastrophic loss  
- **TVaR:** captures tail risk beyond threshold  
- **Spatial maps:** identify localized climate risk hotspots  
- **SSP585:** shows strong nonlinear amplification of extreme losses  

---

# End-to-End Pipeline

H<sub>i</sub><sup>(k)</sup> → H<sub>w</sub> → D<sub>w</sub> → L<sub>w</sub> → L<sub>event</sub> → L<sub>year</sub>  
→ EP Curve → {AAL, P99, TVaR} → Spatial Risk → Climate Impact  

---

<img width="1500" height="895" alt="image" src="https://github.com/user-attachments/assets/3bf7ffd5-4c7f-4a75-a701-d767c969adfa" />

Figure.9: Exceedance Probability (EP) curves comparing baseline and future climate scenarios (SSP245 and SSP585), showing the relationship between annual exceedance probability and total loss (₹ Millions).

Insight: Both SSP scenarios exhibit a clear rightward shift relative to the baseline, indicating higher losses across all probability levels, with SSP585 showing the strongest amplification. The divergence becomes more pronounced in the tail (low-probability, high-impact events), highlighting that climate change disproportionately increases extreme losses rather than just average risk.

<img width="1234" height="765" alt="image" src="https://github.com/user-attachments/assets/bb71e0a7-c3ec-425e-82ca-1893f4471498" />

Figure.10: Comparison of key risk metrics—Expected Annual Loss (AAL), 99th percentile loss (P99), and Tail Value at Risk (TVaR)—across baseline, SSP245, and SSP585 climate scenarios.

Insight: All risk metrics increase under future climate scenarios, with SSP585 showing the largest escalation. While AAL exhibits moderate growth, the sharper rise in P99 and TVaR indicates that climate change disproportionately amplifies extreme and tail losses, highlighting a significant increase in catastrophic flood risk rather than just average impacts.

<img width="1500" height="403" alt="image" src="https://github.com/user-attachments/assets/3b3c68ad-3c18-42ce-9cf6-35068f645947" />

Figure.11: Spatial distribution of extreme losses (P99) under SSP245 and SSP585, along with relative climate impact showing the normalized difference between scenarios.

Insight: Extreme losses intensify and expand spatially under SSP585 compared to SSP245, with clear amplification in urban and flood-prone zones. The relative impact map highlights localized hotspots where climate change disproportionately increases risk, revealing strong spatial heterogeneity rather than uniform escalation.

Climate change amplifies flood risk nonlinearly, with disproportionate escalation in extreme losses and strong spatial heterogeneity, making tail-risk-aware and location-specific adaptation strategies essential.

#  Key Innovations

- **Graph-Conditioned Generative Hazard Modeling**  
  Introduced a Conditional Graph-Based Denoising Diffusion Probabilistic Model (DDPM) to generate spatially coherent rainfall extremes, capturing complex dependencies beyond traditional statistical methods.

- **Integration of Extreme Value Theory with Deep Generative Models**  
  Combined Peaks Over Threshold (Generalized Pareto Distribution) with graph diffusion and DDPM to model both **tail extremes** and **spatial structure**, bridging classical hydrology and modern AI.

- **Physics-Informed Spatial Graph Learning**  
  Constructed a spatial graph G = (V, E, W) incorporating adjacency, distance, terrain, and drainage connectivity, enabling physically consistent propagation of hazard across regions.

- **Probabilistic Hazard Index from Generative Ensembles**  
  Developed a novel hazard formulation combining mean, variability, and tail quantiles (P90–P95), capturing **uncertainty + extremes** in a single index.

- **Full End-to-End Catastrophe Modeling Pipeline**  
  Built a complete pipeline:  
  Bias Correction → EVT → Graph Diffusion → Generative Modeling → Vulnerability → Risk → Loss → EP Curves → Climate Impact

- **Climate-Conditioned Risk Simulation (CMIP6 Integration)**  
  Incorporated SSP245 and SSP585 projections (MIROC6) to quantify **future risk amplification**, moving beyond static risk estimation.

- **Stochastic Loss Modeling with Monte Carlo Simulation**  
  Generated thousands of synthetic event years to compute EP curves, AAL, P99, and TVaR — aligning with industry catastrophe modeling standards (RMS, AIR, Munich Re).

- **Spatially Explicit Decision Intelligence Framework**  
  Translated probabilistic risk into **actionable decision zones (Act / Monitor / Low Priority)** using a composite score combining risk, probability, and climate amplification.

---

#  Why This Approach?

- **Traditional flood models are deterministic**  
  They fail to capture uncertainty, variability, and tail behavior of extreme rainfall.

- **Extreme events are increasing nonlinearly**  
  Climate change affects not just averages but **distribution tails**, which classical models underestimate.

- **Spatial dependencies are critical**  
  Flooding is inherently connected through terrain, drainage, and urban structure — ignoring this leads to unrealistic hazard fields.

- **Risk is multi-dimensional**  
  It depends on hazard, exposure, vulnerability, and adaptation — requiring an integrated framework rather than isolated models.

- **Decision-making requires probabilistic insights**  
  Governments and insurers need **likelihood + impact + uncertainty**, not single-value predictions.

- **Urban systems amplify climate risk**  
  Rapid urbanization (impervious surfaces, infrastructure stress) interacts with climate forcing, demanding spatially aware modeling.

---


#  Limitations of HydroGraphDiff

- **Limited historical data (2001–2024)**  
  ~23 years of data may not fully capture rare extreme events (e.g., 100+ year floods), introducing uncertainty in tail estimation.

- **Simplified hydrodynamics**  
  The framework does not explicitly solve physical flow equations, so detailed flood dynamics (flow routing, inundation depth) are approximated via graph diffusion.

- **Climate projection uncertainty (CMIP6 SSPs)**  
  Future rainfall extremes are scaled using CMIP6, but reliance on a single GCM (MIROC6) introduces model-specific bias and limits robustness.
  Using a multi-model ensemble would capture inter-model variability, reduce bias, and enable more reliable probabilistic climate risk estimation.

- **Static exposure and vulnerability assumptions**  
  Asset values and vulnerability are treated as constant, whereas in reality they evolve with urbanization, infrastructure, and adaptation.

---

# Future Improvements

Addressing these limitations would involve:

- **Coupling with hydrodynamic models for physics-informed validation**  
  Integrate shallow water equation-based models to capture flow routing, inundation depth, and channel dynamics, improving physical realism of hazard propagation.

- **Integration of high-resolution, multi-source Earth Observation data**  
  Incorporate SAR, multispectral imagery, LiDAR, and high-resolution DEMs to better resolve urban flooding patterns and improve hazard and exposure estimation.

- **Dynamic exposure and vulnerability modeling**  
  Develop time-evolving representations of population, infrastructure, land-use change, and adaptive capacity to reflect realistic future risk scenarios.

- **Time-resolved climate projections (multi-model, beyond scaling approaches)**  
  Utilize dynamically downscaled CMIP6 multi-model ensembles to capture changes in rainfall intensity, duration, temporal structure, inter-model variability, and compound extreme events, moving beyond static scaling    methods.

- **Improved generative modeling of extreme tails**  
  Enhance robustness of diffusion models using tail-focused training, hybrid EVT–deep learning approaches, and larger datasets to better represent rare, high-impact events.

---

# Final Perspective

HydroGraphDiff represents a next-generation shift in flood risk modeling by unifying extreme value theory, deep generative AI, graph-based spatial modeling, and climate projections into a single probabilistic framework.

Unlike traditional deterministic approaches, it explicitly captures **uncertainty, tail risk, and spatial dependencies**, enabling realistic simulation of extreme events and their cascading impacts across complex urban systems.

By integrating **climate-conditioned risk amplification, stochastic loss modeling, and decision-centric metrics (EP curves, AAL, PML, TVaR)**, the framework bridges climate science, AI, and catastrophe risk analytics—making it directly applicable to insurers, urban planners, and climate resilience decision-making.

Most importantly, HydroGraphDiff lays the foundation for **AI-driven probabilistic climate risk digital twins**, where extensions such as multi-model climate ensembles, physics-informed learning, and dynamic exposure modeling can evolve it into **a scalable, real-time climate risk intelligence platform**.

