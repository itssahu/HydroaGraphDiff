# HydroaGraphDiff: A Generative AI–Driven Probabilistic Flood Risk & Catastrophe Modeling System (Hyderabad, Musi Basin)

HydroGraphDiff is an end-to-end probabilistic flood risk and catastrophe modeling system developed for the Hyderabad (Musi basin) region. The project integrates geospatial data, extreme value theory (EVT), graph-based spatial modeling, and generative AI (Diffusion Models) to simulate flood hazards and quantify climate-driven financial risk at ward-level resolution.

## 1. Data Used
IMERG (satellite rainfall), 
IMD (gauge rainfall), 
DEM (elevation, slope), 
ESA WorldCover (urbanization), 
OSM (roads, hospitals), 
WorldPop (population),
CMIP6(MIROC6-historical, ssp245, ssp585)

## 2. Architecture
<img width="706" height="707" alt="image" src="https://github.com/user-attachments/assets/f649a0e3-a7dc-4819-a2f1-b65e5b8b642e" />

Figure 1. Final architecture of HydroGraphDiff following a standard hazard–vulnerability–risk framework.

Hazard is derived from ERA5 and IMD rainfall using quantile-mapped bias correction and EVT-based extreme modeling to estimate extreme rainfall magnitudes (return levels). 
A spatial graph $G=(V,E,W)$ is constructed, where nodes $V$ represent grid cells, edges $E$ encode spatial adjacency, and weights $W$ capture distance, terrain, and drainage influence. This graph is first used in a graph diffusion step to propagate extreme rainfall across neighboring regions, producing spatially coherent hazard fields that reflect physical spillover and connectivity.
The same graph $G$ is then reused within a Graph Conditioned DDPM, where it plays two roles: (i) as a conditioning structure guiding the model to learn spatial dependencies in hazard fields, and (ii) through a graph Laplacian regularization ($L = D - W$), which enforces smoothness and physical consistency by penalizing unrealistic spatial discontinuities. The diffusion model learns the conditional distribution of hazard fields and generates multiple stochastic realizations, thereby capturing uncertainty and variability in extreme rainfall patterns.
Vulnerability is constructed from geographic features (terrain, infrastructure, population) into sensitivity, exposure, and adaptive capacity. Risk is computed as $R = H \times V$, with uncertainty, extreme probability, climate scenarios (SSP245/585), and loss exceedance curves (EP, PML, TVaR) providing a comprehensive probabilistic climate risk assessment.

## 3. Bias Correction, Extreme Value Theorem and Graph Diffusion

Figure 2: Extreme rainfall modeling and spatial enhancement pipeline

<p align="center">
  <img src="<img width="1500" height="787" alt="image" src="https://github.com/user-attachments/assets/c760d4a8-1a16-424f-a24f-4449c8c04ff9" />" width="32%" />
  <img src="<img width="1086" height="855" alt="image" src="https://github.com/user-attachments/assets/09d1c585-526d-42ce-9a0b-b8c2a4d52e6f" />" width="32%" />
  <img src="<img width="1500" height="675" alt="image" src="https://github.com/user-attachments/assets/b338e860-1162-4687-be8d-b4aef3ff8aeb" />" width="32%" />
</p>

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

Figure 2. Final architecture of the Graph Conditioned DDPM in HydroGraphDiff. A for-
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

Figure 3. Diffusion dynamics in HydroGraphDiff for RL100 extreme rainfall scenario.
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

  Figure 4. Graph-conditioned DDPM generates diverse stochastic realizations of RL100(100 year return level) extreme rainfall, capturing spatial variability and uncertainty across the urban flood landscape.

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

Figure 5. Ward-level flood vulnerability components for Hyderabad, showing spatial patterns of adaptive capacity, sensitivity, potential impact, and resulting structural vulnerability.
Insight: High vulnerability emerges where elevated sensitivity and exposure coincide with low adaptive capacity, highlighting flood risk hotspots along the Musi river corridor.




