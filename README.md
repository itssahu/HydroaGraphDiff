# HydroaGraphDiff: A Generative AI–Driven Probabilistic Flood Risk & Catastrophe Modeling System (Hyderabad, Musi Basin)

HydroGraphDiff is an end-to-end probabilistic flood risk and catastrophe modeling system developed for the Hyderabad (Musi basin) region. The project integrates geospatial data, extreme value theory (EVT), graph-based spatial modeling, and generative AI (diffusion models) to simulate flood hazards and quantify climate-driven financial risk at ward-level resolution.

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
The same graph $G$ is then reused within a ## Graph Conditioned DDPM, where it plays two roles: (i) as a conditioning structure guiding the model to learn spatial dependencies in hazard fields, and (ii) through a graph Laplacian regularization ($L = D - W$), which enforces smoothness and physical consistency by penalizing unrealistic spatial discontinuities. The diffusion model learns the conditional distribution of hazard fields and generates multiple stochastic realizations, thereby capturing uncertainty and variability in extreme rainfall patterns.
Vulnerability is constructed from geographic features (terrain, infrastructure, population) into sensitivity, exposure, and adaptive capacity. Risk is computed as $R = H \times V$, with uncertainty, extreme probability, climate scenarios (SSP245/585), and loss exceedance curves (EP, PML, TVaR) providing a comprehensive probabilistic climate risk assessment.
