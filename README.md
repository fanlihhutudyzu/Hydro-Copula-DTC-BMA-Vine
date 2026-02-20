# Structural Complexity and Uncertainty in Multivariate Copula-Based Environmental Simulation

## MATLAB Implementation Repository

This repository provides the MATLAB codes and datasets supporting the manuscript:

> *Complexity vs. Efficiency: Are Full-Structure Copula Models Worth the Cost in Multivariate Hydrological Simulation?*

The repository focuses on the **construction, fitting, and stochastic simulation** of three multivariate copula frameworks:

* Gaussian / t Copula (GtC)
* Dependence Tree Copula (DTC)
* Bayesian Model Averaging with Vine Copulas (BMA-Vine)

This implementation is intended to ensure methodological transparency and reproducibility of the model construction and random sample generation procedures presented in the manuscript.

---

# Repository Structure

```
├── Data/
├── GtC-DTC/
└── BMA-Vine/
```

---

# 1. Data Folder

The `Data` folder contains four `.mat` files with observed environmental datasets.

### LTZ_Data.mat

Three-dimensional flood events at Lutaizi Hydrological Station

* Column 1: Peak discharge (m³/s)
* Column 2: Flood volume (10⁸ m³)
* Column 3: Flood duration (h)

### QZ_Data.mat

Three-dimensional flood events at Quzhou Hydrological Station

* Column 1: Peak discharge (m³/s)
* Column 2: Flood volume (10⁸ m³)
* Column 3: Flood duration (h)

### preci_yangzhou.mat

Annual precipitation (1981–2024) from four stations in Yangzhou, China

* Column 1: Yangzhou
* Column 2: Yizheng
* Column 3: Jiangdu
* Column 4: Gaoyou
  Unit: mm

### storm_NL_4D.mat

Storm surge data from the Netherlands (1979–2009)

* Column 1: Significant wave height H1/3 (cm)
* Column 2: Duration (h)
* Column 3: Peak wave period Tp (s)
* Column 4: Water level (cm)

---

# 2. GtC-DTC Folder

Contains MATLAB implementations of Gaussian/t Copula (GtC) and Dependence Tree Copula (DTC).

```
GtC-DTC/
├── 3D/
└── 4D/
```

## 3D Subfolder

* `generate_scenario_3Ddata.m`
  Generates synthetic three-dimensional datasets.

* `scenario_loop_GtDTC_3d.m`
  Performs simulation experiments on synthetic data using GtC and DTC models.

* `GtDTC_3d_obs.m`
  Fits and simulates models for observed 3D datasets.

## 4D Subfolder

* `scenario_loop_GtDTC_4d.m`
  Simulation experiments for synthetic four-dimensional datasets.

* `GtDTC_4d_obs.m`
  Model fitting and simulation for observed 4D datasets.

* `removeties.m`
  Preprocesses storm data to remove ties before copula estimation.

* `copula_gibbs_model.m`
  Core function implementing the DTC model.

* `gaussian_t_copula_model.m`
  Core function implementing Gaussian and t copulas.

---

# 3. BMA-Vine Folder

Contains MATLAB implementations of the Bayesian Model Averaging Vine Copula framework.

```
BMA-Vine/
├── 3D/
└── 4D/
```

## 3D Subfolder

* `generate_scenario_3Ddata.m`
  Generates synthetic three-dimensional datasets.
  
* `scenario_loop_vine_bma_3d.m`
  Performs simulation experiments on 3D synthetic data using BMA-Vine Copula models.
  
* `vine_bma_3d_obs.m`
  Fits and simulates models for 3D observations using BMA-Vine Copula models.

## 4D Subfolder

* `scenario_loop_vine_bma_4d.m`
  Performs simulation experiments on 4D synthetic data using BMA-Vine Copula models.
  
* `vine_bma_4d_obs.m`
  Fits and simulates models for 4D observations using BMA-Vine Copula models.
  
* Additional `.mat` files required internally for BMA-Vine Copula models.

---

# Reproducibility

* Random seeds are explicitly set in the simulation scripts to ensure reproducible results.
* The repository provides complete workflows for:

  * Copula model construction
  * Parameter estimation
  * Random sample generation
  * Scenario-based simulation experiments

Note:
This repository focuses on model construction and stochastic simulation.
Empirical goodness-of-fit comparison codes (e.g., CvM or energy distance statistics) are presented in the manuscript but are not included here.

---


# Requirements

* MATLAB R2020a or later
* No external third-party toolboxes are required.

---


# Contact

Fan Li (lifan@yzu.edu.cn)

College of Hydraulic Science and Engineering, Yangzhou University, China


