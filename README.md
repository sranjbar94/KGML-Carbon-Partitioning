# KGML Carbon Partitioning

**Reference:**  
Ranjbar, S., Desai, A. R., Hoffman, S., & Stoy, P. C. (2025). *Constrained carbon partitioning: a self-trained physics-informed machine learning model to partition carbon measurements into GPP and RECO from eddy covariance measurements*.

# Constrained Carbon Partitioning using KGML

This repository contains code for the study:

**Constrained carbon partitioning: a self-trained physics-informed machine learning model reduces GPP overestimation from eddy covariance measurements**  
*Sadegh Ranjbar1, Ankur R. Desai2, Sophie Hoffman1, Paul C. Stoy1*  
1Department of Biological Systems Engineering, University of Wisconsin – Madison  
2Department of Atmospheric and Oceanic Sciences, University of Wisconsin – Madison  

---

## Overview
Gross Primary Productivity (GPP) is a key component of the carbon cycle but cannot be directly measured.  
This project implements a **knowledge-guided machine learning (KGML)** model to partition Net Ecosystem Exchange (NEE) into GPP and ecosystem respiration (RECO), constrained by physical laws and theoretical expectations.

---

## Features
- Self-supervised, physics-informed KGML model
- Monte Carlo dropout for uncertainty estimation
- Visualizations of energy balance, GPP, RECO, and WUE
- Reproducible workflow for NEON tower datasets

---

## Installation

```bash
git clone <repo-url>
cd kgml_carbon_partitioning
pip install -r requirements.txt


This repository contains code and datasets for the KGML-based carbon partitioning framework. It includes:

- Physics-informed KGML model
- Self-supervised training routines
- Visualization scripts
- Example Jupyter notebook

## Folder Structure

- `data/` : Datasets
- `notebooks/` : Example notebooks
- `src/` : Python modules
- `outputs/` : Figures & trained models

## How to Run

1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare dataset in `data/`

3. Run notebook:  
   ```bash
   jupyter notebook notebooks/KGML_Experiment.ipynb
   ```
