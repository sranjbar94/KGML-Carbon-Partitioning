# KGML Carbon Partitioning

**Reference:**  
Ranjbar, S., Desai, A. R., Hoffman, S., & Stoy, P. C. (2025). *Constrained carbon partitioning: a self-trained physics-informed machine learning model reduces GPP overestimation from eddy covariance measurements*.

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
