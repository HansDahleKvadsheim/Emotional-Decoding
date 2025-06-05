# explanations/ — Feature Importance and Model Explanation

This directory contains code and outputs for generating and analyzing explanations of machine learning models trained on fMRI-derived features. It supports both ROI-based and DFC-based data representations, across all regression models used in the thesis

For methodology and interpretation, refer to Section 4.10 and Chapters 5.3–5.4 in the thesis.

---

## Folder Structure

### ROI/
Contains code and output for ROI-based feature importance analysis.

- `roi_explainer.py`: Script for extracting feature weights from trained models and formatting them for visualization.
- `reverse_atlas.ipynb`:  convert ROI indices into spatial plots.
- Outputs:
  - `importance.csv` per model: Contains emotion-specific feature weights

### DFC/
Contains graph-based explanation tools for models trained on DFC matrices.

#### Main Scripts
- `DFC_MST_analysis.py`: Generates and analyzes MSTs from DFC matrices at network level. 
- `ROI_analysis.py`: Scores node-level contributions from pre-generated MSTs
- `global_metrics.py`: Computes global graph measures (e.g., modularity, efficiency)
- `Jackard_distance.py`: Computes emotion-wise similarity between MSTs using Jaccard distance
- `network_heatmap.py`, `ROI_heatmap.py`: Visualization utilities.

#### Outputs
- `importance.csv` in each model subfolder (e.g., `lasso/`, `ridge/`, `SVR/`, `RF/`, `linear/`)
- `intermediate_results/`: Internal metrics and MSTs
- `output/heatmaps/`: Region-wise and network-wise heatmaps of importance scores
- `output/signatures/`: Signature maps of emotional relevance
- `output/similarity/`: Pairwise emotion similarity matrices
