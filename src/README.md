# src/ — Code for Emotion Decoding from fMRI Data



Refer to Chapter 4 (Methodology) and Chapter 5 (Results) in the thesis for methodological context and interpretation of outputs.

---

## Folder Overview

### sentiment_analysis/
- Contains scripts for automatic emotional labeling of narrative text using GPT-4.
- Includes text files (`alice.txt`, `alice_sentiment_analysis.txt`), plotting scripts, and prompt logic.
- Outputs emotion scores for each text segment.
- Reference: Thesis Section 4.2

### model_training/
- Scripts to train regression models on both ROI and DFC datasets.
- Subfolders divided by representation (ROI / DFC) and linear, ridge, lasso, SVR and RF models.
- Reference: Thesis Sections 4.7–4.9

### hyperparametersearches/
- Scripts for grid/random search of optimal model hyperparameters.
- Organized by representation and model.
- Reference: Appendix C in thesis

### permutation_testing/
- Code for label permutation testing to assess model robustness.
- Mirrors training directory structure.
- Useful for generating null distributions.
- Reference: Thesis Section 4.9 and 5.1.3

### explanations/
- Main module for generating feature importance and XAI outputs.
- Divided into `ROI/` and `DFC/` explanation tools.
  - ROI: direct feature weight analysis.
  - DFC: MST generation, network graph metrics, heatmaps, and similarity matrices.
- Output folders store results grouped by model and emotion.
- Reference: Thesis Sections 4.10, 5.3–5.4


### utils/
- General-purpose scripts and helpers:

---

## Execution Workflow

1. **Run sentiment labeling** using `sentiment_analysis/gpt-4.ipynb`.
2. **Prepare data** (ROI and DFC representations) externally in fMRI_processing folder.
3. **Search hyperparameters** with scripts in `hyperparametersearches/`.
4. **Train models** using `model_training/` scripts.
5. **Run permutation tests** with the corresponding modules in `permutation_testing/`.
6. **Generate explanations** via `explanations/ROI/` or `explanations/DFC/`.
7. **Visualize and interpret** results using saved outputs. 

---


For full methodological and analytical context, refer to the thesis PDF.
