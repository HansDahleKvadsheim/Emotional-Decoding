# Decoding Emotional Responses to Language

This repository contains the full codebase for the Master's thesis **“Decoding Emotional Responses to Language”** by August Sætre Aasvær and Hans Dahle Kvadsheim, completed at NTNU in Spring 2025.

---

## Overview

This project investigates whether classical regression models can decode emotional responses from fMRI data. It compares **ROI-based** and **Dynamic Functional Connectivity (DFC)** representations, uses **Large Language Models (LLMs)** for automated emotional labeling, and introduces **XAI-based techniques** for interpreting model behavior.

For full methodology, refer to **Chapters 1 and 4** of the thesis.

---

##  Repository Structure

emotional_decoding/  
│  
├── data/                  # (not included) Processed ROI & DFC datasets + (included) emotion labels  
├── fMRI_processing/       # Scripts for converting raw fMRI to ROI/DFC representations  
├── src/                   # Main source code for modeling and analysis  
│   ├── sentiment_analysis/ # LLM-based emotional labeling (see thesis §4.2)  
│   ├── models/             # Model training & evaluation scripts (§4.7–4.9)  
│   ├── explanations/       # XAI-based model explanations (§4.10)  
│   └── utils/              # Shared utilities  
└── README.md               # This file  





## Project Pipeline

1. **Emotional Labeling**  
   Naturalistic narrative (Alice in Wonderland) was segmented (~8s) and labeled with LLM on eight Plutchik emotions.  
   → See *Thesis Section 4.2*

2. **ROI Extraction**  
   Using Schaefer 2018 400-parcel atlas, ROIs were extracted from the Alice fMRI dataset.  
   → See *Thesis Section 4.3*

3. **Dynamic Functional Connectivity (DFC)**  
   DFC matrices were computed using a 22-second sliding window.  
   → See *Thesis Section 4.5*

4. **Model Training**  
   Classical regressors (Linear, Ridge, Lasso, SVR, RFR) trained on both ROI and DFC data.  
   → See *Thesis Sections 4.7–4.9*

5. **Model Explanation**  
   Feature importances were analyzed:
   - ROI: region-wise scores
   - DFC: MST-based graph structures  
   → See *Thesis Section 4.10*



## Dataset

Due to size constraints, this repository does **not** include data files.

- **fMRI Data Source**:  
  Download from [OpenNeuro: ds002322](https://openneuro.org/datasets/ds002322/versions/1.0.4)

- **Generate your own ROI/DFC representations** using scripts in `fMRI_processing/`.



> Aasvær, A. S., & Kvadsheim, H. D. (2025). *Decoding Emotional Responses to Language*. Master’s Thesis, Norwegian University of Science and Technology (NTNU).
