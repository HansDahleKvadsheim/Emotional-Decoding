# fMRI_processing/ — ROI and DFC Extraction Code

This folder contains scripts used to convert raw fMRI data into ROI time series and dynamic functional connectivity (DFC) matrices

## Acknowledgment
The original version of this code was generously provided by **Shuer Ye**. We thank him for sharing it. The scripts have been modified to accommodate our project’s requirements, specifically:
- The use of **segment-aligned story stimuli**
- Formatting adjustments for compatibility with our ROI and DFC pipelines

---

## File Overview


### `ROIextract.m`
- Loads the preprocessed fMRI BOLD data
- Extracts average time series for each of the 400 parcels
- Outputs ROI matrix used in regression and feature attribution

### `dfc_generation.m`
- Main DFC pipeline script
- Segments the ROI time series into DFC segments of adjustable windows lengths

---

## Usage Notes

- Scripts are written in **MATLAB**.
- The segment timing must match the labeled emotional segments defined in `sentiment_analysis/`.

---

## Output

- `ROIextract.m`: Matrix of dimensions [n_timepoints x n_ROIs]
- `dfc_generation.m`: 3D array of shape [n_ROIs x n_ROIs x n_windows]
