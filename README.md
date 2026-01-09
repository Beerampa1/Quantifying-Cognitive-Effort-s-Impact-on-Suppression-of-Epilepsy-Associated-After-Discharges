# Quantifying-Cognitive-Efforts-Impact-on-Suppression-of-Epilepsy-Associated-After-Discharges
This project investigates how cognitive effort influences the suppression of epilepsy-associated after discharges (ADs) in intracranial EEG (iEEG) recordings collected during pre-surgical functional brain mapping. Using data from epileptic patients, the code reproduces all preprocessing, analysis, modeling pipelines presented in the paper.


# Project Workflow Overview

This project provides a **complete, reproducible pipeline** for analyzing intracranial EEG (iEEG) data to study how **cognitive effort modulates the suppression of epilepsy-associated after-discharges (ADs)**.  

The workflow supports:
- **Interactive exploration**
- **Batch feature extraction**
- **Predictive model training**

All components are accessible through a **GUI-driven application**, enabling both exploratory neuroscience analysis and reproducible machine-learning experiments.

---

## High-Level Workflow

1. **Launch Application**
2. **Select Processing Route**
3. **Load Trial Metadata (Excel) + EEG Data (H5)**
4. **Select Trial & Time Windows**
5. **Run Signal / Network / Fractal Analyses**
6. **Extract Features (Batch or Interactive)**
7. **Train & Evaluate Predictive Models**

---

## Application Entry Point

```bash
python main.py
```

## Route Selector (GUI Navigation)

The **Route Selector** is the main entry point of the application and allows users to choose between multiple analysis and processing paths:

- **Process Route**  
  → Interactive, trial-level signal exploration

- **Feature Extraction Route**  
  → Batch feature generation across all trials

- **FODN / DFA / MF-DFA Post-Processing**  
  → Visualization and statistical summaries

- **MF-DFA Overlapping Analysis**  
  → Sliding-window fractal analysis

- **Model Training Route**  
  → Train and evaluate machine-learning classifiers

---

## Interactive Processing Route (Trial-Level)

### Main Files
- `gui/main_gui.py`
- `gui/channel_plot_window.py`
- `gui/analysis_window.py`

### Workflow
1. Load **Excel trial metadata**
2. Load **iEEG H5 file**
3. Select a **patient trial**
4. Visualize raw signals and individual channels
5. Launch advanced analyses:
   - DFA / MF-DFA analysis
   - Overlapping MF-DFA windows
   - FODN network analysis

This route is intended for **manual inspection**, **parameter tuning**, and **exploratory analysis**.

---

## Feature Extraction (Batch Processing)

### Main Files
- `gui/feature_extraction_window.py`
- `utils/feature_extractor.py`
- `utils/file_utils.py`

### What It Does
- Iterates over **all trials**
- Loads precomputed results:
  - FODN outputs
  - DFA / MF-DFA outputs
- Extracts:
  - Global summary features
  - Top-K channel-level features
- Produces a unified dataset:

```text
master_features.csv
```


## FODN Analysis (Network Dynamics)

### Core Logic
- `utils/fodn_code.py`
- `utils/fodn_utils.py`

### GUI Components
- `gui/fodn_analysis_window.py`
- `gui/fodn_post_window.py`
- `gui/eigen_vector_heatmap.py`

### Outputs
- Fractional orders (α)
- Coupling matrices
- Leading eigenvalues and eigenvectors
- Sparsity and other network metrics

These features characterize **network-level suppression dynamics** associated with cognitive effort.

---

## DFA / MF-DFA Analysis (Fractal Dynamics)

### Core Logic
- `utils/fast_mfdfa.py`  
  *(NumPy / Numba accelerated for high performance)*

### GUI Components
- `gui/dfa_post_window.py`
- `gui/mfdfa_analysis_window.py`
- `gui/mfdfa_overlap_window.py`

### Capabilities
- Single-window DFA / MF-DFA
- Overlapping (sliding-window) MF-DFA
- Per-channel and per-segment summaries

These analyses capture **scale-dependent signal complexity** in iEEG recordings.

---

## Model Training & Evaluation

### Main File
- `gui/model_training_window.py`

### Workflow
1. Load `master_features.csv`
2. Select the label column
3. Configure preprocessing:
   - NaN handling
   - Feature standardization
4. Train a logistic regression model
5. Display evaluation metrics:
   - Accuracy
   - ROC-AUC
   - Confusion matrix
   - Top contributing features

This route validates whether extracted features can **predict AD suppression under cognitive effort**.


## Key Findings Summary

This study shows that cognitive effort has a measurable and meaningful impact on suppressing epilepsy-associated after-discharges (ADs) in intracranial EEG recordings. We found that fractional-order exponents (α), dominant network eigenvectors, and Hurst exponents all change systematically before and after cognitive questioning, indicating that both the temporal complexity of brain signals and the underlying network structure are sensitive to cognitive effort. When these features were combined in a logistic regression model, the system achieved 77% accuracy under leave-one-out cross-validation despite a limited number of trials, with no clear signs of overfitting. Together, these results suggest that cognitive processing influences AD suppression through distributed network mechanisms rather than isolated brain regions. Overall, the findings support the idea that cognitive effort can serve as a non-invasive, low-cost approach to modulating epileptic activity and provide a foundation for future work aimed at understanding when, where, and how cognitive engagement most effectively suppresses after-discharges.

