# Quantifying-Cognitive-Effort-s-Impact-on-Suppression-of-Epilepsy-Associated-After-Discharges
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
