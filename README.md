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
