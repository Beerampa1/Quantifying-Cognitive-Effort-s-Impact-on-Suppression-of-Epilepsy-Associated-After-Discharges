# Quantifying-Cognitive-Effort-s-Impact-on-Suppression-of-Epilepsy-Associated-After-Discharges
This project investigates how cognitive effort influences the suppression of epilepsy-associated after discharges (ADs) in intracranial EEG (iEEG) recordings collected during pre-surgical functional brain mapping. Using data from epileptic patients, the code reproduces all preprocessing, analysis, modeling pipelines presented in the paper.

This repository contains the full analysis pipeline used to study how cognitive effort affects the suppression of epilepsy-associated after-discharges.

The workflow supports interactive exploration, batch feature extraction, and model training, all accessible through a GUI-driven application.

High-Level Workflow

Launch Application

Select Processing Route

Load Trial Metadata (Excel) + EEG Data (H5)

Select Trial & Time Windows

Run Signal / Network / Fractal Analyses

Extract Features (Batch or Interactive)

Train & Evaluate Predictive Models

Application Entry Point
python main.py


This launches the Route Selector, which acts as the main navigation hub for all analysis pipelines.

Route Selector (GUI Navigation)

The route selector allows users to choose between different analysis paths:

Process Route → interactive trial-level exploration

Feature Extraction Route → batch feature generation

FODN / DFA / MF-DFA Post-Processing → visualization & summaries

MF-DFA Overlapping Analysis → sliding-window fractal analysis

Model Training Route → train & evaluate classifiers

Interactive Processing Route (Trial-Level)

Main files

gui/main_gui.py

gui/channel_plot_window.py

gui/analysis_window.py

Workflow

Load Excel trial metadata

Load iEEG H5 file

Select a patient trial

Visualize signals and channels

Launch:

DFA / MF-DFA analysis

Overlapping MF-DFA windows

FODN network analysis

This route is designed for manual inspection, parameter tuning, and exploratory analysis.

Feature Extraction (Batch Processing)

Main files

gui/feature_extraction_window.py

utils/feature_extractor.py

utils/file_utils.py

What it does

Iterates over all trials

Loads precomputed:

FODN outputs

DFA / MF-DFA outputs

Extracts:

Summary features

Top-K channel features

Produces a unified master_features.csv

This CSV is the central dataset used for modeling.

FODN Analysis (Network Dynamics)

Core logic

utils/fodn_code.py

utils/fodn_utils.py

GUI

gui/fodn_analysis_window.py

gui/fodn_post_window.py

gui/eigen_vector_heatmap.py

Outputs

Fractional orders (α)

Coupling matrices

Leading eigenvalues / eigenvectors

Sparsity and network metrics

Used to characterize network-level suppression dynamics during cognitive effort.

DFA / MF-DFA Analysis (Fractal Dynamics)

Core logic

utils/fast_mfdfa.py (NumPy / Numba accelerated)

GUI

gui/dfa_post_window.py

gui/mfdfa_analysis_window.py

gui/mfdfa_overlap_window.py

Capabilities

Single-window DFA / MF-DFA

Overlapping (sliding-window) MF-DFA

Per-channel and per-segment summaries

These features capture scale-dependent signal complexity.

Model Training & Evaluation

Main file

gui/model_training_window.py

Workflow

Load master_features.csv

Select label column

Configure preprocessing:

NaN handling

Standardization

Train logistic regression

Display:

Accuracy

ROC-AUC

Confusion matrix

Top contributing features

This route validates whether extracted features predict AD suppression under cognitive effort.
