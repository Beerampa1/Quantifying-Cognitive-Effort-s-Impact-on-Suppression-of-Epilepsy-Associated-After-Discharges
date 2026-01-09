# file: gui/batch_processing_window.py
"""
BatchProcessingWindow (prototype / temp)

This window implements a "batch processing route" that:
  1) Loads a root directory containing EEG files in per-trial subfolders
  2) Loads a trials Excel sheet describing trial IDs, EEG subfolders, start/end times, labels
  3) For each valid trial:
        - loads EEG data (currently placeholder loader)
        - extracts a time snippet
        - runs FODN on a set of channels (excluding artifacts like EKG/DC)
        - runs MF-DFA on one representative channel
  4) Saves a master features CSV for downstream modeling.

⚠️ NOTE: load_eeg_data() in this file is a placeholder that generates random data.
         Replace it with your real H5/EDF reader (or call utils.file_utils.load_h5_file).
"""

import sys
import os

import pandas as pd
import numpy as np
import scipy.linalg as LA
from scipy import stats

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QMessageBox, QProgressDialog, QLineEdit, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

# Your FODN runner (project-specific)
from utils.fodn_utils import run_fodn_analysis


# ────────────────────────────────────────────────────────────────
# MF-DFA (prototype implementation)
# ────────────────────────────────────────────────────────────────
def run_mfdfa(x, scales=None, q_vals=None, m=1):
    """
    Compute MF-DFA metrics for a 1D signal x.

    Parameters
    ----------
    x : array-like
        1D EEG snippet (single channel).
    scales : np.ndarray
        Window scales (in samples) used for fluctuation function computation.
    q_vals : np.ndarray
        q-orders for multifractal spectrum estimation.
    m : int
        Polynomial detrending order.

    Returns
    -------
    H : float
        Hurst exponent (slope of log2(F) vs log2(scale)).
    F : np.ndarray
        Fluctuation function values across scales.
    Hq : np.ndarray
        Generalized Hurst exponents across q.
    Fq : np.ndarray
        q-dependent fluctuation functions (len(q_vals) x len(scales)).

    Notes
    -----
    This is a straightforward (slow) reference implementation using np.polyfit.
    Your production app uses utils.fast_mfdfa.run_mfdfa_fast for speed.
    """
    if scales is None:
        scales = np.array([4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192])
    if q_vals is None:
        q_vals = np.array([-5, -3, -2, -1, 0, 1, 2, 3, 5])

    x = np.array(x, dtype=float)
    N = len(x)

    # MF-DFA integrates the demeaned signal to form the "profile"
    Y = np.cumsum(x - np.mean(x))

    F = np.zeros(len(scales))
    Fq = np.zeros((len(q_vals), len(scales)))

    for i, scale in enumerate(scales):
        scale = int(scale)

        # Number of full segments of length "scale"
        segments = int(np.floor(N / scale))
        RMS = np.zeros(segments)

        # Compute detrended RMS per segment
        for v in range(segments):
            idx_start = v * scale
            idx_stop = (v + 1) * scale
            seg_idx = np.arange(idx_start, idx_stop)

            # Fit polynomial trend of order m to the profile segment
            coeffs = np.polyfit(seg_idx, Y[idx_start:idx_stop], m)
            fit_vals = np.polyval(coeffs, seg_idx)

            # RMS of residuals
            RMS[v] = np.sqrt(np.mean((Y[idx_start:idx_stop] - fit_vals) ** 2))

        # Classic DFA fluctuation function for this scale
        F[i] = np.sqrt(np.mean(RMS ** 2))

        # MF-DFA: q-dependent fluctuation function
        for j, q in enumerate(q_vals):
            if q == 0:
                # q=0 uses log averaging
                Fq[j, i] = np.exp(0.5 * np.mean(np.log(RMS ** 2)))
            else:
                Fq[j, i] = (np.mean(RMS ** q)) ** (1.0 / q)

    # Estimate H as slope in log-log space
    log_scales = np.log2(scales)
    log_F = np.log2(F)
    slope, intercept, _, _, _ = stats.linregress(log_scales, log_F)
    H = slope

    # Estimate Hq similarly from log2(Fq(q,scale)) vs log2(scale)
    Hq = np.zeros(len(q_vals))
    for j, q in enumerate(q_vals):
        log_Fq = np.log2(Fq[j, :])
        slope_q, _, _, _, _ = stats.linregress(log_scales, log_Fq)
        Hq[j] = slope_q

    return H, F, Hq, Fq


# ────────────────────────────────────────────────────────────────
# EEG loader (placeholder)
# ────────────────────────────────────────────────────────────────
def load_eeg_data(eeg_file_path):
    """
    Placeholder EEG loader.

    ⚠️ This currently generates random EEG-like data for testing the batch pipeline.
    Replace with real file loading from:
      - H5: utils.file_utils.load_h5_file()
      - EDF: your EDF reader

    Returns
    -------
    data : np.ndarray
        EEG signals shape (channels, samples)
    time_array : np.ndarray
        Time vector shape (samples,)
    channel_names : list[str]
        Channel name strings
    """
    fs = 1000
    duration = 2 * 3600  # 2 hours (seconds)
    num_samples = duration * fs

    data = np.random.randn(64, num_samples)
    time_array = np.linspace(0, duration, num_samples)
    channel_names = [f"Channel {i}" for i in range(64)]
    return data, time_array, channel_names


# ────────────────────────────────────────────────────────────────
# Trial time parsing
# ────────────────────────────────────────────────────────────────
def parse_time_str(tstr):
    """
    Convert a time string "HH:MM:SS.xx" to total seconds.

    Example
    -------
    "0:41:37.00" -> 0*3600 + 41*60 + 37.00 = 2497 seconds
    """
    try:
        parts = tstr.split(":")
        if len(parts) != 3:
            raise ValueError("Time format should be HH:MM:SS.xx")

        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    except Exception as e:
        raise ValueError(f"Error parsing time string '{tstr}': {e}")


# ────────────────────────────────────────────────────────────────
# GUI: Batch processing route window
# ────────────────────────────────────────────────────────────────
class BatchProcessingWindow(QDialog):
    """
    GUI window to batch-process many trials and export a features CSV.

    Major steps:
      - user selects root directory containing EEG trial subfolders
      - user loads Excel sheet describing trials
      - app iterates rows, extracts snippet, computes FODN + MF-DFA features
      - saves one CSV containing one row per trial
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing Route")
        self.showMaximized()

        # Cache EEG loads so repeated references to the same EEG file are faster
        self.eeg_cache = {}

        self.initUI()

    def initUI(self):
        """
        Build the window layout:
          - root directory selector
          - trials Excel loader
          - parameter panels for FODN / MF-DFA
          - start/end time column selectors
          - process button + status label
        """
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # Instructions at top of window
        info_label = QLabel(
            "Batch Processing Route:\n"
            "1) Select the root directory\n"
            "2) Load the trials Excel file (with columns:\n"
            "   'Patient_Session_Trial', 'EEG_File_Sub_Folder', 'AD_Start', 'AD_End', 'Math_Score')\n"
            "3) Set analysis parameters and time snippet columns (optional)\n"
            "4) Click 'Process Trials' to generate features CSV."
        )
        info_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(info_label)

        # ---- Root directory selection ----
        root_layout = QHBoxLayout()
        self.btn_select_root = QPushButton("Select Root Directory")
        self.btn_select_root.setFixedHeight(40)
        self.btn_select_root.clicked.connect(self.select_root_directory)

        self.lbl_root = QLabel("No root directory selected.")
        root_layout.addWidget(self.btn_select_root)
        root_layout.addWidget(self.lbl_root)
        main_layout.addLayout(root_layout)

        # ---- Trials file loader ----
        trials_layout = QHBoxLayout()
        self.btn_load_trials = QPushButton("Load Trials Excel File")
        self.btn_load_trials.setFixedHeight(40)
        self.btn_load_trials.clicked.connect(self.load_trials_file)

        self.lbl_trials = QLabel("No trials file loaded.")
        trials_layout.addWidget(self.btn_load_trials)
        trials_layout.addWidget(self.lbl_trials)
        main_layout.addLayout(trials_layout)

        # ---- Analysis parameters ----
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QFormLayout(params_group)

        # FODN parameters (passed into run_fodn_analysis)
        self.edit_chunk = QLineEdit("1.0")
        self.edit_numfract = QLineEdit("50")
        self.edit_niter = QLineEdit("10")
        self.edit_lambda = QLineEdit("0.5")

        params_layout.addRow(QLabel("FODN Chunk Size (sec):"), self.edit_chunk)
        params_layout.addRow(QLabel("FODN numFract:"), self.edit_numfract)
        params_layout.addRow(QLabel("FODN niter:"), self.edit_niter)
        params_layout.addRow(QLabel("FODN lambdaUse:"), self.edit_lambda)

        # MF-DFA parameters (for the prototype run_mfdfa in this file)
        self.edit_scales = QLineEdit("4 8 16 32 64 128 256 1024 2048 4096 8192")
        self.edit_q_vals = QLineEdit("-5 -3 -2 -1 0 1 2 3 5")
        self.edit_m_order = QLineEdit("1")

        params_layout.addRow(QLabel("MF-DFA scales:"), self.edit_scales)
        params_layout.addRow(QLabel("MF-DFA q values:"), self.edit_q_vals)
        params_layout.addRow(QLabel("MF-DFA m order:"), self.edit_m_order)

        main_layout.addWidget(params_group)

        # ---- Time snippet column selectors ----
        time_group = QGroupBox("Time Snippet Columns")
        time_layout = QFormLayout(time_group)

        # Column names in the Excel file that contain snippet boundaries
        self.edit_start_col = QLineEdit("AD_Start")
        self.edit_end_col = QLineEdit("AD_End")

        time_layout.addRow(QLabel("Start Time Column:"), self.edit_start_col)
        time_layout.addRow(QLabel("End Time Column:"), self.edit_end_col)

        main_layout.addWidget(time_group)

        # ---- Execute ----
        self.btn_process = QPushButton("Process Trials & Save Features CSV")
        self.btn_process.setFixedHeight(40)
        self.btn_process.clicked.connect(self.process_trials)
        main_layout.addWidget(self.btn_process)

        # Status text at bottom
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("", 9))
        main_layout.addWidget(self.status_label)

        # Store paths
        self.trials_file = None
        self.root_directory = None
        self.features_csv = None

    # ────────────────────────────────────────────────────────────
    # UI actions
    # ────────────────────────────────────────────────────────────
    def select_root_directory(self):
        """Prompt user to choose the folder that contains EEG trial subfolders."""
        directory = QFileDialog.getExistingDirectory(self, "Select Root Directory", "")
        if directory:
            self.root_directory = directory
            self.lbl_root.setText(directory)

    def load_trials_file(self):
        """Prompt user to choose the Excel file that lists trial metadata and labels."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Trials Excel File", "", "Excel Files (*.xlsx *.xls)"
        )
        if file_path:
            self.trials_file = file_path
            self.lbl_trials.setText(file_path)

    # ────────────────────────────────────────────────────────────
    # Core processing logic
    # ────────────────────────────────────────────────────────────
    def process_trials(self):
        """
        Main batch loop:
          - load Excel sheet
          - validate required columns
          - iterate each trial row
          - load EEG, slice snippet, run FODN + MF-DFA
          - build one features row per trial
          - save CSV
        """
        if not self.trials_file or not self.root_directory:
            QMessageBox.warning(
                self, "Missing Inputs",
                "Please load the trials file and select the root directory."
            )
            return

        # Load Excel with header on second row (header=1)
        try:
            trials_df = pd.read_excel(self.trials_file, header=1)
            trials_df.columns = trials_df.columns.str.strip()
            print("Trials file columns:", list(trials_df.columns))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load trials file: {e}")
            return

        # Confirm the required columns are present
        needed_cols = [
            "Patient_Session_Trial",
            "EEG_File_Sub_Folder",
            self.edit_start_col.text().strip(),
            self.edit_end_col.text().strip(),
            "Math_Score",
        ]
        for col in needed_cols:
            if col not in trials_df.columns:
                QMessageBox.warning(
                    self, "Trials File Error",
                    f"Trials file must contain '{col}' column."
                )
                return

        # Progress dialog for a long-running batch job
        progress = QProgressDialog(
            "Processing trials...", "Cancel", 0, len(trials_df), self
        )
        progress.setWindowTitle("Processing Trials")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        features_list = []
        self.eeg_cache.clear()  # reset cache for this run

        for idx, row in trials_df.iterrows():
            progress.setValue(idx)
            if progress.wasCanceled():
                break

            # --- Parse row metadata ---
            patient_trial = str(row.get("Patient_Session_Trial", f"Trial_{idx+1}")).strip()
            subfolder = str(row["EEG_File_Sub_Folder"]).strip()

            start_time_str = str(row[self.edit_start_col.text().strip()]).strip()
            end_time_str = str(row[self.edit_end_col.text().strip()]).strip()
            math_score = str(row["Math_Score"]).strip()

            # Some rows represent "AD changes" or non-comparable cases → skip
            if "MC" in math_score.upper():
                print(f"Skipping trial {patient_trial}: Math_Score indicates MC (ad changes).")
                continue

            # Convert Excel time strings into seconds
            try:
                start_time = parse_time_str(start_time_str)
                end_time = parse_time_str(end_time_str)
            except Exception as e:
                print(f"Skipping trial {patient_trial}: Time conversion error: {e}")
                continue

            # Label definition: example rule (M1 = 1 else 0)
            label = 1 if math_score.startswith("M1") else 0

            # --- Find EEG file on disk ---
            candidate1 = os.path.join(self.root_directory, subfolder, "signal.h5")
            candidate2 = os.path.join(self.root_directory, subfolder, "SIGNAL.h5")
            if os.path.exists(candidate1):
                full_eeg_path = candidate1
            elif os.path.exists(candidate2):
                full_eeg_path = candidate2
            else:
                print(f"Skipping trial {patient_trial}: EEG file not found in subfolder '{subfolder}'.")
                continue

            # --- Load EEG (with cache) ---
            if full_eeg_path in self.eeg_cache:
                eeg_data, time_array, channel_names = self.eeg_cache[full_eeg_path]
            else:
                try:
                    eeg_data, time_array, channel_names = load_eeg_data(full_eeg_path)
                    self.eeg_cache[full_eeg_path] = (eeg_data, time_array, channel_names)
                except Exception as e:
                    print(f"Skipping trial {patient_trial}: Failed to load EEG data - {e}")
                    continue

            # --- Slice snippet based on start/end seconds ---
            fs = 1000  # assumed sample rate
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)

            # Clip end index to file length
            end_idx = min(end_idx, eeg_data.shape[1])
            snippet_data = eeg_data[:, start_idx:end_idx]

            # ─────────────────────────────
            # FODN feature extraction
            # ─────────────────────────────
            # Exclude known artifact / non-EEG channels
            unwanted = ["EKG1", "EKG2", "X1 DC1", "X1 DC2", "X1 DC3", "X1 DC4"]
            fodn_indices = [
                i for i, name in enumerate(channel_names) if name not in unwanted
            ]
            if not fodn_indices:
                print(f"Skipping trial {patient_trial}: No valid channels for FODN.")
                continue

            fodn_data = snippet_data[fodn_indices, :]

            try:
                fodn_out = run_fodn_analysis(
                    fodn_data,
                    chunk_size=float(self.edit_chunk.text().strip()),
                    numFract=int(self.edit_numfract.text().strip()),
                    niter=int(self.edit_niter.text().strip()),
                    lambdaUse=float(self.edit_lambda.text().strip())
                )
            except Exception as e:
                print(f"Skipping trial {patient_trial}: FODN analysis error - {e}")
                continue

            # Typical outputs expected from run_fodn_analysis
            alpha_vals = fodn_out["alpha"]
            coupling_mat = fodn_out["coupling_matrix"]

            mean_alpha = np.mean(alpha_vals)
            var_alpha = np.var(alpha_vals)

            # Leading eigenvalue magnitude as a coupling summary
            try:
                eigvals = LA.eigvals(coupling_mat)
                leading_eig = np.max(np.abs(eigvals))
            except Exception:
                leading_eig = np.nan

            # ─────────────────────────────
            # MF-DFA feature extraction
            # ─────────────────────────────
            # In this prototype, MF-DFA is run on *one* representative channel:
            # the first channel included in FODN.
            mfdfa_channel = fodn_indices[0]
            mfdfa_data = snippet_data[mfdfa_channel, :]

            try:
                scales = np.array([float(s) for s in self.edit_scales.text().strip().split()])
                q_vals = np.array([float(s) for s in self.edit_q_vals.text().strip().split()])
                m_order = int(self.edit_m_order.text().strip())
                H, F, Hq, Fq = run_mfdfa(mfdfa_data, scales=scales, q_vals=q_vals, m=m_order)
            except Exception as e:
                print(f"Skipping trial {patient_trial}: MF-DFA error - {e}")
                continue

            mfdfa_H = H
            mfdfa_Hq_mean = np.mean(Hq)

            # Consolidate features into one row
            feat_dict = {
                "Patient_Session_Trial": patient_trial,
                "EEG_File_Sub_Folder": subfolder,
                "Start_Time": start_time,
                "End_Time": end_time,
                "Label": label,

                # FODN features
                "MeanAlpha": mean_alpha,
                "VarAlpha": var_alpha,
                "LeadingEig": leading_eig,

                # MF-DFA features
                "MF_DFA_H": mfdfa_H,
                "MF_DFA_Hq_mean": mfdfa_Hq_mean,
            }
            features_list.append(feat_dict)
            print(f"Processed trial {patient_trial} in subfolder '{subfolder}'")

        # End progress bar
        progress.setValue(len(trials_df))

        if not features_list:
            QMessageBox.warning(self, "No Trials Processed", "No valid trials were processed.")
            return

        # Save out one CSV with one row per trial
        features_df = pd.DataFrame(features_list)
        save_path, _ = QFileDialog.getSaveFileName(
            self, "Save Features CSV", "features.csv", "CSV Files (*.csv)"
        )
        if save_path:
            features_df.to_csv(save_path, index=False)
            self.status_label.setText(
                f"Processed {len(features_list)} trials. Features saved to: {save_path}"
            )
        else:
            QMessageBox.warning(self, "Save Cancelled", "Features CSV was not saved.")


# ────────────────────────────────────────────────────────────────
# Standalone test entry point (useful during development)
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = BatchProcessingWindow()
    window.showMaximized()
    sys.exit(app.exec_())
