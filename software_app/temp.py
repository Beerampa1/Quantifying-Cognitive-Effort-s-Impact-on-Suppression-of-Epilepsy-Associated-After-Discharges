# file: gui/batch_processing_window.py

import sys
import os
import pandas as pd
import numpy as np
import scipy.linalg as LA
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QMessageBox, QProgressDialog, QLineEdit, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from utils.fodn_utils import run_fodn_analysis
from scipy import stats

def run_mfdfa(x, scales=None, q_vals=None, m=1):
    if scales is None:
        scales = np.array([4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192])
    if q_vals is None:
        q_vals = np.array([-5, -3, -2, -1, 0, 1, 2, 3, 5])
    x = np.array(x, dtype=float)
    N = len(x)
    Y = np.cumsum(x - np.mean(x))
    F = np.zeros(len(scales))
    Fq = np.zeros((len(q_vals), len(scales)))
    
    for i, scale in enumerate(scales):
        scale = int(scale)
        segments = int(np.floor(N / scale))
        RMS = np.zeros(segments)
        for v in range(segments):
            idx_start = v * scale
            idx_stop = (v + 1) * scale
            seg_idx = np.arange(idx_start, idx_stop)
            coeffs = np.polyfit(seg_idx, Y[idx_start:idx_stop], m)
            fit_vals = np.polyval(coeffs, seg_idx)
            RMS[v] = np.sqrt(np.mean((Y[idx_start:idx_stop] - fit_vals) ** 2))
        F[i] = np.sqrt(np.mean(RMS ** 2))
        for j, q in enumerate(q_vals):
            if q == 0:
                Fq[j, i] = np.exp(0.5 * np.mean(np.log(RMS ** 2)))
            else:
                Fq[j, i] = (np.mean(RMS ** q)) ** (1.0 / q)
    
    log_scales = np.log2(scales)
    log_F = np.log2(F)
    slope, intercept, _, _, _ = stats.linregress(log_scales, log_F)
    H = slope
    
    Hq = np.zeros(len(q_vals))
    for j, q in enumerate(q_vals):
        log_Fq = np.log2(Fq[j, :])
        slope_q, _, _, _, _ = stats.linregress(log_scales, log_Fq)
        Hq[j] = slope_q
    
    return H, F, Hq, Fq

def load_eeg_data(eeg_file_path):
    # Replace with your actual loader for .h5 or .edf files.
    fs = 1000
    duration = 2 * 3600  # 2 hours in seconds
    num_samples = duration * fs
    data = np.random.randn(64, num_samples)
    time_array = np.linspace(0, duration, num_samples)
    channel_names = [f"Channel {i}" for i in range(64)]
    return data, time_array, channel_names

def parse_time_str(tstr):
    """
    Convert a time string "HH:MM:SS.xx" to total seconds.
    Example: "0:41:37.00" -> 0*3600 + 41*60 + 37.00 = 2497 seconds.
    """
    try:
        parts = tstr.split(':')
        if len(parts) != 3:
            raise ValueError("Time format should be HH:MM:SS.xx")
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    except Exception as e:
        raise ValueError(f"Error parsing time string '{tstr}': {e}")

class BatchProcessingWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing Route")
        self.showMaximized()
        self.eeg_cache = {}  # Cache to store loaded EEG files
        self.initUI()
    
    def initUI(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)
        main_layout.setContentsMargins(5, 5, 5, 5)
        
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
        
        # Root directory selection.
        root_layout = QHBoxLayout()
        self.btn_select_root = QPushButton("Select Root Directory")
        self.btn_select_root.setFixedHeight(40)
        self.btn_select_root.clicked.connect(self.select_root_directory)
        self.lbl_root = QLabel("No root directory selected.")
        root_layout.addWidget(self.btn_select_root)
        root_layout.addWidget(self.lbl_root)
        main_layout.addLayout(root_layout)
        
        # Trials file loading.
        trials_layout = QHBoxLayout()
        self.btn_load_trials = QPushButton("Load Trials Excel File")
        self.btn_load_trials.setFixedHeight(40)
        self.btn_load_trials.clicked.connect(self.load_trials_file)
        self.lbl_trials = QLabel("No trials file loaded.")
        trials_layout.addWidget(self.btn_load_trials)
        trials_layout.addWidget(self.lbl_trials)
        main_layout.addLayout(trials_layout)
        
        # Analysis Parameters Panel.
        params_group = QGroupBox("Analysis Parameters")
        params_layout = QFormLayout(params_group)
        self.edit_chunk = QLineEdit("1.0")
        self.edit_numfract = QLineEdit("50")
        self.edit_niter = QLineEdit("10")
        self.edit_lambda = QLineEdit("0.5")
        params_layout.addRow(QLabel("FODN Chunk Size (sec):"), self.edit_chunk)
        params_layout.addRow(QLabel("FODN numFract:"), self.edit_numfract)
        params_layout.addRow(QLabel("FODN niter:"), self.edit_niter)
        params_layout.addRow(QLabel("FODN lambdaUse:"), self.edit_lambda)
        self.edit_scales = QLineEdit("4 8 16 32 64 128 256 1024 2048 4096 8192")
        self.edit_q_vals = QLineEdit("-5 -3 -2 -1 0 1 2 3 5")
        self.edit_m_order = QLineEdit("1")
        params_layout.addRow(QLabel("MF-DFA scales:"), self.edit_scales)
        params_layout.addRow(QLabel("MF-DFA q values:"), self.edit_q_vals)
        params_layout.addRow(QLabel("MF-DFA m order:"), self.edit_m_order)
        main_layout.addWidget(params_group)
        
        # Time Snippet Columns Panel.
        time_group = QGroupBox("Time Snippet Columns")
        time_layout = QFormLayout(time_group)
        self.edit_start_col = QLineEdit("AD_Start")
        self.edit_end_col = QLineEdit("AD_End")
        time_layout.addRow(QLabel("Start Time Column:"), self.edit_start_col)
        time_layout.addRow(QLabel("End Time Column:"), self.edit_end_col)
        main_layout.addWidget(time_group)
        
        # Process Trials Button.
        self.btn_process = QPushButton("Process Trials & Save Features CSV")
        self.btn_process.setFixedHeight(40)
        self.btn_process.clicked.connect(self.process_trials)
        main_layout.addWidget(self.btn_process)
        
        # Status label.
        self.status_label = QLabel("")
        self.status_label.setFont(QFont("", 9))
        main_layout.addWidget(self.status_label)
        
        # Variables to hold file paths.
        self.trials_file = None
        self.root_directory = None
        self.features_csv = None

    def select_root_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Root Directory", "")
        if directory:
            self.root_directory = directory
            self.lbl_root.setText(directory)
    
    def load_trials_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Trials Excel File", "", "Excel Files (*.xlsx *.xls)"
        )
        if file_path:
            self.trials_file = file_path
            self.lbl_trials.setText(file_path)
    
    def process_trials(self):
        if not self.trials_file or not self.root_directory:
            QMessageBox.warning(self, "Missing Inputs", "Please load the trials file and select the root directory.")
            return
        
        # Load the trials file with header from the second row.
        try:
            trials_df = pd.read_excel(self.trials_file, header=1)
            trials_df.columns = trials_df.columns.str.strip()
            print("Trials file columns:", list(trials_df.columns))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load trials file: {e}")
            return
        
        # Expected columns.
        needed_cols = ["Patient_Session_Trial", "EEG_File_Sub_Folder", 
                       self.edit_start_col.text().strip(), self.edit_end_col.text().strip(),
                       "Math_Score"]
        for col in needed_cols:
            if col not in trials_df.columns:
                QMessageBox.warning(self, "Trials File Error", f"Trials file must contain '{col}' column.")
                return
        
        # Set up progress dialog.
        progress = QProgressDialog("Processing trials...", "Cancel", 0, len(trials_df), self)
        progress.setWindowTitle("Processing Trials")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        features_list = []
        # Clear EEG cache.
        self.eeg_cache.clear()
        
        for idx, row in trials_df.iterrows():
            progress.setValue(idx)
            if progress.wasCanceled():
                break
            
            patient_trial = str(row.get("Patient_Session_Trial", f"Trial_{idx+1}")).strip()
            subfolder = str(row["EEG_File_Sub_Folder"]).strip()
            start_time_str = str(row[self.edit_start_col.text().strip()]).strip()
            end_time_str = str(row[self.edit_end_col.text().strip()]).strip()
            math_score = str(row["Math_Score"]).strip()
            
            # If math score indicates "MC", skip this trial.
            if "MC" in math_score.upper():
                print(f"Skipping trial {patient_trial}: Math_Score indicates MC (ad changes).")
                continue
            
            try:
                start_time = parse_time_str(start_time_str)
                end_time = parse_time_str(end_time_str)
            except Exception as e:
                print(f"Skipping trial {patient_trial}: Time conversion error: {e}")
                continue
            
            label = 1 if math_score.startswith("M1") else 0
            
            # Construct full EEG file path: root_directory + subfolder + ("signal.h5" or "SIGNAL.h5")
            candidate1 = os.path.join(self.root_directory, subfolder, "signal.h5")
            candidate2 = os.path.join(self.root_directory, subfolder, "SIGNAL.h5")
            if os.path.exists(candidate1):
                full_eeg_path = candidate1
            elif os.path.exists(candidate2):
                full_eeg_path = candidate2
            else:
                print(f"Skipping trial {patient_trial}: EEG file not found in subfolder '{subfolder}'.")
                continue
            
            # Check cache: if file already loaded, reuse it.
            if full_eeg_path in self.eeg_cache:
                eeg_data, time_array, channel_names = self.eeg_cache[full_eeg_path]
            else:
                try:
                    eeg_data, time_array, channel_names = load_eeg_data(full_eeg_path)
                    self.eeg_cache[full_eeg_path] = (eeg_data, time_array, channel_names)
                except Exception as e:
                    print(f"Skipping trial {patient_trial}: Failed to load EEG data - {e}")
                    continue
            
            fs = 1000
            start_idx = int(start_time * fs)
            end_idx = int(end_time * fs)
            if end_idx > eeg_data.shape[1]:
                end_idx = eeg_data.shape[1]
            snippet_data = eeg_data[:, start_idx:end_idx]
            
            # FODN processing: exclude unwanted channels.
            unwanted = ["EKG1", "EKG2", "X1 DC1", "X1 DC2", "X1 DC3", "X1 DC4"]
            fodn_indices = [i for i, name in enumerate(channel_names) if name not in unwanted]
            if not fodn_indices:
                print(f"Skipping trial {patient_trial}: No valid channels for FODN.")
                continue
            fodn_data = snippet_data[fodn_indices, :]
            try:
                fodn_out = run_fodn_analysis(fodn_data, chunk_size=float(self.edit_chunk.text().strip()),
                                             numFract=int(self.edit_numfract.text().strip()),
                                             niter=int(self.edit_niter.text().strip()),
                                             lambdaUse=float(self.edit_lambda.text().strip()))
            except Exception as e:
                print(f"Skipping trial {patient_trial}: FODN analysis error - {e}")
                continue
            alpha_vals = fodn_out["alpha"]
            coupling_mat = fodn_out["coupling_matrix"]
            mean_alpha = np.mean(alpha_vals)
            var_alpha = np.var(alpha_vals)
            try:
                eigvals = LA.eigvals(coupling_mat)
                leading_eig = np.max(np.abs(eigvals))
            except Exception:
                leading_eig = np.nan
            
            # MF-DFA processing: use the first valid channel.
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
            
            feat_dict = {
                "Patient_Session_Trial": patient_trial,
                "EEG_File_Sub_Folder": subfolder,
                "Start_Time": start_time,
                "End_Time": end_time,
                "Label": label,
                "MeanAlpha": mean_alpha,
                "VarAlpha": var_alpha,
                "LeadingEig": leading_eig,
                "MF_DFA_H": mfdfa_H,
                "MF_DFA_Hq_mean": mfdfa_Hq_mean
            }
            features_list.append(feat_dict)
            print(f"Processed trial {patient_trial} in subfolder '{subfolder}'")
        
        progress.setValue(len(trials_df))
        if not features_list:
            QMessageBox.warning(self, "No Trials Processed", "No valid trials were processed.")
            return
        
        features_df = pd.DataFrame(features_list)
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Features CSV", "features.csv", "CSV Files (*.csv)")
        if save_path:
            features_df.to_csv(save_path, index=False)
            self.status_label.setText(f"Processed {len(features_list)} trials. Features saved to: {save_path}")
        else:
            QMessageBox.warning(self, "Save Cancelled", "Features CSV was not saved.")

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    window = BatchProcessingWindow()
    window.showMaximized()
    sys.exit(app.exec_())
