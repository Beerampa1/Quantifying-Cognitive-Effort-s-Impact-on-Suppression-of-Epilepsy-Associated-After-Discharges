# GUI + worker thread for extracting features from TrialResults folders.
# Features include:
#   - FODN features (alpha, eigenvalues, spectral radius, sparseness, top-K per channel)
#   - DFA/MF-DFA features (H, ΔHq, Hq per q, top-K per channel)
#   - DFA Overlap features (same as DFA but computed from DFA_Overlap outputs)
# Output is written to a CSV, with a separate log file.

import re
import numpy as np
import csv
from pathlib import Path
from typing import List, Dict, Tuple

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog, QFileDialog, QVBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QSpinBox, QCheckBox, QTabWidget, QWidget,
    QTextEdit, QProgressDialog, QMessageBox
)

# Reads labels from Excel (trial -> label)
from utils.label_strategies import ExcelMathScoreLabeler

# Extracts FODN/DFA features from result folders
from utils.feature_extractor import FeatureExtractor


# ---------------- overlap helpers (local, no change to utils) ----------------

def _find_innermost_overlap_dirs(trial_dir: Path, chunk_size: str) -> List[Path]:
    """
    Find overlap DFA output folders for a given trial.

    We look for folders like:
        DFA_Overlap_*_c{chunk}s*

    Goal:
      Return the *innermost* overlap directories that directly contain Channel_* subfolders.
      (Sometimes overlap outputs are nested.)

    Returns:
      List of Path objects (overlap roots).
    """
    pattern = f"DFA_Overlap_*_c{chunk_size}s*"

    # Search both shallow and deep
    candidates = list(trial_dir.glob(pattern)) + list(trial_dir.rglob(pattern))
    if not candidates:
        return []

    finals = []
    for d in candidates:
        try:
            # "final" overlap dir should contain Channel_* folders
            if any(c.is_dir() and c.name.startswith("Channel_") for c in d.iterdir()):
                finals.append(d)
        except Exception:
            pass

    if finals:
        # de-duplicate and stable sort
        uniq = sorted(set(finals), key=lambda p: str(p))
        return uniq

    # Fallback: if no Channel_* directly found, pick deepest matches
    candidates.sort(key=lambda p: len(p.parts), reverse=True)
    if not candidates:
        return []
    deepest = len(candidates[0].parts)
    return [p for p in candidates if len(p.parts) == deepest]


def _summarize_dfa_overlap(
    dfa_ov_root: Path,
    q_list: List[float]
) -> Tuple[Dict[str, float], List[str], Dict[str, List[float]]]:
    """
    Summarize DFA Overlap results in one overlap folder.

    Folder structure:
      DFA_Overlap.../
        Channel_*/ 
          chunk*/ 
            Hurst.csv     -> scalar H
            Hq_vs_q.csv   -> (q, Hq) (optional)

    Outputs:
      feats dict with keys like DFAov_H_mean/std, DFAov_DeltaHq_mean/std, DFAov_Hq{q}_mean/std
      ch_names list of channel folder names
      per_ch_H dict: channel -> list of H values (used for top-K)
    """
    # Collect global values across all channels/chunks
    H_list: List[float] = []                    # all H values
    dHq_list: List[float] = []                  # ΔHq per chunk (max-min)
    Hq_acc: Dict[float, List[float]] = {q: [] for q in q_list}  # Hq values per q

    # Channel folders present in overlap root
    ch_names = [
        p.name for p in dfa_ov_root.iterdir()
        if p.is_dir() and p.name.startswith("Channel_")
    ]

    # Per-channel H accumulation (for Top-K later)
    per_ch_H = {name: [] for name in ch_names}

    # Iterate channels
    for ch_dir in sorted(p for p in dfa_ov_root.iterdir() if p.is_dir() and p.name.startswith("Channel_")):
        # Iterate chunk subfolders
        for chunk in sorted(p for p in ch_dir.iterdir() if p.is_dir()):
            h_file = chunk / "Hurst.csv"
            if not h_file.exists():
                continue

            # ----- Load H (Hurst exponent) -----
            try:
                H_val = np.loadtxt(h_file, delimiter=",")
                H = float(np.ravel(H_val)[0]) if np.ndim(H_val) else float(H_val)
            except Exception:
                continue

            H_list.append(H)
            if ch_dir.name in per_ch_H:
                per_ch_H[ch_dir.name].append(H)

            # ----- Load Hq (optional) -----
            hq_file = chunk / "Hq_vs_q.csv"
            if hq_file.exists():
                try:
                    # Attempt to load including header (some files may not have header)
                    arr = np.loadtxt(hq_file, delimiter=",", ndmin=2)

                    # If badly shaped, try skipping header row
                    if arr.shape[1] < 2:
                        arr = np.loadtxt(hq_file, delimiter=",", skiprows=1, ndmin=2)

                    qs = arr[:, 0]
                    Hqs = arr[:, 1]

                    # ΔHq = max(Hq) - min(Hq) for this chunk
                    dHq_list.append(float(np.nanmax(Hqs) - np.nanmin(Hqs)))

                    # Store Hq value closest to each requested q
                    for q in q_list:
                        idx = int(np.nanargmin(np.abs(qs - q))) if qs.size else None
                        if idx is not None and 0 <= idx < Hqs.size:
                            Hq_acc[q].append(float(Hqs[idx]))

                except Exception:
                    # Ignore malformed/missing Hq data
                    pass

    feats: Dict[str, float] = {}

    # ----- Global H mean/std -----
    if H_list:
        feats["DFAov_H_mean"] = float(np.nanmean(H_list))  # computed mean H
        feats["DFAov_H_std"]  = float(np.nanstd(H_list))   # computed std H

    # ----- Global ΔHq mean/std -----
    if dHq_list:
        feats["DFAov_DeltaHq_mean"] = float(np.nanmean(dHq_list))  # computed mean ΔHq
        feats["DFAov_DeltaHq_std"]  = float(np.nanstd(dHq_list))   # computed std ΔHq

    # ----- Hq per q mean/std -----
    for q in q_list:
        vals = Hq_acc.get(q, [])
        if vals:
            feats[f"DFAov_Hq{q}_mean"] = float(np.nanmean(vals))   # mean Hq at q
            feats[f"DFAov_Hq{q}_std"]  = float(np.nanstd(vals))    # std Hq at q

    return feats, ch_names, per_ch_H


def _topk_channel_features_from_means(
    means: np.ndarray,
    stds: np.ndarray,
    ch_names: List[str],
    top_pct: int
) -> Dict[str, float]:
    """
    Select top-K channels based on mean H (largest mean values).

    Returns:
      dict with keys:
        DFAov_ch1_H_mean, DFAov_ch1_H_std, ...
    """
    out: Dict[str, float] = {}
    if means.size == 0:
        return out

    n_ch = means.size
    K = max(1, int(n_ch * top_pct / 100))  # top percentage -> count

    # Indices of top K mean values
    idx = np.argsort(means)[-K:]

    # Rank results (rank 1 = top channel)
    rank = 1
    for j in idx:
        out[f"DFAov_ch{rank}_H_mean"] = float(means[j])
        out[f"DFAov_ch{rank}_H_std"]  = float(stds[j])
        rank += 1

    return out


# -----------------------------------------------------------------------------


class ExtractWorker(QThread):
    """
    Background worker thread:
    - loops over all trial folders
    - extracts features (FODN, DFA, DFA overlap)
    - writes a CSV and log
    """
    progress = pyqtSignal(int)     # emits percent progress
    log_line = pyqtSignal(str)     # emits text log lines to UI
    finished = pyqtSignal(str, str)  # emits (csv_path, log_path)

    def __init__(self, *,
                 root_dir: Path,
                 excel_path: Path,
                 save_csv: Path,
                 chunk_size: str,
                 top_pct: int,
                 q_list: List[float],
                 opts_fodn: dict,
                 opts_dfa: dict,
                 skip_unlabeled: bool):
        super().__init__()

        # Store configuration
        self.root_dir       = root_dir
        self.excel_path     = excel_path
        self.save_csv       = save_csv
        self.chunk_size     = chunk_size
        self.top_pct        = top_pct
        self.q_list         = q_list
        self.opts_fodn      = opts_fodn
        self.opts_dfa       = opts_dfa
        self.skip_unlabeled = skip_unlabeled

    def run(self):
        """Main worker execution (runs in background thread)"""

        # 1) Load labels from Excel
        try:
            labeler = ExcelMathScoreLabeler(self.excel_path)
        except Exception as e:
            self.log_line.emit(f"ERROR reading Excel: {e}")
            self.finished.emit("", "")
            return

        # 2) Find trial folders
        trials = sorted(self.root_dir.glob("P*_S*_T*"))
        total = len(trials)
        if total == 0:
            self.log_line.emit("No trial folders found.")
            self.finished.emit("", "")
            return

        # 3) Build CSV header dynamically based on selected options
        header = ["Trial"]

        # ---------- FODN summary fields ----------
        if self.opts_fodn["alpha_mean_std"]:
            header += ["FODN_alpha_mean", "FODN_alpha_std"]
        if self.opts_fodn["alpha_top"]:
            header += ["FODN_alpha_top_mean", "FODN_alpha_top_std"]
        if self.opts_fodn["lead_eig"]:
            header += ["FODN_lead_eig_mean", "FODN_lead_eig_std"]
        if self.opts_fodn["spectral_radius"]:
            header += ["FODN_spectral_radius_mean", "FODN_spectral_radius_std"]
        if self.opts_fodn["sparseness"]:
            header += ["FODN_sparseness"]

        # ---------- FODN per-channel top-K fields ----------
        if self.opts_fodn["pc_mean"] or self.opts_fodn["pc_std"]:
            # Probe one trial to infer number of channels
            n_ch = 0
            for td in trials:
                try:
                    fodn = list(td.glob(f"FODN*_c{self.chunk_size}s"))
                    seg = sorted(p for p in fodn[0].iterdir() if p.is_dir())[0]
                    chunk = sorted(p for p in seg.iterdir() if p.is_dir())[0]
                    C = np.loadtxt(chunk / "Coupling_Data.csv", delimiter=",")
                    n_ch = C.shape[0]
                    break
                except Exception:
                    continue

            if n_ch != 0:
                K = max(1, int(n_ch * self.top_pct / 100))
                for r in range(1, K + 1):
                    if self.opts_fodn["pc_mean"]:
                        header.append(f"FODN_ch{r}_alpha_mean")
                    if self.opts_fodn["pc_std"]:
                        header.append(f"FODN_ch{r}_alpha_std")

        # ---------- DFA (non-overlap) summary fields ----------
        if self.opts_dfa["H_mean_std"]:
            header += ["DFA_H_mean", "DFA_H_std"]
        if self.opts_dfa["DeltaHq"]:
            header += ["DFA_DeltaHq_mean", "DFA_DeltaHq_std"]
        if self.opts_dfa["Hq_per_q"]:
            for q in self.q_list:
                header += [f"DFA_Hq{q}_mean", f"DFA_Hq{q}_std"]

        # ---------- DFA per-channel top-K fields ----------
        if self.opts_dfa["pc_mean"] or self.opts_dfa["pc_std"]:
            n_ch = 0
            for td in trials:
                dfa = list(td.glob(f"DFA*_c{self.chunk_size}s"))
                if dfa:
                    seg = sorted(p for p in dfa[0].iterdir() if p.is_dir())[0]
                    chs = [p for p in seg.iterdir() if p.is_dir()]
                    n_ch = len(chs)
                    break
            if n_ch != 0:
                K = max(1, int(n_ch * self.top_pct / 100))
                for r in range(1, K + 1):
                    if self.opts_dfa["pc_mean"]:
                        header.append(f"DFA_ch{r}_H_mean")
                    if self.opts_dfa["pc_std"]:
                        header.append(f"DFA_ch{r}_H_std")

        # ---------- DFA Overlap summary fields (prefix DFAov_) ----------
        if self.opts_dfa["H_mean_std"]:
            header += ["DFAov_H_mean", "DFAov_H_std"]
        if self.opts_dfa["DeltaHq"]:
            header += ["DFAov_DeltaHq_mean", "DFAov_DeltaHq_std"]
        if self.opts_dfa["Hq_per_q"]:
            for q in self.q_list:
                header += [f"DFAov_Hq{q}_mean", f"DFAov_Hq{q}_std"]

        # ---------- DFA Overlap per-channel top-K fields ----------
        if self.opts_dfa["pc_mean"] or self.opts_dfa["pc_std"]:
            # Probe one trial to infer overlap channel count
            n_ch_ov = 0
            for td in trials:
                ov_dirs = _find_innermost_overlap_dirs(td, self.chunk_size)
                if ov_dirs:
                    chs = [
                        p for p in ov_dirs[0].iterdir()
                        if p.is_dir() and p.name.startswith("Channel_")
                    ]
                    n_ch_ov = len(chs)
                    break
            if n_ch_ov != 0:
                K = max(1, int(n_ch_ov * self.top_pct / 100))
                for r in range(1, K + 1):
                    if self.opts_dfa["pc_mean"]:
                        header.append(f"DFAov_ch{r}_H_mean")
                    if self.opts_dfa["pc_std"]:
                        header.append(f"DFAov_ch{r}_H_std")

        # Add label column at the end
        header += ["Label"]

        # 4) Open output CSV + log file and write header
        csv_path = self.save_csv
        log_path = self.save_csv.with_name(self.save_csv.stem + "_log.txt")

        csv_f = open(csv_path, "w", newline="")
        log_f = open(log_path, "w", encoding="utf-8")

        try:
            writer = csv.DictWriter(csv_f, fieldnames=header)
            writer.writeheader()

            # 5) Process each trial
            for i, td in enumerate(trials, 1):
                tid = td.name
                try:
                    # Get label for this trial
                    lbl = ExcelMathScoreLabeler.get.__wrapped__(ExcelMathScoreLabeler, labeler, tid) \
                          if hasattr(ExcelMathScoreLabeler.get, "__wrapped__") else labeler.get(tid)

                    # Optionally skip unlabeled trials
                    if lbl is None and self.skip_unlabeled:
                        log_f.write(f"SKIP unlabeled: {tid}\n")
                        self.progress.emit(int(i / total * 100))
                        continue

                    feats = {"Trial": tid}

                    # ----- FODN features -----
                    fodn_dirs = list(td.glob(f"FODN*_c{self.chunk_size}s"))
                    if fodn_dirs:
                        feats.update(
                            FeatureExtractor.extract_fodn_features(
                                fodn_dirs[0], self.top_pct, self.opts_fodn
                            )
                        )
                    else:
                        log_f.write(f"WARN no FODN folder for {tid}\n")

                    # ----- DFA (non-overlap) features -----
                    dfa_dirs = list(td.glob(f"DFA*_c{self.chunk_size}s"))
                    if dfa_dirs:
                        feats.update(
                            FeatureExtractor.extract_dfa_features(
                                dfa_dirs[0], self.top_pct, self.q_list, self.opts_dfa
                            )
                        )
                    else:
                        log_f.write(f"WARN no DFA folder for {tid}\n")

                    # ----- DFA Overlap features (auto-detect overlap dirs) -----
                    ov_dirs = _find_innermost_overlap_dirs(td, self.chunk_size)
                    if ov_dirs:
                        # Aggregate across multiple overlap dirs (if present)
                        agg_feats: Dict[str, List[float]] = {}

                        # Pool per-channel stats across overlap dirs
                        pooled_means_map: Dict[str, List[float]] = {}
                        pooled_stds_map: Dict[str, List[float]] = {}

                        for od in ov_dirs:
                            try:
                                f_one, ch_names, per_ch_H = _summarize_dfa_overlap(od, self.q_list)

                                # Collect summary values (later averaged)
                                for k, v in f_one.items():
                                    agg_feats.setdefault(k, []).append(v)

                                # Per-channel H arrays -> mean/std per channel for this overlap dir
                                for ch, arr in per_ch_H.items():
                                    arr_np = np.array(arr, dtype=float) if arr else np.array([])
                                    m = float(np.nanmean(arr_np)) if arr_np.size else np.nan
                                    s = float(np.nanstd(arr_np))  if arr_np.size else np.nan
                                    pooled_means_map.setdefault(ch, []).append(m)
                                    pooled_stds_map.setdefault(ch, []).append(s)

                            except Exception as e:
                                log_f.write(f"WARN overlap parse failed in {od}: {e}\n")
                                continue

                        # Finalize summary aggregation (average across overlap dirs)
                        for k, vals in agg_feats.items():
                            if vals:
                                feats[k] = float(np.nanmean(vals))

                        # Top-K per-channel for overlap (using mean of means)
                        if (self.opts_dfa["pc_mean"] or self.opts_dfa["pc_std"]) and pooled_means_map:
                            ch_sorted = sorted(pooled_means_map.keys())
                            means = np.array([np.nanmean(pooled_means_map[ch]) for ch in ch_sorted], dtype=float)
                            stds  = np.array([np.nanmean(pooled_stds_map[ch])  for ch in ch_sorted], dtype=float)

                            topk_feats = _topk_channel_features_from_means(
                                means, stds, ch_sorted, self.top_pct
                            )
                            feats.update(topk_feats)

                    else:
                        log_f.write(f"WARN no DFA_Overlap folder for {tid}\n")

                    # Save label into output row
                    feats["Label"] = lbl if lbl is not None else ""

                    # Make sure every header column exists
                    for k in header:
                        feats.setdefault(k, "")

                    # Write row to CSV
                    writer.writerow(feats)
                    log_f.write(f"OK {tid}\n")

                except Exception as e:
                    # Write full traceback to log if trial fails
                    import traceback, textwrap
                    tb = textwrap.indent(traceback.format_exc(), "  ")
                    log_f.write(f"ERROR {tid}: {e}\n{tb}\n")

                finally:
                    # Update progress bar
                    self.progress.emit(int(i / total * 100))

        finally:
            csv_f.close()
            log_f.close()

        # 6) Signal completion back to the GUI
        self.finished.emit(str(csv_path), str(log_path))


class FeatureExtractionWindow(QDialog):
    """
    GUI window that lets the user:
      - pick TrialResults root + Excel label file
      - choose chunk size and top-k percentage
      - select which features to extract (FODN/DFA)
      - run extraction in a background thread
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Feature Extraction")
        self.resize(700, 580)  # GUI size
        self.worker = None
        self._build_ui()

    def _build_ui(self):
        """Build all form fields, tabs, and output log area"""
        main = QVBoxLayout(self)
        grid = QGridLayout()
        main.addLayout(grid)

        # ----- TrialResults root -----
        grid.addWidget(QLabel("TrialResults root:"), 0, 0)
        self.le_root = QLineEdit()
        self.le_root.setReadOnly(True)
        btn_r = QPushButton("Browse…")
        btn_r.clicked.connect(self._choose_root)
        grid.addWidget(self.le_root, 0, 1)
        grid.addWidget(btn_r, 0, 2)

        # ----- Excel label file -----
        grid.addWidget(QLabel("Label Excel file:"), 1, 0)
        self.le_excel = QLineEdit()
        self.le_excel.setReadOnly(True)
        btn_e = QPushButton("Browse…")
        btn_e.clicked.connect(self._choose_excel)
        grid.addWidget(self.le_excel, 1, 1)
        grid.addWidget(btn_e, 1, 2)

        # ----- Chunk-size and Top-k percent -----
        grid.addWidget(QLabel("Chunk-size (s):"), 2, 0)
        self.cb_chunk = QComboBox()
        self.cb_chunk.addItems(["0.25", "0.5", "1.0"])
        grid.addWidget(self.cb_chunk, 2, 1)

        grid.addWidget(QLabel("Top-k %:"), 2, 2)
        self.sb_topk = QSpinBox()
        self.sb_topk.setRange(1, 99)
        self.sb_topk.setValue(10)
        grid.addWidget(self.sb_topk, 2, 3)

        # ----- Skip unlabeled trials -----
        self.cb_skip = QCheckBox("Skip unlabeled trials")
        self.cb_skip.setChecked(True)
        main.addWidget(self.cb_skip)

        # ----- Feature selection tabs -----
        tabs = QTabWidget()
        main.addWidget(tabs)

        # ===== FODN tab =====
        tab_f = QWidget()
        vf = QVBoxLayout(tab_f)

        self.cb_f_alpha = QCheckBox("α mean/std")
        self.cb_f_alpha.setChecked(True)

        self.cb_f_alpha_top = QCheckBox("α top-k mean/std")
        self.cb_f_alpha_top.setChecked(True)

        self.cb_f_lead = QCheckBox("Lead eigen-value")
        self.cb_f_lead.setChecked(True)

        self.cb_f_radius = QCheckBox("Spectral radius")
        self.cb_f_sparse = QCheckBox("Sparseness |A|>0.01")

        for cb in (self.cb_f_alpha, self.cb_f_alpha_top, self.cb_f_lead, self.cb_f_radius, self.cb_f_sparse):
            vf.addWidget(cb)

        # Per-channel FODN options
        vm_index = vf.indexOf(self.cb_f_sparse)
        self.cb_f_pc_mean = QCheckBox("Per-channel α mean (top-K)")
        self.cb_f_pc_mean.setChecked(True)
        self.cb_f_pc_std = QCheckBox("Per-channel α std  (top-K)")
        self.cb_f_pc_std.setChecked(True)

        vf.insertWidget(vm_index + 1, self.cb_f_pc_mean)
        vf.insertWidget(vm_index + 2, self.cb_f_pc_std)
        vf.addStretch(1)

        tabs.addTab(tab_f, "FODN")

        # ===== DFA tab =====
        tab_d = QWidget()
        vd = QVBoxLayout(tab_d)

        self.cb_d_H = QCheckBox("H mean/std")
        self.cb_d_H.setChecked(True)

        self.cb_d_dHq = QCheckBox("ΔHq mean/std")
        self.cb_d_dHq.setChecked(True)

        self.cb_d_Hq = QCheckBox("Hq per q")
        self.cb_d_Hq.setChecked(True)

        vd.addWidget(self.cb_d_H)
        vd.addWidget(self.cb_d_dHq)
        vd.addWidget(self.cb_d_Hq)

        # q-values input
        self.le_q = QLineEdit("2,3,4")
        vd.addWidget(QLabel("q-values:"))
        vd.addWidget(self.le_q)

        # Per-channel DFA options
        dm_index = vd.indexOf(self.le_q)
        self.cb_d_pc_mean = QCheckBox("Per-channel H mean (top-K)")
        self.cb_d_pc_mean.setChecked(True)

        self.cb_d_pc_std = QCheckBox("Per-channel H std  (top-K)")
        self.cb_d_pc_std.setChecked(True)

        vd.insertWidget(dm_index + 2, self.cb_d_pc_mean)
        vd.insertWidget(dm_index + 3, self.cb_d_pc_std)
        vd.addStretch(1)

        tabs.addTab(tab_d, "DFA/MF-DFA")

        # ----- Run + log output -----
        self.btn_ext = QPushButton("Extract features")
        self.btn_ext.clicked.connect(self._start)
        main.addWidget(self.btn_ext)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        main.addWidget(self.txt_log, stretch=1)

        # Enable button only when root+excel are selected
        self.le_root.textChanged.connect(self._update_ok)
        self.le_excel.textChanged.connect(self._update_ok)
        self._update_ok()

    def _choose_root(self):
        """Select TrialResults root folder"""
        d = QFileDialog.getExistingDirectory(self, "Select TrialResults root")
        if d:
            self.le_root.setText(d)

    def _choose_excel(self):
        """Select Excel file containing trial labels"""
        f, _ = QFileDialog.getOpenFileName(self, "Select Excel file", filter="Excel (*.xlsx *.xls)")
        if f:
            self.le_excel.setText(f)

    def _update_ok(self):
        """Enable Extract button only if required inputs exist"""
        ok = bool(self.le_root.text() and self.le_excel.text())
        self.btn_ext.setEnabled(ok)

    def _parse_q(self, text: str) -> List[float]:
        """Parse q-value list from UI text field"""
        parts = re.split(r"[,\s]+", text.strip())
        out = []
        for p in parts:
            try:
                out.append(float(p))
            except:
                pass
        return out or [2.0, 3.0, 4.0]

    def _start(self):
        """Start extraction worker"""
        root  = Path(self.le_root.text())
        excel = Path(self.le_excel.text())
        chunk = self.cb_chunk.currentText()
        topk  = self.sb_topk.value()
        qlist = self._parse_q(self.le_q.text())

        # Read feature options from checkboxes
        opts_f = {
            "alpha_mean_std":  self.cb_f_alpha.isChecked(),
            "alpha_top":       self.cb_f_alpha_top.isChecked(),
            "lead_eig":        self.cb_f_lead.isChecked(),
            "spectral_radius": self.cb_f_radius.isChecked(),
            "sparseness":      self.cb_f_sparse.isChecked(),
            "pc_mean":         self.cb_f_pc_mean.isChecked(),
            "pc_std":          self.cb_f_pc_std.isChecked()
        }
        opts_d = {
            "H_mean_std": self.cb_d_H.isChecked(),
            "DeltaHq":    self.cb_d_dHq.isChecked(),
            "Hq_per_q":   self.cb_d_Hq.isChecked(),
            "pc_mean":    self.cb_d_pc_mean.isChecked(),
            "pc_std":     self.cb_d_pc_std.isChecked()
        }

        # Choose output CSV file location
        savef, _ = QFileDialog.getSaveFileName(self, "Save CSV as…", filter="CSV (*.csv)")
        if not savef:
            return
        savep = Path(savef)

        # Progress dialog (0-100)
        self.progress = QProgressDialog("Extracting…", "Cancel", 0, 100, self)
        self.progress.setWindowModality(Qt.WindowModal)
        self.progress.show()

        # Start background worker
        self.worker = ExtractWorker(
            root_dir=root,
            excel_path=excel,
            save_csv=savep,
            chunk_size=chunk,
            top_pct=topk,
            q_list=qlist,
            opts_fodn=opts_f,
            opts_dfa=opts_d,
            skip_unlabeled=self.cb_skip.isChecked()
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log_line.connect(self.txt_log.append)
        self.worker.finished.connect(self._done)
        self.worker.start()

    def _done(self, csv_path: str, log_path: str):
        """Handle completion of extraction"""
        self.progress.close()
        if not csv_path:
            QMessageBox.critical(self, "Error", "Extraction failed. See log.")
        else:
            QMessageBox.information(
                self, "Done",
                f"Features: {csv_path}\nLog:      {log_path}"
            )


if __name__ == "__main__":
    # Standalone run for testing this window
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    w = FeatureExtractionWindow()
    w.show()
    sys.exit(app.exec_())
