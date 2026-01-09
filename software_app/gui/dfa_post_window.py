# file: gui/dfa_post_window.py
# Post-processing GUI for DFA/MFDFA results.
# Loads DFA output folders, aggregates H and Hq values into DataFrames,
# and generates boxplots + exports PNG/CSV.

import os, re
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QComboBox, QProgressDialog, QMessageBox, QSizePolicy, QSpinBox, QGroupBox,
    QGridLayout
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from utils.file_utils import load_h5_file  

# ─────────── helpers ───────────

def _find_dfa_segments(root):
    """
    Search the given root folder and find all DFA result segment folders.

    Expected folder patterns:
      • DFA_<start>-<end>_c<chunk>s
      • DFA_Overlap_<start>-<end>_c<chunk>s_h<hop>s

    Returns:
      List of absolute paths to segment folders.
    """
    pat_plain   = re.compile(r"^DFA_\d+(?:\.\d+)?-\d+(?:\.\d+)?_c\d+(?:\.\d+)?s$")
    pat_overlap = re.compile(r"^DFA_Overlap_\d+(?:\.\d+)?-\d+(?:\.\d+)?_c\d+(?:\.\d+)?s_h\d+(?:\.\d+)?s$")
    segs = []

    # Walk recursively through all directories
    for base, dirs, _ in os.walk(root):
        for d in dirs:
            # If folder name matches DFA pattern, store its path
            if pat_plain.match(d) or pat_overlap.match(d):
                segs.append(os.path.join(base, d))
    return segs


def _parse_seg_name(seg_basename):
    """
    Parse segment folder name and extract timing + configuration values.

    Supports:
      • DFA_<start>-<end>_c<chunk>s
      • DFA_Overlap_<start>-<end>_c<chunk>s_h<hop>s

    Returns:
      (seg_start, seg_end, chunk_s, hop_s or None, is_overlap: bool)
    """
    # Overlapping window format
    m_overlap = re.match(
        r"^DFA_Overlap_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)_c(\d+(?:\.\d+)?)s_h(\d+(?:\.\d+)?)s$",
        seg_basename
    )
    if m_overlap:
        ss, ee, cc, hh = m_overlap.groups()
        return float(ss), float(ee), float(cc), float(hh), True

    # Non-overlapping format
    m_plain = re.match(
        r"^DFA_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)_c(\d+(?:\.\d+)?)s$",
        seg_basename
    )
    if m_plain:
        ss, ee, cc = m_plain.groups()
        return float(ss), float(ee), float(cc), None, False

    # Fallback if name does not match
    return 0.0, 0.0, None, None, False


def _load_dfa_segment(seg_path):
    """
    Load DFA outputs from one segment folder.

    Each segment folder contains:
      Segment/
        Channel_<name>/
          chunk<N>_<t0>-<t1>/
            Hurst.csv        -> scalar H (Hurst exponent)
            Hq_vs_q.csv       -> H(q) vs q (generalized Hurst)

    Returns:
      H_df  : DataFrame with one row per (channel, chunk) containing H
      Hq_df : DataFrame with one row per (channel, chunk, q) containing Hq
    """
    rows_H, rows_Hq = [], []
    seg_tag = os.path.basename(seg_path)

    # Extract segment timing/config from folder name
    seg_start, seg_end, chunk_s, hop_s, is_overlap = _parse_seg_name(seg_tag)

    # Iterate over each channel folder inside the segment
    for chd in os.listdir(seg_path):
        if not chd.startswith("Channel_"):
            continue

        chan = chd.replace("Channel_", "")
        ch_path = os.path.join(seg_path, chd)

        # Iterate over chunk folders (each chunk corresponds to a time block)
        for cd in os.listdir(ch_path):
            cd_path = os.path.join(ch_path, cd)

            # Parse chunk times: chunk<N>_<t0>-<t1>
            mc = re.match(r"^chunk\d+_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$", cd)
            if not mc:
                continue
            t0, t1 = float(mc.group(1)), float(mc.group(2))

            # ----- Load H (Hurst exponent) -----
            hfile = os.path.join(cd_path, "Hurst.csv")
            if os.path.exists(hfile):
                H = np.loadtxt(hfile, delimiter=",").item()
                rows_H.append({
                    "seg_tag": seg_tag,
                    "seg_start": seg_start,
                    "seg_end": seg_end,
                    "chunk_s": chunk_s,
                    "hop_s": hop_s if is_overlap else np.nan,
                    "is_overlap": bool(is_overlap),
                    "chunk_start": t0,
                    "chunk_end": t1,
                    "chan": chan,
                    "H": H,  # computed Hurst value
                })

            # ----- Load Hq (generalized Hurst) -----
            hqfile = os.path.join(cd_path, "Hq_vs_q.csv")
            if os.path.exists(hqfile):
                arr = np.loadtxt(hqfile, delimiter=",", skiprows=1)

                # Ensure 2D array even if file has one row
                arr = np.atleast_2d(arr)

                # Each row: q, Hq
                for q, hq in arr:
                    rows_Hq.append({
                        "seg_tag": seg_tag,
                        "seg_start": seg_start,
                        "seg_end": seg_end,
                        "chunk_s": chunk_s,
                        "hop_s": hop_s if is_overlap else np.nan,
                        "is_overlap": bool(is_overlap),
                        "chunk_start": t0,
                        "chunk_end": t1,
                        "chan": chan,
                        "q": q,
                        "Hq": hq,  # computed H(q)
                    })

    # Convert collected rows into DataFrames
    H_df  = pd.DataFrame(rows_H)
    Hq_df = pd.DataFrame(rows_Hq)
    return H_df, Hq_df


# ─────────── main dialog ───────────

class DFAPostWindow(QDialog):
    """
    GUI tool for post-processing DFA/MFDFA outputs:
    - Load DFA output directory
    - Merge H and Hq results into tables
    - Plot distributions (boxplots)
    - Save plots and tables
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DFA/MFDFA Post-Processing")
        self.setWindowState(Qt.WindowMaximized)

        # DataFrames that hold loaded results
        self.H_df  = pd.DataFrame()   # Hurst exponent data
        self.Hq_df = pd.DataFrame()   # generalized Hurst (Hq) data

        # Optional: channel names loaded from H5 file
        self.channel_names = []

        self._make_ui()

    def _make_ui(self):
        """Build the UI controls and plotting canvas"""
        main = QVBoxLayout(self)

        # ===== Load controls =====
        load_bar = QHBoxLayout()

        # Button to load DFA result root folder
        self.btn_load = QPushButton("Load DFA root…")
        self.btn_load.clicked.connect(self._on_load)

        self.lbl_status = QLabel("No data")

        load_bar.addWidget(self.btn_load)
        load_bar.addWidget(self.lbl_status)
        load_bar.addStretch()

        # Optional: load channel names from H5 file
        self.btn_load_h5 = QPushButton("Load channel names (H5)…")
        self.btn_load_h5.clicked.connect(self._on_load_h5)
        load_bar.addWidget(self.btn_load_h5)

        main.addLayout(load_bar)

        # ===== Plot controls =====
        ctl = QGroupBox("Plot controls")
        grid = QGridLayout(ctl)

        grid.addWidget(QLabel("Plot type:"), 0, 0)

        # Dropdown for plot selection
        self.dd_plot = QComboBox()
        self.dd_plot.addItems([
            "H box-plot per channel",
            "H box-plot vs chunks",
            "Hq box-plot per q-order"
        ])
        grid.addWidget(self.dd_plot, 0, 1, 1, 3)

        # Percent selection for highlighting low variance channels
        grid.addWidget(QLabel("Highlight lowest variance %:"), 1, 0)
        self.spin_pct = QSpinBox()
        self.spin_pct.setRange(1, 100)
        self.spin_pct.setValue(10)
        grid.addWidget(self.spin_pct, 1, 1)

        # Generate plot button
        self.btn_plot = QPushButton("Generate")
        self.btn_plot.clicked.connect(self._make_plot)
        grid.addWidget(self.btn_plot, 1, 3)

        main.addWidget(ctl)

        # ===== Plot canvas =====
        self.fig = plt.Figure(figsize=(8, 5), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main.addWidget(self.canvas, stretch=1)

        # ===== Save controls =====
        save_bar = QHBoxLayout()

        self.btn_png = QPushButton("Save PNG")
        self.btn_png.clicked.connect(self._save_png)

        self.btn_csv = QPushButton("Save CSV")
        self.btn_csv.clicked.connect(self._save_csv)

        save_bar.addWidget(self.btn_png)
        save_bar.addWidget(self.btn_csv)
        save_bar.addStretch()

        main.addLayout(save_bar)

    def _on_load(self):
        """Load and merge all DFA segments from a selected root folder"""
        root = QFileDialog.getExistingDirectory(self, "Select DFA root")
        if not root:
            return

        # Find DFA_* segment folders under root
        segs = _find_dfa_segments(root)
        if not segs:
            QMessageBox.warning(self, "No folders", "No DFA_*_c*s found.")
            return

        # Progress dialog while loading segments
        dlg = QProgressDialog("Loading…", "Cancel", 0, len(segs), self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.show()

        frames_H, frames_Hq = [], []

        # Load each segment and append results
        for i, seg in enumerate(segs):
            H, Hq = _load_dfa_segment(seg)
            frames_H.append(H)
            frames_Hq.append(Hq)

            dlg.setValue(i + 1)
            if dlg.wasCanceled():
                return

        # Combine all segments into final DataFrames
        self.H_df  = pd.concat(frames_H,  ignore_index=True) if frames_H else pd.DataFrame()
        self.Hq_df = pd.concat(frames_Hq, ignore_index=True) if frames_Hq else pd.DataFrame()

        # Count overlap rows if available
        n_overlap = int(self.H_df["is_overlap"].sum()) if not self.H_df.empty else 0

        # Update status label (show counts)
        self.lbl_status.setText(
            f"Loaded H:{len(self.H_df)} rows, Hq:{len(self.Hq_df)} rows  "
            f"• overlap seg rows: {n_overlap}"
        )

        # Clear old plot
        self.fig.clf()
        self.canvas.draw()

    def _on_load_h5(self):
        """Load channel names from a .h5 file (helps label plots nicely)"""
        fn, _ = QFileDialog.getOpenFileName(self, "Select .h5", "", "H5 Files (*.h5)")
        if not fn:
            return

        sig, t, names = load_h5_file(fn)
        if names:
            self.channel_names = names
            QMessageBox.information(self, "OK", f"Got {len(names)} channels")
        else:
            QMessageBox.warning(self, "Fail", "No channel names")

    def _make_plot(self):
        """Generate plot based on selected plot type"""
        if self.H_df.empty:
            QMessageBox.information(self, "No data", "Load DFA data first!")
            return

        # Reset figure/axes
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)

        typ = self.dd_plot.currentText()
        if typ == "H box-plot per channel":
            self._plot_H_box()
        elif typ == "H box-plot vs chunks":
            self._plot_H_vs_chunks()
        else:
            self._plot_Hq_box()

        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_H_box(self):
        """Boxplot of H values per channel, highlight lowest variance channels"""
        chs = sorted(self.H_df.chan.unique())

        # Collect H values for each channel
        data = [self.H_df.loc[self.H_df.chan == c, "H"].values for c in chs]

        # Compute variance per channel (used for highlighting)
        variances = [np.var(d) for d in data]

        pct = self.spin_pct.value()
        num_channels = max(1, int(len(chs) * pct / 100))

        # Find indices of the lowest variance channels
        low_var_indices = np.argsort(variances)[:num_channels]

        # Create boxplot
        bp = self.ax.boxplot(
            data,
            patch_artist=True,
            flierprops={'markersize': 2.5}
        )

        # Highlight lowest-variance channels in red/blue
        for i in low_var_indices:
            bp['boxes'][i].set(color='red', linewidth=2)
            bp['medians'][i].set(color='blue', linewidth=2)

        # Convert channel IDs to real names if available
        labels = [
            self.channel_names[int(c)]
            if c.isdigit() and int(c) < len(self.channel_names)
            else str(c)
            for c in chs
        ]

        self.ax.set_xticks(np.arange(1, len(chs) + 1))
        self.ax.set_xticklabels(labels, rotation=90, fontsize=12)
        self.ax.set_ylabel("H", fontsize=12)
        self.ax.set_title(
            f"H box-plot per channel (Lowest {pct}% variance highlighted)",
            fontsize=28, pad=10
        )
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_ylim(0, 2)

        # Show text summary for first few low-variance channels
        legend_text = f"Lowest {pct}% Variance Channels:\n"
        for i, idx in enumerate(low_var_indices[:10]):
            ch_name = labels[idx]
            var_val = variances[idx]
            legend_text += f"{i+1}. {ch_name} (var={var_val:.4f})\n"
        if len(low_var_indices) > 10:
            legend_text += f"... ({len(low_var_indices) - 10} more)"

        self.ax.text(
            0.01, 0.01, legend_text,
            transform=self.ax.transAxes,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )

    def _plot_H_vs_chunks(self):
        """Boxplot of H distribution across time chunks (all channels pooled per chunk)"""
        grp = self.H_df.groupby(["chunk_start", "chunk_end"])
        arrays, centers = [], []

        # Build one box per chunk interval
        for (a, b), sub in sorted(grp):
            arrays.append(sub.H.values)
            centers.append((a + b) / 2)

        # Choose box width based on spacing between chunk centers
        width = 0.8 * (centers[1] - centers[0]) if len(centers) > 1 else 0.5

        self.ax.boxplot(arrays, positions=centers, widths=width)
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("H")
        self.ax.set_title("H distribution per chunk", fontsize=28, pad=10)
        self.ax.xaxis.set_tick_params(rotation=90)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_ylim(0, 2)

    def _plot_Hq_box(self):
        """Boxplot of Hq distributions per q value"""
        if self.Hq_df.empty:
            QMessageBox.information(self, "No Hq", "No Hq data")
            return

        qs = sorted(self.Hq_df.q.unique())
        data = [self.Hq_df.loc[self.Hq_df.q == q, "Hq"].values for q in qs]

        self.ax.boxplot(data, patch_artist=True)
        self.ax.set_xticks(np.arange(1, len(qs) + 1))
        self.ax.set_xticklabels([f"{q:.0f}" for q in qs], fontsize=12)
        self.ax.set_xlabel("q order")
        self.ax.set_ylabel("Hq")
        self.ax.set_title("Hq box-plot per q", fontsize=28, pad=10)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_ylim(0, 2)

    def _save_png(self):
        """Save current plot to PNG"""
        fn, _ = QFileDialog.getSaveFileName(self, "Save PNG", "plot.png", "PNG Files (*.png)")
        if fn:
            self.fig.savefig(fn)

    def _save_csv(self):
        """Export loaded H and Hq tables into a single CSV file (two sections)"""
        fn, _ = QFileDialog.getSaveFileName(self, "Save CSV", "data.csv", "CSV Files (*.csv)")
        if not fn:
            return

        # Dump both tables into one file
        with open(fn, "w") as f:
            f.write("### H data\n")
            self.H_df.to_csv(f, index=False)
            f.write("\n### Hq data\n")
            self.Hq_df.to_csv(f, index=False)
