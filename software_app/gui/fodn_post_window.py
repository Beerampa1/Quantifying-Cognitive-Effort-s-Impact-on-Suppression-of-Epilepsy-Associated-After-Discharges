# file: gui/fodn_post_window.py

import os, re
import numpy as np
import pandas as pd
from collections import defaultdict
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QFileDialog,
    QComboBox, QSpinBox, QProgressDialog, QMessageBox, QGroupBox,
    QGridLayout, QSizePolicy
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from scipy import linalg as LA
from utils.file_utils import load_h5_file

# ───────────── helpers ───────────── #

def _find_segments(root):
    pat = re.compile(r'^FODN_\d+(\.\d+)?-\d+(\.\d+)?_c\d+(\.\d+)?s$')
    out = []
    for d, dirs, _ in os.walk(root):
        for sub in dirs:
            if pat.match(sub):
                out.append(os.path.join(d, sub))
    return out

def _load_segment(seg_path):
    """Return DataFrame with one row per (chan,chunk): seg_tag, times, alpha, ev_ht."""
    rows = []
    # extract seg_tag
    seg_tag = os.path.basename(seg_path)
    # parse times
    m = re.match(r'FODN_(\d+(\.\d+)?)-(\d+(\.\d+)?)_', seg_tag)
    seg_start, seg_end = (float(m.group(1)), float(m.group(3))) if m else (0.0, 0.0)

    chunk_pat = re.compile(r'^chunk\d+_(\d+(\.\d+)?)-(\d+(\.\d+)?)$')
    for cd in os.listdir(seg_path):
        cpath = os.path.join(seg_path, cd)
        if not os.path.isdir(cpath): continue
        mc = chunk_pat.match(cd)
        if not mc: continue
        t0, t1 = float(mc.group(1)), float(mc.group(3))

        # load α
        alpha_f = os.path.join(cpath, "Alpha_Data.csv")
        if not os.path.exists(alpha_f): continue
        alpha = np.loadtxt(alpha_f, delimiter=",")

        # load coupling, compute eigen‑heat
        coup_f = os.path.join(cpath, "Coupling_Data.csv")
        if os.path.exists(coup_f):
            try:
                C = np.loadtxt(coup_f, delimiter=",")
                w, v = LA.eig(C)
                ev = np.abs(v[:, np.argmax(w.real)])
                # steps 1+2: abs + normalize
                # ev = np.abs(ev)
                ev = (ev - ev.min())/(ev.ptp() + 1e-12)
            except Exception:
                ev = np.full_like(alpha, np.nan)
        else:
            ev = np.full_like(alpha, np.nan)

        # build rows
        for ch_idx, a in enumerate(alpha):
            rows.append({
                'seg_tag':   seg_tag,
                'seg_start': seg_start,
                'seg_end':   seg_end,
                'chunk_start': t0,
                'chunk_end':   t1,
                'chan':      ch_idx,
                'alpha':     a,
                'ev_ht':     ev[ch_idx]
            })
    return pd.DataFrame(rows)


# ───────────── main dialog ───────────── #

class FODNPostWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FODN post‑processing viewer")
        self.setWindowState(Qt.WindowMaximized)
        self.df = pd.DataFrame()
        self.channel_names = []
        self._make_ui()

    def _make_ui(self):
        main = QVBoxLayout(self)

        # --- load bar
        load_bar = QHBoxLayout()
        self.btn_load = QPushButton("Load FODN root…")
        self.btn_load.clicked.connect(self._on_load)
        self.lbl_root = QLabel("no data")
        load_bar.addWidget(self.btn_load)
        load_bar.addWidget(self.lbl_root)
        load_bar.addStretch()
        self.btn_load_h5 = QPushButton("Load channel names (H5)…")
        self.btn_load_h5.clicked.connect(self._on_load_h5)
        load_bar.addWidget(self.btn_load_h5)
        main.addLayout(load_bar)

        # --- controls
        ctl = QGroupBox("Plot controls")
        grid = QGridLayout(ctl)
        grid.addWidget(QLabel("Plot type:"),0,0)
        self.dd_plot = QComboBox()
        self.dd_plot.addItems([
            "α box‑plot (per channel)",
            "α box‑plot vs chunks",
            "Eigen‑heat → per‑channel α (top %)",
            "Eigen‑heat → combined α (top %)",
            "Eigen‑heat → per‑channel α (bottom %)",
            "Eigen‑heat → combined α (bottom %)"
        ])
        grid.addWidget(self.dd_plot,0,1,1,3)
        grid.addWidget(QLabel("Eigen‑heat top X %:"),1,0)
        self.spin_pct = QSpinBox()
        self.spin_pct.setRange(1,100)
        self.spin_pct.setValue(10)
        grid.addWidget(self.spin_pct,1,1)
        self.btn_plot = QPushButton("Generate")
        self.btn_plot.clicked.connect(self._make_plot)
        grid.addWidget(self.btn_plot,1,3)
        main.addWidget(ctl)

        # --- canvas
        self.fig = plt.Figure(figsize=(8,5), dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        main.addWidget(self.canvas, stretch=1)

        # --- save
        save_bar = QHBoxLayout()
        self.btn_png = QPushButton("Save PNG"); self.btn_png.clicked.connect(self._save_png)
        self.btn_csv = QPushButton("Save CSV"); self.btn_csv.clicked.connect(self._save_csv)
        save_bar.addWidget(self.btn_png); save_bar.addWidget(self.btn_csv); save_bar.addStretch()
        main.addLayout(save_bar)

    # ───────── actions ───────── #

    def _on_load(self):
        root = QFileDialog.getExistingDirectory(self,"Select FODN root")
        if not root: return
        segs = _find_segments(root)
        if not segs:
            QMessageBox.warning(self,"No folders","No FODN_*/_c*s folders found."); return

        dlg = QProgressDialog("Loading…","Cancel",0,len(segs),self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.show()
        frames = []
        for i,p in enumerate(segs):
            frames.append(_load_segment(p))
            dlg.setValue(i+1)
            if dlg.wasCanceled(): return

        self.df = pd.concat(frames, ignore_index=True)
        self.lbl_root.setText(f"loaded {len(self.df)} rows")
        self.fig.clf(); self.canvas.draw()

    def _on_load_h5(self):
        fn,_ = QFileDialog.getOpenFileName(self,"Select .h5","",".h5 Files (*.h5)")
        if not fn: return
        sig, t, names = load_h5_file(fn)
        if names:
            self.channel_names = names
            QMessageBox.information(self,"OK",f"Got {len(names)} channels")
        else:
            QMessageBox.warning(self,"Fail","No channel names in that file.")

    def _make_plot(self):
        if self.df.empty:
            QMessageBox.information(self,"No data","Load FODN data first."); return
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        typ = self.dd_plot.currentText()
        if typ.startswith("α box‑plot (per channel)"):
            self._plot_box()
        elif typ.startswith("α box‑plot vs chunks"):
            self._plot_alpha_vs_chunks()
        elif "per‑channel α (top" in typ:
            self._plot_eig(per_channel=True, top_percentile=True)
        elif "combined α (top" in typ:
            self._plot_eig(per_channel=False, top_percentile=True)
        elif "per‑channel α (bottom" in typ:
            self._plot_eig(per_channel=True, top_percentile=False)
        elif "combined α (bottom" in typ:
            self._plot_eig(per_channel=False, top_percentile=False)
        self.fig.tight_layout()
        self.canvas.draw()

    # ───────── plot implementations ───────── #

    def _plot_box(self):
        chan_ids = sorted(self.df.chan.unique())
        data = [self.df.loc[self.df.chan==c, "alpha"].values for c in chan_ids]

        # Variance Calc
        variances = []
        for i, chan_data in enumerate(data):
            if len(chan_data) > 0:
                variances.append((chan_ids[i], np.var(chan_data)))
        
        # Sort variances and select lowest X% based on user selection
        sorted_variances = sorted(variances, key=lambda x: x[1])
        pct = self.spin_pct.value()
        num_to_highlight = max(1, int(len(sorted_variances) * (pct / 100)))
        lowest_variance  = sorted_variances[:num_to_highlight]
        low_var_indices  = [chan_ids.index(chan_id) for chan_id, _ in lowest_variance]
        
        bp = self.ax.boxplot(data,
                             patch_artist=True,
                             flierprops={'markersize': 2.5})
        
        # Highlight the lowest variance channels
        for i in low_var_indices:
            bp['boxes'][i].set(color='red', linewidth=2)
            bp['medians'][i].set(color='blue', linewidth=2)

        labels = [(self.channel_names[c] if c < len(self.channel_names) else f"Ch{c}")
                  for c in chan_ids]
        self.ax.set_xticks(np.arange(1,len(data)+1))
        self.ax.set_xticklabels(labels, rotation=90, fontsize=10)
        self.ax.set_ylabel("α", fontsize=12)
        self.ax.set_xlabel("Channel", fontsize=12)
        self.ax.set_title (f"α Box-Plot Per Channel (Lowest {pct}% Variance Highlighted)",fontsize=28,pad =10)
        self.ax.grid      (True, linestyle='--', alpha=0.7)
        self.ax.set_ylim  (-0.25, 1.5)

        # Show info about up to the first 10 channels
        legend_text = f"Lowest {pct}% Variance Channels:\n"
        for i, (chan_id, var) in enumerate(lowest_variance[:10]):
            chan_label = self.channel_names[chan_id] if chan_id < len(self.channel_names) else f"Ch{chan_id}"
            legend_text += f"{i+1}. {chan_label}: {var:.4f}\n"
        
        if len(lowest_variance) > 10:
            legend_text += f"... ({len(lowest_variance) - 10} more)"
            
        self.ax.text(0.01, 0.01, legend_text, 
                     transform=self.ax.transAxes,
                     verticalalignment='bottom', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    def _plot_alpha_vs_chunks(self):
        grp = self.df.groupby(["chunk_start","chunk_end"])

        arrays = []
        centers = []
        for (a,b), sub in sorted(grp):
            arrays.append(sub.alpha.values)
            centers.append((a+b)/2)
    
        boxplot = self.ax.boxplot(
            arrays, 
            positions=centers, 
            widths=0.8*(centers[1]-centers[0]), 
            flierprops={'markersize': 2.5} 
        )
        self.ax.set_xlabel("Time (s)", fontsize=12)
        self.ax.set_ylabel("α", fontsize=12)
        self.ax.set_title("α Distribution Per Chunk",fontsize=28, pad=10)
        self.ax.xaxis.set_tick_params(rotation=90, labelsize=4)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_ylim(-0.25, 1.5)

    def _plot_eig(self, per_channel, top_percentile=True):
        # Take absolute value, normalize, then calculate mean eigen-heat per channel
        channel_groups = self.df.groupby("chan")
        heat = pd.Series(dtype=float)
        for chan, group in channel_groups:
            chan_heat = group.ev_ht.dropna()
            if not chan_heat.empty:
                chan_heat = chan_heat.abs()
                min_val = chan_heat.min()
                range_val = chan_heat.max() - min_val
                if range_val > 0:
                    chan_heat = (chan_heat - min_val) / range_val
                heat[chan] = chan_heat.mean()
        
        # Filter w/ pct
        pct = self.spin_pct.value()
        if heat.empty:
            sel = []
        else:
            if top_percentile:
                thresh = np.nanpercentile(heat, 100-pct)
                sel = heat[heat>=thresh].index.tolist()
                threshold_label = f"Top {pct}%"
            else:
                thresh = np.nanpercentile(heat, pct)
                sel = heat[heat<=thresh].index.tolist()
                threshold_label = f"Bottom {pct}%"

        if not sel:
            QMessageBox.warning(self,"None",f"No channels {'above' if top_percentile else 'below'} percentile threshold."); 
            return

        if per_channel:
            sel = sorted(sel)
            data = [self.df.loc[self.df.chan==c, "alpha"].values for c in sel]
            
            # Cross Correlation
            alpha_data = np.array([d for d in data if len(d) > 0])
            if alpha_data.shape[0] > 1: 
                max_len = max(len(d) for d in alpha_data)
                padded_data = np.array([np.pad(d, (0, max_len - len(d)), 'constant', constant_values=np.nan) for d in alpha_data])
            corr_matrix = np.zeros((len(sel), len(sel)))
            for i in range(len(sel)):
                for j in range(len(sel)):
                    mask = ~np.isnan(padded_data[i]) & ~np.isnan(padded_data[j])
                    if np.sum(mask) > 1:
                        corr_matrix[i, j] = np.corrcoef(padded_data[i, mask], padded_data[j, mask])[0, 1]
            
            # AVg corr
            mask = ~np.eye(corr_matrix.shape[0], dtype=bool)  
            avg_correlation = np.nanmean(corr_matrix[mask])
            self.fig.clf()
            if len(sel) > 1:
                # Increase the relative size of the correlation matrix by adjusting width_ratios
                gs = self.fig.add_gridspec(1, 2, width_ratios=[3, 2], wspace=0.25)
                self.ax = self.fig.add_subplot(gs[0])
                ax_corr = self.fig.add_subplot(gs[1])
                
                im = ax_corr.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                ax_corr.set_title(f'Correlation Matrix\nAvg: {avg_correlation:.3f}', fontsize=14, pad=10)
                # Adjust colorbar position and size
                cbar = self.fig.colorbar(im, ax=ax_corr, shrink=0.8, pad=0.05)
                if len(sel) < 20: 
                    labels = [self._chan_label(c) for c in sel]
                    ax_corr.set_xticks(np.arange(len(sel)))
                    ax_corr.set_yticks(np.arange(len(sel)))
                    ax_corr.set_xticklabels(labels, rotation=90, fontsize=12)
                    ax_corr.set_yticklabels(labels, fontsize=12)
            else:
                self.ax = self.fig.add_subplot(111)
            
            self.ax.boxplot(data,
                            patch_artist=True,
                            flierprops={'markersize': 2.5}
                            )
            labels = [self._chan_label(c) for c in sel]
            self.ax.set_xticks(np.arange(1,len(data)+1))
            self.ax.set_xticklabels(labels, rotation=90, fontsize=12,pad = 10)
            self.ax.set_ylabel("α", fontsize=12,pad = 10)
            self.ax.set_title(f"α Box-Plots for {threshold_label} Channels (by Eigenvector Selection)",fontsize=20, pad=10)
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.set_ylim(-0.25, 1.5)

        else:
            filtered_df = self.df[self.df.chan.isin(sel)]
            grp = filtered_df.groupby(["chunk_start", "chunk_end"])
            arrays = []
            centers = []
            for (a, b), sub in sorted(grp):
                arrays.append(sub.alpha.values)
                centers.append((a+b)/2)
            if not arrays:
                QMessageBox.warning(self, "No Data", "No data for selected channels and chunks")
                return
            boxplot = self.ax.boxplot(
                arrays, 
                positions=centers, 
                widths=0.8*(centers[1]-centers[0]) if len(centers) > 1 else 0.8, 
                patch_artist=True,
                flierprops={'markersize': 2.5} 
            )
            self.ax.set_xlabel("Time (s)",fontsize=12)
            self.ax.set_ylabel("α", fontsize=12)
            self.ax.set_title(f"α Distribution Across Chunks ({threshold_label} Channels by Eigenvector Selection)",fontsize=20, pad=10)
            self.ax.xaxis.set_tick_params(rotation=90, labelsize=4)
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.set_ylim(-0.25, 1.5)

    def _chan_label(self, idx):
        if idx < len(self.channel_names):
            return self.channel_names[idx]
        else:
            return f"Ch{idx}"

    # ───────── saves ───────── #

    def _save_png(self):
        if self.df.empty: return
        fn,_ = QFileDialog.getSaveFileName(self,"Save PNG","plot.png","PNG Files (*.png)")
        if fn:
            self.fig.savefig(fn)

    def _save_csv(self):
        if self.df.empty: return
        fn,_ = QFileDialog.getSaveFileName(self,"Save CSV","data.csv","CSV Files (*.csv)")
        if fn:
            # save the raw table
            self.df.to_csv(fn, index=False)
