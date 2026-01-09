 # file: gui/dfa_post_window.py

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
    Recursively find segment folders:
      • DFA_<start>-<end>_c<chunk>s
      • DFA_Overlap_<start>-<end>_c<chunk>s_h<hop>s
    Returns a list of absolute paths.
    """
    pat_plain   = re.compile(r"^DFA_\d+(?:\.\d+)?-\d+(?:\.\d+)?_c\d+(?:\.\d+)?s$")
    pat_overlap = re.compile(r"^DFA_Overlap_\d+(?:\.\d+)?-\d+(?:\.\d+)?_c\d+(?:\.\d+)?s_h\d+(?:\.\d+)?s$")
    segs = []
    for base, dirs, _ in os.walk(root):
        for d in dirs:
            if pat_plain.match(d) or pat_overlap.match(d):
                segs.append(os.path.join(base, d))
    return segs

def _parse_seg_name(seg_basename):
    """
    Supports:
      • DFA_<start>-<end>_c<chunk>s
      • DFA_Overlap_<start>-<end>_c<chunk>s_h<hop>s
    Returns (seg_start, seg_end, chunk_s, hop_s or None, is_overlap: bool)
    """
    m_overlap = re.match(
        r"^DFA_Overlap_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)_c(\d+(?:\.\d+)?)s_h(\d+(?:\.\d+)?)s$",
        seg_basename
    )
    if m_overlap:
        ss, ee, cc, hh = m_overlap.groups()
        return float(ss), float(ee), float(cc), float(hh), True

    m_plain = re.match(
        r"^DFA_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)_c(\d+(?:\.\d+)?)s$",
        seg_basename
    )
    if m_plain:
        ss, ee, cc = m_plain.groups()
        return float(ss), float(ee), float(cc), None, False

    return 0.0, 0.0, None, None, False


def _load_dfa_segment(seg_path):
    """
    From one segment folder, loads channel/chunk data into two DataFrames:
      - H_df: one row per (chan,chunk) with H
      - Hq_df: one row per (chan,chunk,q) with Hq
    Also records whether this segment is overlap and (optionally) its hop size.
    """
    rows_H, rows_Hq = [], []
    seg_tag = os.path.basename(seg_path)

    seg_start, seg_end, chunk_s, hop_s, is_overlap = _parse_seg_name(seg_tag)

    # for each Channel_<name> subfolder
    for chd in os.listdir(seg_path):
        if not chd.startswith("Channel_"):
            continue
        chan = chd.replace("Channel_", "")
        ch_path = os.path.join(seg_path, chd)

        # for each chunk subfolder
        for cd in os.listdir(ch_path):
            cd_path = os.path.join(ch_path, cd)

            # parse chunk times: chunk<N>_<t0>-<t1>
            mc = re.match(r"^chunk\d+_(\d+(?:\.\d+)?)-(\d+(?:\.\d+)?)$", cd)
            if not mc:
                continue
            t0, t1 = float(mc.group(1)), float(mc.group(2))

            # H
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
                    "H": H,
                })

            # Hq
            hqfile = os.path.join(cd_path, "Hq_vs_q.csv")
            if os.path.exists(hqfile):
                arr = np.loadtxt(hqfile, delimiter=",", skiprows=1)
                # tolerate single-row files
                arr = np.atleast_2d(arr)
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
                        "Hq": hq,
                    })

    H_df  = pd.DataFrame(rows_H)
    Hq_df = pd.DataFrame(rows_Hq)
    return H_df, Hq_df


# def _load_dfa_segment(seg_path):
#     """
#     From one segment folder, loads channel/chunk data into two DataFrames:
#       - H_df: one row per (chan,chunk) with H
#       - Hq_df: one row per (chan,chunk,q) with Hq
#     """
#     rows_H  = []
#     rows_Hq = []
#     seg_tag = os.path.basename(seg_path)
#     # parse segment times (optional)
#     m = re.match(r"DFA_(\d+(\.\d+)?)-(\d+(\.\d+)?)_", seg_tag)
#     seg_start, seg_end = (float(m.group(1)), float(m.group(3))) if m else (0,0)

#     # for each Channel_<name> subfolder
#     for chd in os.listdir(seg_path):
#         if not chd.startswith("Channel_"): continue
#         chan = chd.replace("Channel_","")
#         ch_path = os.path.join(seg_path, chd)
#         # for each chunk subfolder
#         for cd in os.listdir(ch_path):
#             cd_path = os.path.join(ch_path, cd)
#             # parse chunk times
#             mc = re.match(r"chunk\d+_(\d+(\.\d+)?)-(\d+(\.\d+)?)", cd)
#             if not mc: continue
#             t0, t1 = float(mc.group(1)), float(mc.group(3))
#             # load Hurst
#             hfile = os.path.join(cd_path,"Hurst.csv")
#             if os.path.exists(hfile):
#                 H = np.loadtxt(hfile,delimiter=",").item()
#                 rows_H.append({
#                     "seg_tag":seg_tag, "seg_start":seg_start, "seg_end":seg_end,
#                     "chunk_start":t0, "chunk_end":t1,
#                     "chan":chan, "H":H
#                 })
#             # load Hq
#             hqfile = os.path.join(cd_path,"Hq_vs_q.csv")
#             if os.path.exists(hqfile):
#                 arr = np.loadtxt(hqfile,delimiter=",",skiprows=1)
#                 # arr[:,0]=q, arr[:,1]=Hq
#                 for q,hq in arr:
#                     rows_Hq.append({
#                         "seg_tag":seg_tag, "chunk_start":t0, "chunk_end":t1,
#                         "chan":chan, "q":q, "Hq":hq
#                     })
#     H_df  = pd.DataFrame(rows_H)
#     Hq_df = pd.DataFrame(rows_Hq)
#     return H_df, Hq_df

# ─────────── main dialog ───────────

class DFAPostWindow(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DFA/MFDFA Post‑Processing")
        self.setWindowState(Qt.WindowMaximized)

        self.H_df  = pd.DataFrame()
        self.Hq_df = pd.DataFrame()
        self.channel_names = []

        self._make_ui()

    def _make_ui(self):
        main = QVBoxLayout(self)

        # — load row —
        load_bar = QHBoxLayout()
        self.btn_load = QPushButton("Load DFA root…")
        self.btn_load.clicked.connect(self._on_load)
        self.lbl_status = QLabel("No data")
        load_bar.addWidget(self.btn_load)
        load_bar.addWidget(self.lbl_status)
        load_bar.addStretch()
        self.btn_load_h5 = QPushButton("Load channel names (H5)…")
        self.btn_load_h5.clicked.connect(self._on_load_h5)
        load_bar.addWidget(self.btn_load_h5)
        main.addLayout(load_bar)

        # — controls —
        ctl = QGroupBox("Plot controls")
        grid = QGridLayout(ctl)
        grid.addWidget(QLabel("Plot type:"), 0, 0)
        self.dd_plot = QComboBox()
        self.dd_plot.addItems([
            "H box‑plot per channel",
            "H box‑plot vs chunks",
            "Hq box‑plot per q‑order"
        ])
        grid.addWidget(self.dd_plot, 0, 1, 1, 3)
        grid.addWidget(QLabel("Highlight lowest variance %:"), 1, 0)
        self.spin_pct = QSpinBox()
        self.spin_pct.setRange(1, 100)
        self.spin_pct.setValue(10)
        grid.addWidget(self.spin_pct, 1, 1)
        self.btn_plot = QPushButton("Generate")
        self.btn_plot.clicked.connect(self._make_plot)
        grid.addWidget(self.btn_plot, 1, 3)
        main.addWidget(ctl)

        # — canvas —
        self.fig = plt.Figure(figsize=(8,5),dpi=100)
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        main.addWidget(self.canvas,stretch=1)

        # — save —
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
        root = QFileDialog.getExistingDirectory(self,"Select DFA root")
        if not root: return
        segs = _find_dfa_segments(root)
        if not segs:
            QMessageBox.warning(self,"No folders","No DFA_*_c*s found."); return

        dlg = QProgressDialog("Loading…","Cancel",0,len(segs),self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.show()
        frames_H, frames_Hq = [], []
        for i,seg in enumerate(segs):
            H, Hq = _load_dfa_segment(seg)
            frames_H.append(H); frames_Hq.append(Hq)
            dlg.setValue(i+1)
            if dlg.wasCanceled(): return

        self.H_df  = pd.concat(frames_H,  ignore_index=True) if frames_H else pd.DataFrame()
        self.Hq_df = pd.concat(frames_Hq, ignore_index=True) if frames_Hq else pd.DataFrame()
        n_overlap = int(self.H_df["is_overlap"].sum()) if not self.H_df.empty else 0
        self.lbl_status.setText(
          "Loaded H:{len(self.H_df)} rows, Hq:{len(self.Hq_df)} rows  "
               f"• overlap seg rows: {n_overlap}"
)

        # self.lbl_status.setText(f"Loaded H:{len(self.H_df)} rows, Hq:{len(self.Hq_df)} rows")
        self.fig.clf(); self.canvas.draw()

    def _on_load_h5(self):
        fn,_ = QFileDialog.getOpenFileName(self,"Select .h5","","H5 Files (*.h5)")
        if not fn: return
        sig,t,names = load_h5_file(fn)
        if names:
            self.channel_names = names
            QMessageBox.information(self,"OK",f"Got {len(names)} channels")
        else:
            QMessageBox.warning(self,"Fail","No channel names")

    def _make_plot(self):
        if self.H_df.empty:
            QMessageBox.information(self,"No data","Load DFA data first!"); return
        self.fig.clf()
        self.ax = self.fig.add_subplot(111)
        typ = self.dd_plot.currentText()
        if typ=="H box‑plot per channel":
            self._plot_H_box()
        elif typ=="H box‑plot vs chunks":
            self._plot_H_vs_chunks()
        else:
            self._plot_Hq_box()
        self.fig.tight_layout()
        self.canvas.draw()

    def _plot_H_box(self):
        chs = sorted(self.H_df.chan.unique())
        data = [self.H_df.loc[self.H_df.chan==c,"H"].values for c in chs]
        variances = [np.var(d) for d in data]
        
        pct = self.spin_pct.value()
        num_channels = max(1, int(len(chs) * pct / 100))
        low_var_indices = np.argsort(variances)[:num_channels]

        bp = self.ax.boxplot(data, 
                             patch_artist=True,
                             flierprops={'markersize': 2.5})
        for i in low_var_indices:
            bp['boxes'][i].set(color='red', linewidth=2)
            bp['medians'][i].set(color='blue', linewidth=2)
    
        labels = [self.channel_names[int(c)] if c.isdigit() and
                 int(c)<len(self.channel_names) else str(c)
                 for c in chs]
        
        self.ax.set_xticks(np.arange(1, len(chs)+1))
        self.ax.set_xticklabels(labels, rotation=90, fontsize=12)
        self.ax.set_ylabel("H", fontsize=12)
        self.ax.set_title(f"H box‑plot per channel (Lowest {pct}% variance highlighted)",fontsize=28,pad=10)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_ylim(0, 2)

        # Show info for first 10 channels
        legend_text = f"Lowest {pct}% Variance Channels:\n"
        for i, idx in enumerate(low_var_indices[:10]):
            ch_name = labels[idx]
            var_val = variances[idx]
            legend_text += f"{i+1}. {ch_name} (var={var_val:.4f})\n"
            
        if len(low_var_indices) > 10:
            legend_text += f"... ({len(low_var_indices) - 10} more)"
            
        self.ax.text(0.01, 0.01, legend_text, 
                    transform=self.ax.transAxes,
                    verticalalignment='bottom',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
 

    def _plot_H_vs_chunks(self):
        grp = self.H_df.groupby(["chunk_start","chunk_end"])
        arrays, centers = [], []
        for (a,b),sub in sorted(grp):
            arrays.append(sub.H.values)
            centers.append((a+b)/2)
        width = 0.8*(centers[1]-centers[0]) if len(centers)>1 else 0.5
        self.ax.boxplot(arrays, positions=centers, widths=width)
        self.ax.set_xlabel("time (s)")
        self.ax.set_ylabel("H")
        self.ax.set_title("H distribution per chunk",fontsize=28,pad=10)
        self.ax.xaxis.set_tick_params(rotation=90)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_ylim(0, 2)

    def _plot_Hq_box(self):
        if self.Hq_df.empty:
            QMessageBox.information(self,"No Hq","No Hq data"); return
        qs = sorted(self.Hq_df.q.unique())
        data = [ self.Hq_df.loc[self.Hq_df.q==q,"Hq"].values for q in qs ]
        self.ax.boxplot(data, patch_artist=True)
        self.ax.set_xticks(np.arange(1,len(qs)+1))
        self.ax.set_xticklabels([f"{q:.0f}" for q in qs],fontsize=12)
        self.ax.set_xlabel("q order")
        self.ax.set_ylabel("Hq")
        self.ax.set_title("Hq box‑plot per q",fontsize=28,pad=10)
        self.ax.grid(True, linestyle='--', alpha=0.7)
        self.ax.set_ylim(0, 2)

    def _save_png(self):
        fn,_ = QFileDialog.getSaveFileName(self,"Save PNG","plot.png","PNG Files (*.png)")
        if fn: self.fig.savefig(fn)

    def _save_csv(self):
        fn,_ = QFileDialog.getSaveFileName(self,"Save CSV","data.csv","CSV Files (*.csv)")
        if not fn: return
        # dump H and Hq
        with open(fn,"w") as f:
            f.write("### H data\n")
            self.H_df.to_csv(f,index=False)
            f.write("\n### Hq data\n")
            self.Hq_df.to_csv(f,index=False)
