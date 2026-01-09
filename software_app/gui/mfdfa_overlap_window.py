# file: gui/mfdfa_overlap_window.py
# ----------------------------------------------------------------------
#   • Excel summaries (H & Hq) in one file
#   • parallel image encoding (JPEG/WebP/PNG) with TurboJPEG if available
#   • optional flat images folder to reduce filesystem overhead
#   • preserves your UI & compute paths
# ----------------------------------------------------------------------

import os
import numpy as np
import pandas as pd

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QCheckBox, QScrollArea, QWidget, QPushButton, QFileDialog,
    QProgressDialog, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Optional process-parallel backend
try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

# Optional ultra-fast JPEG encoder
try:
    from turbojpeg import TurboJPEG
    _TURBOJPEG = TurboJPEG()
except Exception:
    _TURBOJPEG = None

# Fast core (Numba / vectorized)
from utils.fast_mfdfa import run_mfdfa_fast as run_mfdfa

from concurrent.futures import ThreadPoolExecutor, as_completed


# --------------------------- fast image encoder ------------------------
def _encode_plot_to_file(path, scales, F, q_vals, Hq,
                         fmt="png", quality=92, compress_level=1):
    """
    Render compact MF-DFA figure and encode quickly.
    fmt: "jpg" | "webp" | "png"
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    import numpy as _np

    fig, ax1 = _plt.subplots(figsize=(5, 3), dpi=100)
    ax1.plot(_np.log2(scales), _np.log2(F), "o-")
    ax1.set_xlabel("log2(scale)")
    ax1.set_ylabel("log2(F)")

    ax2 = ax1.twinx()
    ax2.plot(q_vals, Hq, "s-", color="C1")
    ax2.set_ylabel("Hq")

    fig.tight_layout()

    # draw and read pixels (Matplotlib 3.9+: use buffer_rgba)
    fig.canvas.draw()
    rgba = _np.asarray(fig.canvas.buffer_rgba())  # H x W x 4 (uint8)
    rgb = rgba[..., :3].copy()
    _plt.close(fig)

    try:
        from PIL import Image
    except Exception:
        # Very slow fallback (rare)
        fig2 = _plt.figure(figsize=(5, 3), dpi=100)
        ax = fig2.add_subplot(111)
        ax.axis("off")
        fig2.figimage(rgb, 0, 0)
        fig2.savefig(path)
        _plt.close(fig2)
        return

    fmt_l = fmt.lower()
    if fmt_l in ("jpg", "jpeg"):
        if _TURBOJPEG is not None:
            with open(path, "wb") as f:
                f.write(_TURBOJPEG.encode(rgb, quality=quality, jpeg_subsample=0))
        else:
            Image.fromarray(rgb).save(
                path, format="JPEG", quality=quality, subsampling=0, optimize=False
            )
    elif fmt_l == "webp":
        Image.fromarray(rgb).save(path, format="WEBP", quality=quality, method=4)
    else:
        # PNG (lossless)
        try:
            Image.fromarray(rgb).save(
                path, format="PNG", compress_level=compress_level, optimize=False
            )
        except TypeError:
            Image.fromarray(rgb).save(path, format="PNG")


# ======================== Overlap Analysis Window ======================

class MFDFAOverlapWindow(QDialog):
    """
    MF-DFA with sliding (overlapping) windows.

    chunk_s = window length (sec)
    hop_s   = step between windows (sec) (e.g. 0.5*chunk_s → 50% overlap)

    Provides:
      - Single-channel run with chunk navigation and plot
      - Batch run across selected channels and all boundary segments
      - Save batch outputs to a directory structure + summary Excel
    """
    def __init__(self, parent, signals, time_array,
                 start_time, end_time, channel_names,
                 labeled_times=None):
        super().__init__(parent)
        self.setWindowTitle("MF-DFA (Overlapping Windows)")
        self.setWindowState(Qt.WindowMaximized)

        # Data
        self.X       = signals
        self.t       = time_array
        self.fs      = 1000  # Hz
        self.start_t = start_time
        self.end_t   = end_time
        self.chnames = channel_names

        # Labelled boundaries
        self.bounds = self._build_bounds(labeled_times)

        # Defaults
        self.scales   = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
        self.q_vals   = np.array([-5, -3, -2, -1, 0, 1, 2, 3, 5])
        self.m_order  = 1
        self.chunk_s  = 1.0
        self.hop_s    = 0.5   # 50% overlap by default

        # State
        self.single_chunks = []
        self.chunk_idx     = 0
        self.batch_results = []

        # Fast save options (used by the “fast overlap save” version)
        self.IMG_FMT      = "jpg"   # "jpg" | "webp" | "png"
        self.JPG_QUALITY  = 92
        self.PNG_LEVEL    = 1       # 0..9 (1 is fast)
        self.FLAT_IMAGES  = True    # put all images in one folder for speed

        self._make_ui()

    # ----------------- helpers -----------------
    def _build_bounds(self, labeled):
        """
        Build a list of boundary markers inside [start_t, end_t].
        Ensures ("start", start_t) and ("end", end_t) exist.
        """
        if not labeled:
            return [("start", self.start_t), ("end", self.end_t)]
        seq = sorted(labeled, key=lambda x: x[1])
        seq = [(lab, t) for lab, t in seq if self.start_t <= t <= self.end_t]
        if not seq or seq[0][1] > self.start_t:
            seq.insert(0, ("start", self.start_t))
        if seq[-1][1] < self.end_t:
            seq.append(("end", self.end_t))
        return seq

    def _make_ui(self):
        root = QVBoxLayout(self)

        # ----- top: single-channel -----
        top = QVBoxLayout()
        root.addLayout(top, stretch=2)

        bar = QHBoxLayout()
        bar.addWidget(QLabel("Channel:"))
        self.dd_chan = QComboBox()
        self.dd_chan.addItems(self.chnames)
        bar.addWidget(self.dd_chan)

        self.btn_run1 = QPushButton("Run")
        self.btn_run1.clicked.connect(self._run_single)
        bar.addWidget(self.btn_run1)

        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        self.btn_prev.clicked.connect(lambda: self._jump_chunk(-1))
        self.btn_next.clicked.connect(lambda: self._jump_chunk(1))
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.lbl_nav = QLabel("")
        bar.addWidget(self.btn_prev)
        bar.addWidget(self.btn_next)
        bar.addWidget(self.lbl_nav)
        top.addLayout(bar)

        g_par = QGroupBox("MF-DFA parameters")
        gl = QGridLayout(g_par)
        self.ed_sc  = QLineEdit(" ".join(map(str, self.scales)))
        self.ed_q   = QLineEdit(" ".join(map(str, self.q_vals)))
        self.ed_m   = QLineEdit(str(self.m_order))
        self.ed_csz = QLineEdit(str(self.chunk_s))
        self.ed_hop = QLineEdit(str(self.hop_s))

        gl.addWidget(QLabel("Scales"),   0, 0); gl.addWidget(self.ed_sc,  0, 1)
        gl.addWidget(QLabel("q vals"),   0, 2); gl.addWidget(self.ed_q,   0, 3)
        gl.addWidget(QLabel("m"),        1, 0); gl.addWidget(self.ed_m,   1, 1)
        gl.addWidget(QLabel("Chunk s"),  1, 2); gl.addWidget(self.ed_csz, 1, 3)
        gl.addWidget(QLabel("Hop s"),    2, 0); gl.addWidget(self.ed_hop, 2, 1)
        top.addWidget(g_par)

        self.grid_plot = QGridLayout()
        top.addLayout(self.grid_plot, stretch=1)

        savebar = QHBoxLayout()
        self.btn_save_png = QPushButton("Save PNG")
        self.btn_save_csv = QPushButton("Save CSV")
        self.btn_save_png.setEnabled(False)
        self.btn_save_csv.setEnabled(False)
        self.btn_save_png.clicked.connect(self._save_png_single)
        self.btn_save_csv.clicked.connect(self._save_csv_single)
        savebar.addWidget(self.btn_save_png)
        savebar.addWidget(self.btn_save_csv)
        top.addLayout(savebar)

        # ----- bottom: batch -----
        bot = QVBoxLayout()
        root.addLayout(bot, stretch=1)

        g_cl = QGroupBox("Channels for run-all")
        vcl = QVBoxLayout(g_cl)
        sc = QScrollArea()
        sc.setWidgetResizable(True)
        frame = QWidget()
        fl = __import__("gui.flow_layout", fromlist=["FlowLayout"]).FlowLayout(frame)
        self.chk_map = {}
        for n in self.chnames:
            cb = QCheckBox(n)
            cb.setChecked(not (n.startswith("EKG") or n.startswith("X1 DC")))
            self.chk_map[n] = cb
            fl.addWidget(cb)
        sc.setWidget(frame)
        vcl.addWidget(sc)
        bot.addWidget(g_cl)

        hb = QHBoxLayout()
        self.btn_run_all  = QPushButton("Run All")
        self.btn_save_all = QPushButton("Save All")
        self.btn_run_all.clicked.connect(self._run_all)
        self.btn_save_all.clicked.connect(self._save_all)
        self.lbl_stat = QLabel("")
        hb.addWidget(self.btn_run_all)
        hb.addWidget(self.btn_save_all)
        hb.addWidget(self.lbl_stat)
        bot.addLayout(hb)

    # ----------------- params -----------------
    def _read_params(self):
        """
        Read and validate MF-DFA parameters from the UI.
        """
        try:
            self.scales  = np.array(list(map(int,   self.ed_sc.text().split())))
            self.q_vals  = np.array(list(map(float, self.ed_q.text().split())))
            self.m_order = int(self.ed_m.text())
            self.chunk_s = float(self.ed_csz.text())
            self.hop_s   = float(self.ed_hop.text())
        except Exception:
            raise ValueError("Check scale/q/m/chunk/hop inputs.")

        if self.chunk_s <= 0 or self.hop_s <= 0:
            raise ValueError("Chunk and Hop must be > 0.")
        if self.hop_s > self.chunk_s:
            QMessageBox.information(self, "Note", "Hop > chunk → gaps between windows.")

    # ------------- single-channel -------------
    def _run_single(self):
        """
        Run overlapping MF-DFA for the selected channel across boundary segments.
        """
        try:
            self._read_params()
        except ValueError as e:
            QMessageBox.warning(self, "Bad params", str(e))
            return

        ch = self.dd_chan.currentIndex()
        self.single_chunks.clear()

        n_seg = len(self.bounds) - 1
        dlg = QProgressDialog("MF-DFA (overlap)…", "Cancel", 0, n_seg, self)
        dlg.show()

        csz = int(self.chunk_s * self.fs)
        hop = max(1, int(self.hop_s * self.fs))

        max_scale = int(np.max(self.scales))
        if max_scale > csz:
            QMessageBox.information(
                self, "Adjust scales",
                f"Max scale ({max_scale}) > chunk samples ({csz}). "
                "Increase chunk or reduce scales."
            )

        for i in range(n_seg):
            t0, t1 = self.bounds[i][1], self.bounds[i+1][1]
            i0, i1 = int(t0 * self.fs), int(t1 * self.fs)
            snippet = self.X[ch, i0:i1]
            L = snippet.size
            if L < csz:
                dlg.setValue(i + 1)
                continue

            n_win = 1 + (L - csz) // hop
            for k in range(n_win):
                s0 = k * hop
                seg = snippet[s0:s0 + csz]
                if seg.size < max_scale:
                    continue

                seg = np.ascontiguousarray(seg, dtype=np.float64)
                H, F, Hq, _ = run_mfdfa(seg, self.scales, self.q_vals, self.m_order)
                chunk_start = t0 + (s0 / self.fs)
                self.single_chunks.append(dict(
                    H=H, F=F, Hq=Hq,
                    t0=chunk_start,
                    t1=chunk_start + self.chunk_s
                ))

            dlg.setValue(i + 1)
            dlg.repaint()
            if dlg.wasCanceled():
                return

        if not self.single_chunks:
            QMessageBox.information(self, "None", "No valid chunks.")
            return

        self.chunk_idx = 0
        self._refresh_single_plot()
        self.btn_save_png.setEnabled(True)
        self.btn_save_csv.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)

    def _refresh_single_plot(self):
        """
        Display the current single-channel overlap chunk.
        """
        while self.grid_plot.count():
            w = self.grid_plot.takeAt(0).widget()
            if w:
                w.setParent(None)

        rec = self.single_chunks[self.chunk_idx]
        fig = plt.Figure(figsize=(7, 4), dpi=100)

        ax = fig.add_subplot(211)
        ax.plot(np.log2(self.scales), np.log2(rec["F"]), "o-")
        ax.set_xlabel("log2(scale)")
        ax.set_ylabel("log2(F)")

        ax2 = fig.add_subplot(212)
        ax2.plot(self.q_vals, rec["Hq"], "s-")
        ax2.set_xlabel("q")
        ax2.set_ylabel("Hq")

        fig.tight_layout()
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.grid_plot.addWidget(self.canvas, 0, 0)

        self.lbl_nav.setText(
            f"Chunk {self.chunk_idx+1}/{len(self.single_chunks)}  "
            f"[{rec['t0']:.2f}–{rec['t1']:.2f}s]"
        )

    def _jump_chunk(self, step):
        if not self.single_chunks:
            return
        self.chunk_idx = (self.chunk_idx + step) % len(self.single_chunks)
        self._refresh_single_plot()

    def _save_png_single(self):
        fn, _ = QFileDialog.getSaveFileName(
            self, "PNG",
            f"chunk{self.chunk_idx+1}.png",
            "PNG Files (*.png)"
        )
        if fn:
            self.canvas.figure.savefig(fn)

    def _save_csv_single(self):
        if not self.single_chunks:
            return
        rec = self.single_chunks[self.chunk_idx]
        folder = QFileDialog.getExistingDirectory(
            self, "Save CSVs for this chunk into…", ""
        )
        if not folder:
            return

        np.savetxt(
            os.path.join(folder, "F_vs_scale.csv"),
            np.column_stack([self.scales, rec["F"]]),
            delimiter=",", header="scale,F", comments=""
        )
        np.savetxt(
            os.path.join(folder, "Hq_vs_q.csv"),
            np.column_stack([self.q_vals, rec["Hq"]]),
            delimiter=",", header="q,Hq", comments=""
        )

    # ---------------- batch (overlap) ----------------
    def _run_all(self):
        """
        Batch overlapping MF-DFA across selected channels and all boundary segments.
        """
        try:
            self._read_params()
        except ValueError as e:
            QMessageBox.warning(self, "Bad parameters", str(e))
            return

        ch_ids = [i for i, n in enumerate(self.chnames) if self.chk_map[n].isChecked()]
        if not ch_ids:
            QMessageBox.warning(self, "Select channels", "Tick at least one channel.")
            return

        self.batch_results.clear()

        csz = int(self.chunk_s * self.fs)
        hop = max(1, int(self.hop_s * self.fs))
        n_segs = len(self.bounds) - 1

        max_scale = int(np.max(self.scales))
        if max_scale > csz:
            QMessageBox.information(
                self, "Adjust scales",
                f"Max scale ({max_scale}) > chunk samples ({csz}). "
                "Increase chunk or reduce scales."
            )

        # Progress estimate
        prog_total = 0
        for i in range(n_segs):
            t0, t1 = self.bounds[i][1], self.bounds[i+1][1]
            L = int((t1 - t0) * self.fs)
            if L >= csz:
                prog_total += (1 + (L - csz) // hop) * len(ch_ids)

        prog = QProgressDialog("Running MF-DFA (overlap)…", "Cancel",
                               0, max(prog_total, 1), self)
        prog.setWindowModality(Qt.WindowModal)
        prog.show()

        tick = 0
        for seg_i in range(n_segs):
            t0, t1 = self.bounds[seg_i][1], self.bounds[seg_i+1][1]
            i0, i1 = int(t0 * self.fs), int(t1 * self.fs)
            seg_tag = f"{self.bounds[seg_i][0]}-{self.bounds[seg_i+1][0]}"

            for ch in ch_ids:
                data = self.X[ch, i0:i1]
                L = data.size
                if L < csz:
                    continue

                n_win = 1 + (L - csz) // hop
                for k in range(n_win):
                    s0 = k * hop
                    snippet = data[s0:s0 + csz]
                    if snippet.size < max_scale:
                        continue

                    snippet = np.ascontiguousarray(snippet, dtype=np.float64)
                    H, F, Hq, _ = run_mfdfa(snippet, self.scales, self.q_vals, self.m_order)
                    chunk_start = t0 + (s0 / self.fs)

                    self.batch_results.append(dict(
                        ch=self.chnames[ch],
                        seg_name=seg_tag,
                        seg_start=chunk_start,
                        seg_end=chunk_start + self.chunk_s,
                        F=F,
                        Hq=Hq,
                        H=H,
                        scales=self.scales.copy(),
                        q_vals=self.q_vals.copy()
                    ))

                    tick += 1
                    prog.setValue(tick)
                    if prog.wasCanceled():
                        self.lbl_stat.setText("Batch run cancelled")
                        return

        prog.close()
        self.lbl_stat.setText(f"Batch run finished • {len(self.batch_results)} chunks")

    def _save_all(self):
        """
        Save overlap batch results (CSV per chunk + Excel summary).

        Folder:
          DFA_{start}-{end}_c{chunk}s_h{hop}s/
            Channel_<name>/
              chunk<idx>_<t0>-<t1>/
                Hurst.csv
                F_vs_scale.csv
                Hq_vs_q.csv
            Hurst_Summary.xlsx (or CSV fallback)

        Summary sheet columns:
          Channel, ChunkIndex, SegStart, SegEnd, H
        """
        if not self.batch_results:
            QMessageBox.information(self, "Nothing to save", "Run batch first.")
            return

        dest_root = QFileDialog.getExistingDirectory(self, "Select save location")
        if not dest_root:
            return

        top_dir = os.path.join(
            dest_root,
            f"DFA_{self.start_t:.2f}-{self.end_t:.2f}_c{self.chunk_s}s_h{getattr(self, 'hop_s', 0):g}s"
        )
        os.makedirs(top_dir, exist_ok=True)

        # Group by channel
        from collections import defaultdict
        by_channel = defaultdict(list)
        for r in self.batch_results:
            by_channel[r["ch"]].append(r)

        summary_rows = []

        for ch, results in by_channel.items():
            # Chronological within channel
            results.sort(key=lambda x: (float(x["seg_start"]), float(x["seg_end"])))

            ch_dir = os.path.join(top_dir, f"Channel_{ch.replace(':', '_')}")
            os.makedirs(ch_dir, exist_ok=True)

            for idx, r in enumerate(results, start=1):
                seg_start = float(r["seg_start"])
                seg_end   = float(r["seg_end"])

                ck_dir = os.path.join(ch_dir, f"chunk{idx}_{seg_start:.2f}-{seg_end:.2f}")
                os.makedirs(ck_dir, exist_ok=True)

                np.savetxt(os.path.join(ck_dir, "Hurst.csv"), [float(r["H"])], delimiter=",")

                np.savetxt(
                    os.path.join(ck_dir, "F_vs_scale.csv"),
                    np.column_stack([np.asarray(r["scales"]), np.asarray(r["F"])]),
                    delimiter=",", header="scale,F", comments=""
                )
                np.savetxt(
                    os.path.join(ck_dir, "Hq_vs_q.csv"),
                    np.column_stack([np.asarray(r["q_vals"]), np.asarray(r["Hq"])]),
                    delimiter=",", header="q,Hq", comments=""
                )

                summary_rows.append({
                    "Channel": ch,
                    "ChunkIndex": idx,
                    "SegStart": seg_start,
                    "SegEnd": seg_end,
                    "H": float(r["H"]),
                })

        excel_path = None
        try:
            df = pd.DataFrame(summary_rows)
            if not df.empty:
                df.sort_values(["Channel", "ChunkIndex", "SegStart"], inplace=True, kind="mergesort")

            excel_path = os.path.join(top_dir, "Hurst_Summary.xlsx")
            df.to_excel(excel_path, sheet_name="Summary", index=False)
        except Exception as e:
            try:
                import csv
                csv_path = os.path.join(top_dir, "Hurst_Summary.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f, fieldnames=["Channel", "ChunkIndex", "SegStart", "SegEnd", "H"]
                    )
                    writer.writeheader()
                    for row in summary_rows:
                        writer.writerow(row)
                excel_path = csv_path
            except Exception:
                QMessageBox.warning(
                    self, "Summary save failed",
                    f"Could not write Excel/CSV summary:\n{e}"
                )

        if excel_path:
            self.lbl_stat.setText(
                f"Saved {len(self.batch_results)} chunks → {top_dir}\n"
                f"Summary: {os.path.basename(excel_path)}"
            )
        else:
            self.lbl_stat.setText(
                f"Saved {len(self.batch_results)} chunks → {top_dir}\n"
                "Summary: (failed)"
            )

