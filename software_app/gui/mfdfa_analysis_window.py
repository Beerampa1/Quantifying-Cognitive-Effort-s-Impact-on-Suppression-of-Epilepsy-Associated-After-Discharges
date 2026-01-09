# file: gui/mfdfa_analysis_window.py
import os
import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QGroupBox, QLabel, QLineEdit,
    QComboBox, QCheckBox, QScrollArea, QWidget, QPushButton, QFileDialog,
    QProgressDialog, QMessageBox, QSizePolicy
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# Parallel backend:
#   If joblib is installed, we can parallelize the batch MF-DFA computations.
#   If not, we fall back to a single-threaded loop.
try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

# Fast MF-DFA core:
#   run_mfdfa_fast is assumed to compute MF-DFA metrics for a 1D signal segment.
#   It returns (H, F, Hq, Fq) (based on how it's used below).
from utils.fast_mfdfa import run_mfdfa_fast as run_mfdfa


class MFDFAAnalysisWindow(QDialog):
    """
    MF-DFA analysis GUI window (fast implementation).

    Features:
      1) Single-channel run:
           - pick one channel
           - compute MF-DFA for each chunk inside each boundary segment
           - navigate chunk-by-chunk and visualize:
               * log2(F) vs log2(scale)
               * Hq vs q

      2) Batch run (Run All):
           - pick many channels via checkboxes
           - compute MF-DFA for every chunk of every selected channel
           - (optionally) parallelize computation using joblib threads
           - save chunk outputs to a nested folder structure + an H summary sheet

    Inputs:
      - signals: ndarray shape (n_channels, n_samples)
      - time_array: ndarray shape (n_samples,) (not heavily used here, but kept)
      - start_time, end_time: float seconds
      - channel_names: list[str]
      - labeled_times: optional list[(label, time_seconds)] defining internal boundaries
    """
    def __init__(self, parent, signals, time_array,
                 start_time, end_time, channel_names,
                 labeled_times=None):
        super().__init__(parent)

        # Window chrome
        self.setWindowTitle("MF-DFA Analysis (Fast)")
        self.setWindowState(Qt.WindowMaximized)

        # -------------------- data/state --------------------
        # Raw iEEG data and time base
        self.X       = signals         # shape: (channels, samples)
        self.t       = time_array      # shape: (samples,)
        self.fs      = 1000            # sampling rate (Hz) used for time->index conversion
        self.start_t = start_time      # global start time (s) for this window
        self.end_t   = end_time        # global end time (s) for this window
        self.chnames = channel_names   # list of human-readable channel names

        # Internal boundaries used for iterating segments (start/end and optional markers)
        self.bounds = self._build_bounds(labeled_times)

        # -------------------- default MF-DFA parameters --------------------
        # scales: window sizes used in MF-DFA fluctuation function
        self.scales   = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
        # q values: multifractal moments
        self.q_vals   = np.array([-5, -3, -2, -1, 0, 1, 2, 3, 5])
        # polynomial detrending order
        self.m_order  = 1
        # chunk size (seconds) for splitting each boundary segment
        self.chunk_s  = 1.0

        # -------------------- runtime results --------------------
        # Single-channel chunk results (list of dicts: {"H":..., "F":..., "Hq":...})
        self.single_chunks = []
        # Index into self.single_chunks for navigation
        self.chunk_idx     = 0

        # Batch results: list of dicts per chunk with channel + timing + arrays
        self.batch_results = []

        # Build UI widgets + wiring
        self._make_ui()

    # ---------------- helpers ----------------
    def _build_bounds(self, labeled):
        """
        Create a clean list of boundary markers inside [start_t, end_t].

        If labeled_times is missing/empty:
            returns [("start", start_t), ("end", end_t)]

        If labeled_times exists:
          1) sort by time
          2) keep only times inside [start_t, end_t]
          3) ensure start and end markers exist
        """
        if not labeled:
            return [("start", self.start_t), ("end", self.end_t)]

        seq = sorted(labeled, key=lambda x: x[1])
        seq = [(lab, t) for lab, t in seq if self.start_t <= t <= self.end_t]

        # Ensure explicit start/end boundaries
        if not seq or seq[0][1] > self.start_t:
            seq.insert(0, ("start", self.start_t))
        if seq[-1][1] < self.end_t:
            seq.append(("end", self.end_t))
        return seq

    def _make_ui(self):
        """
        Build the full UI:
          - Top: single-channel controls + parameters + plot area + save buttons
          - Bottom: batch controls with channel selection + run/save buttons
        """
        root = QVBoxLayout(self)

        # ===================== single-channel section =====================
        top = QVBoxLayout()
        root.addLayout(top, stretch=2)

        # Row: channel dropdown + Run + chunk navigation
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Channel:"))

        # Channel selector (single-channel mode)
        self.dd_chan = QComboBox()
        self.dd_chan.addItems(self.chnames)
        bar.addWidget(self.dd_chan)

        # Run MF-DFA for the currently selected channel
        self.btn_run1 = QPushButton("Run")
        bar.addWidget(self.btn_run1)
        self.btn_run1.clicked.connect(self._run_single)

        # Chunk navigation buttons (enabled after a run)
        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        self.btn_prev.clicked.connect(lambda: self._jump_chunk(-1))
        self.btn_next.clicked.connect(lambda: self._jump_chunk(1))
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)

        # Label showing current chunk index / total chunks
        self.lbl_nav = QLabel("")
        bar.addWidget(self.btn_prev)
        bar.addWidget(self.btn_next)
        bar.addWidget(self.lbl_nav)

        top.addLayout(bar)

        # MF-DFA parameter entry group
        g_par = QGroupBox("MF-DFA parameters")
        gl = QGridLayout(g_par)

        # Note: these are plain text fields. We parse them on run.
        self.ed_sc  = QLineEdit(" ".join(map(str, self.scales)))
        self.ed_q   = QLineEdit(" ".join(map(str, self.q_vals)))
        self.ed_m   = QLineEdit(str(self.m_order))
        self.ed_csz = QLineEdit(str(self.chunk_s))

        gl.addWidget(QLabel("Scales"), 0, 0); gl.addWidget(self.ed_sc, 0, 1)
        gl.addWidget(QLabel("q vals"), 0, 2); gl.addWidget(self.ed_q, 0, 3)
        gl.addWidget(QLabel("m"),      1, 0); gl.addWidget(self.ed_m, 1, 1)
        gl.addWidget(QLabel("Chunk s"),1, 2); gl.addWidget(self.ed_csz, 1, 3)
        top.addWidget(g_par)

        # Plot container for single-channel visualization
        self.grid_plot = QGridLayout()
        top.addLayout(self.grid_plot, stretch=1)

        # Save buttons for the *currently displayed* single-channel chunk
        savebar = QHBoxLayout()
        self.btn_save_png = QPushButton("Save PNG")
        self.btn_save_csv = QPushButton("Save CSV")
        for b in (self.btn_save_png, self.btn_save_csv):
            b.setEnabled(False)  # enabled after a successful run
        self.btn_save_png.clicked.connect(self._save_png_single)
        self.btn_save_csv.clicked.connect(self._save_csv_single)
        savebar.addWidget(self.btn_save_png)
        savebar.addWidget(self.btn_save_csv)
        top.addLayout(savebar)

        # ===================== batch section =====================
        bot = QVBoxLayout()
        root.addLayout(bot, stretch=1)

        # Channel selection panel for batch mode (run-all)
        g_cl = QGroupBox("Channels for run-all")
        vcl = QVBoxLayout(g_cl)

        sc = QScrollArea()
        sc.setWidgetResizable(True)

        frame = QWidget()
        from gui.flow_layout import FlowLayout
        fl = FlowLayout(frame)

        # Map channel name -> checkbox (checked means included in batch)
        self.chk_map = {}
        for n in self.chnames:
            cb = QCheckBox(n)

            # Default: check all, but exclude EKG and X1 DC channels by name pattern
            cb.setChecked(True)
            cb.setChecked(not (n.startswith("EKG") or n.startswith("X1 DC")))

            self.chk_map[n] = cb
            fl.addWidget(cb)

        sc.setWidget(frame)
        vcl.addWidget(sc)
        bot.addWidget(g_cl)

        # Batch action buttons + status label
        hb = QHBoxLayout()
        self.btn_run_all = QPushButton("Run All")
        self.btn_save_all = QPushButton("Save All")
        self.lbl_stat = QLabel("")
        hb.addWidget(self.btn_run_all)
        hb.addWidget(self.btn_save_all)
        hb.addWidget(self.lbl_stat)

        self.btn_run_all.clicked.connect(self._run_all)
        self.btn_save_all.clicked.connect(self._save_all)

        bot.addLayout(hb)

    # ---------------- single ----------------
    def _run_single(self):
        """
        Run MF-DFA for the selected single channel.

        Workflow:
          1) Parse parameters from the UI
          2) For each boundary segment [bounds[i], bounds[i+1]]:
               - slice that channel’s data by time
               - split into fixed-length chunks (chunk_s)
               - run MF-DFA for each chunk
               - store results in self.single_chunks
          3) If results exist, show first chunk and enable navigation + save
        """
        # Parse parameters from line edits
        try:
            self.scales  = np.array(list(map(int, self.ed_sc.text().split())))
            self.q_vals  = np.array(list(map(float, self.ed_q.text().split())))
            self.m_order = int(self.ed_m.text())
            self.chunk_s = float(self.ed_csz.text())
        except Exception:
            QMessageBox.warning(self, "Bad params", "Check scale/q/m/chunk inputs.")
            return

        # Channel index from dropdown
        ch = self.dd_chan.currentIndex()

        # Clear previous single-channel run results
        self.single_chunks.clear()

        # Loop over boundary segments
        p_segs = len(self.bounds) - 1
        dlg = QProgressDialog("MF-DFA…", "Cancel", 0, p_segs, self)
        dlg.show()

        for i in range(p_segs):
            # Segment start/end times
            t0, t1 = self.bounds[i][1], self.bounds[i + 1][1]

            # Convert time to sample indices
            idx0, idx1 = int(t0 * self.fs), int(t1 * self.fs)

            # Extract the channel snippet for this boundary segment
            snippet = self.X[ch, idx0:idx1]

            # Chunk size in samples
            csz = int(self.chunk_s * self.fs)

            # Iterate whole chunks only (floor division)
            for k in range(len(snippet) // csz):
                seg = snippet[k * csz:(k + 1) * csz]

                # MF-DFA requires segment length >= max scale
                if seg.size < self.scales.max():
                    continue

                # Run MF-DFA core; keep only what we need for plotting/saving
                H, F, Hq, Fq = run_mfdfa(seg, self.scales, self.q_vals, self.m_order)
                self.single_chunks.append(dict(H=H, F=F, Hq=Hq))

            dlg.setValue(i + 1)
            dlg.repaint()
            if dlg.wasCanceled():
                return

        # If nothing valid was computed, notify and stop
        if not self.single_chunks:
            QMessageBox.information(self, "None", "No valid chunks.")
            return

        # Show first computed chunk
        self.chunk_idx = 0
        self._refresh_single_plot()

        # Enable save/navigation now that we have results
        self.btn_save_png.setEnabled(True)
        self.btn_save_csv.setEnabled(True)
        self.btn_prev.setEnabled(True)
        self.btn_next.setEnabled(True)

    def _refresh_single_plot(self):
        """
        Redraw the single-channel plot panel for the current chunk index.

        Displays:
          - Top subplot: log2(F) vs log2(scale)
          - Bottom subplot: Hq vs q
        """
        # Clear old widgets from plot grid
        while self.grid_plot.count():
            w = self.grid_plot.takeAt(0).widget()
            if w:
                w.setParent(None)

        # Current chunk record
        rec = self.single_chunks[self.chunk_idx]

        # Build a new matplotlib figure for embedding
        fig = plt.Figure(figsize=(7, 4), dpi=100)

        # Subplot 1: fluctuation function
        ax = fig.add_subplot(211)
        ax.plot(np.log2(self.scales), np.log2(rec["F"]), "o-")
        ax.set_xlabel("log2(scale)")
        ax.set_ylabel("log2(F)")

        # Subplot 2: multifractal H(q)
        ax2 = fig.add_subplot(212)
        ax2.plot(self.q_vals, rec["Hq"], "s-")
        ax2.set_xlabel("q")
        ax2.set_ylabel("Hq")

        fig.tight_layout()

        # Embed into Qt
        self.canvas = FigureCanvas(fig)
        self.grid_plot.addWidget(self.canvas, 0, 0)

        # Update navigation label
        self.lbl_nav.setText(f"Chunk {self.chunk_idx+1}/{len(self.single_chunks)}")

    def _jump_chunk(self, step):
        """
        Move the single-channel chunk index by 'step' (wrap-around),
        then refresh the plot.
        """
        if not self.single_chunks:
            return
        self.chunk_idx = (self.chunk_idx + step) % len(self.single_chunks)
        self._refresh_single_plot()

    def _save_png_single(self):
        """
        Save the currently displayed single-channel chunk plot as a PNG.
        """
        fn, _ = QFileDialog.getSaveFileName(
            self, "PNG", f"chunk{self.chunk_idx+1}.png", "PNG Files (*.png)"
        )
        if fn:
            self.canvas.figure.savefig(fn)

    def _save_csv_single(self):
        """
        Save numeric outputs for the current single-channel chunk to CSV files:
          - F_vs_scale.csv: (scale, F)
          - Hq_vs_q.csv: (q, Hq)

        Note:
          - User selects a destination folder; filenames are fixed.
        """
        rec = self.single_chunks[self.chunk_idx]
        folder = QFileDialog.getExistingDirectory(self, "Save CSVs for this chunk into…", "")
        if not folder:
            return

        np.savetxt(
            os.path.join(folder, "F_vs_scale.csv"),
            np.column_stack([self.scales, rec["F"]]),
            delimiter=",",
            header="scale,F",
            comments=""
        )
        np.savetxt(
            os.path.join(folder, "Hq_vs_q.csv"),
            np.column_stack([self.q_vals, rec["Hq"]]),
            delimiter=",",
            header="q,Hq",
            comments=""
        )

    # ---------------- batch ----------------
    def _run_all(self):
        """
        Run MF-DFA for all selected channels across all boundary segments.

        Steps:
          1) Parse parameters
          2) Determine selected channel indices from checkboxes
          3) Build a 'jobs' list of (channel_index, 1D_segment, seg_start_time)
             where each job corresponds to one chunk of one channel
          4) Execute jobs:
               - in parallel with joblib if available
               - otherwise sequentially
          5) Store results in self.batch_results and show a summary status message
        """
        # Parse parameters
        try:
            self.scales = np.array(list(map(int, self.ed_sc.text().split())))
            self.q_vals = np.array(list(map(float, self.ed_q.text().split())))
            self.m_order = int(self.ed_m.text())
            self.chunk_s = float(self.ed_csz.text())
        except ValueError:
            QMessageBox.warning(self, "Bad parameters", "Check inputs.")
            return

        # Selected channels for batch processing
        ch_ids = [i for i, n in enumerate(self.chnames) if self.chk_map[n].isChecked()]
        if not ch_ids:
            QMessageBox.warning(self, "Select channels", "Tick at least one.")
            return

        # Clear prior batch results
        self.batch_results.clear()

        # Build jobs list
        n_segs = len(self.bounds) - 1
        csz = int(self.chunk_s * self.fs)  # chunk length in samples
        jobs = []

        for seg_i in range(n_segs):
            t0, t1 = self.bounds[seg_i][1], self.bounds[seg_i + 1][1]
            i0, i1 = int(t0 * self.fs), int(t1 * self.fs)

            # For each selected channel, slice the boundary segment and chunk it
            for ch in ch_ids:
                data = self.X[ch, i0:i1]
                n_ck = (i1 - i0) // csz  # number of whole chunks

                for k in range(n_ck):
                    snip = data[k * csz:(k + 1) * csz]
                    if snip.size < self.scales.max():
                        continue

                    # Segment start time for this chunk (seconds)
                    jobs.append((ch, snip, t0 + k * self.chunk_s))

        # Progress dialog counts jobs
        prog = QProgressDialog("Running MF-DFA…", "Cancel", 0, max(len(jobs), 1), self)
        prog.show()

        def _compute_one(ch, snip, st):
            """
            Compute MF-DFA for a single (channel, chunk).
            Returns a dict storing both scalars and arrays needed for saving.
            """
            H, F, Hq, _ = run_mfdfa(snip, self.scales, self.q_vals, self.m_order)
            return dict(
                ch=self.chnames[ch],
                seg_start=st,
                seg_end=st + self.chunk_s,
                F=F,
                Hq=Hq,
                H=H,
                scales=self.scales.copy(),
                q_vals=self.q_vals.copy(),
            )

        results = []
        if jobs:
            if _HAVE_JOBLIB:
                # Parallel run using threads (good for numpy-heavy work depending on BLAS/GIL)
                out = Parallel(n_jobs=-1, prefer="threads")(
                    delayed(_compute_one)(ch, snip, st) for ch, snip, st in jobs
                )
                results.extend(out)
                prog.setValue(len(jobs))
            else:
                # Sequential fallback
                for idx, (ch, snip, st) in enumerate(jobs, 1):
                    results.append(_compute_one(ch, snip, st))
                    prog.setValue(idx)
                    prog.repaint()

        prog.close()

        self.batch_results.extend(results)
        self.lbl_stat.setText(f"Batch run finished • {len(self.batch_results)} chunks")

    def _save_all(self):
        """
        Save all batch results to disk in a structured folder layout, plus a summary sheet.

        Layout:
          <dest_root>/
            DFA_<start>-<end>_c<chunk>s/
              Channel_<channel_name_sanitized>/
                chunk1_<t0>-<t1>/
                  Hurst.csv
                  F_vs_scale.csv
                  Hq_vs_q.csv
                chunk2_...
              Hurst_Summary.xlsx  (or CSV fallback)

        Notes:
          - Summary file contains ONE row per chunk (Channel, ChunkIndex, SegStart, SegEnd, H).
          - Channel directory name replaces ":" with "_" to avoid path issues.
        """
        if not self.batch_results:
            QMessageBox.information(self, "Nothing to save", "Run batch first.")
            return

        dest_root = QFileDialog.getExistingDirectory(self, "Select save location")
        if not dest_root:
            return

        # Top output directory for this run (named like DFA_10.00-20.00_c1.0s)
        top_dir = os.path.join(
            dest_root,
            f"DFA_{self.start_t:.2f}-{self.end_t:.2f}_c{self.chunk_s}s"
        )
        os.makedirs(top_dir, exist_ok=True)

        # Group results by channel name (string)
        from collections import defaultdict
        by_channel = defaultdict(list)
        for r in self.batch_results:
            by_channel[r["ch"]].append(r)

        # Accumulate summary rows (scalar H only)
        summary_rows = []

        # Write per-channel directories and per-chunk CSV outputs
        for ch, results in by_channel.items():
            # Sort chunks by time for consistency
            results.sort(key=lambda x: (float(x["seg_start"]), float(x["seg_end"])))

            # Sanitize channel name for folder creation
            ch_dir = os.path.join(top_dir, f"Channel_{ch.replace(':', '_')}")
            os.makedirs(ch_dir, exist_ok=True)

            for idx, r in enumerate(results, 1):
                seg_start = float(r["seg_start"])
                seg_end = float(r["seg_end"])

                # Create chunk folder like chunk3_12.50-13.50
                ck_dir = os.path.join(ch_dir, f"chunk{idx}_{seg_start:.2f}-{seg_end:.2f}")
                os.makedirs(ck_dir, exist_ok=True)

                # Save scalar H (Hurst)
                np.savetxt(os.path.join(ck_dir, "Hurst.csv"), [r["H"]], delimiter=",")

                # Save fluctuation function vs scale
                np.savetxt(
                    os.path.join(ck_dir, "F_vs_scale.csv"),
                    np.column_stack([r["scales"], r["F"]]),
                    delimiter=",",
                    header="scale,F",
                    comments=""
                )

                # Save H(q) vs q
                np.savetxt(
                    os.path.join(ck_dir, "Hq_vs_q.csv"),
                    np.column_stack([r["q_vals"], r["Hq"]]),
                    delimiter=",",
                    header="q,Hq",
                    comments=""
                )

                # Summary row (one per chunk)
                summary_rows.append({
                    "Channel": ch,
                    "ChunkIndex": idx,     # 1-based index within channel
                    "SegStart": seg_start,
                    "SegEnd": seg_end,
                    "H": float(r["H"]),
                })

        # Write summary: prefer Excel (single sheet), fallback to CSV
        excel_path = None
        try:
            import pandas as pd
            df = pd.DataFrame(summary_rows)

            # Stable sort to preserve chunk ordering within channel
            df.sort_values(["Channel", "ChunkIndex", "SegStart"], inplace=True, kind="mergesort")

            excel_path = os.path.join(top_dir, "Hurst_Summary.xlsx")
            df.to_excel(excel_path, sheet_name="Summary", index=False)
        except Exception as e:
            # If Excel writer engine isn't available, fallback to a plain CSV.
            try:
                import csv
                csv_path = os.path.join(top_dir, "Hurst_Summary.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["Channel", "ChunkIndex", "SegStart", "SegEnd", "H"]
                    )
                    writer.writeheader()
                    for row in summary_rows:
                        writer.writerow(row)
                excel_path = csv_path
            except Exception:
                QMessageBox.warning(
                    self,
                    "Summary save failed",
                    f"Could not write Excel/CSV summary:\n{e}"
                )

        # Update UI status
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
