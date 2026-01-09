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

# Parallel backend
try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except Exception:
    _HAVE_JOBLIB = False

# Import fast core
from utils.fast_mfdfa import run_mfdfa_fast as run_mfdfa


class MFDFAAnalysisWindow(QDialog):
    def __init__(self, parent, signals, time_array,
                 start_time, end_time, channel_names,
                 labeled_times=None):
        super().__init__(parent)
        self.setWindowTitle("MF-DFA Analysis (Fast)")
        self.setWindowState(Qt.WindowMaximized)
        # self.save_mode = "by_channel"  

        # Data
        self.X       = signals
        self.t       = time_array
        self.fs      = 1000
        self.start_t = start_time
        self.end_t   = end_time
        self.chnames = channel_names

        # Boundaries
        self.bounds = self._build_bounds(labeled_times)

        # Defaults
        self.scales   = np.array([4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
        self.q_vals   = np.array([-5, -3, -2, -1, 0, 1, 2, 3, 5])
        self.m_order  = 1
        self.chunk_s  = 1.0

        # State
        self.single_chunks = []
        self.chunk_idx     = 0
        self.batch_results = []

        # UI
        self._make_ui()

    # ---------------- helpers ----------------
    def _build_bounds(self, labeled):
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

        # ----- single-channel -----
        top = QVBoxLayout(); root.addLayout(top, stretch=2)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Channel:"))
        self.dd_chan = QComboBox(); self.dd_chan.addItems(self.chnames)
        bar.addWidget(self.dd_chan)

        self.btn_run1 = QPushButton("Run"); bar.addWidget(self.btn_run1)
        self.btn_run1.clicked.connect(self._run_single)

        self.btn_prev = QPushButton("◀")
        self.btn_next = QPushButton("▶")
        self.btn_prev.clicked.connect(lambda: self._jump_chunk(-1))
        self.btn_next.clicked.connect(lambda: self._jump_chunk(1))
        self.btn_prev.setEnabled(False); self.btn_next.setEnabled(False)
        self.lbl_nav = QLabel("")
        bar.addWidget(self.btn_prev); bar.addWidget(self.btn_next); bar.addWidget(self.lbl_nav)
        top.addLayout(bar)

        g_par = QGroupBox("MF-DFA parameters")
        gl = QGridLayout(g_par)
        self.ed_sc  = QLineEdit(" ".join(map(str, self.scales)))
        self.ed_q   = QLineEdit(" ".join(map(str, self.q_vals)))
        self.ed_m   = QLineEdit(str(self.m_order))
        self.ed_csz = QLineEdit(str(self.chunk_s))
        gl.addWidget(QLabel("Scales"), 0, 0); gl.addWidget(self.ed_sc, 0, 1)
        gl.addWidget(QLabel("q vals"), 0, 2); gl.addWidget(self.ed_q, 0, 3)
        gl.addWidget(QLabel("m"),      1, 0); gl.addWidget(self.ed_m, 1, 1)
        gl.addWidget(QLabel("Chunk s"),1, 2); gl.addWidget(self.ed_csz, 1, 3)
        top.addWidget(g_par)

        self.grid_plot = QGridLayout(); top.addLayout(self.grid_plot, stretch=1)

        savebar = QHBoxLayout()
        self.btn_save_png = QPushButton("Save PNG")
        self.btn_save_csv = QPushButton("Save CSV")
        for b in (self.btn_save_png, self.btn_save_csv): b.setEnabled(False)
        self.btn_save_png.clicked.connect(self._save_png_single)
        self.btn_save_csv.clicked.connect(self._save_csv_single)
        savebar.addWidget(self.btn_save_png); savebar.addWidget(self.btn_save_csv)
        top.addLayout(savebar)

        # ----- batch -----
        bot = QVBoxLayout(); root.addLayout(bot, stretch=1)
        g_cl = QGroupBox("Channels for run-all")
        vcl = QVBoxLayout(g_cl)
        sc = QScrollArea(); sc.setWidgetResizable(True)
        frame = QWidget()
        from gui.flow_layout import FlowLayout
        fl = FlowLayout(frame)
        self.chk_map = {}
        for n in self.chnames:
            cb = QCheckBox(n); cb.setChecked(True)
            cb.setChecked(not (n.startswith("EKG") or n.startswith("X1 DC")))
            self.chk_map[n] = cb
            fl.addWidget(cb)
        sc.setWidget(frame); vcl.addWidget(sc); bot.addWidget(g_cl)

        hb = QHBoxLayout()
        self.btn_run_all = QPushButton("Run All")
        self.btn_save_all = QPushButton("Save All")
        self.lbl_stat = QLabel("")
        hb.addWidget(self.btn_run_all); hb.addWidget(self.btn_save_all); hb.addWidget(self.lbl_stat)
        self.btn_run_all.clicked.connect(self._run_all)
        self.btn_save_all.clicked.connect(self._save_all)
        bot.addLayout(hb)

    # ---------------- single ----------------
    def _run_single(self):
        try:
            self.scales  = np.array(list(map(int, self.ed_sc.text().split())))
            self.q_vals  = np.array(list(map(float, self.ed_q.text().split())))
            self.m_order = int(self.ed_m.text())
            self.chunk_s = float(self.ed_csz.text())
        except Exception:
            QMessageBox.warning(self, "Bad params", "Check scale/q/m/chunk inputs."); return

        ch = self.dd_chan.currentIndex()
        self.single_chunks.clear()
        p_segs = len(self.bounds)-1
        dlg = QProgressDialog("MF-DFA…", "Cancel", 0, p_segs, self); dlg.show()

        for i in range(p_segs):
            t0, t1 = self.bounds[i][1], self.bounds[i+1][1]
            idx0, idx1 = int(t0*self.fs), int(t1*self.fs)
            snippet = self.X[ch, idx0:idx1]
            csz = int(self.chunk_s*self.fs)
            for k in range(len(snippet)//csz):
                seg = snippet[k*csz:(k+1)*csz]
                if seg.size < self.scales.max(): continue
                H,F,Hq,Fq = run_mfdfa(seg, self.scales, self.q_vals, self.m_order)
                self.single_chunks.append(dict(H=H,F=F,Hq=Hq))
            dlg.setValue(i+1); dlg.repaint()
            if dlg.wasCanceled(): return

        if not self.single_chunks:
            QMessageBox.information(self,"None","No valid chunks.");return
        self.chunk_idx=0; self._refresh_single_plot()
        self.btn_save_png.setEnabled(True); self.btn_save_csv.setEnabled(True)
        self.btn_prev.setEnabled(True); self.btn_next.setEnabled(True)

    def _refresh_single_plot(self):
        while self.grid_plot.count():
            w=self.grid_plot.takeAt(0).widget()
            if w: w.setParent(None)
        rec=self.single_chunks[self.chunk_idx]
        fig=plt.Figure(figsize=(7,4),dpi=100)
        ax=fig.add_subplot(211); ax.plot(np.log2(self.scales), np.log2(rec["F"]),"o-")
        ax.set_xlabel("log2(scale)"); ax.set_ylabel("log2(F)")
        ax2=fig.add_subplot(212); ax2.plot(self.q_vals, rec["Hq"],"s-")
        ax2.set_xlabel("q"); ax2.set_ylabel("Hq")
        fig.tight_layout()
        self.canvas=FigureCanvas(fig)
        self.grid_plot.addWidget(self.canvas,0,0)
        self.lbl_nav.setText(f"Chunk {self.chunk_idx+1}/{len(self.single_chunks)}")

    def _jump_chunk(self, step):
        if not self.single_chunks: return
        self.chunk_idx=(self.chunk_idx+step)%len(self.single_chunks)
        self._refresh_single_plot()

    def _save_png_single(self):
        fn,_=QFileDialog.getSaveFileName(self,"PNG",f"chunk{self.chunk_idx+1}.png","PNG Files (*.png)")
        if fn: self.canvas.figure.savefig(fn)

    def _save_csv_single(self):
        rec=self.single_chunks[self.chunk_idx]
        folder=QFileDialog.getExistingDirectory(self,"Save CSVs for this chunk into…","")
        if not folder: return
        np.savetxt(os.path.join(folder,"F_vs_scale.csv"),np.column_stack([self.scales,rec["F"]]),
                   delimiter=",",header="scale,F",comments="")
        np.savetxt(os.path.join(folder,"Hq_vs_q.csv"),np.column_stack([self.q_vals,rec["Hq"]]),
                   delimiter=",",header="q,Hq",comments="")

    # ---------------- batch ----------------
    def _run_all(self):
        try:
            self.scales=np.array(list(map(int,self.ed_sc.text().split())))
            self.q_vals=np.array(list(map(float,self.ed_q.text().split())))
            self.m_order=int(self.ed_m.text()); self.chunk_s=float(self.ed_csz.text())
        except ValueError:
            QMessageBox.warning(self,"Bad parameters","Check inputs.");return

        ch_ids=[i for i,n in enumerate(self.chnames) if self.chk_map[n].isChecked()]
        if not ch_ids: QMessageBox.warning(self,"Select channels","Tick at least one.");return
        self.batch_results.clear()

        n_segs=len(self.bounds)-1; csz=int(self.chunk_s*self.fs); jobs=[]
        for seg_i in range(n_segs):
            t0,t1=self.bounds[seg_i][1],self.bounds[seg_i+1][1]
            i0,i1=int(t0*self.fs),int(t1*self.fs)
            for ch in ch_ids:
                data=self.X[ch,i0:i1]; n_ck=(i1-i0)//csz
                for k in range(n_ck):
                    snip=data[k*csz:(k+1)*csz]
                    if snip.size<self.scales.max(): continue
                    jobs.append((ch,snip,t0+k*self.chunk_s))

        prog=QProgressDialog("Running MF-DFA…","Cancel",0,max(len(jobs),1),self); prog.show()
        def _compute_one(ch,snip,st):
            H,F,Hq,_=run_mfdfa(snip,self.scales,self.q_vals,self.m_order)
            return dict(ch=self.chnames[ch], seg_start=st, seg_end=st+self.chunk_s,
                        F=F,Hq=Hq,H=H,scales=self.scales.copy(),q_vals=self.q_vals.copy())

        results=[]
        if jobs:
            if _HAVE_JOBLIB:
                out=Parallel(n_jobs=-1,prefer="threads")(delayed(_compute_one)(ch,snip,st) for ch,snip,st in jobs)
                results.extend(out); prog.setValue(len(jobs))
            else:
                for idx,(ch,snip,st) in enumerate(jobs,1):
                    results.append(_compute_one(ch,snip,st))
                    prog.setValue(idx); prog.repaint()
        prog.close()
        self.batch_results.extend(results)
        self.lbl_stat.setText(f"Batch run finished • {len(self.batch_results)} chunks")

    def _save_all(self):
        if not self.batch_results:
            QMessageBox.information(self, "Nothing to save", "Run batch first.")
            return

        dest_root = QFileDialog.getExistingDirectory(self, "Select save location")
        if not dest_root:
            return

        top_dir = os.path.join(dest_root, f"DFA_{self.start_t:.2f}-{self.end_t:.2f}_c{self.chunk_s}s")
        os.makedirs(top_dir, exist_ok=True)
        # with open(os.path.join(top_dir, "parameters.txt"), "w") as fh:
        #     fh.write(
        #         f"scales      : {self.ed_sc.text()}\n"
        #         f"q values    : {self.ed_q.text()}\n"
        #         f"m order     : {self.ed_m.text()}\n"
        #         f"chunk (sec) : {self.chunk_s}\n"
        #         f"time window : {self.start_t:.2f}-{self.end_t:.2f}\n"
        #         "channels analysed:\n" + "\n".join(sorted(set(r["ch"] for r in self.batch_results))) + "\n"
        #     )


        from collections import defaultdict
        by_channel = defaultdict(list)
        for r in self.batch_results:
            by_channel[r["ch"]].append(r)

        # Accumulate rows for the single summary sheet (only scalar H)
        summary_rows = []

        for ch, results in by_channel.items():
            # sort chunks within the channel by time
            results.sort(key=lambda x: (float(x["seg_start"]), float(x["seg_end"])))

            ch_dir = os.path.join(top_dir, f"Channel_{ch.replace(':', '_')}")
            os.makedirs(ch_dir, exist_ok=True)

            for idx, r in enumerate(results, 1):
                seg_start = float(r["seg_start"])
                seg_end = float(r["seg_end"])

                # --- existing per-chunk outputs (unchanged) ---
                ck_dir = os.path.join(ch_dir, f"chunk{idx}_{seg_start:.2f}-{seg_end:.2f}")
                os.makedirs(ck_dir, exist_ok=True)

                np.savetxt(os.path.join(ck_dir, "Hurst.csv"), [r["H"]], delimiter=",")
                np.savetxt(
                    os.path.join(ck_dir, "F_vs_scale.csv"),
                    np.column_stack([r["scales"], r["F"]]),
                    delimiter=",",
                    header="scale,F",
                    comments=""
                )
                np.savetxt(
                    os.path.join(ck_dir, "Hq_vs_q.csv"),
                    np.column_stack([r["q_vals"], r["Hq"]]),
                    delimiter=",",
                    header="q,Hq",
                    comments=""
                )
                # ----------------------------------------------

                # Row for the summary sheet (only H)
                summary_rows.append({
                    "Channel": ch,
                    "ChunkIndex": idx,   # 1-based within channel
                    "SegStart": seg_start,
                    "SegEnd": seg_end,
                    "H": float(r["H"]),
                })

        # Write the single-sheet summary
        excel_path = None
        try:
            import pandas as pd
            df = pd.DataFrame(summary_rows)
            # Nice ordering
            df.sort_values(["Channel", "ChunkIndex", "SegStart"], inplace=True, kind="mergesort")
            excel_path = os.path.join(top_dir, "Hurst_Summary.xlsx")
            df.to_excel(excel_path, sheet_name="Summary", index=False)
        except Exception as e:
            # Fallback to CSV if pandas/engine not available
            try:
                import csv
                csv_path = os.path.join(top_dir, "Hurst_Summary.csv")
                with open(csv_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["Channel","ChunkIndex","SegStart","SegEnd","H"])
                    writer.writeheader()
                    for row in summary_rows:
                        writer.writerow(row)
                excel_path = csv_path
            except Exception:
                QMessageBox.warning(self, "Summary save failed", f"Could not write Excel/CSV summary:\n{e}")

        if excel_path:
            self.lbl_stat.setText(
                f"Saved {len(self.batch_results)} chunks → {top_dir}\nSummary: {os.path.basename(excel_path)}"
            )
        else:
            self.lbl_stat.setText(f"Saved {len(self.batch_results)} chunks → {top_dir}\nSummary: (failed)")











# def _run_all(self):
#     try:
#         self.scales  = np.array(list(map(int, self.ed_sc.text().split())))
#         self.q_vals  = np.array(list(map(float, self.ed_q.text().split())))
#         self.m_order = int(self.ed_m.text())
#         self.chunk_s = float(self.ed_csz.text())
#     except ValueError:
#         QMessageBox.warning(self, "Bad parameters", "Check inputs.")
#         return

#     ch_ids = [i for i, n in enumerate(self.chnames) if self.chk_map[n].isChecked()]
#     if not ch_ids:
#         QMessageBox.warning(self, "Select channels", "Tick at least one.")
#         return

#     self.batch_results.clear()

#     n_segs = len(self.bounds) - 1
#     csz = int(self.chunk_s * self.fs)
#     jobs = []

#     for seg_i in range(n_segs):
#         tag_left,  t0 = self.bounds[seg_i]
#         tag_right, t1 = self.bounds[seg_i + 1]
#         seg_name = f"{tag_left}-{tag_right}"     # e.g. "start-end"
#         i0, i1 = int(t0 * self.fs), int(t1 * self.fs)

#         for ch in ch_ids:
#             data = self.X[ch, i0:i1]
#             n_ck = (i1 - i0) // csz
#             for k in range(n_ck):
#                 snip = data[k * csz:(k + 1) * csz]
#                 if snip.size < self.scales.max():
#                     continue
#                 # pass segment label + segment window with the job
#                 jobs.append((ch, snip, t0 + k * self.chunk_s, t0, t1, seg_name))

#     prog = QProgressDialog("Running MF-DFA…", "Cancel", 0, max(len(jobs), 1), self)
#     prog.show()

#     def _compute_one(ch, snip, chunk_start, seg_t0, seg_t1, seg_name):
#         H, F, Hq, _ = run_mfdfa(snip, self.scales, self.q_vals, self.m_order)
#         return dict(
#             ch=self.chnames[ch],
#             seg_start=chunk_start,
#             seg_end=chunk_start + self.chunk_s,
#             seg_name=seg_name,           # new
#             seg_t0=float(seg_t0),        # new (segment window)
#             seg_t1=float(seg_t1),        # new
#             F=F, Hq=Hq, H=H,
#             scales=self.scales.copy(),
#             q_vals=self.q_vals.copy()
#         )

#     results = []
#     if jobs:
#         try:
#             from joblib import Parallel, delayed
#             out = Parallel(n_jobs=-1, prefer="threads")(
#                 delayed(_compute_one)(*j) for j in jobs
#             )
#             results.extend(out)
#             prog.setValue(len(jobs))
#         except Exception:
#             for idx, j in enumerate(jobs, 1):
#                 results.append(_compute_one(*j))
#                 prog.setValue(idx); prog.repaint()

#     prog.close()
#     self.batch_results.extend(results)
#     self.lbl_stat.setText(f"Batch run finished • {len(self.batch_results)} chunks")











# def _save_all(self):
#     if not self.batch_results:
#         QMessageBox.information(self, "Nothing to save", "Run batch first.")
#         return

#     dest_root = QFileDialog.getExistingDirectory(self, "Select save location")
#     if not dest_root:
#         return

#     top_dir = os.path.join(dest_root, f"DFA_{self.start_t:.2f}-{self.end_t:.2f}_c{self.chunk_s}s")
#     os.makedirs(top_dir, exist_ok=True)

#     # --- always write a top-level parameters.txt (handy even if commented later) ---
#     try:
#         with open(os.path.join(top_dir, "parameters.txt"), "w") as fh:
#             fh.write(
#                 f"scales      : {self.ed_sc.text()}\n"
#                 f"q values    : {self.ed_q.text()}\n"
#                 f"m order     : {self.ed_m.text()}\n"
#                 f"chunk (sec) : {self.chunk_s}\n"
#                 f"time window : {self.start_t:.2f}-{self.end_t:.2f}\n"
#             )
#     except Exception:
#         pass

#     # ---- build summary rows across all channels (one global sheet) ----
#     summary_rows = []

#     if self.save_mode == "by_segment":
#         # -------- Script-1 style hierarchy: segment → Channel_* → chunk* --------
#         from collections import defaultdict

#         # group results by (seg_name, seg_t0, seg_t1)
#         seg_groups = defaultdict(list)
#         for r in self.batch_results:
#             seg_key = (r.get("seg_name", "segment"), float(r["seg_t0"]), float(r["seg_t1"]))
#             seg_groups[seg_key].append(r)

#         total = len(self.batch_results)
#         prog = QProgressDialog("Saving chunks…", "Cancel", 0, total, self)
#         prog.setWindowModality(Qt.WindowModal); prog.show()
#         done = 0

#         for (seg_name, seg_t0, seg_t1), seg_recs in seg_groups.items():
#             seg_dir = os.path.join(top_dir, f"DFA_{seg_t0:.2f}-{seg_t1:.2f}_c{self.chunk_s}s")
#             os.makedirs(seg_dir, exist_ok=True)

#             # parameters.txt per segment (Script-1 behavior)
#             try:
#                 with open(os.path.join(seg_dir, "parameters.txt"), "w") as fh:
#                     fh.write(
#                         f"scales      : {self.ed_sc.text()}\n"
#                         f"q values    : {self.ed_q.text()}\n"
#                         f"m order     : {self.ed_m.text()}\n"
#                         f"chunk (sec) : {self.chunk_s}\n\n"
#                         f"segment tag : {seg_name}\n"
#                         f"segment time: {seg_t0:.2f}-{seg_t1:.2f}\n\n"
#                         "channels analysed:\n" +
#                         "\n".join(sorted({r['ch'] for r in seg_recs})) + "\n"
#                     )
#             except Exception:
#                 pass

#             # group by channel inside this segment
#             chan_groups = defaultdict(list)
#             for r in seg_recs:
#                 chan_groups[r["ch"]].append(r)

#             for ch, items in chan_groups.items():
#                 items.sort(key=lambda z: (float(z["seg_start"]), float(z["seg_end"])))
#                 ch_dir = os.path.join(seg_dir, f"Channel_{ch.replace(':', '_')}")
#                 os.makedirs(ch_dir, exist_ok=True)

#                 for idx, r in enumerate(items, 1):
#                     seg_start = float(r["seg_start"]); seg_end = float(r["seg_end"])
#                     ck_dir = os.path.join(ch_dir, f"chunk{idx}_{seg_start:.2f}-{seg_end:.2f}")
#                     os.makedirs(ck_dir, exist_ok=True)

#                     np.savetxt(os.path.join(ck_dir, "Hurst.csv"), [r["H"]], delimiter=",")
#                     np.savetxt(
#                         os.path.join(ck_dir, "F_vs_scale.csv"),
#                         np.column_stack([r["scales"], r["F"]]),
#                         delimiter=",", header="scale,F", comments=""
#                     )
#                     np.savetxt(
#                         os.path.join(ck_dir, "Hq_vs_q.csv"),
#                         np.column_stack([r["q_vals"], r["Hq"]]),
#                         delimiter=",", header="q,Hq", comments=""
#                     )

#                     # add to global summary
#                     summary_rows.append({
#                         "Segment": seg_name,
#                         "Channel": ch,
#                         "ChunkIndex": idx,
#                         "SegStart": seg_start,
#                         "SegEnd": seg_end,
#                         "H": float(r["H"]),
#                         "Folder": ck_dir
#                     })

#                     done += 1
#                     prog.setValue(done)
#                     if prog.wasCanceled():
#                         self.lbl_stat.setText("Save cancelled")
#                         return
#         prog.close()

#     else:
#         # -------- Script-2 style: top/Channel_* → chunk* (flat) --------
#         from collections import defaultdict
#         by_channel = defaultdict(list)
#         for r in self.batch_results:
#             by_channel[r["ch"]].append(r)

#         for ch, results in by_channel.items():
#             results.sort(key=lambda x: (float(x["seg_start"]), float(x["seg_end"])))
#             ch_dir = os.path.join(top_dir, f"Channel_{ch.replace(':', '_')}")
#             os.makedirs(ch_dir, exist_ok=True)

#             for idx, r in enumerate(results, 1):
#                 seg_start = float(r["seg_start"]); seg_end = float(r["seg_end"])
#                 ck_dir = os.path.join(ch_dir, f"chunk{idx}_{seg_start:.2f}-{seg_end:.2f}")
#                 os.makedirs(ck_dir, exist_ok=True)

#                 np.savetxt(os.path.join(ck_dir, "Hurst.csv"), [r["H"]], delimiter=",")
#                 np.savetxt(
#                     os.path.join(ck_dir, "F_vs_scale.csv"),
#                     np.column_stack([r["scales"], r["F"]]),
#                     delimiter=",", header="scale,F", comments=""
#                 )
#                 np.savetxt(
#                     os.path.join(ck_dir, "Hq_vs_q.csv"),
#                     np.column_stack([r["q_vals"], r["Hq"]]),
#                     delimiter=",", header="q,Hq", comments=""
#                 )

#                 # add to global summary
#                 summary_rows.append({
#                     "Segment": r.get("seg_name", ""),     # empty if not using labels
#                     "Channel": ch,
#                     "ChunkIndex": idx,
#                     "SegStart": seg_start,
#                     "SegEnd": seg_end,
#                     "H": float(r["H"]),
#                     "Folder": ck_dir
#                 })

#     # ---- write global summary Excel (or CSV fallback) ----
#     excel_path = None
#     try:
#         import pandas as pd
#         df = pd.DataFrame(summary_rows)
#         df.sort_values(["Segment", "Channel", "ChunkIndex", "SegStart"], inplace=True, kind="mergesort")
#         excel_path = os.path.join(top_dir, "Hurst_Summary.xlsx")
#         df.to_excel(excel_path, sheet_name="Summary", index=False)
#     except Exception as e:
#         try:
#             import csv
#             csv_path = os.path.join(top_dir, "Hurst_Summary.csv")
#             with open(csv_path, "w", newline="") as f:
#                 cols = ["Segment","Channel","ChunkIndex","SegStart","SegEnd","H","Folder"]
#                 import csv as _csv
#                 w = _csv.DictWriter(f, fieldnames=cols)
#                 w.writeheader()
#                 for row in summary_rows:
#                     w.writerow({k: row.get(k, "") for k in cols})
#             excel_path = csv_path
#         except Exception:
#             QMessageBox.warning(self, "Summary save failed", f"Could not write Excel/CSV summary:\n{e}")

#     if excel_path:
#         self.lbl_stat.setText(
#             f"Saved {len(self.batch_results)} chunks → {top_dir}\nSummary: {os.path.basename(excel_path)}"
#         )
#     else:
#         self.lbl_stat.setText(f"Saved {len(self.batch_results)} chunks → {top_dir}\nSummary: (failed)")































# # file: gui/mfdfa_analysis_window.py

# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
# from PyQt5.QtWidgets import (
#     QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QMessageBox, QSizePolicy,
#     QComboBox, QLineEdit, QGroupBox, QToolButton, QFileDialog, QWidget, QScrollArea, QGridLayout
# )
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QFont
# from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

# class CollapsiblePanel(QGroupBox):
#     def __init__(self, title="Parameters", parent=None):
#         super().__init__(parent)
#         self.setTitle("")
#         self.toggle_button = QToolButton(text=title, checkable=True, checked=False)
#         self.toggle_button.setStyleSheet("QToolButton { border: none; padding: 2px; }")
#         self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
#         self.toggle_button.setArrowType(Qt.RightArrow)
#         self.toggle_button.clicked.connect(self.on_toggle)

#         self.content = QWidget()
#         self.content.setVisible(False)
#         self.content.setContentsMargins(0, 0, 0, 0)

#         lay = QHBoxLayout()
#         lay.addWidget(self.toggle_button)
#         lay.addWidget(self.content)
#         lay.setContentsMargins(0, 0, 0, 0)
#         self.setLayout(lay)
#         self.setMaximumHeight(self.toggle_button.sizeHint().height())

#     def on_toggle(self):
#         checked = self.toggle_button.isChecked()
#         self.content.setVisible(checked)
#         self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
#         if checked:
#             self.setMaximumHeight(16777215)
#         else:
#             self.setMaximumHeight(self.toggle_button.sizeHint().height())

# def run_mfdfa(x, scales=None, q_vals=None, m=1):
#     """
#     Perform multifractal detrended fluctuation analysis on a 1D signal x.
    
#     Parameters:
#         x: 1D numpy array (the time series)
#         scales: array-like of scale values (window lengths).
#                 Default: [4,8,16,32,64,128,256,1024,2048,4096,8192]
#         q_vals: array-like of q orders.
#                 Default: [-5,-3,-2,-1,0,1,2,3,5]
#         m: order of the detrending polynomial (default 1)
        
#     Returns:
#         H: Estimated Hurst exponent (slope from log-log regression of F vs. scale)
#         F: fluctuation function array (one per scale)
#         Hq: array of Hurst exponents for each q in q_vals
#         Fq: 2D array (len(q_vals) x len(scales)) of fluctuation functions for each q.
#     """
#     if scales is None:
#         scales = np.array([4,8,16,32,64,128,256,1024,2048,4096,8192])
#     if q_vals is None:
#         q_vals = np.array([-5,-3,-2,-1,0,1,2,3,5])
        
#     x = np.array(x, dtype=float)
#     N = len(x)
#     Y = np.cumsum(x - np.mean(x))
#     F = np.zeros(len(scales))
#     Fq = np.zeros((len(q_vals), len(scales)))
    
#     for i, scale in enumerate(scales):
#         scale = int(scale)
#         segments = int(np.floor(N / scale))
#         RMS = np.zeros(segments)
#         for v in range(segments):
#             idx_start = v * scale
#             idx_stop = (v+1) * scale
#             segment_indices = np.arange(idx_start, idx_stop)
#             coeffs = np.polyfit(segment_indices, Y[idx_start:idx_stop], m)
#             fit_vals = np.polyval(coeffs, segment_indices)
#             RMS[v] = np.sqrt(np.mean((Y[idx_start:idx_stop] - fit_vals)**2))
#         F[i] = np.sqrt(np.mean(RMS**2))
#         for j, q in enumerate(q_vals):
#             if q == 0:
#                 Fq[j, i] = np.exp(0.5 * np.mean(np.log(RMS**2)))
#             else:
#                 Fq[j, i] = (np.mean(RMS**q))**(1.0/q)
#     log_scales = np.log2(scales)
#     log_F = np.log2(F)
#     slope, intercept, _, _, _ = stats.linregress(log_scales, log_F)
#     H = slope
#     Hq = np.zeros(len(q_vals))
#     for j, q in enumerate(q_vals):
#         log_Fq = np.log2(Fq[j, :])
#         slope_q, _, _, _, _ = stats.linregress(log_scales, log_Fq)
#         Hq[j] = slope_q
#     return H, F, Hq, Fq

# class MFDFAAnalysisWindow(QDialog):
#     def __init__(self, parent, signals, time_array, start_time, end_time, channel_names):
#         """
#         signals: 2D numpy array (channels, timepoints)
#         time_array: 1D numpy array (seconds)
#         start_time, end_time: floats, selected time window from second window
#         channel_names: list of channel names from H5 file
#         """
#         super().__init__(parent)
#         self.setWindowTitle("MF-DFA Analysis Window")
#         self.setWindowState(Qt.WindowMaximized)
#         self.signals = signals
#         self.time_array = time_array
#         self.fs = 1000
#         self.start_time = start_time
#         self.end_time = end_time
#         self.all_channel_names = channel_names

#         # MF-DFA will be run on one channel at a time.
#         self.selected_channel = None

#         # Default MF-DFA parameters
#         self.scales = np.array([4, 8, 16, 32, 64, 128, 256, 1024, 2048, 4096, 8192])
#         self.q_vals = np.array([-5, -3, -2, -1, 0, 1, 2, 3, 5])
#         self.m_order = 1

#         # Results: H, F, Hq, Fq
#         self.mfdfa_results = None

#         self.initUI()

#     def initUI(self):
#         main_layout = QVBoxLayout(self)
#         main_layout.setSpacing(5)
#         main_layout.setContentsMargins(5, 5, 5, 5)

#         top_row = QHBoxLayout()
#         top_row.setSpacing(5)
#         self.param_panel = CollapsiblePanel("MF-DFA Parameters")
#         param_layout = QHBoxLayout(self.param_panel.content)
#         param_layout.setContentsMargins(0, 0, 0, 0)
#         lbl_scales = QLabel("Scales:")
#         self.edit_scales = QLineEdit(" ".join(map(str, self.scales)))
#         self.edit_scales.setMaximumWidth(200)
#         lbl_q = QLabel("q values:")
#         self.edit_q = QLineEdit(" ".join(map(str, self.q_vals)))
#         self.edit_q.setMaximumWidth(200)
#         lbl_m = QLabel("m (order):")
#         self.edit_m = QLineEdit(str(self.m_order))
#         self.edit_m.setMaximumWidth(50)
#         for widget in [lbl_scales, self.edit_scales, lbl_q, self.edit_q, lbl_m, self.edit_m]:
#             widget.setFont(QFont("", 8))
#         param_layout.addWidget(lbl_scales)
#         param_layout.addWidget(self.edit_scales)
#         param_layout.addWidget(lbl_q)
#         param_layout.addWidget(self.edit_q)
#         param_layout.addWidget(lbl_m)
#         param_layout.addWidget(self.edit_m)
#         param_layout.addStretch()
#         top_row.addWidget(self.param_panel)

#         # Channel Selector (dropdown)
#         channel_box = QGroupBox("Select Channel")
#         channel_layout = QHBoxLayout(channel_box)
#         self.channel_dropdown = QComboBox()
#         for name in self.all_channel_names:
#             self.channel_dropdown.addItem(name)
#         channel_layout.addWidget(self.channel_dropdown)
#         top_row.addWidget(channel_box)
#         main_layout.addLayout(top_row)

#         self.btn_run = QPushButton("Run MF-DFA Analysis")
#         self.btn_run.setMaximumHeight(30)
#         self.btn_run.clicked.connect(self.run_mfdfa)
#         main_layout.addWidget(self.btn_run)

#         self.plot_container = QWidget()
#         self.plot_layout = QVBoxLayout(self.plot_container)
#         main_layout.addWidget(self.plot_container)

#         save_layout = QHBoxLayout()
#         self.btn_save_plot = QPushButton("Save Plot")
#         self.btn_save_csv = QPushButton("Save CSV")
#         self.btn_save_plot.setMaximumHeight(30)
#         self.btn_save_csv.setMaximumHeight(30)
#         save_layout.addWidget(self.btn_save_plot)
#         save_layout.addWidget(self.btn_save_csv)
#         main_layout.addLayout(save_layout)
#         self.btn_save_plot.clicked.connect(lambda: self.save_plot("png"))
#         self.btn_save_csv.clicked.connect(lambda: self.save_plot("csv", data=True))

#         self.results_label = QLabel("")
#         self.results_label.setFont(QFont("", 9))
#         main_layout.addWidget(self.results_label)

#     def run_mfdfa(self):
#         # Get parameters from input fields
#         try:
#             scales_str = self.edit_scales.text().strip()
#             self.scales = np.array([float(s) for s in scales_str.split()])
#             q_str = self.edit_q.text().strip()
#             self.q_vals = np.array([float(s) for s in q_str.split()])
#             self.m_order = int(self.edit_m.text())
#         except Exception as e:
#             QMessageBox.warning(self, "Parameter Error", "Please enter valid MF-DFA parameters.")
#             return

#         # Get selected channel from dropdown.
#         self.selected_channel = self.channel_dropdown.currentText()
#         try:
#             ch_index = self.all_channel_names.index(self.selected_channel)
#         except ValueError:
#             QMessageBox.warning(self, "Channel Error", "Selected channel not found.")
#             return

#         start_idx = int(self.start_time * self.fs)
#         end_idx = int(self.end_time * self.fs)
#         data_segment = self.signals[ch_index, start_idx:end_idx]

#         try:
#             H, F, Hq, Fq = run_mfdfa(data_segment, scales=self.scales, q_vals=self.q_vals, m=self.m_order)
#             self.mfdfa_results = {"H": H, "F": F, "Hq": Hq, "Fq": Fq}
#             self.results_label.setText(f"Hurst Exponent (H): {H:.3f}\nMultifractal exponents (Hq): {Hq}")
#         except Exception as e:
#             QMessageBox.critical(self, "MF-DFA Analysis Error", str(e))
#             return

#         # Plot DFA regression: log2(scales) vs. log2(F) with regression line.
#         log_scales = np.log2(self.scales)
#         log_F = np.log2(F)
#         slope, intercept, _, _, _ = stats.linregress(log_scales, log_F)
#         reg_line = intercept + slope * log_scales

#         if hasattr(self, "canvas") and self.canvas is not None:
#             self.plot_layout.removeWidget(self.canvas)
#             self.canvas.setParent(None)
#         fig = plt.Figure(figsize=(8,5), dpi=100)
#         ax = fig.add_subplot(111)
#         ax.plot(log_scales, log_F, 'o', label="Data")
#         ax.plot(log_scales, reg_line, 'r-', label=f"Fit (slope={slope:.3f})")
#         ax.set_xlabel("log2(scale)")
#         ax.set_ylabel("log2(F)")
#         ax.set_title("DFA Regression")
#         ax.legend()
#         fig.tight_layout()
#         self.canvas = FigureCanvas(fig)
#         self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.plot_layout.addWidget(self.canvas)

#     def save_plot(self, fmt, data=False):
#         import time
#         suggested_name = f"MF-DFA_plot_{int(time.time())}.{fmt}"
#         fileName, _ = QFileDialog.getSaveFileName(self, "Save Plot", suggested_name,
#                                                   f"{fmt.upper()} Files (*.{fmt});;All Files (*)")
#         if fileName:
#             if data and self.mfdfa_results is not None:
#                 # Save the fluctuation function F and the multifractal exponents Hq as CSV.
#                 np.savetxt(fileName.replace(f".{fmt}", "_F.csv"), self.mfdfa_results["F"], delimiter=",")
#                 np.savetxt(fileName.replace(f".{fmt}", "_Hq.csv"), self.mfdfa_results["Hq"], delimiter=",")
#             else:
#                 if hasattr(self, "canvas") and self.canvas is not None:
#                     self.canvas.figure.savefig(fileName)

# if __name__ == "__main__":
#     from PyQt5.QtWidgets import QApplication
#     from scipy import stats
#     app = QApplication(sys.argv)
#     # Create fake data for testing: 64 channels, 60 seconds at 1000 Hz.
#     fake_signals = np.random.randn(64, 60*1000)
#     fake_time = np.linspace(0, 60, 60*1000)
#     fake_channel_names = [f"Channel {i}" for i in range(64)]
#     win = MFDFAAnalysisWindow(None, fake_signals, fake_time, 10, 20, fake_channel_names)
#     win.show()
#     sys.exit(app.exec_())
