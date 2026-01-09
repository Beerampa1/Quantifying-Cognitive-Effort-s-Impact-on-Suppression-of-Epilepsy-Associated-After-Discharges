# fs_dfa_viewer_c05_summary_by_outcome_trailwise.py
# Folder-based viewer (no DB).
# Plots:
#   • H box-plot per chunk (x = SegStart; dashed line at first W2 of each trail)
#   • H box-plot per channel
# Summary button (per your spec):
#   • Do Success and Fail SEPARATELY (no mixing)
#   • For each outcome:
#       - For EACH TRAIL: Top-10 LOWEST Var(H) channels (computed within that trail)
#       - ACROSS trails: channels common to ALL trails
#       - Rank channels by # of trails' Top-10 membership
#   • Finally: channels common in BOTH outcomes' "common-to-all-trails" sets
#
# Requirements: PyQt5, pandas, openpyxl, matplotlib, numpy

import os, re, sys, numpy as np, pandas as pd
from typing import Optional, Tuple, List, Dict
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QComboBox, QCheckBox, QSpinBox, QMessageBox, QSizePolicy,
    QPlainTextEdit
)
from PyQt5.QtCore import Qt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import FuncFormatter  # NEW

H_MIN = 0.5  # drop H <= this (ignore tiny/invalid H)

# ── path → metadata ──────────────────────────────────────────────────────
TRAIL_REGEXES = [
    r"\bP\d+_S\d+[a-z]?_(?:T|P)\d+\b",
    r"\bP\d+_ST\d+_(?:T|P)\d+\b",
]
def find_trail(path: str) -> Optional[str]:
    for rgx in TRAIL_REGEXES:
        m = re.search(rgx, path, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None

def outcome_window(path: str) -> Tuple[str, str]:
    s = path.replace("\\", "/").lower()
    if re.search(r"(?<![a-z])fail(?![a-z])", s):
        outc = "Fail"
    elif re.search(r"(?<![a-z])(success|sucess|succ)(?![a-z])", s):
        outc = "Success"
    else:
        outc = "Unknown"
    if re.search(r"(^|[\/_\-])w1($|[\/_\-])", s):
        win = "W1"
    elif re.search(r"(^|[\/_\-])w2($|[\/_\-])", s):
        win = "W2"
    else:
        win = "Unknown"
    return outc, win

def is_c05(path: str) -> bool:
    return bool(re.search(r"_c0\.?5s\b", path.lower()))

# ── Excel reading (robust to header variants) ────────────────────────────
def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "h" not in df.columns:
        for alt in ["hurst", "hurstvalue", "h "]:
            if alt in df.columns:
                df.rename(columns={alt: "h"}, inplace=True); break
    if "channel" not in df.columns:
        for alt in ["chan", "ch", "channel ", "channels"]:
            if alt in df.columns:
                df.rename(columns={alt: "channel"}, inplace=True); break
    if "chunkindex" not in df.columns:
        for alt in ["chunk_index", "index", "idx"]:
            if alt in df.columns:
                df.rename(columns={alt: "chunkindex"}, inplace=True); break
    if "segstart" not in df.columns:
        for alt in ["start", "t0", "start_sec", "segment_start", "chunk_start", "seg start"]:
            if alt in df.columns:
                df.rename(columns={alt: "segstart"}, inplace=True); break
    if "segend" not in df.columns:
        for alt in ["end", "t1", "end_sec", "segment_end", "chunk_end", "seg end"]:
            if alt in df.columns:
                df.rename(columns={alt: "segend"}, inplace=True); break
    return df

def read_summary_df(path: str) -> Optional[pd.DataFrame]:
    try:
        xls = pd.ExcelFile(path)
        sheet = next((s for s in xls.sheet_names if s.lower()=="summary"), xls.sheet_names[0])
        df = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    except Exception:
        return None
    df = normalize_headers(df)
    if "h" not in df.columns or "channel" not in df.columns:
        return None
    out = pd.DataFrame()
    out["Channel"]     = df["channel"].astype(str).str.strip()
    out["H"]           = pd.to_numeric(df["h"], errors="coerce")
    out["chunk_start"] = pd.to_numeric(df.get("segstart", np.nan), errors="coerce")
    out = out.dropna(subset=["H", "Channel"])
    return out

# ── collect rows from folder tree (c0.5s only) ──────────────────────────
def collect(base: str) -> pd.DataFrame:
    rows: List[dict] = []
    for root, _, files in os.walk(base):
        for f in files:
            if f.lower() != "hurst_summary.xlsx":
                continue
            full = os.path.join(root, f)
            if not is_c05(full):
                continue
            sdf = read_summary_df(full)
            if sdf is None or sdf.empty:
                continue
            outc, win = outcome_window(full)
            trail     = find_trail(full)
            for _, r in sdf.iterrows():
                cs = r.get("chunk_start", np.nan)
                if not np.isfinite(cs):
                    continue
                h = float(r["H"])
                if not np.isfinite(h) or h <= H_MIN:
                    continue
                rows.append({
                    "Source": full,
                    "Outcome": outc,      # "Success" / "Fail" / "Unknown"
                    "Window": win,        # "W1" / "W2" / "Unknown"
                    "Trail": trail,       # e.g., P1_S5_T1
                    "Channel": str(r["Channel"]),
                    "H": h,
                    "chunk_start": float(cs),
                })
    return pd.DataFrame(rows)

# ── plotting helpers ────────────────────────────────────────────────────
def _nice_box_width(xs: np.ndarray) -> float:
    if xs.size <= 1:
        return 0.4
    diffs = np.diff(np.sort(xs))
    w = 0.8 * np.median(diffs)
    return float(np.clip(w, 0.05, 5.0))

def _var_per_channel(df: pd.DataFrame) -> pd.DataFrame:
    """Per-channel sample variance of H inside df."""
    if df.empty:
        return pd.DataFrame(columns=["Channel", "Var(H)"])
    g = df.groupby("Channel")["H"]
    out = pd.DataFrame({"Var(H)": g.var(ddof=1)}).dropna().reset_index()
    return out

# ── GUI ─────────────────────────────────────────────────────────────────
class Viewer(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DFA viewer — c0.5s (per chunk • per channel • outcome-wise summary)")
        self.setWindowState(Qt.WindowMaximized)

        self.df = pd.DataFrame()

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        btn = QPushButton("Open folder…"); btn.clicked.connect(self._open_folder)
        top.addWidget(btn)

        top.addWidget(QLabel("Trail:"))
        self.dd_trail = QComboBox(); top.addWidget(self.dd_trail, 1)

        top.addWidget(QLabel("View:"))
        self.dd_plot = QComboBox()
        self.dd_plot.addItems([
            "H box-plot per chunk",
            "H box-plot per channel",
        ])
        top.addWidget(self.dd_plot, 1)

        self.chk_w2 = QCheckBox("dotted W2 start(s)"); self.chk_w2.setChecked(True)
        top.addWidget(self.chk_w2)

        draw = QPushButton("Draw"); draw.clicked.connect(self._draw)
        top.addWidget(draw)

        # Summary button (OUTCOME-SEPARATE, TRAIL-WISE → CROSS-TRAIL)
        btn_sum = QPushButton("Summary…")
        btn_sum.clicked.connect(self._run_summary_outcome_trailwise)
        top.addWidget(btn_sum)

        # NEW: Save PNG button
        btn_save = QPushButton("Save PNG…")   # NEW
        btn_save.clicked.connect(self._save_png)  # NEW
        top.addWidget(btn_save)               # NEW

        root.addLayout(top)

        # CHANGED: Paper-friendly default figure size; crisp canvas dpi
        self.fig = plt.Figure(figsize=(7.2, 3.6), dpi=150)  # CHANGED
        self.ax  = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        root.addWidget(self.canvas, 1)

        self.lbl = QLabel("Choose a folder to begin"); root.addWidget(self.lbl)
        self.dd_trail.currentIndexChanged.connect(self._draw)
        self.dd_plot.currentIndexChanged.connect(self._draw)

    # ── folder -> load
    def _open_folder(self):
        base = QFileDialog.getExistingDirectory(self, "Choose base folder", "")
        if not base: return
        self.df = collect(base)

        if self.df.empty:
            QMessageBox.information(self, "No data",
                "Found no usable _c0.5s hurst_summary.xlsx rows with SegStart (chunk start).")
            self.dd_trail.clear(); self.fig.clf(); self.canvas.draw(); return

        trails = sorted([t for t in self.df["Trail"].dropna().unique()])
        self.dd_trail.clear(); self.dd_trail.addItem("<all>")
        for t in trails: self.dd_trail.addItem(t)

        nW1 = int((self.df["Window"]=="W1").sum())
        nW2 = int((self.df["Window"]=="W2").sum())
        nChunks = len(self.df["chunk_start"].unique())
        self.lbl.setText(f"Rows={len(self.df)}  •  W1={nW1}  •  W2={nW2}  •  Trails={len(trails)}  •  Chunks={nChunks}")
        self._draw()

    # ── dispatcher
    def _draw(self):
        if self.df.empty: return
        trail = self.dd_trail.currentText()
        d = self.df if trail in ("", None, "<all>") else self.df[self.df["Trail"]==trail]

        self.fig.clf(); self.ax = self.fig.add_subplot(111)
        what = self.dd_plot.currentText()
        if what.startswith("H box-plot per chunk"):
            self._plot_per_chunk(d)
        else:
            self._plot_per_channel(d)
        self.fig.tight_layout(); self.canvas.draw_idle()

    # ── plots
    def _plot_per_chunk(self, d: pd.DataFrame):

        sub = d.dropna(subset=["chunk_start"]).copy()
        if sub.empty:
            self.ax.text(0.5,0.5,"No SegStart values available in Summary sheets.",
                        ha="center", va="center", transform=self.ax.transAxes)
            return

        g = sub.groupby("chunk_start", dropna=True)
        arrays, xpos, windows = [], [], []
        for cs, gg in sorted(g, key=lambda t: float(t[0])):
            vals = gg["H"].to_numpy(float)
            if vals.size == 0:
                continue
            arrays.append(vals)
            xpos.append(float(cs))

            # Window label
            win = gg["Window"].iloc[0] if len(gg) > 0 else "Unknown"
            windows.append(win)

        if not arrays:
            self.ax.text(0.5,0.5,"No per-chunk groups with H values.",
                        ha="center", va="center", transform=self.ax.transAxes)
            return

        xpos = np.asarray(xpos, float)
        w = _nice_box_width(xpos)

        bp = self.ax.boxplot(
            arrays, positions=xpos, widths=w, patch_artist=True,
            flierprops={'markersize': 3, 'alpha': .35},
            boxprops=dict(lw=1.4),
            whiskerprops=dict(lw=1.2),
            capprops=dict(lw=1.2),
            medianprops=dict(color='black', lw=1.8)
        )

        # ---- W1 / W2 Color Coding ----
        w1_color = '#FFFFE0'  # Light yellow
        w2_color = '#E6D5E6'  # Light purple

        for i, window in enumerate(windows):
            color = (
                w1_color if window == "W1"
                else w2_color if window == "W2"
                else '#FFFFFF'
            )
            bp['boxes'][i].set(facecolor=color)

        # ---- W2 boundary markers ----
        if self.chk_w2.isChecked():
            w2_lines = []
            for tr, gg in sub.groupby("Trail"):
                w2starts = gg.loc[gg["Window"]=="W2", "chunk_start"].dropna()
                if not w2starts.empty:
                    w2_lines.append(float(w2starts.min()))

            for x in sorted(set(w2_lines)):
                self.ax.axvline(x, color='black', ls='--', lw=1.0, alpha=.9)

        # ---- Axis Labels ----
        self.ax.set_xlabel("Time (seconds)", fontsize=18, fontweight='bold')
        self.ax.set_ylabel("H values", fontsize=18, fontweight='bold')

        self.ax.set_ylim(0, 2)
        self.ax.set_xlim(xpos.min() - w, xpos.max() + w)

        self.ax.set_title(
            "Temporal Evolution of Hurst Exponents",
            fontsize=24,
            fontweight='bold',
            pad=15
        )

        # ---- Grid ----
        self.ax.grid(True, axis='y', ls='--', alpha=.35)

        # ---- X Tick Formatting ----
        self.ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, pos: f"{x:.2f}")
        )

        # Bigger tick labels
        self.ax.tick_params(axis='x', rotation=90, labelsize=14)
        self.ax.tick_params(axis='y', labelsize=14)


    def _plot_per_channel(self, d: pd.DataFrame):
        g = d.groupby("Channel")
        data, labels = [], []
        for ch, gg in sorted(g, key=lambda t: (str(t[0]).lower() if t[0] is not None else "")):
            vals = gg["H"].to_numpy(float)
            if vals.size == 0: continue
            data.append(vals)
            labels.append(ch if ch else "<blank>")

        if not data:
            self.ax.text(0.5,0.5,"No channel data.",ha="center",va="center",
                         transform=self.ax.transAxes); return

        bp = self.ax.boxplot(
            data, patch_artist=True,
            flierprops={'markersize':2.5,'alpha':.35},
            boxprops=dict(lw=1.4), whiskerprops=dict(lw=1.2),
            capprops=dict(lw=1.2), medianprops=dict(color='black', lw=1.8)
        )
        
        # Color boxes uniformly (light gray)
        for box in bp['boxes']:
            box.set(facecolor='#E8E8E8')

        self.ax.set_xticks(np.arange(1, len(labels)+1))
        self.ax.set_xticklabels(labels, rotation=90, fontsize=14)
        self.ax.set_ylabel("H"); self.ax.set_ylim(0,2)
        self.ax.set_title(f"H per channel")
        self.ax.grid(True, ls='--', alpha=.35)

    # ── NEW: Save current figure as PNG (publication quality) ────────────
    def _save_png(self):  # NEW
        if self.fig is None:
            QMessageBox.information(self, "Save PNG", "Nothing to save yet."); return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save plot as PNG", "plot.png", "PNG Image (*.png)"
        )
        if not path:
            return
        if not path.lower().endswith(".png"):
            path += ".png"
        try:
            # Save at high DPI with tight bounding box and white background
            self.fig.savefig(path, dpi=900, bbox_inches="tight", facecolor="white")
            QMessageBox.information(self, "Save PNG", f"Saved figure to:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Save PNG", f"Failed to save figure:\n{e}")

    # ── Summary (Outcome-separate, trail-wise Top-10 → cross-trail) ──────
    def _run_summary_outcome_trailwise(self):
        """
        Outcome-separated (Success / Fail):
          For each outcome:
            - For each TRAIL in that outcome:
                * compute Var(H) per Channel within that trail,
                * take that trail's Top-10 LOWEST-variance channels.
            - Across its trails:
                * list channels common to ALL trails,
                * rank channels by how many trails' Top-10 lists they appear in.
        Finally:
            - intersection of the two outcomes' "common-to-all-trails" channel sets.
        """
        if self.df.empty:
            QMessageBox.information(self, "Summary", "No data loaded yet.")
            return

        df = self.df.copy()
        if df["Trail"].isna().all():
            QMessageBox.information(self, "Summary", "No Trail IDs were parsed from paths.")
            return

        def per_outcome_summary(df_outcome: pd.DataFrame, outcome_name: str):
            # group by Trail within this outcome
            per_trail_top10: Dict[str, List[str]] = {}
            for trail, sub in df_outcome.groupby("Trail"):
                vt = _var_per_channel(sub)
                if vt.empty:
                    continue
                vt = vt.sort_values("Var(H)", ascending=True)
                per_trail_top10[str(trail)] = vt["Channel"].head(10).tolist()

            trails = sorted(per_trail_top10.keys())
            if not trails:
                return {
                    "outcome": outcome_name,
                    "trails": [],
                    "common_all": [],
                    "appearance_tbl": pd.DataFrame(columns=["Channel", "#Trails"]).
                        sort_index(),  # keep shape compatible
                    "per_trail_top10": per_trail_top10
                }

            sets = [set(per_trail_top10[t]) for t in trails]
            # channels common to ALL trails for this outcome
            common_all = sorted(set.intersection(*sets)) if len(sets) > 1 else sorted(list(sets[0]))

            # count appearances across trails (how many Top-10 lists)
            from collections import Counter
            counter = Counter()
            for s in sets:
                counter.update(s)
            appearance_tbl = (pd.DataFrame([{"Channel": ch, "#Trails": c} for ch, c in counter.items()])
                                .sort_values(["#Trails","Channel"], ascending=[False, True])
                                .reset_index(drop=True))

            return {
                "outcome": outcome_name,
                "trails": trails,
                "common_all": common_all,
                "appearance_tbl": appearance_tbl,
                "per_trail_top10": per_trail_top10
            }

        succ_sum = per_outcome_summary(df[df["Outcome"]=="Success"], "Success")
        fail_sum = per_outcome_summary(df[df["Outcome"]=="Fail"],    "Fail")

        # intersection of outcome consensus sets (common to ALL in Success) ∩ (common to ALL in Fail)
        common_both = sorted(set(succ_sum["common_all"]).intersection(set(fail_sum["common_all"])))

        # Pretty printing helpers
        def fmt_list(title, items):
            if not items: return f"{title}\n(none)\n"
            return f"{title}\n" + "\n".join(f"  - {x}" for x in items) + "\n"

        def fmt_tbl(title, df_show: pd.DataFrame, k=40):
            hdr = title + "\n"
            if df_show is None or df_show.empty: return hdr + "(none)\n"
            show = df_show.copy().head(k)
            return hdr + show.to_string(index=False) + "\n"

        def fmt_per_trail(title, per_trail: Dict[str, List[str]]):
            lines = [title]
            if not per_trail:
                lines.append("(none)\n"); return "\n".join(lines)
            for t in sorted(per_trail.keys()):
                chs = per_trail[t]
                lines.append(f"  {t}: " + (", ".join(chs) if chs else "(none)"))
            lines.append("")  # blank line
            return "\n".join(lines)

        # Build report text
        lines = []
        # SUCCESS block
        lines.append("=== SUCCESS (outcome-wise, trail-wise Top-10) ===")
        lines.append(f"Trails considered: {len(succ_sum['trails'])}")
        lines.append(fmt_list("Channels common to ALL Success trails:", succ_sum["common_all"]))
        lines.append(fmt_tbl("Channels ranked by # of Success trails' Top-10 membership:",
                             succ_sum["appearance_tbl"]))
        lines.append(fmt_per_trail("Per-trail Top-10 (Success):", succ_sum["per_trail_top10"]))

        # FAIL block
        lines.append("=== FAIL (outcome-wise, trail-wise Top-10) ===")
        lines.append(f"Trails considered: {len(fail_sum['trails'])}")
        lines.append(fmt_list("Channels common to ALL Fail trails:", fail_sum["common_all"]))
        lines.append(fmt_tbl("Channels ranked by # of Fail trails' Top-10 membership:",
                             fail_sum["appearance_tbl"]))
        lines.append(fmt_per_trail("Per-trail Top-10 (Fail):", fail_sum["per_trail_top10"]))

        # COMMON BETWEEN OUTCOMES
        lines.append("=== COMMON BETWEEN OUTCOMES ===")
        lines.append(fmt_list("Channels that are common-to-ALL in BOTH Success and Fail:",
                              common_both))

        # Show dialog
        dlg = QDialog(self)
        dlg.setWindowTitle("Summary — Outcome-separated Trail-wise Top-10 → Common Channels")
        dlg.resize(1100, 800)
        lay = QVBoxLayout(dlg)
        txt = QPlainTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText("\n".join(lines))
        lay.addWidget(txt)
        btns = QHBoxLayout()
        closeb = QPushButton("Close"); closeb.clicked.connect(dlg.accept)
        btns.addStretch(1); btns.addWidget(closeb)
        lay.addLayout(btns)
        dlg.exec_()

# ── main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    mpl.rcParams["axes.titlesize"] = 10
    app = QApplication(sys.argv)
    win = Viewer(); win.show()
    sys.exit(app.exec_())