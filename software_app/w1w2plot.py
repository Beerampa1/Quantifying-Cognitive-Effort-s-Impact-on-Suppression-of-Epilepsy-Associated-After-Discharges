# #!/usr/bin/env python3
# """
# Pick your 'Hurst' folder, read EVERY H value from all Hurst_Summary.xlsx files
# (no averaging), and plot TWO boxplots (c0.5s and c1s) with groups:
#   - W1 Fail, W2 Fail, W1 Success, W2 Success

# Y-axis is fixed to 0..2.5.

# Outputs (inside <Hurst>/_hurst_plots):
#   - Hurst_boxplot_c0.5s.png
#   - Hurst_boxplot_c1s.png
#   - Hurst_pooled.csv   (all rows used to make the plots)

# Requires: pandas, matplotlib, openpyxl
#     pip install pandas matplotlib openpyxl
# """

# import os
# import re
# import sys
# from typing import Optional, Tuple, List

# # -------- folder picker (GUI if available; console fallback) --------
# def pick_directory(title: str = "Select your 'Hurst' folder") -> Optional[str]:
#     try:
#         import tkinter as tk
#         from tkinter import filedialog
#         root = tk.Tk()
#         root.withdraw()
#         try:
#             root.attributes("-topmost", True)
#         except Exception:
#             pass
#         d = filedialog.askdirectory(title=title)
#         root.update()
#         root.destroy()
#         return d if d else None
#     except Exception:
#         print("\nGUI picker unavailable. Paste the full path to your 'Hurst' folder:")
#         d = input("> ").strip().strip('"').strip("'")
#         return d if d else None

# # ----------------------------- core logic -----------------------------
# import pandas as pd
# import matplotlib.pyplot as plt

# # Trails can look like: P1_S8_P4, P1_S6a_T4, P14_ST2_T1, P38_ST3_P16, etc.
# TRAIL_REGEXES = [
#     r"(?i)\bP\d+_S\d+[a-z]?_(?:T|P)\d+\b",
#     r"(?i)\bP\d+_ST\d+_(?:T|P)\d+\b",
# ]

# def find_trail_in_path(path: str) -> Optional[str]:
#     for rgx in TRAIL_REGEXES:
#         m = re.search(rgx, path)
#         if m:
#             return m.group(0)
#     # Fallback: parent folder of DFA_* is usually the trail folder
#     parts = path.replace("\\", "/").split("/")
#     for i, p in enumerate(parts):
#         if p.startswith("DFA_"):
#             if i - 1 >= 0:
#                 return parts[i - 1]
#     return None

# def parse_seglen_from_path(path: str) -> Optional[str]:
#     """
#     Accept _c0.5s, _c1s, _c1.0s (with dot or comma).
#     Return canonical '0.5s' or '1s' when possible, else e.g. '2s'.
#     """
#     m = re.search(r"(?i)_c([\d]+(?:[.,]\d+)?)s\b", path)
#     if not m:
#         return None
#     val = m.group(1).replace(",", ".")
#     try:
#         f = float(val)
#         if abs(f - 1.0) < 1e-6:
#             return "1s"
#         if abs(f - 0.5) < 1e-6:
#             return "0.5s"
#         return f"{f:g}s"
#     except ValueError:
#         return None

# def parse_metadata_from_path(path: str) -> Tuple[str, str, str, Optional[str]]:
#     """
#     Returns (Outcome, Window, SegLen, Trail)
#     Outcome: 'Success' if 'success' or 'sucess' seen; 'Fail' if 'fail'; else 'Unknown'
#     Window : 'W1'/'W2' if present; else 'Unknown'
#     SegLen : '0.5s'/'1s'/other; else 'Unknown'
#     Trail  : best-effort trail id
#     """
#     lower_parts = [p.lower() for p in path.replace("\\", "/").split("/")]
#     outcome = "Unknown"
#     if any("fail" in p for p in lower_parts):
#         outcome = "Fail"
#     if any(("success" in p) or ("sucess" in p) for p in lower_parts):
#         outcome = "Success"

#     window = "Unknown"
#     if any(p == "w1" for p in lower_parts):
#         window = "W1"
#     if any(p == "w2" for p in lower_parts):
#         window = "W2"

#     seglen = parse_seglen_from_path(path) or "Unknown"
#     trail = find_trail_in_path(path)
#     return outcome, window, seglen, trail

# def read_hurst_summary(xlsx_path: str) -> Optional[pd.DataFrame]:
#     try:
#         xls = pd.ExcelFile(xlsx_path)
#         sheet = "Summary" if "Summary" in xls.sheet_names else xls.sheet_names[0]
#         df = pd.read_excel(xlsx_path, sheet_name=sheet)
#     except Exception:
#         try:
#             df = pd.read_excel(xlsx_path)
#         except Exception:
#             return None
#     return df if ("H" in df.columns) else None

# def collect_from_base_dir(base_dir: str) -> pd.DataFrame:
#     """
#     Collect ALL raw H values (no averaging):
#     Outcome, Window, SegLen, Trail, Channel, H, Source
#     """
#     rows: List[dict] = []
#     for root, _, files in os.walk(base_dir):
#         for f in files:
#             if f.lower() != "hurst_summary.xlsx":
#                 continue
#             fullp = os.path.join(root, f)
#             df = read_hurst_summary(fullp)
#             if df is None or df.empty:
#                 continue
#             outcome, window, seglen, trail = parse_metadata_from_path(fullp)
#             for _, r in df.iterrows():
#                 H = r.get("H")
#                 if pd.isna(H):
#                     continue
#                 rows.append({
#                     "Outcome": outcome,
#                     "Window": window,
#                     "SegLen": seglen,
#                     "Trail": trail,
#                     "Channel": r.get("Channel"),
#                     "H": float(H),
#                     "Source": fullp
#                 })
#     return pd.DataFrame(rows)

# def make_boxplot(df: pd.DataFrame, seglen: str, outfile: str):
#     """
#     One boxplot for seglen ('0.5s' or '1s') with groups:
#     W1 Fail, W2 Fail, W1 Success, W2 Success
#     Uses ALL H values. y-axis fixed 0..2.5.
#     """
#     target = df[df["SegLen"].str.lower() == seglen.lower()]
#     groups = [
#         ("W1", "Fail",    "W1 Fail"),
#         ("W2", "Fail",    "W2 Fail"),
#         ("W1", "Success", "W1 Success"),
#         ("W2", "Success", "W2 Success"),
#     ]
#     data, labels, counts = [], [], []
#     for w, o, lab in groups:
#         vals = target[(target["Window"] == w) & (target["Outcome"] == o)]["H"].dropna().values
#         if len(vals) > 0:
#             data.append(vals)
#             labels.append(lab)
#             counts.append(len(vals))

#     plt.figure(figsize=(9, 6))
#     if not data:
#         plt.title(f"Hurst Coefficient Boxplot (c{seglen}) — no data found")
#         plt.xlabel("Group")
#         plt.ylabel("Hurst (H)")
#         plt.ylim(0, 2.5)
#         plt.tight_layout()
#         plt.savefig(outfile, dpi=150, bbox_inches="tight")
#         plt.close()
#         return

#     plt.boxplot(data, labels=labels, showmeans=True)
#     ax = plt.gca()
#     ax.set_xticks(range(1, len(labels) + 1))
#     ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, counts)])
#     ax.set_ylim(0, 2.5)
#     plt.title(f"Hurst Coefficients (c{seglen})")
#     plt.xlabel("Group")
#     plt.ylabel("Hurst (H)")
#     plt.tight_layout()
#     plt.savefig(outfile, dpi=150, bbox_inches="tight")
#     plt.close()

# def main():
#     base_dir = pick_directory("Select your 'Hurst' folder (contains Sucess/Success_Trails & Fail_Trails)")
#     if not base_dir or not os.path.isdir(base_dir):
#         print("No folder selected or invalid path. Exiting.")
#         sys.exit(1)

#     print(f"\nScanning: {base_dir}")
#     pooled = collect_from_base_dir(base_dir)
#     if pooled.empty:
#         print("No usable Hurst_Summary.xlsx files found (or no 'H' values). Exiting.")
#         sys.exit(1)

#     outdir = os.path.join(base_dir, "_hurst_plots")
#     os.makedirs(outdir, exist_ok=True)

#     # Save ALL raw rows used for plotting
#     out_csv = os.path.join(outdir, "Hurst_pooled.csv")
#     pooled.to_csv(out_csv, index=False)
#     print(f"Saved pooled CSV: {out_csv}")

#     # Make the two plots (fixed y from 0 to 2.5)
#     out_05 = os.path.join(outdir, "Hurst_boxplot_c0.5s.png")
#     out_1s = os.path.join(outdir, "Hurst_boxplot_c1s.png")
#     make_boxplot(pooled, "0.5s", out_05)
#     print(f"Saved: {out_05}")
#     make_boxplot(pooled, "1s", out_1s)
#     print(f"Saved: {out_1s}")

#     # Quick summary to console
#     stats = (
#         pooled.groupby(["SegLen", "Window", "Outcome"], dropna=False)["H"]
#               .agg(["count", "mean", "median"])
#               .reset_index()
#               .sort_values(["SegLen", "Window", "Outcome"])
#     )
#     with pd.option_context('display.max_rows', None, 'display.width', 140):
#         print("\nAggregation summary (using ALL H values):")
#         print(stats.to_string(index=False))

# if __name__ == "__main__":
#     main()



#!/usr/bin/env python3
"""
Pick your 'Hurst' folder, scan every Hurst_Summary.xlsx, average H per Channel per Trail,
and plot TWO boxplots (one for c0.5s, one for c1s) with groups:
  - W1 Fail, W2 Fail, W1 Success, W2 Success

Y-axis is fixed to 0..2.5.

Outputs (inside <Hurst>/_hurst_plots):
  - Hurst_boxplot_c0.5s.png
  - Hurst_boxplot_c1s.png
  - Hurst_averaged_per_channel_per_trail.csv   (the data used for the boxplots)

Requires: pandas, matplotlib, openpyxl
    pip install pandas matplotlib openpyxl
"""

import os
import re
import sys
from typing import Optional, Tuple, List

# -------- folder picker (GUI if available; console fallback) --------
def pick_directory(title: str = "Select your 'Hurst' folder") -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        d = filedialog.askdirectory(title=title)
        root.update()
        root.destroy()
        return d if d else None
    except Exception:
        print("\nGUI picker unavailable. Paste the full path to your 'Hurst' folder:")
        d = input("> ").strip().strip('"').strip("'")
        return d if d else None

# ----------------------------- core logic -----------------------------
import pandas as pd
import matplotlib.pyplot as plt

# Trails can look like: P1_S8_P4, P1_S6a_T4, P14_ST2_T1, P38_ST3_P16, etc.
TRAIL_REGEXES = [
    r"(?i)\bP\d+_S\d+[a-z]?_(?:T|P)\d+\b",
    r"(?i)\bP\d+_ST\d+_(?:T|P)\d+\b",
]

def find_trail_in_path(path: str) -> Optional[str]:
    for rgx in TRAIL_REGEXES:
        m = re.search(rgx, path)
        if m:
            return m.group(0)
    # Fallback: parent folder of DFA_* is usually the trail folder
    parts = path.replace("\\", "/").split("/")
    for i, p in enumerate(parts):
        if p.startswith("DFA_"):
            if i - 1 >= 0:
                return parts[i - 1]
    return None

def parse_seglen_from_path(path: str) -> Optional[str]:
    """
    Accept _c0.5s, _c1s, _c1.0s (with dot or comma).
    Return canonical '0.5s' or '1s' when possible, else e.g. '2s'.
    """
    m = re.search(r"(?i)_c([\d]+(?:[.,]\d+)?)s\b", path)
    if not m:
        return None
    val = m.group(1).replace(",", ".")
    try:
        f = float(val)
        if abs(f - 1.0) < 1e-6:
            return "1s"
        if abs(f - 0.5) < 1e-6:
            return "0.5s"
        return f"{f:g}s"
    except ValueError:
        return None

def parse_metadata_from_path(path: str) -> Tuple[str, str, str, Optional[str]]:
    """
    Returns (Outcome, Window, SegLen, Trail)
    Outcome: 'Success' if 'success' or 'sucess' seen; 'Fail' if 'fail'; else 'Unknown'
    Window : 'W1'/'W2' if present; else 'Unknown'
    SegLen : '0.5s'/'1s'/other; else 'Unknown'
    Trail  : best-effort trail id
    """
    lower_parts = [p.lower() for p in path.replace("\\", "/").split("/")]
    outcome = "Unknown"
    if any("fail" in p for p in lower_parts):
        outcome = "Fail"
    if any(("success" in p) or ("sucess" in p) for p in lower_parts):
        outcome = "Success"

    window = "Unknown"
    if any(p == "w1" for p in lower_parts):
        window = "W1"
    if any(p == "w2" for p in lower_parts):
        window = "W2"

    seglen = parse_seglen_from_path(path) or "Unknown"
    trail = find_trail_in_path(path)
    return outcome, window, seglen, trail

def read_hurst_summary(xlsx_path: str) -> Optional[pd.DataFrame]:
    try:
        xls = pd.ExcelFile(xlsx_path)
        sheet = "Summary" if "Summary" in xls.sheet_names else xls.sheet_names[0]
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
    except Exception:
        try:
            df = pd.read_excel(xlsx_path)
        except Exception:
            return None
    return df if ("H" in df.columns) else None

def collect_from_base_dir(base_dir: str) -> pd.DataFrame:
    """
    Aggregate raw rows: Outcome, Window, SegLen, Trail, Channel, H, Source
    """
    rows: List[dict] = []
    for root, _, files in os.walk(base_dir):
        for f in files:
            if f.lower() != "hurst_summary.xlsx":
                continue
            fullp = os.path.join(root, f)
            df = read_hurst_summary(fullp)
            if df is None or df.empty:
                continue
            outcome, window, seglen, trail = parse_metadata_from_path(fullp)
            for _, r in df.iterrows():
                H = r.get("H")
                if pd.isna(H):
                    continue
                rows.append({
                    "Outcome": outcome,
                    "Window": window,
                    "SegLen": seglen,
                    "Trail": trail,
                    "Channel": r.get("Channel"),
                    "H": float(H),
                    "Source": fullp
                })
    return pd.DataFrame(rows)

def average_per_channel_per_trail(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse many H rows -> one value per (SegLen, Window, Outcome, Trail, Channel)
    using MEAN (average).
    """
    return (
        df.groupby(["SegLen", "Window", "Outcome", "Trail", "Channel"], dropna=False)["H"]
          .mean()
          .reset_index()
    )

def make_boxplot(df: pd.DataFrame, seglen: str, outfile: str):
    """
    One boxplot for seglen ('0.5s' or '1s') with groups:
    W1 Fail, W2 Fail, W1 Success, W2 Success
    Shows means, and fixes y-limits to 0..2.5
    """
    target = df[df["SegLen"].str.lower() == seglen.lower()]
    groups = [
        ("W1", "Fail",    "W1 Fail"),
        ("W2", "Fail",    "W2 Fail"),
        ("W1", "Success", "W1 Success"),
        ("W2", "Success", "W2 Success"),
    ]
    data, labels, counts = [], [], []
    for w, o, lab in groups:
        vals = target[(target["Window"] == w) & (target["Outcome"] == o)]["H"].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(lab)
            counts.append(len(vals))

    plt.figure(figsize=(9, 6))
    if not data:
        plt.title(f"Hurst Coefficient Boxplot (c{seglen}) — no data found")
        plt.xlabel("Group")
        plt.ylabel("Hurst (H)")
        plt.ylim(0, 2.5)
        plt.tight_layout()
        plt.savefig(outfile, dpi=150, bbox_inches="tight")
        plt.close()
        return

    plt.boxplot(data, labels=labels, showmeans=True)
    ax = plt.gca()
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, counts)])
    ax.set_ylim(0, 2.5)
    plt.title(f"Hurst Coefficients (c{seglen})")
    plt.xlabel("Group")
    plt.ylabel("Hurst (H)")
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

def main():
    base_dir = pick_directory("Select your 'Hurst' folder (contains Sucess/Success_Trails & Fail_Trails)")
    if not base_dir or not os.path.isdir(base_dir):
        print("No folder selected or invalid path. Exiting.")
        sys.exit(1)

    print(f"\nScanning: {base_dir}")
    raw = collect_from_base_dir(base_dir)
    if raw.empty:
        print("No usable Hurst_Summary.xlsx files found (or no 'H' values). Exiting.")
        sys.exit(1)

    # Average H across rows -> one value per Channel per Trail (per seg/window/outcome)
    averaged = average_per_channel_per_trail(raw)

    outdir = os.path.join(base_dir, "_hurst_plots")
    os.makedirs(outdir, exist_ok=True)

    # Save the averaged data used for plotting
    out_csv = os.path.join(outdir, "Hurst_averaged_per_channel_per_trail.csv")
    averaged.to_csv(out_csv, index=False)
    print(f"Saved averaged CSV: {out_csv}")

    # Make the two plots (fixed y from 0 to 2.5)
    out_05 = os.path.join(outdir, "Hurst_boxplot_c0.5s.png")
    out_1s = os.path.join(outdir, "Hurst_boxplot_c1s.png")
    make_boxplot(averaged, "0.5s", out_05)
    print(f"Saved: {out_05}")
    make_boxplot(averaged, "1s", out_1s)
    print(f"Saved: {out_1s}")

    # Quick summary to console
    stats = (
        averaged.groupby(["SegLen", "Window", "Outcome"], dropna=False)["H"]
                .agg(["count", "mean", "median"])
                .reset_index()
                .sort_values(["SegLen", "Window", "Outcome"])
    )
    with pd.option_context('display.max_rows', None, 'display.width', 140):
        print("\nAggregation summary (on averaged per-channel-per-trail values):")
        print(stats.to_string(index=False))

if __name__ == "__main__":
    main()
