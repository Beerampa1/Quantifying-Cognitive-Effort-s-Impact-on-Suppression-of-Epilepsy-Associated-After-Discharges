# extract_dfa_overlap.py
import csv
import re
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

# Your existing utilities
from utils.label_strategies import ExcelMathScoreLabeler
from utils.feature_extractor import FeatureExtractor



def parse_q_list(text: str) -> List[float]:
    parts = re.split(r"[,\s]+", text.strip())
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            pass
    return out or [2.0, 3.0, 4.0]


def find_innermost_overlap_dirs(trial_dir: Path, chunk_str: str, hop_str: str) -> List[Path]:
    """
    Find 'innermost' DFA_Overlap_*_c{chunk}s_h{hop}s folders under a trial.
    If nested, we keep the deepest occurrences (the ones that contain Channel_* subdirs).
    """
    pattern = f"DFA_Overlap_*_c{chunk_str}s_h{hop_str}s"
    candidates = list(trial_dir.rglob(pattern))
    if not candidates:
        return []

    # Keep only those that look like final analysis folders (contain Channel_* subdirs)
    finals = []
    for d in candidates:
        if any(p.is_dir() and p.name.startswith("Channel_") for p in d.iterdir()):
            finals.append(d)

    if finals:
        return sorted(finals)

    # Fallback: if no Channel_* inside, return deepest by path depth
    candidates.sort(key=lambda p: len(p.parts), reverse=True)
    deepest_depth = len(candidates[0].parts)
    return [p for p in candidates if len(p.parts) == deepest_depth]


def infer_channel_count_from_overlap_dir(overlap_dir: Path) -> int:
    """Count Channel_* subdirectories as a proxy for number of channels."""
    try:
        return sum(1 for p in overlap_dir.iterdir() if p.is_dir() and p.name.startswith("Channel_"))
    except Exception:
        return 0


def build_header(
    trials: List[Path],
    chunk_str: str,
    hop_str: str,
    top_pct: int,
    q_list: List[float],
    opts_dfa: Dict[str, bool],
) -> List[str]:
    header = ["Trial", "OverlapFolder"]  # include which folder we used

    # DFA summary fields
    if opts_dfa.get("H_mean_std", True):
        header += ["DFA_H_mean", "DFA_H_std"]
    if opts_dfa.get("DeltaHq", True):
        header += ["DFA_DeltaHq_mean", "DFA_DeltaHq_std"]
    if opts_dfa.get("Hq_per_q", True):
        for q in q_list:
            header += [f"DFA_Hq{q}_mean", f"DFA_Hq{q}_std"]

    # Per-channel top-K: infer a typical channel count
    n_ch = 0
    for td in trials:
        inns = find_innermost_overlap_dirs(td, chunk_str, hop_str)
        if inns:
            n_ch = infer_channel_count_from_overlap_dir(inns[0])
            if n_ch:
                break

    if n_ch and (opts_dfa.get("pc_mean") or opts_dfa.get("pc_std")):
        K = max(1, int(n_ch * top_pct / 100))
        for r in range(1, K + 1):
            if opts_dfa.get("pc_mean"):
                header.append(f"DFA_ch{r}_H_mean")
            if opts_dfa.get("pc_std"):
                header.append(f"DFA_ch{r}_H_std")

    header += ["Label"]
    return header


def extract_for_trial(
    trial_dir: Path,
    labeler: ExcelMathScoreLabeler,
    chunk_str: str,
    hop_str: str,
    top_pct: int,
    q_list: List[float],
    opts_dfa: Dict[str, bool],
    skip_unlabeled: bool,
) -> List[Dict[str, str]]:
    """
    Returns one or multiple rows per trial:
    - If there are multiple innermost overlap folders (e.g., many subsegments),
      we write one row per folder and average inside FeatureExtractor (as it already does).
    """
    rows = []
    tid = trial_dir.name
    lbl = labeler.get(tid)
    if lbl is None and skip_unlabeled:
        return rows  # empty

    inns = find_innermost_overlap_dirs(trial_dir, chunk_str, hop_str)
    if not inns:
        # No overlap folder found for this trial
        rows.append({"Trial": tid, "OverlapFolder": "", "Label": "" if lbl is None else lbl})
        return rows

    for ov in inns:
        feats = {"Trial": tid, "OverlapFolder": str(ov)}
        try:
            # Reuse your existing DFA extractor on the overlap folder.
            feats.update(FeatureExtractor.extract_dfa_features(ov, top_pct, q_list, opts_dfa))
        except Exception as e:
            feats["__ERROR__"] = f"{e}"

        feats["Label"] = "" if lbl is None else lbl
        rows.append(feats)

    return rows


def run_extract_overlap(
    root_dir: Path,
    excel_path: Path,
    save_csv: Path,
    chunk_str: str,   # e.g. "0.5"
    hop_str: str,     # e.g. "0.25"
    top_pct: int,
    q_list: List[float],
    opts_dfa: Dict[str, bool],
    skip_unlabeled: bool = True,
) -> Path:
    # Load labels
    labeler = ExcelMathScoreLabeler(excel_path)

    # Discover trial folders
    trials = sorted(root_dir.glob("P*_S*_T*"))
    if not trials:
        raise RuntimeError("No trial folders (P*_S*_T*) found under: " + str(root_dir))

    # Build header
    header = build_header(trials, chunk_str, hop_str, top_pct, q_list, opts_dfa)

    # Prepare log
    log_path = save_csv.with_name(save_csv.stem + "_log.txt")
    with open(save_csv, "w", newline="") as csv_f, open(log_path, "w", encoding="utf-8") as log_f:
        writer = csv.DictWriter(csv_f, fieldnames=header)
        writer.writeheader()

        for td in trials:
            try:
                rows = extract_for_trial(
                    td, labeler, chunk_str, hop_str, top_pct, q_list, opts_dfa, skip_unlabeled
                )
                if not rows:
                    log_f.write(f"SKIP unlabeled: {td.name}\n")
                    continue

                for r in rows:
                    # Make sure all header keys exist
                    for k in header:
                        r.setdefault(k, "")
                    writer.writerow(r)

                log_f.write(f"OK {td.name} ({len(rows)} row(s))\n")
            except Exception as e:
                import traceback, textwrap
                tb = textwrap.indent(traceback.format_exc(), "  ")
                log_f.write(f"ERROR {td.name}: {e}\n{tb}\n")

    return save_csv


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Extract DFA overlap features into a CSV.")
    ap.add_argument("--root", required=True, type=Path, help="TrialResults root (contains P*_S*_T*).")
    ap.add_argument("--labels", required=True, type=Path, help="Excel file with labels.")
    ap.add_argument("--out", required=True, type=Path, help="Output CSV path.")
    ap.add_argument("--chunk", required=True, help="Chunk-size string as in folder (e.g., 0.5).")
    ap.add_argument("--hop", required=True, help="Hop-size string as in folder (e.g., 0.25).")
    ap.add_argument("--topk", type=int, default=10, help="Top-k %% of channels (default 10).")
    ap.add_argument("--q", type=str, default="2,3,4", help="Comma/space q-list (default '2,3,4').")
    ap.add_argument("--skip-unlabeled", action="store_true", default=False)
    # DFA feature toggles (mirroring your GUI)
    ap.add_argument("--dfa-H-mean-std", action="store_true", default=True)
    ap.add_argument("--dfa-DeltaHq", action="store_true", default=True)
    ap.add_argument("--dfa-Hq-per-q", action="store_true", default=True)
    ap.add_argument("--dfa-pc-mean", action="store_true", default=True)
    ap.add_argument("--dfa-pc-std", action="store_true", default=True)

    args = ap.parse_args()

    q_list = parse_q_list(args.q)
    opts_dfa = {
        "H_mean_std": args.dfa_H_mean_std,
        "DeltaHq": args.dfa_DeltaHq,
        "Hq_per_q": args.dfa_Hq_per_q,
        "pc_mean": args.dfa_pc_mean,
        "pc_std": args.dfa_pc_std,
    }

    out_csv = run_extract_overlap(
        root_dir=args.root,
        excel_path=args.labels,
        save_csv=args.out,
        chunk_str=args.chunk,
        hop_str=args.hop,
        top_pct=args.topk,
        q_list=q_list,
        opts_dfa=opts_dfa,
        skip_unlabeled=args.skip_unlabeled,
    )
    print("Saved:", out_csv)























