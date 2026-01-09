# extract_dfa_overlap.py
# Script to extract DFA (overlapping-window) features from trial result folders
# and save them into a single CSV file (with labels from an Excel sheet).

import csv
import re
from pathlib import Path
from typing import List, Dict

import numpy as np

# Reads trial labels (success/fail/etc.) from Excel
from utils.label_strategies import ExcelMathScoreLabeler

# Computes DFA feature summaries from a DFA output folder
from utils.feature_extractor import FeatureExtractor


def parse_q_list(text: str) -> List[float]:
    """
    Parse q values from a user string like "2,3,4" or "2 3 4".
    If parsing fails or empty, fallback to [2.0, 3.0, 4.0].
    """
    parts = re.split(r"[,\s]+", text.strip())
    out = []
    for p in parts:
        try:
            out.append(float(p))
        except Exception:
            # Ignore invalid values
            pass
    return out or [2.0, 3.0, 4.0]


def find_innermost_overlap_dirs(trial_dir: Path, chunk_str: str, hop_str: str) -> List[Path]:
    """
    Find the *final* DFA_Overlap result folders inside a trial directory.

    Why "innermost":
    - Sometimes DFA outputs can be nested.
    - We want the deepest folder that contains Channel_* subfolders.

    Returns:
      List of overlap directories (sorted).
    """
    # Example folder name pattern:
    # DFA_Overlap_0-10_c0.5s_h0.25s
    pattern = f"DFA_Overlap_*_c{chunk_str}s_h{hop_str}s"
    candidates = list(trial_dir.rglob(pattern))
    if not candidates:
        return []

    # Keep folders that look like final outputs (contain Channel_* dirs)
    finals = []
    for d in candidates:
        if any(p.is_dir() and p.name.startswith("Channel_") for p in d.iterdir()):
            finals.append(d)

    # If found "final" folders, return them
    if finals:
        return sorted(finals)

    # Fallback: choose deepest folders by path depth
    candidates.sort(key=lambda p: len(p.parts), reverse=True)
    deepest_depth = len(candidates[0].parts)
    return [p for p in candidates if len(p.parts) == deepest_depth]


def infer_channel_count_from_overlap_dir(overlap_dir: Path) -> int:
    """
    Count Channel_* subdirectories.
    This acts as a proxy for number of channels in the results.
    """
    try:
        return sum(
            1 for p in overlap_dir.iterdir()
            if p.is_dir() and p.name.startswith("Channel_")
        )
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
    """
    Build the CSV header dynamically based on enabled feature options.
    Also includes per-channel Top-K columns (based on percent of channels).
    """
    header = ["Trial", "OverlapFolder"]  # track which overlap folder produced the row

    # ----- DFA summary stats -----
    if opts_dfa.get("H_mean_std", True):
        header += ["DFA_H_mean", "DFA_H_std"]  # computed mean/std of H
    if opts_dfa.get("DeltaHq", True):
        header += ["DFA_DeltaHq_mean", "DFA_DeltaHq_std"]  # computed mean/std of ΔHq
    if opts_dfa.get("Hq_per_q", True):
        # computed mean/std of Hq for each q value
        for q in q_list:
            header += [f"DFA_Hq{q}_mean", f"DFA_Hq{q}_std"]

    # ----- Per-channel Top-K features -----
    # Infer a typical channel count by checking first trial that has overlap outputs
    n_ch = 0
    for td in trials:
        inns = find_innermost_overlap_dirs(td, chunk_str, hop_str)
        if inns:
            n_ch = infer_channel_count_from_overlap_dir(inns[0])
            if n_ch:
                break

    # If channel count is known, add Top-K columns
    if n_ch and (opts_dfa.get("pc_mean") or opts_dfa.get("pc_std")):
        K = max(1, int(n_ch * top_pct / 100))  # top percentage converted to count
        for r in range(1, K + 1):
            if opts_dfa.get("pc_mean"):
                header.append(f"DFA_ch{r}_H_mean")  # per-channel mean H (top-ranked)
            if opts_dfa.get("pc_std"):
                header.append(f"DFA_ch{r}_H_std")   # per-channel std H (top-ranked)

    # Final label column (from Excel)
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
    Extract features for ONE trial directory.

    Notes:
    - Some trials can contain multiple "innermost" overlap folders.
      In that case we return multiple rows (one per overlap folder).
    - Labels are pulled from Excel using the trial folder name.

    Returns:
      List of CSV rows (dicts).
    """
    rows = []
    tid = trial_dir.name

    # Get label for this trial (e.g., success/fail)
    lbl = labeler.get(tid)

    # Optionally skip trials without labels
    if lbl is None and skip_unlabeled:
        return rows  # empty

    # Find overlap result folders for this trial
    inns = find_innermost_overlap_dirs(trial_dir, chunk_str, hop_str)
    if not inns:
        # No overlap results found → still write a row with label if available
        rows.append({"Trial": tid, "OverlapFolder": "", "Label": "" if lbl is None else lbl})
        return rows

    # Extract features for each overlap folder
    for ov in inns:
        feats = {"Trial": tid, "OverlapFolder": str(ov)}
        try:
            # Compute DFA overlap features using existing extractor
            feats.update(FeatureExtractor.extract_dfa_features(ov, top_pct, q_list, opts_dfa))
        except Exception as e:
            # Store error message in row if feature extraction fails
            feats["__ERROR__"] = f"{e}"

        feats["Label"] = "" if lbl is None else lbl
        rows.append(feats)

    return rows


def run_extract_overlap(
    root_dir: Path,
    excel_path: Path,
    save_csv: Path,
    chunk_str: str,   # e.g. "0.5" (must match folder naming)
    hop_str: str,     # e.g. "0.25" (must match folder naming)
    top_pct: int,
    q_list: List[float],
    opts_dfa: Dict[str, bool],
    skip_unlabeled: bool = True,
) -> Path:
    """
    Main driver:
    - Load labels from Excel
    - Find trial folders
    - Extract overlap DFA features
    - Write CSV and log file
    """
    # Load trial labels from Excel
    labeler = ExcelMathScoreLabeler(excel_path)

    # Discover trial folders (expected naming pattern)
    trials = sorted(root_dir.glob("P*_S*_T*"))
    if not trials:
        raise RuntimeError("No trial folders (P*_S*_T*) found under: " + str(root_dir))

    # Build output CSV header (based on enabled options)
    header = build_header(trials, chunk_str, hop_str, top_pct, q_list, opts_dfa)

    # Write an additional log file to track skipped/failed trials
    log_path = save_csv.with_name(save_csv.stem + "_log.txt")

    with open(save_csv, "w", newline="") as csv_f, open(log_path, "w", encoding="utf-8") as log_f:
        writer = csv.DictWriter(csv_f, fieldnames=header)
        writer.writeheader()

        # Process each trial folder
        for td in trials:
            try:
                rows = extract_for_trial(
                    td, labeler, chunk_str, hop_str, top_pct, q_list, opts_dfa, skip_unlabeled
                )

                # Skip unlabeled trials if requested
                if not rows:
                    log_f.write(f"SKIP unlabeled: {td.name}\n")
                    continue

                # Write each row to CSV
                for r in rows:
                    # Ensure all keys exist (avoids KeyError in DictWriter)
                    for k in header:
                        r.setdefault(k, "")
                    writer.writerow(r)

                log_f.write(f"OK {td.name} ({len(rows)} row(s))\n")

            except Exception as e:
                # Log full traceback if something unexpected fails
                import traceback, textwrap
                tb = textwrap.indent(traceback.format_exc(), "  ")
                log_f.write(f"ERROR {td.name}: {e}\n{tb}\n")

    return save_csv


if __name__ == "__main__":
    # Command-line interface for running feature extraction

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

    # DFA feature toggles (similar to GUI options)
    ap.add_argument("--dfa-H-mean-std", action="store_true", default=True)
    ap.add_argument("--dfa-DeltaHq", action="store_true", default=True)
    ap.add_argument("--dfa-Hq-per-q", action="store_true", default=True)
    ap.add_argument("--dfa-pc-mean", action="store_true", default=True)
    ap.add_argument("--dfa-pc-std", action="store_true", default=True)

    args = ap.parse_args()

    # Parse q list string into float list
    q_list = parse_q_list(args.q)

    # Build options dict for feature extraction
    opts_dfa = {
        "H_mean_std": args.dfa_H_mean_std,
        "DeltaHq": args.dfa_DeltaHq,
        "Hq_per_q": args.dfa_Hq_per_q,
        "pc_mean": args.dfa_pc_mean,
        "pc_std": args.dfa_pc_std,
    }

    # Run extraction and save CSV
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
