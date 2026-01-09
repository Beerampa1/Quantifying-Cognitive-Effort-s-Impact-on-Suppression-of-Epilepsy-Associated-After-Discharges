#!/usr/bin/env python
"""
FODN α box-plot for W1 / W2
--------------------------------------

This script:

1. Asks you to select a root folder that contains FODN_*_c*s result folders.
2. Recursively finds all FODN_*_c*s folders.
3. For each chunk folder (chunkX_start-end):
       - Loads Alpha_Data.csv (α per channel)
       - Records the chunk start time (in seconds)
4. Sorts all chunks by their start time across all FODN folders.
5. Asks for the time (in seconds) at which W2 starts.
   If you cancel, the script uses the mid-time between first and last chunk.
6. Plots α-distribution per chunk as box-plots:

   X-axis = actual chunk start time (seconds)
   Y-axis = α (per channel)
   W1 chunks = light yellow boxes
   W2 chunks = light purple boxes
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


# ----------------------------------------------------------------------
# Find FODN folders
# ----------------------------------------------------------------------
def find_fodn_folders(start_folder):
    """
    Recursively search 'start_folder' for subfolders that match:
        FODN_<start>-<end>_c<chunk_size>s
    """
    results = []
    pattern = r"^FODN_\d+(\.\d+)?-\d+(\.\d+)?_c\d+(\.\d+)?s$"

    for root, dirs, files in os.walk(start_folder):
        for d in dirs:
            if re.match(pattern, d):
                results.append(os.path.join(root, d))
    return results


# ----------------------------------------------------------------------
# Load α data + times from FODN folders
# ----------------------------------------------------------------------
def load_alpha_by_time(root_folder):
    """
    From root_folder, find all FODN_*_c*s folders, then for each chunk
    load Alpha_Data.csv and associate it with the chunk start time.

    Returns
    -------
    times : np.ndarray, shape (num_chunks,)
        Start time (seconds) of each chunk/window.
    alpha_arrays : list of np.ndarray
        alpha_arrays[i] is a 1D array of α values for chunk i (per channel).
    """
    fodn_folders = find_fodn_folders(root_folder)
    if not fodn_folders:
        return None, None

    chunk_pattern = r"chunk(\d+)_([0-9.]+)-([0-9.]+)"

    time_alpha_pairs = []

    for fodn in fodn_folders:
        for d in os.listdir(fodn):
            chunk_dir = os.path.join(fodn, d)
            if not os.path.isdir(chunk_dir):
                continue

            m = re.match(chunk_pattern, d)
            if not m:
                continue

            _, start_str, end_str = m.groups()
            start_time = float(start_str)

            alpha_file = os.path.join(chunk_dir, "Alpha_Data.csv")
            if not os.path.exists(alpha_file):
                print(f"[WARN] Missing Alpha_Data.csv in {chunk_dir}")
                continue

            try:
                arr = pd.read_csv(alpha_file, header=None).values.flatten()
            except Exception as e:
                print(f"[WARN] Could not read {alpha_file}: {e}")
                continue

            arr = arr[arr > 0]

            if arr.size == 0:
                print(f"[WARN] Empty Alpha_Data.csv in {chunk_dir}")
                continue

            time_alpha_pairs.append((start_time, arr))

    if not time_alpha_pairs:
        return None, None

    # sort by time
    time_alpha_pairs.sort(key=lambda x: x[0])
    times = np.array([t for t, _ in time_alpha_pairs])
    alpha_arrays = [a for _, a in time_alpha_pairs]

    return times, alpha_arrays


# ----------------------------------------------------------------------
# Plot α box-plots for W1 / W2
# ----------------------------------------------------------------------
def plot_alpha_w1w2(times, alpha_arrays, w2_start):
    """
    Plot α distribution per chunk, colouring W1 (times < w2_start)
    and W2 (times >= w2_start).

    Parameters
    ----------
    times : np.ndarray, shape (num_chunks,)
    alpha_arrays : list[np.ndarray]
    w2_start : float
        Time (seconds) where W2 starts.
    """
    num_chunks = len(alpha_arrays)
    if num_chunks == 0:
        print("[ERROR] No alpha data to plot.")
        return

    # positions for boxplots (actual time)
    positions = times

    # approximate width: 80% of min spacing, or a default
    if len(times) > 1:
        diffs = np.diff(times)
        min_step = np.min(diffs)
        width = 0.8 * min_step
    else:
        width = 0.4  # arbitrary

    fig, ax = plt.subplots(figsize=(16, 6))

    fig.suptitle("Temporal Evaluation of Alpha Values", fontsize=22, fontweight="bold")


    # boxplot returns a dict of artists; we do all in one go
    bp = ax.boxplot(
        alpha_arrays,
        positions=positions,
        widths=width,
        patch_artist=True,
        flierprops={'markersize': 2.5}
    )

    # colour W1 / W2
    # light yellow & light purple
    color_w1 = "#fff9c4"  # light yellow
    color_w2 = "#e1bee7"  # light purple

    for i, box in enumerate(bp['boxes']):
        if times[i] < w2_start:
            box.set(facecolor=color_w1, edgecolor="black")
        else:
            box.set(facecolor=color_w2, edgecolor="black")

    # Optionally colour medians slightly darker
    for med in bp['medians']:
        med.set(color="black", linewidth=1.5)

    ax.set_xlabel("Time (seconds)", fontsize=18, fontweight="bold")
    ax.set_ylabel("Fractional Order Exponent values (α)", fontsize=18, fontweight="bold")

    ax.set_ylim(0, 1.6)

    # vertical line at W2 start
    ax.axvline(w2_start, color='k', linestyle='--', linewidth=1)
    # ax.text(
    #     w2_start, ax.get_ylim()[1],
    #     "  W2 start",
    #     va="top", ha="left"
    # )
    from matplotlib.ticker import MaxNLocator

    # rotate x-labels if many chunks
    ax.tick_params(axis='x', rotation=90, labelsize=12)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()

    # --- Ask user where to SAVE the figure ---
    save_folder = filedialog.askdirectory(
        title="Select a folder to save the 300 DPI figure"
    )

    if save_folder:
        save_path = os.path.join(save_folder, "Fig1.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[INFO] Figure saved at 300 DPI to:\n{save_path}")
    else:
        print("[INFO] No save folder selected. Figure not saved.")

    plt.show()

# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo(
        "FODN α W1/W2 viewer",
        "Select the ROOT folder that contains FODN_*_c*s result folders."
    )

    folder = filedialog.askdirectory(title="Select folder with FODN_*_c*s subfolders")
    if not folder:
        print("No folder selected. Exiting.")
        return

    print(f"[INFO] Selected root folder:\n  {folder}")

    times, alpha_arrays = load_alpha_by_time(folder)
    if times is None or alpha_arrays is None:
        messagebox.showerror(
            "No data",
            "No valid Alpha_Data.csv files found under FODN_*_c*s folders."
        )
        return

    print(f"[INFO] Loaded {len(times)} chunks.")

    # ask for W2 start time
    t_min, t_max = float(times[0]), float(times[-1])
    default_mid = 0.5 * (t_min + t_max)

    answer = simpledialog.askstring(
        "W2 start time",
        f"Enter W2 start time in seconds.\n\n"
        f"Range: [{t_min:.2f}, {t_max:.2f}]\n"
        f"If you leave it blank or cancel, midpoint ({default_mid:.2f}s) is used."
    )

    if not answer:
        w2_start = default_mid
    else:
        try:
            w2_start = float(answer)
        except ValueError:
            messagebox.showwarning(
                "Invalid input",
                f"Could not parse '{answer}'. Using midpoint {default_mid:.2f}s."
            )
            w2_start = default_mid

    print(f"[INFO] Using W2 start time = {w2_start:.2f} s")

    plot_alpha_w1w2(times, alpha_arrays, w2_start)


if __name__ == "__main__":
    main()
