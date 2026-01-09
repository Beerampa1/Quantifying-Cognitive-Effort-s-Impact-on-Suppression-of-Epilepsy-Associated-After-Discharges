#!/usr/bin/env python
"""
Standalone FODN Eigenvector Heatmap Generator
---------------------------------------------

This script:

1. Asks the user to select a root folder that contains FODN_*_c*s result folders.
2. Recursively finds all FODN_*_c*s folders.
3. For each chunk folder (chunkX_start-end):
       - Loads Coupling_Data.csv
       - Computes the dominant eigenvector
       - Records the chunk start time (in seconds)
4. Sorts all chunks by their start time across all FODN folders.
5. Optionally asks for a channel-name file (TXT or CSV) and uses those
   as Y-axis labels if the length matches the number of eigenvector components.
6. Plots a heatmap:

   X-axis = actual chunk start time (seconds)
   Y-axis = eigenvector component (channel / channel name)
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
# Dominant eigenvector
# ----------------------------------------------------------------------
def compute_dominant_eigenvector(matrix):
    """
    Compute the dominant (largest eigenvalue) eigenvector of 'matrix'.
    Returns the real part of that eigenvector.
    """
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    idx_max = np.argmax(eigenvalues.real)
    return eigenvectors[:, idx_max].real


# ----------------------------------------------------------------------
# Load eigenvectors + times from FODN folders
# ----------------------------------------------------------------------
def load_eigenvectors_with_time(root_folder):
    """
    From root_folder, find all FODN_*_c*s folders, then for each chunk
    load Coupling_Data.csv, compute the dominant eigenvector, and
    associate it with the chunk start time.

    Returns
    -------
    times : np.ndarray, shape (num_windows,)
        Start time (seconds) of each chunk/window.
    evectors : np.ndarray, shape (num_channels, num_windows)
        Dominant eigenvectors stacked as columns.
    """
    fodn_folders = find_fodn_folders(root_folder)
    if not fodn_folders:
        return None, None

    chunk_pattern = r"chunk(\d+)_([0-9.]+)-([0-9.]+)"
    time_evec_pairs = []

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

            coupling_file = os.path.join(chunk_dir, "Coupling_Data.csv")
            if not os.path.exists(coupling_file):
                print(f"[WARN] Missing Coupling_Data.csv in: {chunk_dir}")
                continue

            matrix = pd.read_csv(coupling_file, header=None).values

            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                print(f"[WARN] Non-square matrix in: {chunk_dir}  shape={matrix.shape}")
                continue

            eig = compute_dominant_eigenvector(matrix)
            time_evec_pairs.append((start_time, eig))


    if not time_evec_pairs:
        return None, None

    # Sort by time
    time_evec_pairs.sort(key=lambda x: x[0])
    times = np.array([t for t, _ in time_evec_pairs])
    evectors = np.column_stack([v for _, v in time_evec_pairs])

    return times, evectors


# ----------------------------------------------------------------------
# Optional: load channel names
# ----------------------------------------------------------------------
def load_channel_names(num_channels):
    """
    Ask the user for an optional channel-name file (txt or csv).
    Returns
    -------
    names : list[str] or None
        If user cancels or length mismatch, returns None.
    """
    answer = messagebox.askyesno(
        "Channel names",
        "Do you want to load channel names for the Y-axis?\n\n"
        "If yes, choose a TXT (one name per line) or CSV file."
    )
    if not answer:
        return None

    fname = filedialog.askopenfilename(
        title="Select channel-name file",
        filetypes=[
            ("Text / CSV", "*.txt *.csv"),
            ("All files", "*.*")
        ]
    )
    if not fname:
        return None

    names = None
    try:
        if fname.lower().endswith(".txt"):
            with open(fname, "r", encoding="utf-8") as f:
                names = [line.strip() for line in f if line.strip()]
        else:
            df = pd.read_csv(fname, header=None)
            if df.shape[0] == 1:
                names = df.iloc[0, :].astype(str).tolist()
            elif df.shape[1] == 1:
                names = df.iloc[:, 0].astype(str).tolist()
            else:
                # If it's a matrix, just flatten first row
                names = df.iloc[0, :].astype(str).tolist()
    except Exception as e:
        messagebox.showerror("Channel name load error", f"Could not read file:\n{e}")
        return None

    if names is None or len(names) != num_channels:
        messagebox.showwarning(
            "Channel names mismatch",
            f"Loaded {len(names) if names else 0} names, "
            f"but there are {num_channels} eigenvector components.\n\n"
            "Using channel indices instead."
        )
        return None

    return names


# ----------------------------------------------------------------------
# Plot eigenvector heatmap
# ----------------------------------------------------------------------
def plot_heatmap(times, evectors, channel_names=None, w2_start=None):
    """
    Plot eigenvector heatmap with time on X-axis and channel names (if provided)
    on Y-axis.

    Now includes:
    - Ask user for a folder to save the heatmap
    - Save at 300 DPI
    """
    num_channels, num_windows = evectors.shape

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("Eigenvector Heatmap", fontsize=22, fontweight="bold")

    # extent = [x_min, x_max, y_max, y_min] to flip Y-axis
    im = ax.imshow(
        evectors,
        aspect='auto',
        cmap='viridis',
        interpolation='nearest',
        extent=[times[0], times[-1], num_channels, 0]
    )

    ax.set_xlabel("Time (seconds)", fontsize=18, fontweight="bold")
    ax.set_ylabel(
        "Channels" if channel_names is not None else "Eigenvector Component (Channel Index)",
        fontsize=18,
        fontweight="bold"
    )

    # X ticks
    if num_windows <= 20:
        xticks = times
    else:
        step = max(1, num_windows // 15)
        xticks = times[::step]

    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.2f}" for t in xticks], rotation=45, fontsize=10, ha="right")

    # Y ticks
    yticks = np.arange(num_channels) + 0.5
    ax.set_yticks(yticks)
    if channel_names is not None:
        ax.set_yticklabels(
            channel_names,
            fontsize=5,               # smaller
            color="#444444",           # lighter dark gray
            fontweight="normal",
            ha="right"                 # right-align for clarity
        )
    else:
        ax.set_yticklabels(
            np.arange(num_channels),
            fontsize=5,
            color="#444444",
            fontweight="normal",
            ha="right"
        )
    ax.tick_params(axis='y', pad=4)
    

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Eigenvector value")
       

        

    # === Highlight the chosen point on the x-axis (NO vertical line) ===
    if w2_start is not None:
        # Get current ticks
        xticks = list(ax.get_xticks())

        # Check if a tick is already very close to w2_start
        tol = (times[-1] - times[0]) * 0.01  # 1% of total range
        already_there = any(abs(t - w2_start) < tol for t in xticks)

        # If not, add w2_start as an extra tick
        if not already_there:
            xticks.append(w2_start)
            xticks = sorted(xticks)

        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{t:.2f}" for t in xticks],
                           rotation=45, fontsize=10, ha="right")

        # Color and bold the tick label at w2_start
        for tick, t in zip(ax.get_xticklabels(), xticks):
            if abs(t - w2_start) < tol:
                tick.set_color("red")
                tick.set_fontweight("bold")

    plt.tight_layout()



       # -------------------------------------------------------------
    # NEW: Ask user where to save the image (folder selection)
    # -------------------------------------------------------------
    save_folder = filedialog.askdirectory(title="Select folder to save heatmap (PNG)")
    if save_folder:
        save_path = os.path.join(save_folder, "Eigenvector_Heatmap.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[SAVED] High-resolution heatmap saved to:\n{save_path}")
    else:
        print("[INFO] No folder selected — heatmap not saved.")

    plt.show()

    


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo(
        "FODN eigenvector heatmap",
        "Select the root folder that contains FODN_*_c*s result folders."
    )

    folder = filedialog.askdirectory(title="Select folder with FODN_*_c*s subfolders")
    if not folder:
        print("No folder selected. Exiting.")
        return

    print(f"[INFO] Selected root folder:\n  {folder}")

    times, evectors = load_eigenvectors_with_time(folder)
    if times is None or evectors is None:
        messagebox.showerror(
            "No data",
            "No valid Coupling_Data.csv files found under FODN_*_c*s folders."
        )
        return

    print(f"[INFO] Loaded {len(times)} chunks.")
    print(f"[INFO] Each eigenvector has {evectors.shape[0]} components (channels).")
        # ask for W2 start time (same style as alpha script)
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


    # Optionally load channel names
    channel_names = load_channel_names(evectors.shape[0])
   

    # --- NEW: let user remove noisy channels manually -----------------
    if channel_names is not None:
        prompt = (
            "Enter comma-separated channel names to EXCLUDE "
            "(e.g., DP1,DP2,EKG1), or leave blank to keep all:"
        )
        ans_noise = simpledialog.askstring("Remove noisy channels", prompt)
        if ans_noise:
            bad_names = [s.strip() for s in ans_noise.split(",") if s.strip()]
            keep_idx = [i for i, nm in enumerate(channel_names) if nm not in bad_names]
            if not keep_idx:
                messagebox.showwarning(
                    "No channels left",
                    "All channels were excluded. Nothing to plot."
                )
                return
            evectors = evectors[keep_idx, :]
            channel_names = [channel_names[i] for i in keep_idx]
            print(f"[INFO] Excluded channels: {bad_names}")
            print(f"[INFO] Remaining channels: {len(channel_names)}")
    else:
        prompt = (
            "No channel names loaded.\n\n"
            "Enter comma-separated channel INDICES (0-based) to EXCLUDE "
            "(e.g., 0,1,2), or leave blank to keep all:"
        )
        ans_noise = simpledialog.askstring("Remove noisy channels", prompt)
        if ans_noise:
            try:
                bad_idx = sorted(
                    {int(s.strip()) for s in ans_noise.split(",") if s.strip()}
                )
            except ValueError:
                messagebox.showwarning(
                    "Invalid indices",
                    "Could not parse some indices. No channels were removed."
                )
            else:
                num_ch = evectors.shape[0]
                keep_idx = [i for i in range(num_ch) if i not in bad_idx and 0 <= i < num_ch]
                if not keep_idx:
                    messagebox.showwarning(
                        "No channels left",
                        "All channels were excluded. Nothing to plot."
                    )
                    return
                evectors = evectors[keep_idx, :]
                print(f"[INFO] Excluded channel indices: {bad_idx}")
                print(f"[INFO] Remaining channels: {evectors.shape[0]}")
    # ------------------------------------------------------------------

    # Plot
    plot_heatmap(times, evectors, channel_names=channel_names, w2_start=w2_start)


if __name__ == "__main__":
    main()
