# utils/feature_extractor.py

import numpy as np
from pathlib import Path
from typing import List, Dict

class FeatureExtractor:
    @staticmethod
    def extract_fodn_features(
        fodn_root: Path,
        top_pct: int,
        opts: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Compute both summary and top-K per-channel α features from FODN output.
        opts keys:
          - alpha_mean_std
          - alpha_top            (summary)
          - lead_eig
          - spectral_radius
          - sparseness
          - pc_mean              (per-channel mean for top-K)
          - pc_std               (per-channel std for top-K)
        """
        alphas, eig_vals, eig_vecs = [], [], []

        n_ch_ref = None
        # gather A, eigenvalues, eigenvectors
        for seg in sorted(p for p in fodn_root.iterdir() if p.is_dir()):
            for chunk in sorted(p for p in seg.iterdir() if p.is_dir()):
                a_file = chunk / "Alpha_Data.csv"
                c_file = chunk / "Coupling_Data.csv"
                if not a_file.exists() or not c_file.exists():
                    continue

                A = np.loadtxt(a_file, delimiter=",")
                C = np.loadtxt(c_file, delimiter=",")

                # ── sanity checks ───────────────────────────────────────────
                if A.ndim != 2 or C.ndim != 2 or C.shape[0] != C.shape[1]:
                    # malformed csv → skip this chunk
                    continue

                if n_ch_ref is None:
                    n_ch_ref = A.shape[1]
                if A.shape[1] != n_ch_ref:
                    # different channel count → skip
                    continue
                # ────────────────────────────────────────────────────────────

                w, v = np.linalg.eig(C)
                lead = np.max(np.abs(w))
                vec  = np.abs(v[:, np.argmax(w.real)])

                alphas.append(A)
                eig_vals.append(lead)
                eig_vecs.append(vec)

        if not alphas:
            return {}

        # stack into arrays
        A_mat = np.vstack(alphas)     # shape: (chunks×channels)
        V    = np.array(eig_vals)     # shape: (chunks,)
        E    = np.vstack(eig_vecs)    # shape: (chunks×channels)

        out: Dict[str, float] = {}

        # — summary metrics —
        if opts.get("alpha_mean_std"):
            out["FODN_alpha_mean"] = float(A_mat.mean())
            out["FODN_alpha_std"]  = float(A_mat.std())
        if opts.get("alpha_top"):
            # summary top-K α (across channels ranked by E)
            heat = E.mean(axis=0)
            k = max(1, int(len(heat)*top_pct/100))
            idx = np.argsort(heat)[-k:]
            top_vals = A_mat[:, idx]
            out["FODN_alpha_top_mean"] = float(top_vals.mean())
            out["FODN_alpha_top_std"]  = float(top_vals.std())
        if opts.get("lead_eig"):
            out["FODN_lead_eig_mean"] = float(V.mean())
            out["FODN_lead_eig_std"]  = float(V.std())
        if opts.get("spectral_radius"):
            out["FODN_spectral_radius_mean"] = float(V.mean())
            out["FODN_spectral_radius_std"]  = float(V.std())
        if opts.get("sparseness"):
            thresh = 0.01
            sparsity = np.count_nonzero(np.abs(A_mat) > thresh) / A_mat.size
            out["FODN_sparseness"] = float(sparsity)

        # — per-channel top-K features —
        if opts.get("pc_mean") or opts.get("pc_std"):
            heat = E.mean(axis=0)
            n_ch = heat.size
            k    = max(1, int(n_ch * top_pct/100))
            idx  = np.argsort(heat)[-k:]

            for rank, ch in enumerate(idx, 1):
                if opts.get("pc_mean"):
                    out[f"FODN_ch{rank}_alpha_mean"] = float(A_mat[:, ch].mean())
                if opts.get("pc_std"):
                    out[f"FODN_ch{rank}_alpha_std"]  = float(A_mat[:, ch].std())

        return out


    @staticmethod
    def extract_dfa_features(
        dfa_root: Path,
        top_pct: int,
        q_list: List[float],
        opts: Dict[str, bool]
    ) -> Dict[str, float]:
        """
        Compute summary and top-K per-channel H features from DFA output.
        opts keys:
          - H_mean_std
          - DeltaHq
          - Hq_per_q
          - pc_mean   (per-channel H mean for top-K)
          - pc_std    (per-channel H std for top-K)
        """
        # -- summary containers --
        H_list = []
        dHq_list = []
        Hq_acc = {q: [] for q in q_list}

        # -- for per-channel collection --
        # gather channel dirs from first segment
        seg_dirs = sorted(p for p in dfa_root.iterdir() if p.is_dir())
        if not seg_dirs:
            return {}

        first_seg = seg_dirs[0]
        ch_names  = sorted(p.name for p in first_seg.iterdir() if p.is_dir())
        per_ch_H = {name: [] for name in ch_names}

        # -- iterate all segments & channels --
        for seg in seg_dirs:
            for ch in sorted(p for p in seg.iterdir() if p.is_dir()):
                name = ch.name
                for chunk in sorted(p for p in ch.iterdir() if p.is_dir()):
                    h_file  = chunk/"Hurst.csv"
                    hq_file = chunk/"Hq_vs_q.csv"
                    if not h_file.exists() or not hq_file.exists():
                        continue

                    H = float(np.loadtxt(h_file, delimiter=","))
                    arr = np.loadtxt(hq_file, delimiter=",", skiprows=1)
                    qs = arr[:,0]; Hqs = arr[:,1]

                    # summary
                    H_list.append(H)
                    dHq_list.append(float(Hqs.max() - Hqs.min()))
                    if opts.get("Hq_per_q"):
                        for q in q_list:
                            idx = int(np.argmin(np.abs(qs - q)))
                            Hq_acc[q].append(float(Hqs[idx]))

                    # per-channel
                    if name in per_ch_H:
                        per_ch_H[name].append(H)

        out: Dict[str, float] = {}

        # — summary metrics —
        if H_list:
            if opts.get("H_mean_std"):
                out["DFA_H_mean"] = float(np.mean(H_list))
                out["DFA_H_std"]  = float(np.std(H_list))
            if opts.get("DeltaHq"):
                out["DFA_DeltaHq_mean"] = float(np.mean(dHq_list))
                out["DFA_DeltaHq_std"]  = float(np.std(dHq_list))
            if opts.get("Hq_per_q"):
                for q in q_list:
                    vals = Hq_acc.get(q, [])
                    if vals:
                        out[f"DFA_Hq{q}_mean"] = float(np.mean(vals))
                        out[f"DFA_Hq{q}_std"]  = float(np.std(vals))

        # — per-channel top-K H features —
        if opts.get("pc_mean") or opts.get("pc_std"):
            means = []
            stds  = []
            for name in ch_names:
                arr = np.array(per_ch_H[name]) if per_ch_H[name] else np.array([np.nan])
                means.append(float(np.nanmean(arr)))
                stds .append(float(np.nanstd(arr)))
            means = np.array(means); stds = np.array(stds)
            n_ch = means.size
            k    = max(1, int(n_ch * top_pct/100))
            idx  = np.argsort(means)[-k:]
            for rank, j in enumerate(idx, 1):
                if opts.get("pc_mean"):
                    out[f"DFA_ch{rank}_H_mean"] = float(means[j])
                if opts.get("pc_std"):
                    out[f"DFA_ch{rank}_H_std"]  = float(stds[j])

        return out
