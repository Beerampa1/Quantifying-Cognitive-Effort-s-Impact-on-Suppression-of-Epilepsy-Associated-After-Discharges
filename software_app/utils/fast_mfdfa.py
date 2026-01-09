# utils/fast_mfdfa.py
import os
import numpy as np

USE_NUMBA = os.getenv("MF_BACKEND", "numba").lower() == "numba"
try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False
    USE_NUMBA = False

def _vandermonde(n, m):
    i = np.arange(n, dtype=float)
    cols = [i**p for p in range(m, -1, -1)]
    return np.stack(cols, axis=1)

def _precompute_pinv(X):
    Xt = X.T
    return np.linalg.pinv(Xt @ X) @ Xt

def run_mfdfa_numpy(x, scales, q_vals, m=1):
    eps = np.finfo(float).eps
    x = np.asarray(x, dtype=float)
    Y = np.cumsum(x - x.mean())
    F  = np.full(len(scales), np.nan, dtype=float)
    Fq = np.full((len(q_vals), len(scales)), np.nan, dtype=float)

    for si, s in enumerate(scales):
        s = int(s)
        segs = Y.size // s
        if segs == 0:
            continue
        X    = _vandermonde(s, m)
        pinv = _precompute_pinv(X)
        RMS  = np.empty(segs, dtype=float)
        for v in range(segs):
            seg  = Y[v*s:(v+1)*s]
            coef = pinv @ seg
            fit  = X @ coef
            res  = seg - fit
            r    = np.sqrt(np.mean(res*res))
            RMS[v] = r if r > eps else eps

        F[si] = np.sqrt(np.mean(RMS*RMS))
        for ji, q in enumerate(q_vals):
            if q == 0:
                Fq[ji, si] = np.exp(0.5 * np.mean(np.log(RMS**2)))
            else:
                Fq[ji, si] = (np.mean(RMS**q))**(1.0/q)

    lg_s = np.log2(scales)
    lg_F = np.log2(F + eps)
    good = np.isfinite(lg_F)
    H = np.polyfit(lg_s[good], lg_F[good], 1)[0] if good.sum() > 1 else np.nan

    Hq = np.empty(len(q_vals), dtype=float)
    for ji in range(len(q_vals)):
        lg = np.log2(Fq[ji] + eps)
        good = np.isfinite(lg)
        Hq[ji] = np.polyfit(lg_s[good], lg[good], 1)[0] if good.sum() > 1 else np.nan
    return H, F, Hq, Fq

if HAVE_NUMBA:
    @njit(cache=True, fastmath=True)
    def _vandermonde_nb(n, m):
        X = np.empty((n, m+1), dtype=np.float64)
        for c in range(m+1):
            p = m - c
            for i in range(n):
                X[i, c] = i**p
        return X

    @njit(cache=True, fastmath=True)
    def _pinv_nb(X):
        Xt = X.T
        A  = Xt @ X
        U, s, Vt = np.linalg.svd(A)
        s_inv = np.zeros_like(s)
        for i in range(s.size):
            if s[i] > 1e-12:
                s_inv[i] = 1.0/s[i]
        A_pinv = (Vt.T * s_inv) @ U.T
        return A_pinv @ Xt

    @njit(cache=True, fastmath=True, parallel=True)
    def _core_nb(Y, scales, q_vals, m, eps):
        S = scales.size
        Q = q_vals.size
        F  = np.full(S, np.nan)
        Fq = np.full((Q, S), np.nan)
        for si in prange(S):
            s = int(scales[si])
            segs = Y.size // s
            if segs == 0:
                continue
            X    = _vandermonde_nb(s, m)
            pinv = _pinv_nb(X)
            RMS  = np.empty(segs)
            for v in range(segs):
                seg  = Y[v*s:(v+1)*s]
                coef = pinv @ seg
                fit  = X @ coef
                res  = seg - fit
                r    = np.sqrt(np.mean(res*res))
                RMS[v] = r if r > eps else eps
            F[si] = np.sqrt(np.mean(RMS*RMS))
            for ji in range(Q):
                q = q_vals[ji]
                if q == 0.0:
                    ssum = 0.0
                    for r in RMS:
                        ssum += np.log(r*r)
                    Fq[ji, si] = np.exp(0.5 * (ssum / RMS.size))
                else:
                    ssum = 0.0
                    for r in RMS:
                        ssum += (r**q)
                    Fq[ji, si] = (ssum / RMS.size) ** (1.0/q)
        return F, Fq

    def run_mfdfa_numba(x, scales, q_vals, m=1):
        eps = np.finfo(float).eps
        x = np.asarray(x, dtype=np.float64)
        Y = np.cumsum(x - x.mean())
        scales = np.asarray(scales, dtype=np.float64)
        q_vals = np.asarray(q_vals, dtype=np.float64)
        F, Fq = _core_nb(Y, scales, q_vals, m, eps)
        lg_s = np.log2(scales)
        lg_F = np.log2(F + eps)
        good = np.isfinite(lg_F)
        H = np.polyfit(lg_s[good], lg_F[good], 1)[0] if good.sum() > 1 else np.nan
        Hq = np.empty(len(q_vals))
        for ji in range(len(q_vals)):
            lg = np.log2(Fq[ji] + eps)
            good = np.isfinite(lg)
            Hq[ji] = np.polyfit(lg_s[good], lg[good], 1)[0] if good.sum() > 1 else np.nan
        return H, F, Hq, Fq

def run_mfdfa_fast(x, scales, q_vals, m=1):
    if USE_NUMBA and HAVE_NUMBA:
        return run_mfdfa_numba(x, scales, q_vals, m)
    return run_mfdfa_numpy(x, scales, q_vals, m)
