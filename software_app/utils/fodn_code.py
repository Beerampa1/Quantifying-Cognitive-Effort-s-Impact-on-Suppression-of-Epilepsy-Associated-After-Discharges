# utils/fodn_code.py
# ----------------------------------------------------------------------
# Fractional-Order Dynamic Network (FODN) utilities.
#
# This file contains:
#   1) HaarWaveletTransform
#        • A simple 1D Haar wavelet transform used to estimate fractional order.
#   2) fracOrdUU
#        • Estimates fractional orders per channel
#        • Builds fractional differencing ("zVec") via Gamma-function coefficients
#        • Iteratively estimates:
#            - A: coupling/state transition matrix between channels
#            - u: sparse "input"/innovation terms (via LASSO / ADMM)
#
# Notes / caveats:
#   • This code assumes 1D signals per channel: X shape (numCh, K)
#   • Several methods use dense linear algebra; sparse mode is partially supported.
#   • Some parts look like legacy code (e.g., sparse branch in _getLassoSoln),
#     but are preserved and documented rather than re-written here.
# ----------------------------------------------------------------------

import numpy as np
from scipy.special import gamma
import scipy.linalg as LA
import scipy.sparse as spSparse
import scipy.sparse.linalg as sLA
import time


class HaarWaveletTransform(object):
    """
    Minimal 1D Haar wavelet transform.

    Given a 1D signal X of length N, the transform produces:
      • W: approximation coefficients per scale (stored row-wise)
      • D: detail coefficients per scale (stored row-wise)

    This is used later to estimate the fractional order via a log-variance
    vs scale regression of Haar detail coefficients.
    """
    def __init__(self, X):
        # Store original shape and signal as numpy array
        self._N = np.shape(X)
        self.X = np.array(X)

        # Normalize/validate dimensionality:
        # - Accepts 1D arrays
        # - Accepts (1, N) or (N, 1) and squeezes them to 1D
        # - Rejects multi-dimensional arrays (> 2D or 2D non-vector)
        try:
            if np.size(self._N) == 1:
                self._N = self._N[0]
            elif np.size(self._N) > 1:
                if self._N[0] == 1 or self._N[1] == 1:
                    self.X = np.squeeze(X)
                    self._N = np.size(self.X)
                else:
                    raise Exception('dimErr')
        except Exception as err:
            errStatus = err.args[0]
            if errStatus == 'dimErr':
                print('Only single dimensional arrays are acceptable')

    def normalize(self):
        """Zero-center the signal (mean removal)."""
        mean = np.mean(self.X)
        self.X -= mean

    def _dwthaar(self, Signal):
        """
        One-level Haar "DWT" step:
          • C: approximation (low-pass) coefficients
          • S: detail (high-pass) coefficients

        Implementation:
          - Pair adjacent samples and compute average and difference.
          - Applies scaling so the transform is energy-consistent.
        """
        # Use only full pairs
        NUse = int(np.floor(np.size(Signal) / 2))

        # Pairwise average
        C = (Signal[:2 * NUse:2] + Signal[1:2 * NUse:2]) / 2

        # Difference from the average (detail)
        S = Signal[:2 * NUse:2] - C

        # Scaling factors (specific to the original author's convention)
        C = 2 * C / np.sqrt(2)
        S = -2 * S / np.sqrt(2)
        return C, S

    def transform(self):
        """
        Compute multi-scale Haar transform up to floor(log2(N)) levels.

        Returns
        -------
        W : np.ndarray
            Approximation coefficients per level (rows), zero-padded
        D : np.ndarray
            Detail coefficients per level (rows), zero-padded
        """
        Nby2 = int(np.floor(self._N / 2))
        W = np.zeros((Nby2, Nby2))  # rows correspond to levels
        D = np.zeros((Nby2, Nby2))

        j = self._N
        Signal = self.X

        # Number of dyadic scales: floor(log2(N))
        for i in range(int(np.floor(np.log2(self._N)))):
            j = int(np.floor(j / 2))
            w, d = self._dwthaar(Signal)
            W[i, :j] = w
            D[i, :j] = d
            # Next scale operates on approximation coefficients
            Signal = w

        return W, D


class fracOrdUU(object):
    """
    Fractional-order model with sparse innovations / inputs.

    Typical usage:
        model = fracOrdUU(numFract=50, niter=10, lambdaUse=0.5)
        model.fit(X)  # X shape: (numCh, K)

    High-level algorithm (as implemented):
      1) Estimate fractional order per channel using Haar detail variances.
      2) Build zVec via fractional differencing (Gamma coefficients).
      3) Initial least-squares estimate for A (coupling matrix).
      4) If B not provided, choose B heuristically from A.
      5) Iteratively:
           - Solve for sparse u(:,k) with LASSO (ADMM) per time step.
           - Re-estimate A with least squares on (zVec - B*u).
    """
    def __init__(self, numInp=[], numFract=50, niter=10, B=[], lambdaUse=0.5, verbose=0):
        self.verbose = verbose
        self._order = []                    # fractional order per channel
        self._numCh = []                    # number of channels
        self._K = []                        # number of samples per channel
        self._numInp = numInp               # number of "inputs" (columns of B)
        self._numFract = numFract           # number of fractional coefficients (memory length)
        self._lambdaUse = lambdaUse         # sparsity penalty strength (LASSO)
        self._niter = niter                 # number of outer iterations
        self._BMat = B                      # input mixing matrix B (numCh x numInp)
        self._zVec = []                     # fractional differenced signals (numCh x K)
        self._AMat = []                     # A estimates across iterations (niter+1 x numCh x numCh)
        self._u = []                        # sparse input sequence (numInp x K)
        self._performSparseComputation = False
        self._preComputedVars = []

    # ------------------------------------------------------------------
    # Fractional order estimation via Haar wavelet variance regression
    # ------------------------------------------------------------------
    def _getFractionalOrder(self, x):
        """
        Estimate fractional order d from log2(variance(detail_coeffs)) vs scale.

        The method:
          • Compute Haar detail coefficients at each dyadic scale.
          • For each scale i, compute unbiased variance of detail coeffs.
          • Fit a line p[0] to (scale_index, log2(var)).
          • Return p[0]/2 (author's convention).
        """
        numScales = int(np.floor(np.log2(self._K)))
        log_wavelet_scales = np.zeros((numScales,))
        scale = np.arange(1, numScales + 1)

        Wt = HaarWaveletTransform(x)
        Wt.normalize()
        _, W = Wt.transform()

        j = int(np.floor(self._K / 2))
        for i in range(numScales - 1):
            y = W[i, :j]
            variance = np.var(y, ddof=1)  # unbiased estimate (ddof=1)

            # Avoid log(0) or log(negative) due to numerical issues
            if variance <= 0:
                variance = 1e-10

            log_wavelet_scales[i] = np.log2(variance)
            j = int(np.floor(j / 2))

        p = np.polyfit(scale[:numScales - 1], log_wavelet_scales[:numScales - 1], 1)
        return p[0] / 2

    def _estimateOrder(self, X):
        """Estimate fractional order for each channel in X."""
        self._order = np.empty((self._numCh,))
        for i in range(self._numCh):
            self._order[i] = self._getFractionalOrder(X[i, :])

    # ------------------------------------------------------------------
    # Fractional differencing / pre-processing
    # ------------------------------------------------------------------
    def _updateZVec(self, X):
        """
        Build zVec from X via fractional differencing coefficients.

        For each channel i:
          • Compute preFactVec[j] = Γ(-d + j) / (Γ(-d) Γ(j+1))
          • Convolve with signal X[i,:]
          • Take first K samples for zVec (same length as input)
        """
        self._zVec = np.empty((self._numCh, self._K))
        j = np.arange(0, self._numFract + 1)

        for i in range(self._numCh):
            preFactVec = gamma(-self._order[i] + j) / gamma(-self._order[i]) / gamma(j + 1)
            y = np.convolve(X[i, :], preFactVec)
            self._zVec[i, :] = y[:self._K]

    # ------------------------------------------------------------------
    # Heuristic B selection
    # ------------------------------------------------------------------
    def _setHeuristicBMat(self, A):
        """
        Choose B (numCh x numInp) heuristically from A.

        Idea:
          • Threshold A to form B candidates where |A| > 0.01
          • Use QR to find independent columns
          • If not enough independent columns exist, fallback to a simple
            [I; 0] structure (first numInp channels as inputs).
        """
        B = np.zeros((self._numCh, self._numCh))
        B[np.abs(A) > 0.01] = A[np.abs(A) > 0.01]

        # QR decomposition to assess column independence
        _, r = LA.qr(B)
        colInd = np.where(np.abs(np.diag(r)) > 1e-7)

        if np.size(colInd[0]) < self._numInp:
            # Fallback: take first numInp channels as "inputs"
            self._BMat = np.vstack((np.eye(self._numInp),
                                    np.zeros((self._numCh - self._numInp, self._numInp))))
        else:
            colInd = colInd[0][:self._numInp]
            self._BMat = B[:, colInd]

        if np.linalg.matrix_rank(B) < self._numInp:
            raise Exception('rank deficient B')

    # ------------------------------------------------------------------
    # Least squares A estimation
    # ------------------------------------------------------------------
    def _performLeastSq(self, Y, X):
        """
        Least squares estimate for A in:
            Y[k] ≈ A @ X[k-1]  (channel-wise)

        Inputs
        ------
        Y, X : arrays of shape (K, numCh)
            (Note: calling code transposes zVec and X accordingly.)

        Returns
        -------
        A : np.ndarray (numCh x numCh)
            Estimated coupling matrix
        mse : float
            Mean MSE across channels
        """
        # Build lagged X (XUse[k] = X[k-1], with XUse[0]=0)
        XUse = np.vstack((np.zeros((1, self._numCh)), X[:-1, :]))

        # Regularization to avoid singularity in (XUse.T XUse)
        reg = 1e-8 * np.eye(XUse.shape[1])

        # A = Y^T X (X^T X)^-1  (with ridge-like stabilization)
        A = np.matmul(np.matmul(Y.T, XUse), LA.inv(np.matmul(XUse.T, XUse) + reg))

        # Channel-wise MSE
        mse = LA.norm(Y - np.matmul(XUse, A.T), axis=0) ** 2 / self._K
        return A, np.mean(mse)

    # ------------------------------------------------------------------
    # ADMM / LASSO helpers
    # ------------------------------------------------------------------
    def _factor(self, A, rho):
        """
        Pre-factorization helper used in ADMM for LASSO.
        Returns (L, U) where U = L^T (Cholesky factorization).

        Dense vs sparse:
          • If _performSparseComputation is True, uses sparse CSC storage.
          • Otherwise uses dense arrays.
        """
        m, n = np.shape(A)

        if self._performSparseComputation:
            if m >= n:
                L_val = LA.cholesky(np.matmul(A.T, A) + rho * spSparse.eye(n), lower=True)
            else:
                L_val = LA.cholesky(spSparse.eye(m) + 1 / rho * np.matmul(A, A.T), lower=True)

            L_val = spSparse.csc_matrix(L_val)
            U_val = spSparse.csc_matrix(L_val.T)

        else:
            if m >= n:
                L_val = LA.cholesky(np.matmul(A.T, A) + rho * np.eye(n), lower=True)
            else:
                L_val = LA.cholesky(np.eye(m) + 1 / rho * np.matmul(A, A.T), lower=True)

            U_val = L_val.T

        return L_val, U_val

    def _shrinkage(self, x, kappa):
        """
        Soft-thresholding operator:
            S_kappa(x) = sign(x) * max(|x| - kappa, 0)
        Used as the proximal operator for the L1 norm.
        """
        return np.maximum(0, x - kappa) - np.maximum(0, -x - kappa)

    def _objective(self, A, b, lambdaUse, x, z):
        """LASSO objective value for monitoring ADMM progress."""
        return 0.5 * np.sum((np.matmul(A, x) - b) ** 2) + lambdaUse * LA.norm(z, ord=1)

    class _history(object):
        """Container for ADMM diagnostics per iteration."""
        def __init__(self, N):
            self._objval = np.empty((N,))
            self._r_norm = np.empty((N,))
            self._s_norm = np.empty((N,))
            self._eps_pri = np.empty((N,))
            self._eps_dual = np.empty((N,))

    class _preComputedVars_(object):
        """
        Cache for matrices used repeatedly in LASSO solves:
          • L and U are Cholesky factors
          • LInv and UInv are their inverses (dense)
        """
        def __init__(self):
            self._lasso_L = []
            self._lasso_U = []
            self._lasso_LInv = []
            self._lasso_UInv = []

        def _updateLassoLUMat(self, A, rho):
            """
            Precompute and store (L, U) and their inverses for repeated solves.
            Note: uses a fresh fracOrdUU() instance only to access _factor().
            """
            self._lasso_L, self._lasso_U = fracOrdUU()._factor(A, rho)
            self._lasso_LInv = LA.inv(self._lasso_L)
            self._lasso_UInv = LA.inv(self._lasso_U)

    def _getLassoSoln(self, b, lambdaUse):
        """
        Solve the LASSO problem:
            minimize 0.5 ||A x - b||_2^2 + lambda ||x||_1
        where A is BMat and b is the current residual (per time step).

        Uses ADMM with fixed iteration limits and tolerances.

        Returns
        -------
        z : np.ndarray
            The sparse solution (ADMM's z variable) squeezed to 1D.
        """
        A = self._BMat
        b = np.reshape(b, (np.size(b), 1))

        MAX_ITER = 100
        ABSTOL = 1e-4
        RELTOL = 1e-2

        m, n = np.shape(A)
        Atb = np.matmul(A.T, b)

        # ADMM parameter; common heuristic is rho ~ 1/lambda
        rho = 1 / lambdaUse
        alpha = 1  # over-relaxation parameter (1 = none)

        # ADMM variables
        z = np.zeros((n, 1))
        u = np.zeros((n, 1))

        # Precomputed inverses for fast solves
        LInv = self._preComputedVars._lasso_LInv
        UInv = self._preComputedVars._lasso_UInv

        history = self._history(MAX_ITER)

        for k in range(MAX_ITER):
            # x-update: solve (A^T A + rho I)x = Atb + rho(z - u)
            q = Atb + rho * (z - u)

            # Dense branch: x = U^{-1} L^{-1} q
            # Sparse branch: appears to be legacy / incomplete; left as-is.
            if self._performSparseComputation:
                if m >= n:
                    x = sLA.inv(UInv)
                else:
                    x = q / rho - np.matmul(
                        A.T,
                        sLA.inv(UInv) * (sLA.inv(LInv) * np.matmul(A, q))
                    ) / rho ** 2
            else:
                if m >= n:
                    x = np.matmul(UInv, np.matmul(LInv, q))
                else:
                    x = q / rho - np.matmul(
                        A.T,
                        np.matmul(LA.inv(UInv), np.matmul(LA.inv(LInv), np.matmul(A, q)))
                    ) / rho ** 2

            # z-update with relaxation + soft thresholding
            zold = np.array(z)
            x_hat = alpha * x + (1 - alpha) * zold
            z = self._shrinkage(x_hat + u, lambdaUse / rho)

            # dual update
            u += x_hat - z

            # Diagnostics / stopping criteria
            history._objval[k] = self._objective(A, b, lambdaUse, x, z)
            history._r_norm[k] = LA.norm(x - z)
            history._s_norm[k] = LA.norm(-rho * (z - zold))
            history._eps_pri[k] = (np.sqrt(n) * ABSTOL + RELTOL * np.max((LA.norm(x), LA.norm(-z))))
            history._eps_dual[k] = np.sqrt(n) * ABSTOL + RELTOL * LA.norm(rho * u)

            if (history._r_norm[k] < history._eps_pri[k] and
                history._s_norm[k] < history._eps_dual[k]):
                break

        return np.squeeze(z)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(self, X):
        """
        Fit the fractional-order model to multichannel data X.

        Parameters
        ----------
        X : array-like, shape (numCh, K)
            Multichannel time series data.

        Side effects / stored results
        -----------------------------
        self._order : estimated fractional orders per channel
        self._zVec  : fractional differenced signals
        self._AMat  : A estimates for each iteration (0..niter)
        self._BMat  : B matrix (provided or heuristic)
        self._u     : sparse inputs per time sample
        """
        X = np.array(X, dtype='float')
        self._numCh, self._K = np.shape(X)

        # Default: numInp ~ half the channels
        if np.size(self._numInp) == 0:
            self._numInp = int(np.floor(self._numCh / 2))

        # Store A across iterations (including initial estimate at index 0)
        self._AMat = np.empty((self._niter + 1, self._numCh, self._numCh))

        # Sparse inputs (numInp x K)
        self._u = np.zeros((self._numInp, self._K))

        try:
            # Basic sanity checks
            if self._numCh == 1:
                raise Exception('oneSensor')
            if self._K < self._numCh:
                raise Exception('lessData')
            if np.size(self._BMat) > 0:
                if np.shape(self._BMat) != (self._numCh, self._numInp):
                    raise Exception('BMatDim')

            # 1) Fractional order per channel
            self._estimateOrder(X)

            # 2) Build fractional differenced series zVec
            self._updateZVec(X)

            # 3) Initial A from least squares: zVec ≈ A * X_lag
            self._AMat[0, :, :], mse = self._performLeastSq(self._zVec.T, X.T)

            # 4) Choose B if not provided
            if np.size(self._BMat) == 0:
                self._setHeuristicBMat(self._AMat[0, :, :])

            # 5) Precompute LASSO solver matrices for repeated ADMM calls
            self._preComputedVars = self._preComputedVars_()
            self._preComputedVars._updateLassoLUMat(self._BMat, 1 / self._lambdaUse)

            t0 = time.time()
            if self.verbose > 0:
                print('beginning mse = %f' % (mse))

            mseIter = np.empty((self._niter + 1,))
            mseIter[0] = mse

            # Outer loop: alternate between u (LASSO) and A (least squares)
            for iterInd in range(self._niter):
                # Estimate sparse inputs u[:, k] for k=1..K-1
                for kInd in range(1, self._K):
                    # residual yUse = z[k] - A*x[k-1]
                    yUse = self._zVec[:, kInd] - np.matmul(self._AMat[iterInd, :, :], X[:, kInd - 1])
                    self._u[:, kInd] = self._getLassoSoln(yUse, self._lambdaUse)

                # Re-estimate A using corrected zVec = zVec - B*u
                self._AMat[iterInd + 1, :, :], mseIter[iterInd + 1] = self._performLeastSq(
                    (self._zVec - np.matmul(self._BMat, self._u)).T,
                    X.T
                )

                if self.verbose > 0:
                    print('iter ind = %d, mse = %f' % (iterInd, mseIter[iterInd + 1]))

            print('time taken = %f' % (time.time() - t0))

        except Exception as err:
            # Friendly error messages for common failure modes
            errStatus = err.args[0]
            if errStatus == 'oneSensor':
                print('The number of sensors must be > 1, retry...')
            elif errStatus == 'lessData':
                print('The number of data points are less than number of sensors, retry...')
            elif errStatus == 'BMatDim':
                print('size of B should be consistent with the number of channels and number of inputs')
            else:
                print('some different error')
