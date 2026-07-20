"""Alignment algorithms."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from numba import jit


@dataclass
class AlignmentResult:
    """Container for an alignment result.

    Attributes:
        path (np.ndarray): Warping path of shape (L, 2). path[l] = (n, m).
        cost_matrix (np.ndarray): Raw cost matrix (N, M).
        accumulated_cost (np.ndarray or None): Accumulated cost matrix (N, M) if available.
        total_cost (float): Minimum accumulated alignment cost.
        meta (dict): Additional algorithm-specific info.
    """
    path: np.ndarray
    cost_matrix: np.ndarray
    accumulated_cost: np.ndarray = None
    total_cost: float = None
    meta: dict = None


class Aligner(ABC):
    """Base class for alignment algorithms."""

    @abstractmethod
    def align(self, C):
        """Compute an alignment from a cost matrix.

        Args:
            C (np.ndarray): Cost matrix of shape (N, M).

        Returns:
            AlignmentResult
        """
        raise NotImplementedError


@jit(nopython=True)
def _accumulated_cost_basic(C):
    N, M = C.shape
    D = np.zeros((N, M))
    D[0, 0] = C[0, 0]
    for n in range(1, N):
        D[n, 0] = D[n-1, 0] + C[n, 0]
    for m in range(1, M):
        D[0, m] = D[0, m-1] + C[0, m]
    for n in range(1, N):
        for m in range(1, M):
            D[n, m] = C[n, m] + min(D[n-1, m], D[n, m-1], D[n-1, m-1])
    return D


@jit(nopython=True)
def _backtrack_basic(D):
    N, M = D.shape
    n, m = N - 1, M - 1
    P = [(n, m)]
    while n > 0 or m > 0:
        if n == 0:
            cell = (0, m - 1)
        elif m == 0:
            cell = (n - 1, 0)
        else:
            val = min(D[n-1, m-1], D[n-1, m], D[n, m-1])
            if val == D[n-1, m-1]:
                cell = (n-1, m-1)
            elif val == D[n-1, m]:
                cell = (n-1, m)
            else:
                cell = (n, m-1)
        P.append(cell)
        n, m = cell
    P.reverse()
    out = np.empty((len(P), 2), dtype=np.int64)
    for i, (a, b) in enumerate(P):
        out[i, 0] = a
        out[i, 1] = b
    return out


class BasicDTW(Aligner):
    """Classical DTW with step set {(1,0),(0,1),(1,1)} and uniform weights.

    Reference: Müller, FMP Notebooks, C3/C3S2_DTWbasic.
    """
    def align(self, C):
        D = _accumulated_cost_basic(C)
        P = _backtrack_basic(D)
        return AlignmentResult(
            path=P, cost_matrix=C,
            accumulated_cost=D, total_cost=float(D[-1, -1]),
            meta={'algorithm': 'BasicDTW'}
        )


class WeightedDTW(Aligner):
    """DTW with configurable step weights.

    Args:
        step_weights (tuple): (w_diag, w_vert, w_horiz) weights applied to
            the three step choices. Default (1,1,1) matches BasicDTW.
    """
    def __init__(self, step_weights=(1.0, 1.0, 1.0)):
        self.step_weights = step_weights

    def align(self, C):
        wd, wv, wh = self.step_weights
        N, M = C.shape
        D = np.full((N, M), np.inf)
        D[0, 0] = C[0, 0]
        for n in range(1, N):
            D[n, 0] = D[n-1, 0] + wv * C[n, 0]
        for m in range(1, M):
            D[0, m] = D[0, m-1] + wh * C[0, m]
        for n in range(1, N):
            for m in range(1, M):
                D[n, m] = min(
                    D[n-1, m-1] + wd * C[n, m],
                    D[n-1, m]   + wv * C[n, m],
                    D[n, m-1]   + wh * C[n, m],
                )
        P = _backtrack_basic(D)
        return AlignmentResult(
            path=P, cost_matrix=C, accumulated_cost=D,
            total_cost=float(D[-1, -1]),
            meta={'algorithm': 'WeightedDTW', 'step_weights': self.step_weights}
        )


class LibrosaDTW(Aligner):
    """Wrapper around librosa.sequence.dtw for easy access to advanced variants.

    Args:
        step_sizes_sigma (np.ndarray or None): Allowed steps.
        weights_add (np.ndarray or None): Additive weights for each step.
        weights_mul (np.ndarray or None): Multiplicative weights.
        subseq (bool): Enable subsequence DTW.
        backtrack (bool): Whether to compute the optimal path.
        global_constraints (bool): Use Sakoe-Chiba band.
        band_rad (float): Sakoe-Chiba band radius (fraction of max(N,M)).
    """
    def __init__(self, step_sizes_sigma=None, weights_add=None, weights_mul=None,
                 subseq=False, backtrack=True, global_constraints=False, band_rad=0.25):
        import librosa
        self._librosa = librosa
        self.step_sizes_sigma = step_sizes_sigma
        self.weights_add = weights_add
        self.weights_mul = weights_mul
        self.subseq = subseq
        self.backtrack = backtrack
        self.global_constraints = global_constraints
        self.band_rad = band_rad

    def align(self, C):
        D, wp = self._librosa.sequence.dtw(
            C=C,
            step_sizes_sigma=self.step_sizes_sigma,
            weights_add=self.weights_add,
            weights_mul=self.weights_mul,
            subseq=self.subseq,
            backtrack=self.backtrack,
            global_constraints=self.global_constraints,
            band_rad=self.band_rad,
        )
        P = np.asarray(wp[::-1])  # librosa returns path reversed
        return AlignmentResult(
            path=P, cost_matrix=C, accumulated_cost=D,
            total_cost=float(D[P[-1, 0], P[-1, 1]]),
            meta={'algorithm': 'LibrosaDTW', 'subseq': self.subseq}
        )
