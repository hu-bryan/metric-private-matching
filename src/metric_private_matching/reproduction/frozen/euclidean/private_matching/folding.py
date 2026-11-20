"""Public-task folding map.

The basis is built from public Q only. Private points may be folded in the
simulator's evaluation layer; the server never sees true R_fold or noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_matching.assignment import pairwise_euclidean

DEFAULT_RANK_EPS_MULTIPLIER = 1.0


@dataclass(frozen=True)
class FoldingMap:
    """Thin public basis and folded dimension.

    ``B`` has shape ``(d, k)`` with orthonormal columns spanning the centered
    public-task row space, excluding the origin row ``q_1``.
    """

    B: np.ndarray
    k: int
    m: int
    q1: np.ndarray
    rank_tol: float
    singular_values: np.ndarray
    unexpected_rank_loss: bool
    full_span: bool

    @property
    def d(self) -> int:
        return int(self.B.shape[0])

    def center(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float64) - self.q1

    def fold_public(self, Q: np.ndarray) -> np.ndarray:
        P = self.center(Q)
        coords = P @ self.B
        if self.full_span:
            return np.ascontiguousarray(coords)
        radial = np.zeros((P.shape[0], 1), dtype=np.float64)
        return np.concatenate([coords, radial], axis=1)

    def fold_private(self, R: np.ndarray) -> np.ndarray:
        """Evaluation/client coordinates. Not a server input."""
        X = self.center(R)
        coords = X @ self.B
        if self.full_span:
            return np.ascontiguousarray(coords)
        residual = X - (coords @ self.B.T)
        radius = np.linalg.norm(residual, axis=1, keepdims=True)
        return np.concatenate([coords, radius], axis=1)


def numerical_rank(
    singular_values: np.ndarray,
    nrows: int,
    ncols: int,
    *,
    multiplier: float = DEFAULT_RANK_EPS_MULTIPLIER,
) -> tuple[int, float]:
    s = np.asarray(singular_values, dtype=np.float64)
    if s.size == 0 or not np.any(np.isfinite(s)):
        return 0, 0.0
    peak = float(np.max(s))
    if peak <= 0.0:
        return 0, 0.0
    tol = peak * max(int(nrows), int(ncols), 1) * float(np.finfo(np.float64).eps) * float(multiplier)
    rank = int(np.sum(s > tol))
    return rank, float(tol)


def folding_from_public(
    Q: np.ndarray,
    *,
    rank_eps_multiplier: float = DEFAULT_RANK_EPS_MULTIPLIER,
) -> FoldingMap:
    Q = np.asarray(Q, dtype=np.float64)
    if Q.ndim != 2:
        raise ValueError("Q must be 2-D")
    n, d = Q.shape
    if n < 1 or d < 1:
        raise ValueError("Q must have positive shape")
    q1 = np.array(Q[0], dtype=np.float64, copy=True)
    P_rest = Q[1:] - q1
    nrows = int(P_rest.shape[0])
    if nrows == 0 or d == 0:
        B = np.zeros((d, 0), dtype=np.float64)
        sv = np.zeros((0,), dtype=np.float64)
        k, tol = 0, 0.0
    else:
        _, sv, vt = np.linalg.svd(P_rest, full_matrices=False)
        k, tol = numerical_rank(sv, nrows, d, multiplier=rank_eps_multiplier)
        B = np.ascontiguousarray(vt[:k].T) if k else np.zeros((d, 0), dtype=np.float64)
    expected_generic_k = min(max(n - 1, 0), d)
    unexpected = bool(k < expected_generic_k)
    full_span = bool(k == d)
    m = int(d if full_span else k + 1)
    if m != min(k + 1, d):
        raise RuntimeError("folded dimension identity m = min(k+1, d) failed")
    return FoldingMap(
        B=B,
        k=int(k),
        m=int(m),
        q1=q1,
        rank_tol=float(tol),
        singular_values=np.asarray(sv, dtype=np.float64),
        unexpected_rank_loss=unexpected,
        full_span=full_span,
    )


def cost_preservation_error(Q: np.ndarray, R: np.ndarray, fmap: FoldingMap) -> float:
    C = pairwise_euclidean(Q, R)
    C_fold = pairwise_euclidean(fmap.fold_public(Q), fmap.fold_private(R))
    return float(np.max(np.abs(C - C_fold)))
