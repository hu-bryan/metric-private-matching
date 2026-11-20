"""Exact linear assignment and true Euclidean scoring."""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def pairwise_euclidean(public_points: np.ndarray, private_points: np.ndarray) -> np.ndarray:
    """Cost matrix with public rows and private columns, C[i,j] = ||Q[i]-R[j]||_2."""
    Q = np.asarray(public_points, dtype=np.float64)
    R = np.asarray(private_points, dtype=np.float64)
    if Q.ndim != 2 or R.ndim != 2:
        raise ValueError("points must be 2-D arrays")
    if Q.shape[1] != R.shape[1]:
        raise ValueError(f"dimension mismatch: public {Q.shape} vs private {R.shape}")
    return np.asarray(cdist(Q, R, metric="euclidean"), dtype=np.float64)


def exact_assignment(cost: np.ndarray) -> np.ndarray:
    """Return a permutation of column indices, one per row."""
    cost = np.asarray(cost, dtype=np.float64)
    if cost.ndim != 2:
        raise ValueError("cost must be a matrix")
    n_row, n_col = cost.shape
    if n_row != n_col:
        raise ValueError(f"this experiment requires a square cost matrix, got {cost.shape}")
    row_ind, col_ind = linear_sum_assignment(cost)
    assignment = np.empty(n_row, dtype=np.int64)
    assignment[row_ind] = col_ind
    if np.unique(assignment).size != n_row:
        raise RuntimeError("assignment is not one-to-one")
    return assignment


def assignment_cost(cost: np.ndarray, assignment: np.ndarray) -> float:
    cost = np.asarray(cost, dtype=np.float64)
    assignment = np.asarray(assignment, dtype=np.int64)
    n = cost.shape[0]
    if assignment.shape != (n,):
        raise ValueError("assignment length must match the number of rows")
    return float(cost[np.arange(n), assignment].sum())


def optimal_cost(cost: np.ndarray) -> tuple[np.ndarray, float]:
    assignment = exact_assignment(cost)
    return assignment, assignment_cost(cost, assignment)


def random_permutation_baseline(cost: np.ndarray) -> float:
    """Exact expected total cost of a uniform random permutation: sum(C)/n."""
    cost = np.asarray(cost, dtype=np.float64)
    n = cost.shape[0]
    if n < 1:
        raise ValueError("empty cost matrix")
    return float(cost.sum() / n)
