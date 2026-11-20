"""Public-only PCA and nested farthest-point proxies; no private tuning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_matching.anchor import AnchorGeometry, directional_coefficient
from private_matching.assignment import pairwise_euclidean
from private_matching.folding import FoldingMap, folding_from_public, numerical_rank


@dataclass(frozen=True)
class ProxyGeometry:
    fold: FoldingMap
    anchor: AnchorGeometry
    row_map: np.ndarray
    D_proxy: float
    family: str
    value: int

    def expected_bound(self, n: int, epsilon: float, correction: bool) -> float:
        if self.anchor.B == 0:
            return 2 * self.D_proxy
        coefficient = (2 * np.sqrt(self.anchor.B) + 1)**2 if correction else 2 * self.fold.m
        return float(2 * self.D_proxy + n * coefficient / epsilon)


def make_geometry(Q: np.ndarray, references: np.ndarray, row_map: np.ndarray,
                  fold: FoldingMap, family: str, value: int) -> ProxyGeometry:
    centered = references - fold.q1
    # Numerical rank truncation is public approximation, including in the exact arm.
    proxies = fold.q1 + (centered @ fold.B) @ fold.B.T
    D = float(np.linalg.norm(Q - proxies[row_map], axis=1).sum())
    return ProxyGeometry(fold, AnchorGeometry.from_references(fold.fold_public(references)),
                         row_map, D, family, value)


def endpoint_grid(maximum: int, *, zero: bool = False, step: int = 5) -> list[int]:
    """Linear step-5 interior, with small-value and exact endpoints included."""
    if maximum < 0 or step < 1:
        raise ValueError("invalid grid extent")
    return sorted({maximum, *range(step, maximum + 1, step),
                   *([1] if maximum >= 1 else []), *([0] if zero else [])})


class PublicProxyCatalog:
    """One PCA SVD and one nested public traversal per fixed instance.

    Scan ALL PCA ranks and ALL representative counts for bound-based selection.
    Representative selection uses the valid m_T <= min(s,d) certificate to avoid
    n separate SVDs; selected/swept geometries log their actual rank and bound.
    """

    def __init__(self, Q: np.ndarray):
        self.Q = np.asarray(Q, dtype=np.float64)
        self.n, self.d = Q.shape
        fold = folding_from_public(Q)
        self.exact = make_geometry(Q, Q, np.arange(self.n), fold, "exact", fold.k)
        self.mean = Q.mean(axis=0)
        centered = Q - self.mean
        _, sv, vt = np.linalg.svd(centered, full_matrices=False)
        self.rank, self.tol = numerical_rank(sv, self.n, self.d)
        self.sv, self.basis = sv, vt[:self.rank].T
        coords = centered @ self.basis
        residual = centered - coords @ self.basis.T
        tail = np.zeros((self.n, self.rank + 1))
        if self.rank:
            tail[:, :self.rank] = np.cumsum((coords * coords)[:, ::-1], axis=1)[:, ::-1]
        tail += np.sum(residual * residual, axis=1)[:, None]
        self.pca_errors = np.sqrt(np.maximum(0, tail)).sum(axis=0)
        self.public_distances = pairwise_euclidean(Q, Q)
        self.diameter = float(self.public_distances.max())
        self.order = np.empty(self.n, dtype=int)
        self.cluster_errors = np.empty(self.n)
        selected = np.zeros(self.n, dtype=bool)
        nearest = np.full(self.n, np.inf)
        next_index = 0
        for count in range(self.n):
            self.order[count] = next_index
            selected[next_index] = True
            nearest = np.minimum(nearest, self.public_distances[:, next_index])
            self.cluster_errors[count] = nearest.sum()
            remaining = np.where(selected, -np.inf, nearest)
            next_index = int(np.argmax(remaining))
        self._cache: dict[tuple[str, int], ProxyGeometry] = {}

    def geometry(self, family: str, value: int) -> ProxyGeometry:
        key = family, int(value)
        if key in self._cache:
            return self._cache[key]
        if family == "pca":
            if not 0 <= value <= self.rank:
                raise ValueError("PCA rank outside public rank")
            basis = np.ascontiguousarray(self.basis[:, :value])
            fold = FoldingMap(basis, value, min(value + 1, self.d), self.mean,
                              self.tol, self.sv, value < min(self.n - 1, self.d), value == self.d)
            geom = make_geometry(self.Q, self.Q, np.arange(self.n), fold, family, value)
        elif family == "representatives":
            if not 1 <= value <= self.n:
                raise ValueError("representative count outside 1..n")
            # At s=n use original order, matching the exact mechanism byte-for-byte.
            indices = np.arange(self.n) if value == self.n else self.order[:value]
            refs = self.Q[indices]
            row_map = np.argmin(self.public_distances[:, indices], axis=1)
            fold = self.exact.fold if value == self.n else folding_from_public(refs)
            geom = make_geometry(self.Q, refs, row_map, fold, family, value)
        else:
            raise ValueError(f"unknown proxy family {family}")
        self._cache[key] = geom
        return geom

    def select(self, family: str, epsilon: float, correction: bool) -> tuple[int, float]:
        if family == "pca":
            values = np.arange(self.rank + 1)
            dimensions = np.minimum(values + 1, self.d)
            errors = self.pca_errors
            pairs = np.full(len(values), self.n * (self.n - 1) // 2)
            pairs[0] = 0
        elif family == "representatives":
            values = np.arange(1, self.n + 1)
            dimensions = np.minimum(values, self.d)
            errors = self.cluster_errors
            pairs = values * (values - 1) // 2
        else:
            raise ValueError(f"unknown proxy family {family}")
        coefficients = np.array([directional_coefficient(int(m), int(p))
                                 for m, p in zip(dimensions, pairs)])
        noise = (2 * np.sqrt(coefficients) + 1)**2 if correction else 2 * dimensions
        noise = np.where(pairs == 0, 0, noise)
        bounds = 2 * errors + self.n * noise / epsilon
        index = int(np.argmin(bounds))  # smallest rank/count on ties
        return int(values[index]), float(bounds[index])
