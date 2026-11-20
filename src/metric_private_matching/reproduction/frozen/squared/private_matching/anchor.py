"""Local anchor-distance release and public decoding (follow-up Sections 5–7).

Noise and true folded inputs belong to the evaluator, never the server message.
Repeated secant directions are retained: their count is a valid upper bound on
the number of distinct sign-identified directions, without approximate deduplication.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_matching.mechanisms import sample_euclidean_laplace


def directional_coefficient(m: int, pair_count: int) -> float:
    if m < 1 or pair_count < 0:
        raise ValueError("invalid dimension or secant count")
    return min(float(m), float(np.sqrt(2 * m * np.log(2 * pair_count)))) if pair_count else 0.0


def optimized_split(B: float, epsilon: float) -> tuple[float, float]:
    if not np.isfinite(epsilon) or epsilon <= 0 or not np.isfinite(B) or B <= 0:
        raise ValueError("positive finite epsilon and B are required")
    eps_r = epsilon / (2 * np.sqrt(B) + 1)
    return float(epsilon - eps_r), float(eps_r)


@dataclass(frozen=True)
class AnchorGeometry:
    p: np.ndarray
    left: np.ndarray
    right: np.ndarray
    lengths: np.ndarray
    B: float

    @classmethod
    def from_references(cls, p: np.ndarray) -> "AnchorGeometry":
        p = np.asarray(p, dtype=np.float64)
        if p.ndim != 2 or min(p.shape) < 1 or not np.all(np.isfinite(p)):
            raise ValueError("finite nonempty reference matrix required")
        left, right = np.triu_indices(len(p), 1)
        lengths = np.linalg.norm(p[left] - p[right], axis=1)
        nonzero = lengths > 0
        return cls(p, left[nonzero], right[nonzero], lengths[nonzero],
                   directional_coefficient(p.shape[1], int(nonzero.sum())))

    def max_projection(self, W: np.ndarray) -> np.ndarray:
        """Exact maximum over ALL secants, without a pairs × m × drivers tensor.

        First project onto the reference points, then difference these projections.
        Cost is O(drivers * (references*m + references**2)); pair blocks cap memory.
        """
        projections = np.asarray(W) @ self.p.T
        maxima = np.zeros(len(W))
        for start in range(0, len(self.lengths), 1024):
            part = slice(start, start + 1024)
            values = np.abs(projections[:, self.left[part]] - projections[:, self.right[part]])
            values /= self.lengths[part]
            maxima = np.maximum(maxima, values.max(axis=1))
        return maxima


@dataclass(frozen=True)
class AnchorMessage:
    Y: np.ndarray
    radius: np.ndarray


def nearest_anchor(p: np.ndarray, Y: np.ndarray) -> np.ndarray:
    # Drop the common ||Y||² term so large orthogonal noise cannot erase contrasts.
    scores = np.sum(p * p, axis=1)[:, None] - 2 * (p @ Y.T)
    return np.argmin(scores, axis=0)  # first index wins, shared by client and server


def anchor_client(u: np.ndarray, geometry: AnchorGeometry, epsilon: float,
                  rng: np.random.Generator) -> tuple[AnchorMessage, np.ndarray, np.ndarray]:
    """Return message plus evaluator-only W/eta for the pointwise certificate."""
    if geometry.B == 0:
        raise ValueError("coincident references require no release; use a fixed matching")
    eps_y, eps_r = optimized_split(geometry.B, epsilon)
    W, _ = sample_euclidean_laplace(len(u), u.shape[1], eps_y, rng)
    Y = u + W
    a = nearest_anchor(geometry.p, Y)
    # For a proxy fold this MUST be the distance to the proxy, not original Q[a].
    radius = np.linalg.norm(u - geometry.p[a], axis=1)
    eta = rng.laplace(scale=1 / eps_r, size=len(u))
    return AnchorMessage(Y, np.maximum(0, radius + eta)), W, eta


def anchor_server(message: AnchorMessage, p: np.ndarray) -> np.ndarray:
    """Decode using only public references and the two-part local messages."""
    Y, radius = message.Y, message.radius
    a = nearest_anchor(p, Y)
    result = np.empty((len(p), len(Y)))
    for start in range(0, len(Y), 32):
        part = slice(start, start + 32)
        pa = p[a[part]]
        delta = np.einsum("ijm,ijm->ij", pa[None] - p[:, None],
                          2 * Y[None, part] - p[:, None] - pa[None])
        # No subtraction of large squared distances. Negative roundoff is clipped.
        result[:, part] = np.hypot(radius[None, part], np.sqrt(np.maximum(0, delta)))
    return result


def sample_centered_laplace(n: int, drivers: int, epsilon: float,
                            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Exact cone-law sampler on H for ||v||_osc=(max(v)-min(v))/2.

    The n lower facets of [-1,1]^n project along 1 to a disjoint tiling of
    P([-1,1]^n), up to boundaries. Each facet has the same constant Jacobian.
    Pick a facet uniformly, fix that coordinate to -1, draw the others uniformly,
    and center. This samples the BALL uniformly; normalizing gives cone measure.
    Centering a full-dimensional uniform cube draw would give the wrong law.
    """
    if n < 1 or drivers < 1 or not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("positive dimensions and finite epsilon required")
    if n == 1:
        return np.zeros((1, drivers)), np.zeros(drivers)
    V = rng.uniform(-1, 1, size=(n, drivers))
    V[rng.integers(n, size=drivers), np.arange(drivers)] = -1
    V -= V.mean(axis=0)
    osc = np.ptp(V, axis=0) / 2
    # A zero-radius draw has probability zero; redraw without changing the law.
    for j in np.flatnonzero(osc == 0):
        while osc[j] == 0:
            v = rng.uniform(-1, 1, size=n)
            v[rng.integers(n)] = -1
            V[:, j] = v - v.mean()
            osc[j] = np.ptp(v) / 2
    radii = rng.gamma(n - 1, scale=1 / epsilon, size=drivers)
    return V * (radii / osc)[None], radii


def centered_client(reference_costs: np.ndarray, epsilon: float,
                    rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    noise, radii = sample_centered_laplace(*reference_costs.shape, epsilon, rng)
    return reference_costs - reference_costs.mean(axis=0) + noise, radii
