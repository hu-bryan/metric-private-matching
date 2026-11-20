"""The four mechanisms in *Matching Under Metric Privacy*.

For fixed public Y, privacy uses D(X,X') = sum_i ||X[i]-X'[i]||_2.
The first three mechanisms release local reports. Gumbel matching is central:
only its assignment is private at the stated calibration.
No function normalizes, samples, or fits preprocessing on private inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist

Cost = Literal["euclidean", "squared"]
MECHANISMS = ("original_points", "projected_representation", "distance_vectors", "gumbel")


def _points(value: ArrayLike, name: str) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=float)
    if result.ndim != 2 or min(result.shape) < 1 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite (n, d) array")
    return result


def _inputs(private: ArrayLike, public: ArrayLike):
    x, y = _points(private, "private"), _points(public, "public")
    if x.shape != y.shape:
        raise ValueError("private and public must have the same (n, d) shape")
    return x, y


def _epsilon(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("epsilon must be finite and strictly positive")
    return value


def _cost(value: Cost) -> str:
    if value not in ("euclidean", "squared"):
        raise ValueError("cost must be 'euclidean' or 'squared'")
    return "euclidean" if value == "euclidean" else "sqeuclidean"


def _radial(n: int, d: int, epsilon: float, rng: np.random.Generator):
    if d == 0:
        return np.empty((n, 0))
    direction = rng.normal(size=(n, d))
    norms = np.linalg.norm(direction, axis=1)
    while np.any(norms == 0):
        zero = norms == 0
        direction[zero] = rng.normal(size=(int(zero.sum()), d))
        norms = np.linalg.norm(direction, axis=1)
    radii = rng.gamma(shape=d, scale=1 / epsilon, size=(n, 1))
    return radii * (direction / norms[:, None])


def matching_from_costs(costs: ArrayLike) -> NDArray[np.intp]:
    """Minimize a square cost matrix. This function adds no privacy itself."""
    values = _points(costs, "costs")
    if values.shape[0] != values.shape[1]:
        raise ValueError("costs must be square")
    _, permutation = linear_sum_assignment(values)
    return permutation


def matching_from_points(private: ArrayLike, public: ArrayLike, *, cost: Cost = "euclidean"):
    """Match released points to public points, or solve a nonprivate baseline."""
    x, y = _inputs(private, public)
    return matching_from_costs(cdist(x, y, metric=_cost(cost)))


def privatize_points(private: ArrayLike, epsilon: float, *, rng=None):
    """Release each point with independent radial Laplace noise in R^d.

    ``rng`` accepts a NumPy Generator, integer seed, or None. A fixed seed is
    for reproducible research; deployed privacy requires secret fresh randomness.
    """
    x = _points(private, "private")
    epsilon = _epsilon(epsilon)
    return x + _radial(*x.shape, epsilon, np.random.default_rng(rng))


@dataclass(frozen=True)
class ProjectedReport:
    """Locally private values and their public candidate representation.

    Contains no original private points or unsanitized private residuals.
    """
    values: NDArray[np.float64]
    public_points: NDArray[np.float64]
    noise_dimension: int
    cost: Cost = "euclidean"

    def matching(self, *, cost: Cost | None = None):
        if cost is not None and cost != self.cost:
            raise ValueError('Decode a projected report with the objective used to create it')
        metric = _cost(self.cost)
        # A rank-zero squared-cost report has zero coordinates and needs no noise.
        costs = cdist(self.values, self.public_points, metric=metric)
        return matching_from_costs(costs)


def privatize_projected(private: ArrayLike, public: ArrayLike, epsilon: float,
                       *, cost: Cost = "euclidean", rng=None) -> ProjectedReport:
    """Release a representation determined by the public affine span.

    Euclidean costs retain r projection coordinates and one residual norm when
    r+1<d; otherwise use the original d coordinates. Squared costs use only r
    projection coordinates because private residual squares are row constants.
    """
    x, y = _inputs(private, public)
    epsilon = _epsilon(epsilon)
    _cost(cost)
    center = y.mean(axis=0)
    yc, xc = y - center, x - center
    _, singular, vt = np.linalg.svd(yc, full_matrices=False)
    tolerance = singular[0] * max(y.shape) * np.finfo(float).eps
    rank = int((singular > tolerance).sum())
    d = y.shape[1]
    if (cost == "euclidean" and rank + 1 >= d) or rank == d:
        return ProjectedReport(privatize_points(x, epsilon, rng=rng), y.copy(), d, cost)
    basis = np.ascontiguousarray(vt[:rank].T)
    u, v = xc @ basis, yc @ basis
    if cost == "euclidean":
        u = np.column_stack((u, np.linalg.norm(xc - u @ basis.T, axis=1)))
        v = np.column_stack((v, np.zeros(len(y))))
    dimension = u.shape[1]
    values = u + _radial(len(x), dimension, epsilon, np.random.default_rng(rng))
    return ProjectedReport(values, v, dimension, cost)


def privatize_distances(private: ArrayLike, public: ArrayLike, epsilon: float,
                       *, cost: Cost = "euclidean", rng=None):
    """Release locally private costs using the infinity-norm Laplace law.

    Squared costs subtract ||x-center(Y)||^2 from each row before adding noise
    at rate epsilon/(2 max_j ||Y[j]-center(Y)||). Negative reports are valid.
    """
    x, y = _inputs(private, public)
    epsilon = _epsilon(epsilon)
    metric = _cost(cost)
    generator = np.random.default_rng(rng)
    n = len(x)
    base = cdist(x, y, metric=metric)
    sensitivity = 1.0
    if cost == "squared":
        center = y.mean(axis=0)
        sensitivity = 2 * float(np.linalg.norm(y - center, axis=1).max())
        base = np.sum((y - center)**2, axis=1)[None, :] - 2 * (x - center) @ (y - center).T
    if sensitivity == 0:
        return np.zeros((n, n))
    cube = generator.uniform(-1, 1, (n, n))
    norms = np.max(np.abs(cube), axis=1)
    while np.any(norms == 0):
        zero = norms == 0
        cube[zero] = generator.uniform(-1, 1, (int(zero.sum()), n))
        norms = np.max(np.abs(cube), axis=1)
    noise = cube * (generator.gamma(n, size=n) / norms)[:, None]
    return base + (sensitivity / epsilon) * noise


def original_points(private, public, epsilon, *, cost: Cost = "euclidean", rng=None):
    """Compute a matching from locally privatized original coordinates."""
    x, y = _inputs(private, public)
    _cost(cost)
    return matching_from_points(privatize_points(x, epsilon, rng=rng), y, cost=cost)


def projected_representation(private, public, epsilon, *, cost: Cost = "euclidean", rng=None):
    """Compute a matching from the locally private projected representation."""
    return privatize_projected(private, public, epsilon, cost=cost, rng=rng).matching(cost=cost)


def distance_vectors(private, public, epsilon, *, cost: Cost = "euclidean", rng=None):
    """Compute a matching from locally private distance-vector reports."""
    return matching_from_costs(privatize_distances(private, public, epsilon, cost=cost, rng=rng))


def gumbel_matching(private, public, epsilon, *, cost: Cost = "euclidean", rng=None):
    """Release only the central private assignment; do not release noisy scores.

    Add independent Gumbel edge scores at scale 2/epsilon for Euclidean costs
    and 2*diameter(public)/epsilon for squared costs.
    """
    x, y = _inputs(private, public)
    epsilon = _epsilon(epsilon)
    metric = _cost(cost)
    scale = 2 / epsilon
    base = cdist(x, y, metric=metric)
    if cost == "squared":
        diameter = float(pdist(y).max()) if len(y) > 1 else 0.0
        scale *= diameter
        center = y.mean(axis=0)
        base = np.sum((y - center)**2, axis=1)[None, :] - 2 * (x - center) @ (y - center).T
    noise = np.random.default_rng(rng).gumbel(size=base.shape)
    return matching_from_costs(base - scale * noise)


def match(private: ArrayLike, public: ArrayLike, epsilon: float, *,
          mechanism: str = "gumbel", cost: Cost = "euclidean", rng=None):
    """Match equal-sized point sets under epsilon D metric privacy.

    Parameters
    ----------
    private, public : finite array-like of shape (n, d)
        Private rows and fixed public candidates, respectively.
    epsilon : positive float
        Privacy budget in inverse units of Euclidean displacement.
    mechanism : {'original_points', 'projected_representation', 'distance_vectors', 'gumbel'}
        The first three use local privacy; Gumbel requires a trusted server.
    cost : {'euclidean', 'squared'}
        Assignment objective. The privacy metric remains unsquared Euclidean.
    rng : numpy.random.Generator, int, or None
        Random generator or research seed. Inputs are never modified.

    Returns
    -------
    numpy.ndarray of shape (n,)
        A permutation: private row i is assigned to public[permutation[i]].
    """
    functions = dict(original_points=original_points,
                     projected_representation=projected_representation,
                     distance_vectors=distance_vectors, gumbel=gumbel_matching)
    if mechanism not in functions:
        raise ValueError(f"mechanism must be one of {MECHANISMS}")
    return functions[mechanism](private, public, epsilon, cost=cost, rng=rng)
