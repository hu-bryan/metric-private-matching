"""Three primary mechanisms with an explicit Euclidean-distance server rule.

Local reports are generated from private data. The public server constructs
scores from public points and those reports only. The debiased folded server
is intentionally not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_matching.assignment import pairwise_euclidean
from private_matching.config import AMBIENT_SERVER_RULE, FOLDED_SERVER_RULE

SERVER_RULE_EUCLIDEAN = "euclidean_report_distances"


@dataclass
class LocalReport:
    mechanism: str
    dimension: int
    locations: np.ndarray | None
    cost_noise: np.ndarray | None
    radii: np.ndarray
    server_rule: str


def _redraw_zero_rows(draw_rows, current: np.ndarray) -> np.ndarray:
    out = np.array(current, dtype=np.float64, copy=True)
    if out.size == 0:
        return out
    zero = np.all(out == 0.0, axis=1)
    while np.any(zero):
        idx = np.flatnonzero(zero)
        out[idx] = draw_rows(len(idx))
        zero = np.all(out == 0.0, axis=1)
    return out


def sample_euclidean_laplace(
    n_points: int,
    dim: int,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic Euclidean Laplace: direction times Gamma(shape=dim, scale=1/eps)."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    if dim < 1:
        raise ValueError("Laplace dimension must be >= 1")
    if n_points < 1:
        noise = np.zeros((0, dim), dtype=np.float64)
        return noise, np.zeros((0,), dtype=np.float64)

    def draw(n_rows: int) -> np.ndarray:
        return rng.normal(size=(n_rows, dim))

    gaussian = _redraw_zero_rows(draw, draw(n_points))
    norms = np.linalg.norm(gaussian, axis=1, keepdims=True)
    bad = ~np.isfinite(norms) | (norms == 0.0)
    while np.any(bad):
        idx = np.flatnonzero(bad.ravel())
        gaussian[idx] = draw(idx.size)
        norms = np.linalg.norm(gaussian, axis=1, keepdims=True)
        bad = ~np.isfinite(norms) | (norms == 0.0)
    omega = gaussian / norms
    radii = rng.gamma(shape=dim, scale=1.0 / float(epsilon), size=(n_points, 1))
    noise = radii * omega
    return noise, radii[:, 0]


def sample_supnorm_cone_laplace(
    n_tasks: int,
    n_drivers: int,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Independent column noise: radius times cone measure on the sup-norm sphere."""
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0")
    if n_tasks < 1 or n_drivers < 1:
        raise ValueError("phi-infinity noise requires positive dimensions")

    V = rng.uniform(-1.0, 1.0, size=(n_tasks, n_drivers))
    col_inf = np.max(np.abs(V), axis=0)
    zero_cols = col_inf == 0.0
    while np.any(zero_cols):
        idx = np.flatnonzero(zero_cols)
        V[:, idx] = rng.uniform(-1.0, 1.0, size=(n_tasks, idx.size))
        col_inf = np.max(np.abs(V), axis=0)
        zero_cols = col_inf == 0.0
    omega = V / col_inf[None, :]
    radii = rng.gamma(shape=n_tasks, scale=1.0 / float(epsilon), size=(n_drivers,))
    noise = omega * radii[None, :]
    return noise, radii


def ambient_report(R: np.ndarray, epsilon: float, rng: np.random.Generator) -> LocalReport:
    R = np.asarray(R, dtype=np.float64)
    n, d = R.shape
    noise, radii = sample_euclidean_laplace(n, d, epsilon, rng)
    return LocalReport(
        mechanism="ambient",
        dimension=int(d),
        locations=R + noise,
        cost_noise=None,
        radii=radii,
        server_rule=AMBIENT_SERVER_RULE,
    )


def folded_report(R_fold: np.ndarray, epsilon: float, rng: np.random.Generator) -> LocalReport:
    R_fold = np.asarray(R_fold, dtype=np.float64)
    n, m = R_fold.shape
    noise, radii = sample_euclidean_laplace(n, m, epsilon, rng)
    return LocalReport(
        mechanism="folded",
        dimension=int(m),
        locations=R_fold + noise,
        cost_noise=None,
        radii=radii,
        server_rule=FOLDED_SERVER_RULE,
    )


def phi_infinity_report(
    n_tasks: int,
    n_drivers: int,
    epsilon: float,
    rng: np.random.Generator,
) -> LocalReport:
    noise, radii = sample_supnorm_cone_laplace(n_tasks, n_drivers, epsilon, rng)
    return LocalReport(
        mechanism="phi_infinity",
        dimension=int(n_tasks),
        locations=None,
        cost_noise=noise,
        radii=radii,
        server_rule=SERVER_RULE_EUCLIDEAN,
    )


def server_scores(
    report: LocalReport,
    *,
    Q: np.ndarray | None = None,
    Q_fold: np.ndarray | None = None,
    C: np.ndarray | None = None,
    server_rule: str = SERVER_RULE_EUCLIDEAN,
) -> np.ndarray:
    """Public server: Euclidean report distances, no squared-radius correction."""
    if report.mechanism in {"ambient", "folded"} and server_rule != report.server_rule:
        raise ValueError(
            f"server_rule {server_rule!r} is incompatible with report rule {report.server_rule!r}"
        )
    if report.mechanism == "ambient":
        if Q is None or report.locations is None:
            raise ValueError("ambient server requires public Q and noisy locations")
        if server_rule != AMBIENT_SERVER_RULE:
            raise ValueError(f"unsupported ambient server_rule {server_rule!r}")
        return pairwise_euclidean(Q, report.locations)
    if report.mechanism == "folded":
        if Q_fold is None or report.locations is None:
            raise ValueError("folded server requires public Q_fold and noisy folded reports")
        if server_rule != FOLDED_SERVER_RULE:
            raise ValueError(
                f"unsupported folded server_rule {server_rule!r}; "
                "folded_debiased is not a primary mechanism"
            )
        # Explicit raw Euclidean distances. Do not subtract mu = m(m+1)/eps^2.
        return pairwise_euclidean(Q_fold, report.locations)
    if report.mechanism == "phi_infinity":
        if C is None or report.cost_noise is None:
            raise ValueError("phi-infinity server requires true public costs plus column noise")
        return np.asarray(C, dtype=np.float64) + np.asarray(report.cost_noise, dtype=np.float64)
    raise ValueError(f"unknown mechanism {report.mechanism!r}")


def column_range_sum(error: np.ndarray) -> float:
    error = np.asarray(error, dtype=np.float64)
    return float(np.sum(np.max(error, axis=0) - np.min(error, axis=0)))


def twice_column_infnorm_sum(error: np.ndarray) -> float:
    error = np.asarray(error, dtype=np.float64)
    return float(2.0 * np.sum(np.max(np.abs(error), axis=0)))
