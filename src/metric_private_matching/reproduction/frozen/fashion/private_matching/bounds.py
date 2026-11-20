"""Unconditional configuration-wise expected-excess upper bounds.

H_A = 2 n d / epsilon
H_Phi = 2 n n / epsilon
H_Psi = 2 n m / epsilon

These are folding_representation_review.md Equation (2) / original-server
constants. Do not attach the review's debiased bound or an uncorrected
square-root claim.
"""

from __future__ import annotations

from private_matching.config import ConfigError


def excess_bound(mechanism: str, n: int, d: int, m: int, epsilon: float) -> float:
    if not (epsilon > 0.0):
        raise ConfigError(f"epsilon must be > 0, got {epsilon!r}")
    n = int(n)
    d = int(d)
    m = int(m)
    if mechanism == "ambient":
        return 2.0 * n * d / float(epsilon)
    if mechanism == "phi_infinity":
        return 2.0 * n * n / float(epsilon)
    if mechanism == "folded":
        return 2.0 * n * m / float(epsilon)
    raise ConfigError(f"unknown mechanism {mechanism!r}")


def bounds_for_setting(
    mechanisms: tuple[str, ...],
    n: int,
    d: int,
    m: int,
    epsilons: tuple[float, ...],
) -> dict[str, dict[str, float]]:
    from private_matching.config import epsilon_key

    out: dict[str, dict[str, float]] = {}
    for mechanism in mechanisms:
        out[mechanism] = {
            epsilon_key(eps): excess_bound(mechanism, n, d, m, eps) for eps in epsilons
        }
    return out


def normalize_bound(bound: float, gap: float) -> float | None:
    if not (gap > 0.0):
        return None
    return float(bound) / float(gap)
