"""Span-only anchor release (new arm; the original anchor mechanism is untouched).

Coordinate noise only in the public affine span W = span(Q - q0), k = dim W:

    Y = B^T(x - q0) + Z,  Z ~ Lap_k(eps_y) (Euclidean norm-Laplace),
    a = argmin_i ||Y - p_i||,  T = max(0, ||x - q_a|| + eta),  eta ~ Lap(1/eps_r),
    chat_i = sqrt(T^2 + ||Y - p_i||^2 - ||Y - p_a||^2)   (stable secant form, anchor_server).

Privacy: x -> B^T(x-q0) and, given Y (so fixed a), x -> ||x-q_a|| are 1-Lipschitz; adaptive
composition gives eps_y + eps_r = eps metric privacy. Per client, osc_i(chat_i - c_i) <= 4 M_Q(Z) + |eta|
(PRIOR_FREE_PROOF.md, Section 8), asserted for every realization. k = 0 (coincident public points)
needs no release. For k = d the law equals the original full anchor, whose outcomes are reused.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from private_matching.anchor import AnchorGeometry, AnchorMessage, anchor_server, nearest_anchor, optimized_split
from private_matching.assignment import pairwise_euclidean, exact_assignment, assignment_cost, optimal_cost
from private_matching.mechanisms import sample_euclidean_laplace

SAMPLE_KEYS = ("excess", "output_cost", "oscillation_original", "oscillation_proxy", "realization_bound",
               "proxy_regret", "clipped_radius_fraction", "mean_noise_norm", "max_reconstruction_error",
               "agreement_with_proxy_optimum")


@dataclass(frozen=True)
class SpanGeometry:
    q0: np.ndarray
    B: np.ndarray            # (d, k) orthonormal public span basis
    p: np.ndarray            # (n, k) public coordinates
    proxies: np.ndarray      # q0 + B p_i (equal to Q up to the recorded numerical residual)
    anchor: AnchorGeometry | None
    D_proxy: float

    @property
    def k(self):
        return int(self.B.shape[1])

    @property
    def no_release(self):
        return self.k == 0 or self.anchor is None or self.anchor.B == 0

    @classmethod
    def from_fold(cls, Q, fold):
        """Uses the exact public fold's span basis (q0 = q_1, numerical rank recorded there)."""
        Q = np.asarray(Q, dtype=np.float64)
        B = np.asarray(fold.B, dtype=np.float64)
        p = (Q-fold.q1) @ B
        proxies = fold.q1+p @ B.T
        anchor = AnchorGeometry.from_references(p) if B.shape[1] else None
        return cls(fold.q1, B, p, proxies, anchor, float(np.linalg.norm(Q-proxies, axis=1).sum()))

    def split(self, epsilon):
        return optimized_split(self.anchor.B, epsilon)


def client(X, geom, epsilon, rng):
    """Returns (Y, T) message plus evaluator-only (Z, eta)."""
    eps_y, eps_r = geom.split(epsilon)
    v = (np.asarray(X, dtype=np.float64)-geom.q0) @ geom.B
    unit_Z, unit_radii = sample_euclidean_laplace(len(v), geom.k, 1., rng)
    Z = unit_Z/eps_y
    Y = v+Z
    a = nearest_anchor(geom.p, Y)
    radius = np.linalg.norm(X-geom.proxies[a], axis=1)
    eta = rng.laplace(size=len(v))/eps_r
    return AnchorMessage(Y, np.maximum(0., radius+eta)), Z, eta, unit_radii/eps_y


def decode(message, geom):
    return anchor_server(message, geom.p)


def evaluate_span_anchor(Q, R, C, optimum, fixed_cost, geom, eps, K, rng_for_draw, diameter):
    """K independent draws; per-realization secant bound asserted. rng_for_draw(k) -> Generator."""
    start = time.monotonic()
    n = len(Q)
    samples = {key: [] for key in SAMPLE_KEYS}
    proxy_C = pairwise_euclidean(geom.proxies, R)
    proxy_assignment, proxy_opt = optimal_cost(proxy_C)
    D = geom.D_proxy
    if geom.no_release:
        eps_y = eps_r = 0.
        bound = 2*D  # all public points coincide (up to the recorded residual): every matching ties
    else:
        eps_y, eps_r = geom.split(eps)
        bound = 2*D+n*(4*geom.anchor.B/eps_y+1/eps_r)
    for k in range(1 if geom.no_release else K):
        if geom.no_release:
            assignment, cost, C_hat, radii, clipped, pointwise = np.arange(n), fixed_cost, None, None, None, 2*D
        else:
            message, Z, eta, radii = client(R, geom, eps, rng_for_draw(k))
            C_hat = decode(message, geom)
            clipped = float(np.mean(message.radius == 0))
            per_column = 4*geom.anchor.max_projection(Z)+np.abs(eta)
            osc_columns = np.ptp(C_hat-proxy_C, axis=0)
            tol = 1e-8*max(1., float(np.max(per_column)))
            if np.any(osc_columns > per_column+tol):
                raise FloatingPointError("span-anchor secant oscillation bound failed")
            pointwise = 2*D+float(per_column.sum())
            assignment = exact_assignment(C_hat)
            cost = assignment_cost(C, assignment)
        regret = cost-optimum
        tolerance = 1e-8*max(1., abs(optimum), abs(cost), pointwise)
        if regret < -tolerance or regret > pointwise+tolerance or not np.isfinite(regret):
            raise FloatingPointError("span-anchor realization certificate failed")
        values = dict(excess=max(0., regret), output_cost=cost,
                      oscillation_original=float(np.ptp(C_hat-C, axis=0).sum()) if C_hat is not None else None,
                      oscillation_proxy=float(np.ptp(C_hat-proxy_C, axis=0).sum()) if C_hat is not None else None,
                      realization_bound=pointwise, proxy_regret=assignment_cost(proxy_C, assignment)-proxy_opt,
                      clipped_radius_fraction=clipped,
                      mean_noise_norm=float(np.mean(radii)) if radii is not None else None,
                      max_reconstruction_error=float(np.max(np.abs(C_hat-C))) if C_hat is not None else None,
                      agreement_with_proxy_optimum=float(np.mean(assignment == proxy_assignment)))
        for key in SAMPLE_KEYS:
            samples[key].append(values[key])
    metadata = dict(base_mechanism="span_anchor", decoder="span_anchor_secant_inner_product", privacy_model="local",
                    epsilon=eps, epsilon_y=eps_y, epsilon_r=eps_r, epsilon_other=0., coordinate_noise_dimension=geom.k,
                    coordinate_noise_norm=None if geom.no_release else "l2",
                    expected_noise_norm=geom.k/eps_y if eps_y else None,
                    scalar_noise_count=int(not geom.no_release), scalar_noise_scale=1/eps_r if eps_r else None,
                    report_dimension=0 if geom.no_release else geom.k+1, no_release=geom.no_release, D_proxy=D,
                    B_directional=geom.anchor.B if geom.anchor is not None else None,
                    directional_method="min(k,sqrt(2*k*log(2*nonzero_secant_pairs))) at span dimension k",
                    mechanism_certificate_uncapped=bound, universal_cap=n*diameter,
                    mechanism_certificate_capped=min(bound, n*diameter), span_dimension=geom.k,
                    provenance="deterministic" if geom.no_release else "simulated", status="complete",
                    pointwise_checked=True, pointwise_ok=True, runtime_seconds=time.monotonic()-start,
                    failure_reason=None)
    return dict(metadata=metadata, samples=samples)
