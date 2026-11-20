"""One configuration work unit: all mechanisms, epsilons, and assigned repetitions."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from private_matching.assignment import (
    assignment_cost,
    exact_assignment,
    optimal_cost,
    pairwise_euclidean,
    random_permutation_baseline,
)
from private_matching.bounds import bounds_for_setting
from private_matching.config import ScientificConfig, epsilon_key
from private_matching.folding import cost_preservation_error, folding_from_public
from private_matching.instances import load_instance
from private_matching.manifest import block_id, scientific_fingerprint, work_id
from private_matching.mechanisms import (
    ambient_report,
    column_range_sum,
    folded_report,
    phi_infinity_report,
    server_scores,
    twice_column_infnorm_sum,
)
from private_matching.results import SCHEMA_VERSION
from private_matching.seeds import noise_generator

COST_PRESERVATION_RTOL = 1e-10
COST_PRESERVATION_ATOL = 1e-10
TRANSFER_ATOL = 1e-8


def _empty_mech_eps_map(mechanisms: tuple[str, ...], epsilons: tuple[float, ...]) -> dict[str, dict[str, list[float]]]:
    return {
        mechanism: {epsilon_key(eps): [] for eps in epsilons} for mechanism in mechanisms
    }


def run_work_unit(
    cfg: ScientificConfig,
    s: int,
    n: int,
    d: int,
    *,
    k_start: int = 0,
    k_end: int | None = None,
    record_diagnostics: bool = False,
) -> dict[str, Any]:
    t_all = time.perf_counter()
    k_end = int(cfg.noise_repetitions_K if k_end is None else k_end)
    k_start = int(k_start)
    if not (0 <= k_start < k_end <= int(cfg.noise_repetitions_K)):
        raise ValueError(f"invalid repetition block [{k_start}, {k_end}) for K={cfg.noise_repetitions_K}")

    t0 = time.perf_counter()
    inst = load_instance(cfg, s, n, d)
    C = pairwise_euclidean(inst.Q, inst.R)
    _, O = optimal_cost(C)
    B = random_permutation_baseline(C)
    g = B - O
    fmap = folding_from_public(inst.Q, rank_eps_multiplier=cfg.rank_svd_eps_multiplier)
    Q_fold = fmap.fold_public(inst.Q)
    R_fold = fmap.fold_private(inst.R)
    C_fold_true = pairwise_euclidean(Q_fold, R_fold)
    preserve_err = cost_preservation_error(inst.Q, inst.R, fmap)
    preserve_tol = COST_PRESERVATION_ATOL + COST_PRESERVATION_RTOL * max(1.0, float(np.max(C)))
    if preserve_err > preserve_tol:
        raise FloatingPointError(
            f"folding failed cost preservation: max abs error {preserve_err} > {preserve_tol}"
        )
    preprocess_s = time.perf_counter() - t0

    excess = _empty_mech_eps_map(cfg.mechanisms, cfg.epsilons)
    util = _empty_mech_eps_map(cfg.mechanisms, cfg.epsilons)
    noise_s = {m: 0.0 for m in cfg.mechanisms}
    assign_s = {m: 0.0 for m in cfg.mechanisms}
    diagnostics: dict[str, Any] = {}
    if record_diagnostics:
        diagnostics = {
            "transfer_ok": True,
            "max_transfer_residual": 0.0,
            "negative_excess_flags": 0,
        }

    for k in range(k_start, k_end):
        for epsilon in cfg.epsilons:
            ek = epsilon_key(epsilon)
            for mechanism in cfg.mechanisms:
                rng = noise_generator(cfg.root_seed, s, n, d, mechanism, epsilon, k)
                t_noise = time.perf_counter()
                if mechanism == "ambient":
                    report = ambient_report(inst.R, epsilon, rng)
                    C_hat = server_scores(
                        report, Q=inst.Q, server_rule=cfg.ambient_server_rule
                    )
                    E = C_hat - C
                    radius_bound = 2.0 * float(np.sum(report.radii))
                elif mechanism == "phi_infinity":
                    report = phi_infinity_report(n, n, epsilon, rng)
                    C_hat = server_scores(report, C=C)
                    E = np.asarray(report.cost_noise, dtype=np.float64)
                    radius_bound = 2.0 * float(np.sum(report.radii))
                elif mechanism == "folded":
                    report = folded_report(R_fold, epsilon, rng)
                    C_hat = server_scores(
                        report, Q_fold=Q_fold, server_rule=cfg.folded_server_rule
                    )
                    E = C_hat - C_fold_true
                    radius_bound = 2.0 * float(np.sum(report.radii))
                else:
                    raise ValueError(f"unknown mechanism {mechanism}")
                noise_s[mechanism] += time.perf_counter() - t_noise

                t_as = time.perf_counter()
                assignment = exact_assignment(C_hat)
                U = assignment_cost(C, assignment)
                assign_s[mechanism] += time.perf_counter() - t_as
                X = U - O
                if X < -1e-8:
                    if record_diagnostics:
                        diagnostics["negative_excess_flags"] += 1
                    elif X < -1e-6:
                        raise FloatingPointError(
                            f"materially negative excess {X} for {mechanism} s={s} n={n} d={d}"
                        )
                excess[mechanism][ek].append(float(X))
                util[mechanism][ek].append(float(U))
                if record_diagnostics:
                    range_sum = column_range_sum(E)
                    inf_sum = twice_column_infnorm_sum(E)
                    residual = float(X) - range_sum
                    diagnostics["max_transfer_residual"] = max(
                        diagnostics["max_transfer_residual"], residual
                    )
                    if residual > TRANSFER_ATOL:
                        diagnostics["transfer_ok"] = False
                    if float(X) - inf_sum > TRANSFER_ATOL:
                        diagnostics["transfer_ok"] = False
                    if float(X) - radius_bound > TRANSFER_ATOL:
                        diagnostics["transfer_ok"] = False

    bounds = bounds_for_setting(cfg.mechanisms, n, d, fmap.m, cfg.epsilons)
    record = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "work_id": work_id(cfg, s, n, d),
        "block_id": block_id(cfg, s, n, d, k_start, k_end),
        "scientific_fingerprint": scientific_fingerprint(cfg),
        "seed_scheme_version": cfg.seed_scheme_version,
        "bit_generator": cfg.bit_generator,
        "experiment_name": cfg.name,
        "s": int(s),
        "n": int(n),
        "d": int(d),
        "label": f"Point configuration {int(s) + 1}",
        "k_span": int(fmap.k),
        "m_fold": int(fmap.m),
        "full_span": bool(fmap.full_span),
        "rank_tol": float(fmap.rank_tol),
        "unexpected_rank_loss": bool(fmap.unexpected_rank_loss),
        "cost_preservation_error": float(preserve_err),
        "O": float(O),
        "B": float(B),
        "g": float(g),
        "gap_zero": bool(not (g > 0.0)),
        "K_requested": int(cfg.noise_repetitions_K),
        "k_start": k_start,
        "k_end": k_end,
        "mechanisms": list(cfg.mechanisms),
        "epsilons": [float(e) for e in cfg.epsilons],
        "ambient_server_rule": cfg.ambient_server_rule,
        "folded_server_rule": cfg.folded_server_rule,
        "bounds": bounds,
        "excess": excess,
        "utility": util,
        "timings": {
            "preprocess_s": preprocess_s,
            "noise_and_costs_s": noise_s,
            "assignment_s": assign_s,
            "total_s": time.perf_counter() - t_all,
        },
    }
    if record_diagnostics:
        record["diagnostics"] = diagnostics
    return record
