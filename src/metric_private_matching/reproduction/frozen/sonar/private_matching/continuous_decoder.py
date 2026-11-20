"""Opt-in `continuous_decoder` study: prior-free decoders of the SAME continuous full-fold reports.

Every decoder arm post-processes exactly the reports of the existing full original Psi (`folded`)
baseline, Y_j = f_Q(r_j) + Lap_m(eps) at full eps, regenerated from the same `study.evaluate` stream
and asserted draw-by-draw to reproduce the folded baseline's direct assignment. No prior on R.

* robust_interval: randomized minimax regret for the interval box l=max(0,d-t), u=d+t,
  t = Gamma(m,1/eps) quantile at 1-alpha/n (noise-only union bound, valid for every fixed R).
  Scored by the conditional expected true cost <C,H> of the randomized output (Rao-Blackwellized
  over server randomization); the output protocol samples one permutation from a Birkhoff mixture.
* biascorr_l{lambda}: sqrt(max(d^2 - lambda m(m+1)/eps^2, 0)) then Hungarian (noise-only correction).
* selected_cheap: public finite stress-set selection among direct and the bias corrections using only
  (Q, eps, public seeds); a heuristic, not a guarantee over arbitrary R.
* span_anchor: separate new mechanism (own module and stream); full rank reuses the original anchor.

Q and R are those of robust_fitted (fixed Q; 20 conditional R replicates per regime, NOT independent Q).
Instance ids equal robust_fitted's, so all existing baseline noise streams (and R) are identical to
that study's; the rows are recomputed here, never copied from its checkpoints.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import scipy

from private_matching import robust_fitted as rf
from private_matching.assignment import pairwise_euclidean, exact_assignment, assignment_cost, optimal_cost, \
    random_permutation_baseline
from private_matching.folding import folding_from_public
from private_matching.followup import source_digest
from private_matching.interval_matching import (confidence_radius, interval_bounds, solve_compact_lp, double_oracle,
                                                worst_case_regret, birkhoff_decomposition, permutation_matrix, ds_residual)
from private_matching.login_guard import refuse_login_compute
from private_matching.mechanisms import sample_euclidean_laplace
from private_matching.proxies import PublicProxyCatalog
from private_matching.results import atomic_write_json, read_json
from private_matching.span_anchor import SpanGeometry, evaluate_span_anchor
from private_matching.study import rng_for
from private_matching.study_geometry import stable_id

VERSION = "continuous-decoder-v1"
STUDY_ID = "continuous_decoder"
RESULTS_SUBDIR = "continuous_decoder_20260926"
TRAIN_KINDS = ("near", "crowded", "broad", "shifted")
DECODER_KEYS = ("excess", "output_cost", "oscillation_original", "realization_bound", "mean_noise_norm",
                "max_reconstruction_error", "agreement_with_proxy_optimum")
ROBUST_KEYS = ("excess", "output_cost", "realization_bound", "mean_noise_norm", "agreement_with_proxy_optimum",
               "confidence_radius", "coverage_event", "radius_event", "interval_upper", "interval_lower", "interval_gap",
               "direct_interval_objective", "interval_certificate_term", "solver_seconds", "solver_exact",
               "solver_iterations", "solver_H_residual", "solver_G_residual", "solver_raw_H_residual",
               "solver_raw_G_residual", "support_size", "max_H_entry",
               "bvn_terms", "bvn_leftover", "sampled_output_excess")


def seed_of(*parts) -> int:
    return int(stable_id([VERSION, *parts])[:16], 16)


def cheap_name(lam):
    return "direct" if lam == 0 else f"biascorr_l{lam:g}"


def corrected_costs(d, lam, m, eps):
    """sqrt(max(d^2 - lam m(m+1)/eps^2, 0)); lam=0 returns d itself (byte-identical direct)."""
    if lam == 0:
        return d
    return np.sqrt(np.maximum(d*d-lam*m*(m+1)/eps**2, 0.))


def folded_reports(u, eps, rng):
    """Full-epsilon Euclidean Laplace reports in fold coordinates, exactly as study.evaluate draws them."""
    unit_W, unit_radii = sample_euclidean_laplace(len(u), u.shape[1], 1., rng)
    return u+unit_W/eps, unit_radii/eps


# ----------------------------------------------------------------------------------------------
# Public stress-set selection among cheap decoders (no R argument anywhere).
# ----------------------------------------------------------------------------------------------

def fit(Q, epsilon, lambdas, fit_key, phase, K_train, K_validation, per_kind=3, root_seed=rf.ROOT_SEED,
        P=rf.PARAMETERS):
    """Select among direct/bias corrections by worst mean true regret over public synthetic configs."""
    started = time.monotonic()
    n = len(Q)
    fold = folding_from_public(Q)
    p, m = fold.fold_public(Q), fold.m
    candidates = [0.]+[float(x) for x in lambdas]
    seeds = {role: seed_of("fit", fit_key, phase, root_seed, role) for role in
             ["train_configurations", "train_noise", "validation_configurations", "validation_noise"]}
    result = dict(fit_key=fit_key, epsilon=float(epsilon), phase=phase, seeds=seeds, K_train=K_train,
                  K_validation=K_validation, configurations_per_kind=per_kind, train_kinds=list(TRAIN_KINDS),
                  candidates=[cheap_name(x) for x in candidates], candidate_lambdas=candidates, fold_dimension=m,
                  selection_rule="argmin over candidates of max over public training configurations of mean true "
                  "unsquared matching regret; first candidate (direct) on ties",
                  noise="common folded report draws across candidates within the fit only",
                  scope="finite public stress set; not an assumed law of R and not a guarantee over arbitrary R")
    arrays = dict(Q=Q)
    for split, K in [("train", K_train), ("validation", K_validation)]:
        rng = np.random.default_rng(seeds[f"{split}_configurations"])
        configs = np.array([rf.synthetic_configuration(kind, Q, rng, P) for kind in TRAIN_KINDS for _ in range(per_kind)])
        noise_rng = np.random.default_rng(seeds[f"{split}_noise"])
        scores = np.zeros((len(candidates), len(configs)))
        for s, X in enumerate(configs):
            C = pairwise_euclidean(Q, X)
            optimum = optimal_cost(C)[1]
            u = fold.fold_private(X)
            for _ in range(K):
                Y, _ = folded_reports(u, epsilon, noise_rng)
                d = pairwise_euclidean(p, Y)
                for c, lam in enumerate(candidates):
                    scores[c, s] += assignment_cost(C, exact_assignment(corrected_costs(d, lam, m, epsilon)))-optimum
        scores /= K
        arrays[f"{split}_configurations"], arrays[f"{split}_scores"] = configs, scores
        result[f"{split}_scores"] = scores.tolist()
        result[f"{split}_worst"] = scores.max(axis=1).tolist()
    worst = np.asarray(result["train_worst"])
    selected = int(np.argmin(worst))
    result.update(selected_index=selected, selected=cheap_name(candidates[selected]),
                  selected_lambda=candidates[selected], selected_train_worst=float(worst[selected]),
                  selected_validation_worst=float(result["validation_worst"][selected]),
                  fit_seconds=time.monotonic()-started)
    return result, arrays


def load_or_fit(directory, fingerprint, family, n, epsilon, manifest, create=True):
    fit_id = f"{family}_n{n}_eps{epsilon:g}"
    path = directory/"fits"/f"{fit_id}.json"
    if path.exists():
        record = read_json(path)
        if record.get("fingerprint") != fingerprint:
            raise ValueError(f"fit from different fingerprint: {path}")
        return record
    if not create:
        raise FileNotFoundError(f"missing public fit {path}; run the fit stage first")
    Q = rf.public_Q(family, n, manifest["d"], manifest["root_seed"], manifest["parameters"])
    record, arrays = fit(Q, epsilon, manifest["lambdas"], f"{family}:n{n}:d{manifest['d']}", manifest["phase"],
                         manifest["K_train"], manifest["K_validation"], manifest["train_per_kind"],
                         manifest["root_seed"], manifest["parameters"])
    record.update(fingerprint=fingerprint, fit_id=fit_id, family=family, n=n, d=manifest["d"], fit_path=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    atomic_write_json(path, record)
    print(f"fit {fit_id}: selected {record['selected']} train worst {record['selected_train_worst']:.4f} "
          f"validation worst {record['selected_validation_worst']:.4f} ({record['fit_seconds']:.1f}s)", flush=True)
    return record


# ----------------------------------------------------------------------------------------------
# Design and identity
# ----------------------------------------------------------------------------------------------

def design(pilot=False, **overrides):
    value = dict(version=VERSION, study_id=STUDY_ID, root_seed=rf.ROOT_SEED, d=16, ns=[25, 50],
                 families=list(rf.FAMILIES), epsilons=[4., 16., 64.], regimes=list(rf.REGIMES),
                 replicates=2 if pilot else 20, K=5 if pilot else 50, alphas=[.1, .5] if pilot else [.1],
                 lambdas=[.5, 1.], K_train=10 if pilot else 30, K_validation=10 if pilot else 30, train_per_kind=3,
                 robust_panel=None, robust_solver="compact_lp", birkhoff=True, runtime_probe=pilot,
                 phase="pilot" if pilot else "production", parameters=rf.PARAMETERS,
                 instance_protocol="robust_fitted instance ids, R generators and baseline noise streams (same phase)",
                 baselines=list(rf.BASELINES))
    if pilot:
        value.update(families=["unstructured", "rank2"], ns=[25])
    else:
        # Fixed from pilot 24182233 TIMING only (runtime_projection.json: n=25 robust ~11 CPU min, n=50 ~100 CPU
        # min): the robust arm runs on all three n=25 Q; cheap, span and baseline arms run at n=25 and n=50.
        value.update(robust_panel=[f"{family}:n25" for family in rf.FAMILIES],
                     robust_panel_reason="pilot 24182233 timing: n=50 robust LP ~0.22 s/solve (~100 CPU min) exceeds "
                     "the 30 CPU-minute budget; chosen before any production utility results")
    value.update(overrides)
    return value


def identity(pilot=False, **overrides):
    value = dict(design(pilot, **overrides), source_digest=source_digest(), numpy=np.__version__,
                 scipy=scipy.__version__)
    return value, stable_id(value)


def robust_enabled(manifest, family, n):
    panel = manifest.get("robust_panel")
    return panel is None or f"{family}:n{n}" in panel


def expected_rows(manifest, family, n):
    alphas = len(manifest["alphas"]) if robust_enabled(manifest, family, n) else 0
    return len(rf.BASELINES)+1+len(manifest["lambdas"])+1+alphas


def instance_spec(manifest, family, n, regime, replicate):
    # Identical to robust_fitted.run_unit: same instance id, same R and same baseline noise streams.
    return dict(version=rf.VERSION, family=family, n=n, d=manifest["d"], regime=regime, replicate=replicate,
                phase=manifest["phase"], root_seed=manifest["root_seed"])


# ----------------------------------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------------------------------

def _decoder_row(meta, arm, samples, **fields):
    row = dict(meta, arm_id=arm, arm_label=arm, base_mechanism="folded", requested_base_mechanism="folded",
               approximation_family="exact", privacy_model="local", epsilon_r=0., epsilon_other=0.,
               coordinate_noise_norm="l2", scalar_noise_count=0, scalar_noise_scale=None, no_release=False,
               source_arm_id=None, provenance="simulated", reuse_reason=None, status="complete",
               pointwise_checked=True, pointwise_ok=True, failure_reason=None, radius_budget_fraction=None,
               ablation=None, operating_point="decoder", report_source_arm_id="folded")
    row.update(fields)
    return dict(metadata=row, samples=samples)


def shared_decoder_rows(Q, R, C, optimum, opt_assignment, catalog, folded_row, eps, K, iid, meta, manifest, robust,
                        rng_server):
    """All decoder arms on the folded baseline's exact reports. Returns (rows, arrays)."""
    n = len(Q)
    geom = catalog.exact
    fold = geom.fold
    m = fold.m
    stream = f"folded:{geom.family}:{geom.value}"
    if folded_row["metadata"]["noise_stream"] != stream or folded_row["metadata"]["coordinate_noise_dimension"] != m:
        raise ValueError("folded baseline stream/dimension mismatch")
    u = fold.fold_private(R)
    p = geom.anchor.p
    B = n*catalog.diameter
    lambdas = [float(x) for x in manifest["lambdas"]]
    coupling = stable_id([iid, eps, stream, "shared_folded_reports"])
    common = dict(epsilon=eps, epsilon_y=eps, coordinate_noise_dimension=m, expected_noise_norm=m/eps,
                  report_dimension=m, noise_stream=stream, coupling_group=coupling, D_proxy=geom.D_proxy,
                  coupling_method="identical full-fold reports to the folded (direct) baseline row: same study.evaluate "
                  "stream and draw index; draw-by-draw identity of the direct assignment asserted")
    cheap = {lam: {key: [] for key in DECODER_KEYS} for lam in lambdas}
    robust_samples = {a: {key: [] for key in ROBUST_KEYS} for a in (manifest["alphas"] if robust else [])}
    arrays = dict(reports=np.empty((K, n, m)), radii=np.empty((K, n)), direct_assignment=np.empty((K, n), dtype=np.int64))
    for lam in lambdas:
        arrays[f"assignment_{cheap_name(lam)}"] = np.empty((K, n), dtype=np.int64)
    for a in robust_samples:
        arrays[f"H_alpha{a:g}"] = np.empty((K, n, n))
        arrays[f"G_alpha{a:g}"] = np.empty((K, n, n))
        arrays[f"sampled_assignment_alpha{a:g}"] = np.empty((K, n), dtype=np.int64)
        arrays[f"bvn_offsets_alpha{a:g}"] = [0]
        arrays[f"bvn_weights_alpha{a:g}"] = []
        arrays[f"bvn_perms_alpha{a:g}"] = []
    solver_totals = {a: 0. for a in robust_samples}
    for k in range(K):
        Y, radii = folded_reports(u, eps, rng_for(iid, eps, stream, k))
        d = pairwise_euclidean(p, Y)[geom.row_map]
        direct = exact_assignment(d)
        excess = assignment_cost(C, direct)-optimum
        if max(0., excess) != folded_row["samples"]["excess"][k]:
            raise AssertionError("regenerated reports do not reproduce the folded baseline draw")
        arrays["reports"][k], arrays["radii"][k], arrays["direct_assignment"][k] = Y, radii, direct
        noise_bound = 2*float(radii.sum())+2*geom.D_proxy
        for lam in lambdas:
            C_hat = corrected_costs(d, lam, m, eps)
            assignment = exact_assignment(C_hat)
            cost = assignment_cost(C, assignment)
            regret = cost-optimum
            osc = float(np.ptp(C_hat-C, axis=0).sum())
            # |g(s)-s| <= sqrt(b) with g(s)-s in [-sqrt(b),0], so osc <= 2||W_j|| + sqrt(b) per column.
            bound = noise_bound+n*np.sqrt(lam*m*(m+1))/eps
            tol = 1e-8*max(1., abs(optimum), abs(cost), bound)
            if regret < -tol or regret > osc+tol or osc > bound+tol:
                raise FloatingPointError("bias-corrected decoder certificate failed")
            arrays[f"assignment_{cheap_name(lam)}"][k] = assignment
            for key, value in dict(excess=max(0., regret), output_cost=cost, oscillation_original=osc,
                                   realization_bound=bound, mean_noise_norm=float(radii.mean()),
                                   max_reconstruction_error=float(np.max(np.abs(C_hat-C))),
                                   agreement_with_proxy_optimum=float(np.mean(assignment == opt_assignment))).items():
                cheap[lam][key].append(value)
        for a, samples in robust_samples.items():
            t = confidence_radius(m, eps, a, n)
            l, uu = interval_bounds(d, t)
            if manifest["robust_solver"] == "compact_lp":
                sol = solve_compact_lp(l, uu, fallback=direct)
            else:
                sol = double_oracle(l, uu, initial=[direct, np.arange(n)])
            H = sol["H"]
            solver_totals[a] += sol["seconds"]
            if manifest.get("birkhoff", True):
                weights, perms, leftover = birkhoff_decomposition(H)
                if weights.sum() <= 0:
                    raise FloatingPointError("empty Birkhoff decomposition")
                probabilities = weights/weights.sum()
                # Certify and score the actual normalized output mixture, including numerical repair.
                H = np.zeros_like(H)
                for probability, perm in zip(probabilities, perms):
                    H[np.arange(n), perm] += probability
                sol["upper"], _ = worst_case_regret(H, l, uu)
                sol["gap"] = sol["upper"]-sol["lower"]
            # Feasibility is asserted for the H actually output and scored (not the solver's intermediate).
            sol["H_residual"] = ds_residual(H)
            expected_cost = float(np.sum(C*H))
            regret = expected_cost-optimum
            covered = bool(np.all(C >= l-1e-12) and np.all(C <= uu+1e-12))
            radius_event = bool(radii.max() <= t)
            # Reverse triangle inequality: max radius <= t implies box coverage (converse need not hold).
            if radius_event and not covered:
                raise AssertionError("confidence coverage identity failed")
            F_direct, _ = worst_case_regret(permutation_matrix(direct), l, uu)
            tol = 1e-7*max(1., abs(optimum), abs(expected_cost), abs(sol["upper"]), abs(F_direct))
            if regret < -tol or regret > B+tol:
                raise FloatingPointError("robust expected regret outside [0, n diam Q]")
            if covered and regret > sol["upper"]+tol:
                raise FloatingPointError("interval certificate failed on a covered draw")
            if sol["lower"] > sol["upper"]+tol or (sol["exact"] and sol["upper"] > F_direct+sol["gap"]+tol):
                raise FloatingPointError("interval optimization certificate inconsistent")
            if max(sol["H_residual"], sol["G_residual"]) > 1e-8 or (sol["exact"] and sol["gap"] > tol):
                raise FloatingPointError("interval solver feasibility or optimality tolerance failed")
            if manifest.get("birkhoff", True):
                sampled = perms[rng_server(a, k).choice(len(weights), p=probabilities)]
                mixture_cost = float(sum(w*assignment_cost(C, s) for w, s in zip(probabilities, perms)))
                if abs(mixture_cost-expected_cost) > 1e-6*max(1., expected_cost)+leftover*B:
                    raise FloatingPointError("Birkhoff mixture does not reproduce <C,H>")
                arrays[f"bvn_weights_alpha{a:g}"].extend(probabilities.tolist())
                arrays[f"bvn_perms_alpha{a:g}"].extend(perms.tolist())
                arrays[f"bvn_offsets_alpha{a:g}"].append(len(arrays[f"bvn_weights_alpha{a:g}"]))
                sampled_excess = max(0., assignment_cost(C, sampled)-optimum)
                terms = len(weights)
            else:
                sampled, sampled_excess, terms, leftover = direct, None, None, None
            arrays[f"H_alpha{a:g}"][k] = H
            arrays[f"G_alpha{a:g}"][k] = sol["G"]
            arrays[f"sampled_assignment_alpha{a:g}"][k] = sampled
            for key, value in dict(excess=max(0., regret), output_cost=expected_cost,
                                   realization_bound=min(B, sol["upper"]) if covered else B,
                                   mean_noise_norm=float(radii.mean()),
                                   agreement_with_proxy_optimum=float(np.sum(H[np.arange(n), opt_assignment])/n),
                                   confidence_radius=t, coverage_event=float(covered),
                                   radius_event=float(radius_event), interval_upper=sol["upper"],
                                   interval_lower=sol["lower"], interval_gap=sol["gap"], direct_interval_objective=F_direct,
                                   interval_certificate_term=min(B, sol["upper"]), solver_seconds=sol["seconds"],
                                   solver_exact=float(sol["exact"]), solver_iterations=float(sol["iterations"]),
                                   solver_H_residual=sol["H_residual"], solver_G_residual=sol["G_residual"],
                                   solver_raw_H_residual=sol.get("H_raw_residual"),
                                   solver_raw_G_residual=sol.get("G_raw_residual"),
                                   support_size=float(np.sum(H > 1e-9)), max_H_entry=float(H.max()),
                                   bvn_terms=None if terms is None else float(terms),
                                   bvn_leftover=leftover, sampled_output_excess=sampled_excess).items():
                samples[key].append(value)
    rows = []
    for lam in lambdas:
        name = cheap_name(lam)
        rows.append(_decoder_row(meta, name, cheap[lam], decoder=f"hungarian_on_{name}", decoder_family=name,
                                 bias_lambda=lam, bias_subtracted=lam*m*(m+1)/eps**2, randomized_output=False,
                                 mechanism_certificate_uncapped=2*geom.D_proxy+2*n*m/eps+n*np.sqrt(lam*m*(m+1))/eps,
                                 universal_cap=B, runtime_seconds=None, **common))
    for a, samples in robust_samples.items():
        certificate = float(np.mean(samples["interval_certificate_term"]))+a*B
        rows.append(_decoder_row(meta, f"robust_interval_a{a:g}", samples, decoder="randomized_interval_minimax_regret",
                                 decoder_family="robust_interval", robust_alpha=a,
                                 confidence_radius_value=confidence_radius(m, eps, a, n), randomized_output=True,
                                 evaluation_statistic="conditional expected true cost <C,H> of the randomized output "
                                 "(Rao-Blackwellized over server randomization); one Birkhoff sample also recorded",
                                 robust_solver=manifest["robust_solver"], mechanism_certificate_uncapped=None,
                                 universal_cap=B, interval_expected_certificate=certificate,
                                 solver_total_seconds=solver_totals[a], runtime_seconds=solver_totals[a], **common))
    for key in [k for k in arrays if k.startswith("bvn_")]:
        arrays[key] = np.asarray(arrays[key], dtype=np.int64 if "perms" in key or "offsets" in key else np.float64)
    return rows, arrays


def span_anchor_row(Q, R, C, optimum, fixed_cost, catalog, anchor_row, eps, K, iid, meta):
    geom = SpanGeometry.from_fold(Q, catalog.exact.fold)
    fields = dict(meta, arm_id="span_anchor", arm_label="span_anchor", requested_base_mechanism="span_anchor",
                  approximation_family="exact", operating_point="new_mechanism", span_dimension=geom.k)
    if catalog.exact.fold.full_span and np.array_equal(geom.p, catalog.exact.anchor.p):
        # k=d: identical release law (no height coordinate) and identical secant geometry and split.
        row = dict(fields, **{k: v for k, v in anchor_row["metadata"].items() if k not in fields})
        row.update(base_mechanism="span_anchor", provenance="derived_reused", source_arm_id="anchor",
                   reuse_reason="k=d: span-only anchor equals the full original anchor law", span_dimension=geom.k)
        return dict(metadata=row, samples=anchor_row["samples"])
    stream = f"span_anchor:k{geom.k}"
    result = evaluate_span_anchor(Q, R, C, optimum, fixed_cost, geom, eps, K, lambda k: rng_for(iid, eps, stream, k),
                                  catalog.diameter)
    row = dict(fields, **result["metadata"])
    row.update(noise_stream=None if geom.no_release else stream, coupling_group=None, coupling_method=None,
               source_arm_id=None, reuse_reason=None, radius_budget_fraction=None, ablation=None)
    return dict(metadata=row, samples=result["samples"])


def unit_path(directory, family, n, regime, replicate, eps):
    return directory/"units"/f"{family}_n{n}_{regime}_r{replicate:02d}_eps{eps:g}.json"


def validate_unit(record, manifest, family, n, path):
    if record.get("fingerprint") != manifest["fingerprint"] or record.get("status") != "complete":
        raise ValueError(f"incompatible checkpoint {path}")
    if len(record["rows"]) != expected_rows(manifest, family, n):
        raise ValueError(f"unexpected arm count in {path}")
    for row in record["rows"]:
        x = np.asarray(row["samples"]["excess"], dtype=float)
        K = 1 if row["metadata"].get("provenance") == "deterministic" else manifest["K"]
        if len(x) != K or not np.all(np.isfinite(x)) or np.any(x < 0) or not row["metadata"]["pointwise_ok"] \
                or any(len(v) != K for v in row["samples"].values()):
            raise ValueError(f"invalid checkpoint samples {path}")


def run_unit(manifest, directory, family, n, regime, replicate, eps, Q, catalog, fit_record):
    fp, K, phase = manifest["fingerprint"], manifest["K"], manifest["phase"]
    path = unit_path(directory, family, n, regime, replicate, eps)
    if path.exists():
        record = read_json(path)
        validate_unit(record, manifest, family, n, path)
        return record
    started = time.monotonic()
    R = rf.evaluation_R(regime, Q, family, replicate, phase, manifest["root_seed"], manifest["parameters"])
    iid = stable_id(instance_spec(manifest, family, n, regime, replicate))
    C = pairwise_euclidean(Q, R)
    opt_assignment, optimum = optimal_cost(C)
    fixed_cost = assignment_cost(C, np.arange(n))
    random_cost = random_permutation_baseline(C)
    fold = catalog.exact.fold
    d = manifest["d"]
    analytic = 2 if family == "rank2" else min(n-1, d)
    meta = dict(experiment_version=VERSION, fingerprint=fp, run_phase=phase, study_id=STUDY_ID,
                geometry_family=family, private_regime=regime, n=n, d=d, instance_id=iid,
                instance_seed=rf.seed_of("evaluation_R", family, n, d, regime, replicate, phase, manifest["root_seed"]),
                root_seed=manifest["root_seed"], geometry_replicate=replicate, replicate=replicate,
                geometry_randomized=True, public_Q_fixed=True,
                replication_unit="independent private R draw; public Q fixed per (family, n)",
                instance_protocol=manifest["instance_protocol"],
                configuration_label=f"{family}:n{n}:{regime}", fixed_instance_sweep=True, scaled_changing_instance=False,
                public_affine_rank=fold.k, numerical_rank_tolerance=fold.rank_tol, pca_rank_tolerance=catalog.tol,
                analytic_affine_rank=analytic, intrinsic_rank=2 if family == "rank2" else None,
                cluster_count=manifest["parameters"]["public_cluster_count"] if family == "clustered" else None,
                cluster_spread=manifest["parameters"]["public_cluster_rms"] if family == "clustered" else None,
                exact_fold_residual=catalog.exact.D_proxy,
                rank_status="approximate_rank_truncation" if fold.k < analytic else "expected_rank_retained",
                preprocessing="numerical_projection_residual_recorded",
                scale_convention="public Q centered, RMS radius 1; regime RMS values in these units", public_scale=1.,
                public_rms_radius=rf.rms_radius(Q), public_diameter=catalog.diameter, OPT=optimum,
                fixed_matching_cost=fixed_cost, expected_random_cost=random_cost, random_minus_OPT=random_cost-optimum,
                K_requested=K, matching_tie_rule="scipy_linear_sum_assignment_fixed_input_order",
                fit_id=fit_record["fit_id"], fit_path=fit_record["fit_path"],
                numpy_version=np.__version__, scipy_version=scipy.__version__)
    rows = rf.baseline_rows(Q, R, C, optimum, fixed_cost, catalog, eps, K, iid, meta)
    by_arm = {row["metadata"]["arm_id"]: row for row in rows}
    rows.append(span_anchor_row(Q, R, C, optimum, fixed_cost, catalog, by_arm["anchor"], eps, K, iid, meta))
    server_key = stable_id([iid, eps, "robust_output_sampling"])

    def rng_server(alpha, k):
        return np.random.default_rng(int(stable_id([server_key, alpha, k])[:16], 16))

    decoded, arrays = shared_decoder_rows(Q, R, C, optimum, opt_assignment, catalog, by_arm["folded"], eps, K, iid,
                                          meta, manifest, robust_enabled(manifest, family, n), rng_server)
    rows += decoded
    # Public stress-set selection: copy the selected decoder's samples (derived operating point).
    chosen = fit_record["selected"]
    source = next(r for r in rows if r["metadata"]["arm_id"] == ("folded" if chosen == "direct" else chosen))
    selected = dict(source["metadata"], arm_id="selected_cheap", arm_label="selected_cheap", operating_point="selected",
                    provenance="derived_reused", source_arm_id=source["metadata"]["arm_id"], report_source_arm_id="folded",
                    reuse_reason="public stress-set selection copies the chosen decoder's samples",
                    selected_branch=chosen, selection_origin="public_synthetic_stress_set",
                    candidate_set="direct, " + ", ".join(cheap_name(float(x)) for x in manifest["lambdas"]),
                    fit_selected_candidate_id=chosen, fit_train_worst_mean_regret=fit_record["selected_train_worst"],
                    fit_validation_worst_mean_regret=fit_record["selected_validation_worst"],
                    fit_selection_objective="max over public synthetic configurations of mean true matching regret "
                    "(finite stress set; heuristic, not a guarantee over arbitrary R)", K_train=fit_record["K_train"])
    rows.append(dict(metadata=selected, samples=source["samples"]))
    arrays_path = path.with_suffix(".npz")
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, public_Q=Q, private_R=R, opt_assignment=opt_assignment, **arrays)
    for row in rows:
        row["metadata"]["replay_arrays_path"] = str(arrays_path)
    record = dict(fingerprint=fp, status="complete", rows=rows, public_Q=Q.tolist(), private_R=R.tolist(),
                  fit_id=fit_record["fit_id"], replay_arrays_path=str(arrays_path),
                  elapsed_seconds=time.monotonic()-started)
    validate_unit(record, manifest, family, n, path)
    atomic_write_json(path, record)
    return record


# ----------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------

def results_directory(root, fingerprint, phase):
    return Path(root)/RESULTS_SUBDIR/phase/fingerprint


def all_units(manifest):
    return [(family, n, regime, replicate, eps) for family in manifest["families"] for n in manifest["ns"]
            for regime in manifest["regimes"] for replicate in range(manifest["replicates"])
            for eps in manifest["epsilons"]]


def prepare(root, pilot=False, **overrides):
    ident, fp = identity(pilot, **overrides)
    manifest = dict(ident, fingerprint=fp)
    directory = results_directory(root, fp, manifest["phase"])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"manifest.json").exists():
        if read_json(directory/"manifest.json").get("fingerprint") != fp:
            raise ValueError(f"manifest from different fingerprint in {directory}")
    else:
        atomic_write_json(directory/"manifest.json", manifest)
    # Stage 1: every public fit is completed and saved before any evaluation R is generated.
    for family in manifest["families"]:
        for n in manifest["ns"]:
            for eps in manifest["epsilons"]:
                load_or_fit(directory, fp, family, n, eps, manifest)
    print(f"continuous_decoder {manifest['phase']} prepared -> {directory} (resolved {directory.resolve()})", flush=True)
    return directory, manifest


def run_units(directory, manifest, units):
    timings = []
    catalogs = {}
    for family, n, regime, replicate, eps in units:
        if (family, n) not in catalogs:
            Q = rf.public_Q(family, n, manifest["d"], manifest["root_seed"], manifest["parameters"])
            catalogs[family, n] = (Q, PublicProxyCatalog(Q))
        Q, catalog = catalogs[family, n]
        fit_record = load_or_fit(directory, manifest["fingerprint"], family, n, eps, manifest, create=False)
        started = time.monotonic()
        record = run_unit(manifest, directory, family, n, regime, replicate, eps, Q, catalog, fit_record)
        solver = sum(r["metadata"].get("solver_total_seconds") or 0. for r in record["rows"])
        timings.append(dict(family=family, n=n, regime=regime, replicate=replicate, epsilon=eps,
                            seconds=time.monotonic()-started, robust_solver_seconds=solver))
        print(f"{family} n={n} {regime} r{replicate} eps={eps:g}: {timings[-1]['seconds']:.2f}s "
              f"(robust solver {solver:.2f}s)", flush=True)
    return timings


def run_continuous_decoder(root, pilot=False, stage="all", shard_index=None, shard_count=None, **overrides):
    refuse_login_compute("run the continuous_decoder study")
    directory, manifest = prepare(root, pilot, **overrides)
    if stage == "prepare":
        return directory
    units = all_units(manifest)
    if stage == "shard":
        if shard_index is None or not shard_count or not 0 <= shard_index < shard_count:
            raise ValueError("shard stage needs 0 <= --shard-index < --shard-count")
        timings = run_units(directory, manifest, units[shard_index::shard_count])
        atomic_write_json(directory/"timings"/f"shard_{shard_index:04d}.json", dict(units=timings))
        return directory
    if stage == "finalize":
        return finalize_continuous_decoder(directory)
    if stage != "all":
        raise ValueError(f"unknown stage {stage}")
    timings = run_units(directory, manifest, units)
    atomic_write_json(directory/"timings"/"serial.json", dict(units=timings))
    if manifest.get("runtime_probe"):
        runtime_projection(directory, manifest, timings)
    finalize_continuous_decoder(directory)
    return directory


def runtime_projection(directory, manifest, timings, probe_draws=2):
    """Measure robust solves at n=50 on pilot-phase reports and project production CPU time."""
    probe = []
    production = design(False)
    for family in manifest["families"]:
        for n in sorted(set(production["ns"])):
            Q = rf.public_Q(family, n, manifest["d"], manifest["root_seed"], manifest["parameters"])
            catalog = PublicProxyCatalog(Q)
            fold = catalog.exact.fold
            p = catalog.exact.anchor.p
            for regime in manifest["regimes"]:
                R = rf.evaluation_R(regime, Q, family, 0, "pilot", manifest["root_seed"], manifest["parameters"])
                u = fold.fold_private(R)
                for eps in manifest["epsilons"]:
                    rng = np.random.default_rng(seed_of("runtime_probe", family, n, regime, eps))
                    for _ in range(probe_draws):
                        Y, _ = folded_reports(u, eps, rng)
                        dist = pairwise_euclidean(p, Y)
                        direct = exact_assignment(dist)
                        for alpha in manifest["alphas"]:
                            l, uu = interval_bounds(dist, confidence_radius(fold.m, eps, alpha, n))
                            lp = solve_compact_lp(l, uu, fallback=direct)
                            started = time.monotonic()
                            weights, _, leftover = birkhoff_decomposition(lp["H"])
                            bvn_seconds = time.monotonic()-started
                            entry = dict(family=family, n=n, regime=regime, epsilon=eps, alpha=alpha,
                                         lp_seconds=lp["seconds"], lp_exact=lp["exact"], lp_upper=lp["upper"],
                                         lp_gap=lp["gap"], bvn_seconds=bvn_seconds, bvn_terms=len(weights),
                                         bvn_leftover=leftover)
                            if not probe and n == min(production["ns"]):
                                # One alternative-solver timing is enough; production uses the compact LP.
                                do = double_oracle(l, uu, initial=[direct, np.arange(n)], max_iter=100)
                                entry.update(do_seconds=do["seconds"], do_iterations=do["iterations"],
                                             do_exact=do["exact"], do_upper=do["upper"], do_gap=do["gap"])
                            probe.append(entry)
    per_solve = {}
    for n in production["ns"]:
        part = [e for e in probe if e["n"] == n and e["alpha"] == production["alphas"][0]]
        per_solve[n] = float(np.mean([e["lp_seconds"]+e["bvn_seconds"] for e in part]))
    count_per_n = len(production["families"])*len(production["regimes"])*production["replicates"]*len(production["epsilons"])
    robust_seconds = {n: per_solve[n]*count_per_n*production["K"]*len(production["alphas"]) for n in production["ns"]}
    nonrobust = [t["seconds"]-t["robust_solver_seconds"] for t in timings]
    base_unit = float(np.mean(nonrobust))*production["K"]/manifest["K"]
    projection = dict(probe=probe, mean_seconds_per_robust_solve_incl_birkhoff=per_solve,
                      projected_robust_cpu_seconds_by_n=robust_seconds,
                      projected_nonrobust_cpu_seconds=base_unit*len(all_units(production))*1.5,
                      method="mean probe LP+Birkhoff seconds x production solves; non-robust part scaled from pilot "
                      "units by K ratio with a 1.5x allowance for n=50; not a runtime guarantee")
    atomic_write_json(directory/"runtime_projection.json", projection)
    print(f"runtime projection: {projection['mean_seconds_per_robust_solve_incl_birkhoff']} s/solve; robust CPU "
          f"{robust_seconds}; non-robust {projection['projected_nonrobust_cpu_seconds']:.0f}s", flush=True)
    return projection


def finalize_continuous_decoder(directory):
    """Validate every checkpoint (no simulation), export canonical table, summaries and figures."""
    refuse_login_compute("finalize the continuous_decoder study")
    directory = Path(directory)
    manifest = read_json(directory/"manifest.json")
    for family, n, regime, replicate, eps in all_units(manifest):
        path = unit_path(directory, family, n, regime, replicate, eps)
        validate_unit(read_json(path), manifest, family, n, path)
    print(f"Validated {len(all_units(manifest))} continuous_decoder checkpoints in {directory}", flush=True)
    shards = sorted((directory/"timings").glob("*.json"))
    if shards:
        units = [u for path in shards for u in read_json(path)["units"]]
        atomic_write_json(directory/"timings.json", dict(units=units, serial_seconds=sum(u["seconds"] for u in units),
                                                         robust_solver_seconds=sum(u["robust_solver_seconds"] for u in units),
                                                         files=[str(p) for p in shards]))
    from private_matching.analysis_table import export_table
    from private_matching.continuous_decoder_plots import make_report
    output = directory/"handoff"
    export_table([directory], output)
    make_report(directory, output)
    return output
