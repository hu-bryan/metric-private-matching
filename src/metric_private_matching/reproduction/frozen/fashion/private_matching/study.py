"""Compact research suite. Frozen cube runners and their streams are untouched."""
from __future__ import annotations

import itertools
import math
import time
from pathlib import Path

import numpy as np
import scipy
from private_matching.anchor import (AnchorMessage, anchor_server, nearest_anchor,
                                    optimized_split, centered_client, directional_coefficient)
from private_matching.assignment import (pairwise_euclidean, exact_assignment, assignment_cost,
                                        optimal_cost, random_permutation_baseline)
from private_matching.mechanisms import sample_euclidean_laplace, sample_supnorm_cone_laplace
from private_matching.proxies import PublicProxyCatalog
from private_matching.results import atomic_write_json, read_json
from private_matching.followup import source_digest
from private_matching.login_guard import refuse_login_compute
from private_matching.study_geometry import (cube_approximation_design, design, geometry, instance_units, lower_bound_design,
                                             projection_design, stable_id, interchangeability_design,
                                             dimension_tradeoff_design)

VERSION = "compact-matching-v2"
OPT_IN = {"projection_epsilon", "cube_approximation", "psi_lower_bound", "interchangeability", "dimension_tradeoff",
          "robust_fitted", "continuous_decoder"}


def candidate_key(base, family="exact", value=None):
    return base if family == "exact" else f"{base}:{family}:{value}"


def canonical(base, family, value, catalog):
    # Endpoints use the full fold by convention, including public reference order.
    if base == "fixed" or (family == "pca" and value == 0) or (family == "representatives" and value == 1):
        return "fixed"
    endpoint = family == "exact" or (family == "pca" and value == catalog.rank == catalog.exact.fold.k) or (family == "representatives" and value == catalog.n)
    if endpoint:
        if base == "ambient" and catalog.exact.fold.m == catalog.d and catalog.rank == catalog.exact.fold.k:
            # Codimension-one reflection fixes Q; isotropic laws and costs coincide.
            return "folded"
        return base
    return candidate_key(base, family, value)


def near_far_gap(task_radius, assignment, near):
    """Mean public-task radius given to near drivers minus far drivers (None without groups)."""
    if near is None:
        return None
    driver_radius = np.empty(len(assignment))
    driver_radius[assignment] = task_radius  # assignment[i] is the driver of task i
    return float(driver_radius[near].mean()-driver_radius[~near].mean())


def rng_for(iid, epsilon, stream, k):
    return np.random.default_rng(int(stable_id([VERSION, iid, epsilon, stream, k]), 16))


def evaluate(Q, R, C, optimum, fixed_cost, catalog, base, geom, eps, K, iid,
             clusters=None, populations=None, fraction=None, diagnostic=None,
             coupled_epsilon=False, near=None):
    start = time.monotonic()
    n, d = Q.shape
    no_release = base == "fixed" or (base in {"folded", "anchor"} and geom.anchor.B == 0)
    noiseless = base == "proxy"
    nonprivate = noiseless or diagnostic == "exact_radius"
    ey = er = eo = 0.
    dim = 0 if no_release or noiseless else (d if base == "ambient" else n if base == "phi_infinity" else geom.fold.m)
    norm = None if no_release or noiseless or base == "exponential" else "linf" if base == "phi_infinity" else "oscillation_half_range" if base == "centered" else "l2"
    D = geom.D_proxy if base in {"folded", "anchor", "proxy"} else 0.
    if base == "anchor" and not no_release:
        ey, er = optimized_split(geom.anchor.B, eps) if fraction is None else (eps*(1-fraction), eps*fraction)
        if diagnostic == "exact_radius":
            er = 0.
        bound = 2*D + n*(4*geom.anchor.B/ey + (1/er if er else 0))
    elif base in {"folded", "ambient"} and not no_release:
        ey = eps if diagnostic != "original_split" else eps*(1-fraction)
        bound = 2*D + 2*n*dim/ey
    elif base in {"phi_infinity", "centered"}:
        dim = n if base == "phi_infinity" else n-1
        eo = eps
        bound = 2*n*dim/eps
    elif base == "exponential":
        if n > 8:
            raise ValueError("central enumeration requires n<=8")
        eo, dim = eps, None
        bound = 2*(math.lgamma(n+1)+1)/eps
    else:
        bound = n*catalog.diameter if base == "fixed" else 2*D
    stream = f"{base}:{geom.family}:{geom.value}"
    coupling = None
    if diagnostic:
        # Same unit-rate vector and scalar draws across all four ablations/fractions.
        stream = f"decoder_ablation:{geom.family}:{geom.value}"
        coupling = stable_id([iid, eps, stream])
    elif coupled_epsilon and not no_release and not noiseless:
        # Common random numbers across epsilon: identical unit-rate draws, scaled by budget.
        coupling = stable_id([iid, "epsilon_coupled", stream])
    proxy_C = C
    u = None
    if base in {"folded", "anchor", "proxy"} and not no_release:
        u = geom.fold.fold_private(R)
        proxy_C = pairwise_euclidean(geom.anchor.p, u)[geom.row_map]
        if geom.family == "pca":
            assert geom.fold.m == min(geom.value+1, d)
    proxy_assignment, proxy_opt = optimal_cost(proxy_C)
    task_radius = np.linalg.norm(Q-Q.mean(axis=0), axis=1)
    samples = {key: [] for key in ["excess", "output_cost", "oscillation_original", "oscillation_proxy",
                                    "realization_bound", "proxy_regret", "wrong_cluster_fraction",
                                    "clipped_radius_fraction", "mean_noise_norm", "max_reconstruction_error",
                                    "agreement_with_proxy_optimum", "near_minus_far_task_radius"]}
    permutation_costs = probabilities = None
    if base == "exponential":
        perms = np.array(list(itertools.permutations(range(n))))
        permutation_costs = C[np.arange(n)[None], perms].sum(axis=1)
        weights = np.exp(-eps*(permutation_costs-permutation_costs.min())/2)
        probabilities = weights/weights.sum()
    last_progress = start
    for k in range(1 if no_release or noiseless else K):
        rng = rng_for(iid, "epsilon_coupled" if coupled_epsilon and not diagnostic else eps, stream, k)
        pointwise = bound
        radii = clipped = None
        if no_release:
            assignment = np.arange(n)
            cost = fixed_cost
            C_hat = None
        elif base == "exponential":
            index = rng.choice(len(probabilities), p=probabilities)
            assignment, cost, C_hat = perms[index], float(permutation_costs[index]), None
            pointwise = None
        else:
            if noiseless:
                C_hat = proxy_C
            elif base in {"anchor", "folded", "ambient"}:
                unit_W, unit_radii = sample_euclidean_laplace(n, dim, 1., rng)
                W, radii = unit_W/ey, unit_radii/ey
                if base == "ambient":
                    C_hat = pairwise_euclidean(Q, R+W)
                    pointwise = 2*float(radii.sum())
                elif base == "folded":
                    C_hat = pairwise_euclidean(geom.anchor.p, u+W)[geom.row_map]
                    pointwise = 2*D + 2*float(radii.sum())
                else:
                    Y = u+W
                    anchors = nearest_anchor(geom.anchor.p, Y)
                    radius = np.linalg.norm(u-geom.anchor.p[anchors], axis=1)
                    eta = rng.laplace(size=n)/er if er else np.zeros(n)
                    released = np.maximum(0, radius+eta)
                    clipped = float(np.mean(released == 0))
                    C_hat = anchor_server(AnchorMessage(Y, released), geom.anchor.p)[geom.row_map]
                    pointwise = 2*D + float((4*geom.anchor.max_projection(W)+np.abs(eta)).sum())
            elif base == "phi_infinity":
                W, radii = sample_supnorm_cone_laplace(n, n, eps, rng)
                C_hat, pointwise = C+W, 2*float(radii.sum())
            elif base == "centered":
                C_hat, radii = centered_client(C, eps, rng)
                pointwise = 2*float(radii.sum())
            else:
                raise ValueError(base)
            assignment = exact_assignment(C_hat)
            cost = assignment_cost(C, assignment)
        regret = cost-optimum
        tolerance = 1e-8*max(1., abs(optimum), abs(cost), abs(pointwise or 0))
        osc = float(np.ptp(C_hat-C, axis=0).sum()) if C_hat is not None else None
        osc_proxy = float(np.ptp(C_hat-proxy_C, axis=0).sum()) if C_hat is not None else None
        proxy_regret = assignment_cost(proxy_C, assignment)-proxy_opt
        if regret < -tolerance or not np.isfinite(regret):
            raise FloatingPointError("invalid true-cost regret")
        if pointwise is not None and regret > pointwise+tolerance:
            raise FloatingPointError("realization certificate failed")
        if osc is not None and (regret > osc+tolerance or regret > 2*D+proxy_regret+tolerance):
            raise FloatingPointError("oscillation/proxy transfer failed")
        if base == "anchor" and osc_proxy is not None and osc_proxy > pointwise-2*D+tolerance:
            raise FloatingPointError("anchor secant oscillation certificate failed")
        values = dict(excess=max(0., regret), output_cost=cost, oscillation_original=osc,
                      oscillation_proxy=osc_proxy, realization_bound=pointwise,
                      proxy_regret=proxy_regret if base in {"anchor", "folded", "proxy"} else None,
                      wrong_cluster_fraction=float(np.mean(clusters != populations[assignment])) if clusters is not None else None,
                      clipped_radius_fraction=clipped,
                      mean_noise_norm=float(np.mean(radii)) if radii is not None else None,
                      max_reconstruction_error=float(np.max(np.abs(C_hat-C))) if C_hat is not None else None,
                      # Fraction of tasks given the same driver as the noiseless proxy (OPT for exact arms).
                      agreement_with_proxy_optimum=float(np.mean(assignment == proxy_assignment)),
                      near_minus_far_task_radius=near_far_gap(task_radius, assignment, near))
        for key in samples:
            samples[key].append(values[key])
        if time.monotonic()-last_progress > 30:
            print(f"{iid} eps={eps:g} {stream}: draw {k+1}/{K}", flush=True)
            last_progress = time.monotonic()
    metadata = dict(base_mechanism=base, decoder="private_anchor_distance_inner_product" if base == "anchor" else
                    "euclidean_report_distances" if base in {"ambient", "folded"} else base,
                    privacy_model="nonprivate_diagnostic" if nonprivate else "central" if base == "exponential" else "local",
                    epsilon=eps, epsilon_y=ey, epsilon_r=er, epsilon_other=eo,
                    coordinate_noise_dimension=dim, coordinate_noise_norm=norm,
                    expected_noise_norm=dim/(ey or eo) if dim and (ey or eo) else None,
                    scalar_noise_count=int(base == "anchor" and er > 0), scalar_noise_scale=1/er if er else None,
                    report_dimension=(dim+int(base == "anchor")) if dim is not None else None,
                    no_release=no_release, D_proxy=D, B_directional=geom.anchor.B if base in {"folded", "anchor"} else None,
                    directional_method="min(m,sqrt(2*m*log(2*nonzero_secant_pairs)))" if base == "anchor" else None,
                    mechanism_certificate_uncapped=bound, universal_cap=n*catalog.diameter,
                    mechanism_certificate_capped=min(bound, n*catalog.diameter),
                    noise_stream=stream if not no_release and not noiseless else None,
                    coupling_group=coupling,
                    coupling_method=("common unit-rate draws across epsilon, scaled by 1/budget" if coupling and not diagnostic
                                     else "same unit-rate isotropic vectors scaled by budget; same scalar unit Laplace" if coupling else None),
                    radius_budget_fraction=fraction, ablation=diagnostic,
                    provenance="deterministic" if no_release or noiseless else "simulated",
                    status="complete", pointwise_checked=base != "exponential", pointwise_ok=True,
                    runtime_seconds=time.monotonic()-start, failure_reason=None)
    return dict(metadata=metadata, samples=samples)


def run_case(case, replicate, eps, root_seed, K, directory, fingerprint, centered=False, central=False, context=None, phase="test",
             require_existing=False):
    started = time.monotonic()
    Q, R, meta, clusters, populations = geometry(case, replicate, eps, root_seed)
    n, d = Q.shape
    path = directory / "units" / f"{meta['instance_id']}_eps{eps:g}.json"
    if require_existing and not path.exists():
        raise FileNotFoundError(f"missing unit checkpoint {path}")
    if path.exists():
        record = read_json(path)
        if record.get("fingerprint") != fingerprint or record.get("status") != "complete":
            raise ValueError(f"incompatible checkpoint {path}")
        for row in record["rows"]:
            if row["metadata"].get("status") == "certificate_only":
                continue
            m = row["metadata"]
            expected = 1 if m["no_release"] or m["base_mechanism"] == "proxy" else K
            if (m.get("instance_id") != meta["instance_id"] or m.get("epsilon") != eps or m.get("K_requested") != K
                    or len(row["samples"]["excess"]) != expected or not m.get("pointwise_ok")
                    or not np.all(np.isfinite(row["samples"]["excess"]))
                    or np.any(np.asarray(row["samples"]["excess"]) < 0)
                    or any(len(values) != expected for values in row["samples"].values())):
                raise ValueError(f"invalid checkpoint samples {path}")
        return record
    if context is None:
        context = {}
    if meta["instance_id"] not in context:
        C = pairwise_euclidean(Q, R)
        context[meta["instance_id"]] = (PublicProxyCatalog(Q), C, optimal_cost(C)[1], assignment_cost(C, np.arange(n)))
    catalog, C, optimum, fixed_cost = context[meta["instance_id"]]
    # Structured private rows: first half near (x0.5 off-span), second half far (x4).
    near = (np.arange(n) < n//2) if "private_near_multiplier" in meta else None
    if near is not None:
        meta["OPT_near_minus_far_task_radius"] = near_far_gap(
            np.linalg.norm(Q-Q.mean(axis=0), axis=1), optimal_cost(C)[0], near)
    if case["study_id"] == "interchangeability":
        # Validates the group/partner semantics: 0 when the true optimum respects them.
        meta["OPT_wrong_cluster_fraction"] = float(np.mean(clusters != populations[optimal_cost(C)[0]]))
    meta.update(experiment_version=VERSION, fingerprint=fingerprint, run_phase=phase, public_affine_rank=catalog.exact.fold.k,
                numerical_rank_tolerance=catalog.exact.fold.rank_tol, pca_rank_tolerance=catalog.tol,
                public_diameter=catalog.diameter, OPT=optimum, fixed_matching_cost=fixed_cost,
                expected_random_cost=random_permutation_baseline(C),
                random_minus_OPT=random_permutation_baseline(C)-optimum,
                exact_fold_residual=catalog.exact.D_proxy,
                rank_status="approximate_rank_truncation" if catalog.exact.fold.k < meta["analytic_affine_rank"] else "expected_rank_retained",
                preprocessing="numerical_projection_residual_recorded",
                K_requested=K, matching_tie_rule="scipy_linear_sum_assignment_fixed_input_order",
                numpy_version=np.__version__, scipy_version=scipy.__version__)
    cache, rows = {}, []

    def add(base, family="exact", value=None, label=None, selection=None, selected=None, candidate_set=None,
            fraction=None, diagnostic=None):
        arm = label or candidate_key(base, family, value)
        key = canonical(base, family, value, catalog)
        if diagnostic:
            key = f"{base}:{family}:{value}:ablation:{diagnostic}:{None if diagnostic == 'original_full' else fraction}"
        use_exact = family == "exact" or (family == "pca" and value == catalog.rank == catalog.exact.fold.k) or (family == "representatives" and value == n)
        geom = catalog.exact if use_exact else catalog.geometry(family, value)
        effective_base = "fixed" if key == "fixed" else "folded" if key == "folded" else base
        reused = key in cache
        if not reused:
            evaluated = evaluate(Q, R, C, optimum, fixed_cost, catalog, effective_base, geom, eps, K,
                                 meta["instance_id"], clusters, populations, fraction, diagnostic,
                                 bool(case.get("coupled_epsilon")), near)
            cache[key] = (arm, evaluated)
        source, result = cache[key]
        row_meta = dict(meta, **result["metadata"])
        # Labels describe the requested arm, distribution reuse is explicit.
        row_meta.update(arm_id=arm, arm_label=arm, requested_base_mechanism=base,
                        radius_budget_fraction=fraction, ablation=diagnostic,
                        approximation_family=family, approximation_rank=value if family == "pca" else None,
                        representative_count=value if family == "representatives" else None,
                        source_arm_id=source if reused else None,
                        provenance="derived_reused" if reused else result["metadata"]["provenance"],
                        selected_branch=selected, selection_certificate=selection,
                        selection_origin="public_certificate" if selection is not None else None,
                        candidate_set=candidate_set, operating_point="selected" if selection is not None else "candidate" if family != "exact" else "baseline",
                        reuse_reason="canonical endpoint/no-release/isotropic equivalence or selected candidate" if reused else None)
        if base == "proxy":
            row_meta.update(privacy_model="nonprivate_diagnostic", decoder="noiseless_proxy_matching")
        if base in {"anchor", "folded", "proxy"}:
            row_meta["D_proxy"] = geom.D_proxy
            if key == "fixed":
                row_meta.update(mechanism_certificate_uncapped=2*geom.D_proxy,
                                mechanism_certificate_capped=min(2*geom.D_proxy, n*catalog.diameter))
        row_meta["D_proxy_per_point"] = row_meta["D_proxy"]/n
        rows.append(dict(metadata=row_meta, samples=result["samples"]))
        return arm

    # Full Psi precedes ambient so equivalent distributions use exactly one sample set.
    for base in ([] if case.get("diagnostics_only") else ["fixed", "folded", "ambient", "phi_infinity", "anchor"]):
        add(base)
    if centered:
        add("centered")
    if central and n <= 8:
        add("exponential")
    choices = [(catalog.exact.expected_bound(n, eps, False), "folded"),
               (catalog.exact.expected_bound(n, eps, True), "anchor"), (n*catalog.diameter, "fixed")]
    score, branch = min(choices, key=lambda pair: pair[0])
    if not case.get("diagnostics_only"):
        add(branch, label="auto", selection=score, selected=branch, candidate_set="folded,anchor,fixed; first on ties")
    families = list(case["families"])
    if case.get("secondary_pca") or case.get("cross_comparison"):
        families = list(dict.fromkeys(families+["pca", "representatives"]))
    certificates = []
    for family in families:
        maximum = catalog.rank if family == "pca" else n
        # Archive every cheap public score, without simulating every integer candidate.
        for v in range(0 if family == "pca" else 1, maximum+1):
            m = min(v+1 if family == "pca" else v, d)
            pairs = (n*(n-1)//2 if v else 0) if family == "pca" else v*(v-1)//2
            B = directional_coefficient(m, pairs)
            error = catalog.pca_errors[v] if family == "pca" else catalog.cluster_errors[v-1]
            for base in ["folded", "anchor"]:
                coefficient = ((2*np.sqrt(B)+1)**2 if base == "anchor" else 2*m) if pairs else 0.
                certificates.append(dict(family=family, base=base, value=v,
                                         D_proxy=float(error), selection_dimension_upper_bound=m,
                                         certificate=float(2*error+n*coefficient/eps)))
        values = set(case.get("ranks" if family == "pca" else "counts", [0, 1] if family == "pca" else [1, 2]))
        if case.get("secondary_pca") and family == "pca":
            values = {0, 1}
        if case.get("cross_comparison") and family not in case["families"]:
            values = {1, 4}
        values = {v for v in values if (0 if family == "pca" else 1) <= v <= maximum} | {maximum}
        for value in sorted(values):
            for base in ["folded", "anchor"]:
                add(base, family, value)
            # Actual public-approximation damage: no privacy claim.
            add("proxy", family, value)
        for base in ["folded", "anchor"]:
            value, score = catalog.select(family, eps, base == "anchor")
            candidate = candidate_key(base, family, value)
            if not any(r["metadata"]["arm_id"] == candidate for r in rows):
                add(base, family, value)
            add(base, family, value, label=f"{base}:{family}:selected", selection=score,
                selected=candidate, candidate_set=f"all integers {0 if family == 'pca' else 1}..{maximum}; smallest on ties; uncapped public certificate")
            certificates.append(dict(family=family, base=base, selected_value=value, certificate=score))
    if case.get("diagnostics"):
        for fraction in [.1, .25, .5]:
            for base, diagnostic in [("folded", "original_full"), ("folded", "original_split"),
                                     ("anchor", "noisy_radius"), ("anchor", "exact_radius")]:
                add(base, label=f"diagnostic:{diagnostic}:{fraction}", fraction=fraction, diagnostic=diagnostic)
    evaluated_names = {r["metadata"]["arm_id"] for r in rows}
    for cert in certificates:
        if "value" not in cert:
            continue
        family, base, value = cert["family"], cert["base"], cert["value"]
        name = candidate_key(base, family, value)
        if name in evaluated_names:
            continue
        rows.append(dict(metadata=dict(meta, arm_id=name, arm_label=name, base_mechanism=base,
                         approximation_family=family, approximation_rank=value if family == "pca" else None,
                         representative_count=value if family == "representatives" else None,
                         epsilon=eps, privacy_model="local", status="certificate_only",
                         provenance="public_certificate_only", operating_point="candidate",
                         D_proxy=cert["D_proxy"], D_proxy_per_point=cert["D_proxy"]/n,
                         selection_dimension_upper_bound=cert["selection_dimension_upper_bound"],
                         selection_certificate=cert["certificate"], universal_cap=n*catalog.diameter,
                         selection_origin="public_certificate", candidate_set="all integer candidates; no noise simulation"),
                         samples={"excess": []}))
    record = dict(fingerprint=fingerprint, status="complete", rows=rows, public_selections=certificates,
                  public_Q=Q.tolist(), private_R=R.tolist(), elapsed_seconds=time.monotonic()-started)
    atomic_write_json(path, record)
    print(f"{meta['study_id']} n={n} d={d} rep={replicate} eps={eps:g}: {len(rows)} rows, {len(cache)} evaluations, {record['elapsed_seconds']:.2f}s", flush=True)
    return record


def study_identity(pilot=False, study_filter=None, centered=False, central=False):
    """Cases, K, fingerprint and result directory name; identical in every shard."""
    if study_filter == "projection_epsilon":
        # Opt-in; never part of the default complete suite.
        cases = projection_design(pilot)
    elif study_filter == "cube_approximation":
        cases = cube_approximation_design(pilot)
    elif study_filter == "psi_lower_bound":
        cases = lower_bound_design(pilot)
    elif study_filter == "interchangeability":
        cases = interchangeability_design(pilot)
    elif study_filter == "dimension_tradeoff":
        cases = dimension_tradeoff_design(pilot)
    else:
        cases = design(pilot)
        if central:
            cases.append(dict(study_id="central_benchmark", geometry_family="two_population", n=6, d=6,
                              epsilons=[1., 8.], geometry_replicates=1, L=3., families=[],
                              scaled_changing_instance=False, diagnostics=False))
        cases = [c for c in cases if study_filter is None or c["study_id"] == study_filter]
    if not cases:
        raise ValueError("empty study selection")
    # The cube sweep matches the frozen cube archive's K=100 noise draws.
    K = 5 if pilot else 100 if study_filter in {"cube_approximation", "psi_lower_bound"} else 50
    identity = dict(version=VERSION, cases=cases, K=K, root_seed=20260921,
                    source_digest=source_digest(), numpy=np.__version__, scipy=scipy.__version__,
                    centered=centered, central=central)
    return identity, stable_id(identity)


def run_instances(manifest, directory, units, phase):
    """Run whole instances (every epsilon) so the public catalog is built once each."""
    cases, K, fp = manifest["cases"], manifest["K"], manifest["fingerprint"]
    timings = []
    for index, replicate in units:
        case, context = cases[index], {}
        for eps in case["epsilons"]:
            result = run_case(case, replicate, eps, manifest["root_seed"], K, directory, fp,
                              manifest["centered"], manifest["central"], context, phase)
            timings.append(dict(study_id=case["study_id"], geometry_family=case["geometry_family"], n=case["n"], d=case["d"],
                                replicate=replicate, epsilon=eps, elapsed_seconds=result["elapsed_seconds"], K=K))
    return timings


def finalize_study(directory, handoff=None):
    """Validate every expected checkpoint (no simulation), merge shard timings, export."""
    refuse_login_compute("finalize compact matching studies")
    directory = Path(directory)
    manifest = read_json(directory/"manifest.json")
    if manifest.get("study_id") == "robust_fitted":
        from private_matching.robust_fitted import finalize_robust_fitted
        return finalize_robust_fitted(directory)
    if manifest.get("study_id") == "continuous_decoder":
        from private_matching.continuous_decoder import finalize_continuous_decoder
        return finalize_continuous_decoder(directory)
    phase ="pilot" if directory.parent.name == "pilot" else "production"
    count = 0
    for index, replicate in instance_units(manifest["cases"]):
        case = manifest["cases"][index]
        for eps in case["epsilons"]:
            run_case(case, replicate, eps, manifest["root_seed"], manifest["K"], directory, manifest["fingerprint"],
                     manifest["centered"], manifest["central"], None, phase, require_existing=True)
            count += 1
    shards = sorted((directory/"timings").glob("shard_*.json"))
    if shards:
        units = [u for path in shards for u in read_json(path)["units"]]
        atomic_write_json(directory/"timings.json", dict(units=units, serial_seconds=sum(u["elapsed_seconds"] for u in units),
                                                         shard_files=len(shards)))
    print(f"Validated {count} unit checkpoints in {directory}", flush=True)
    from private_matching.analysis_table import export_table
    studies = {c["study_id"] for c in manifest["cases"]}
    projection = studies == {"projection_epsilon"}
    # Opt-in studies get their own handoff per phase: pilot draws are never pooled.
    opt_in = next(iter(studies)) if len(studies) == 1 and studies <= OPT_IN else None
    output = Path(handoff) if handoff else (directory.parents[2]/f"{opt_in}_handoff"/phase if opt_in
                                            else directory.parents[2]/"analysis_handoff")
    export_table([directory], output)
    if projection:
        from private_matching.projection_plots import make_report
        make_report(output/"analysis_table.csv", output, pilot=phase == "pilot")
    if studies == {"interchangeability"}:
        from private_matching.interchangeability_plots import make_report as interchangeability_report
        interchangeability_report(output/"analysis_table.csv", output, manifest["fingerprint"])
    return output


def run_study(root, pilot=False, study_filter=None, centered=False, central=False,
              stage="all", shard_index=None, shard_count=None):
    """stage: all (serial, original behavior) | prepare | shard | finalize.

    Shards own disjoint whole instances; each unit checkpoint is its own atomic file.
    Only prepare writes the manifest and only finalize writes the table.
    """
    refuse_login_compute("run compact matching studies")
    if study_filter == "robust_fitted":
        # Opt-in restricted fitted-family study with its own versioned directory tree.
        if stage != "all" or centered or central:
            raise ValueError("robust_fitted supports only --stage all, without --centered/--central")
        from private_matching.robust_fitted import run_robust_fitted
        return run_robust_fitted(root, pilot)
    if study_filter == "continuous_decoder":
        # Opt-in prior-free decoder study; prepare runs all public fits, shards own disjoint units.
        if centered or central:
            raise ValueError("continuous_decoder does not take --centered/--central")
        from private_matching.continuous_decoder import run_continuous_decoder
        return run_continuous_decoder(root, pilot, stage, shard_index, shard_count)
    identity, fp = study_identity(pilot, study_filter, centered, central)
    cases, K = identity["cases"], identity["K"]
    directory = Path(root)/VERSION/("pilot" if pilot else "production")/fp
    phase = "pilot" if pilot else "production"
    manifest = dict(identity, fingerprint=fp)
    if stage == "prepare":
        directory.mkdir(parents=True, exist_ok=True)
        atomic_write_json(directory/"manifest.json", manifest)
        print(f"Prepared {directory} with {len(instance_units(cases))} instances", flush=True)
        return directory
    if stage == "shard":
        if shard_index is None or not shard_count or not 0 <= shard_index < shard_count:
            raise ValueError("shard stage needs 0 <= --shard-index < --shard-count")
        if read_json(directory/"manifest.json").get("fingerprint") != fp:
            raise ValueError(f"manifest missing or from different source: {directory}")
        units = instance_units(cases)[shard_index::shard_count]
        timings = run_instances(manifest, directory, units, phase)
        atomic_write_json(directory/"timings"/f"shard_{shard_index:04d}.json", dict(units=timings))
        print(f"Shard {shard_index}/{shard_count}: {len(units)} instances, {len(timings)} units", flush=True)
        return directory
    if stage == "finalize":
        return finalize_study(directory)
    if stage != "all":
        raise ValueError(f"unknown stage {stage}")
    directory.mkdir(parents=True, exist_ok=True)
    atomic_write_json(directory/"manifest.json", manifest)
    timings = run_instances(manifest, directory, instance_units(cases), phase)
    atomic_write_json(directory/"timings.json", dict(units=timings, serial_seconds=sum(t["elapsed_seconds"] for t in timings)))
    if study_filter in OPT_IN:
        finalize_study(directory)
        print(f"Raw checkpoints: {directory / 'units'}", flush=True)
        return directory
    if pilot:
        estimates = []
        for target in design(False):
            matching = [t for t in timings if t["study_id"] == target["study_id"] and t["geometry_family"] == target["geometry_family"]]
            if not matching:
                continue
            measured = min(matching, key=lambda t: abs(t["n"]-target["n"])+abs(t["d"]-target["d"]))
            size_factor = max((target["n"]/measured["n"])**3,
                              target["n"]*target["d"]**2/(measured["n"]*measured["d"]**2))
            seconds = measured["elapsed_seconds"]*(50/K)*size_factor*target["geometry_replicates"]*len(target["epsilons"])
            estimates.append(dict(study_id=target["study_id"], n=target["n"], d=target["d"], seconds=seconds,
                                  measured_n=measured["n"], measured_d=measured["d"]))
        atomic_write_json(directory/"runtime_projection.json", dict(estimates=estimates,
                          projected_serial_seconds=sum(e["seconds"] for e in estimates),
                          method="Whole-unit time times K ratio and max(n^3,n*d^2) size ratio; excludes unmeasured families; not a runtime guarantee",
                          missing_studies=sorted({c["study_id"] for c in design(False)}-{t["study_id"] for t in timings})))
    from private_matching.analysis_table import export_table
    export_table([directory], Path(root)/"analysis_handoff")
    print(f"Raw checkpoints: {directory / 'units'}", flush=True)
    return directory
