"""Compact, explicitly versioned study designs; geometry uses no private tuning."""
from __future__ import annotations

import hashlib
import json
import numpy as np


def stable_id(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:24]


def design(pilot: bool = False) -> list[dict]:
    cases = []
    for n in [32, 64, 128, 256, 512]:
        cases.append(dict(study_id="appendix_scaled", geometry_family="two_population",
                          n=n, d=n, epsilons=[1.], geometry_replicates=1,
                          families=["representatives"], counts=[1, 2, n],
                          secondary_pca=n in [32, 128], scaled_changing_instance=True,
                          diagnostics=n == 32))
    for a_height, b_height in ([(2., 0.)] if pilot else [(2., 0.), (0.5, 0.), (2., 1.)]):
        cases.append(dict(study_id="controlled_epsilon" if (a_height, b_height) == (2., 0.) else "controlled_geometry", geometry_family="two_population",
                          n=32 if pilot else 128, d=32 if pilot else 128,
                          epsilons=([1., 8.] if pilot else [1., 4., 16.]) if (a_height, b_height) == (2., 0.) else [4.], geometry_replicates=1,
                          L=16., private_A_height=a_height, private_B_height=b_height,
                          families=["representatives"], counts=[1, 2],
                          scaled_changing_instance=False, diagnostics=a_height == 2. and b_height == 0.))
    for d in [32, 128, 512]:
        for family, thickness in [("unstructured", 0.), ("low_rank", 0.),
                                  ("near_low_rank", 0.02), ("near_low_rank", 0.2),
                                  ("clustered", 0.02), ("clustered", 0.2)]:
            cases.append(dict(study_id="structured_geometry", geometry_family=family,
                              n=128, d=d, epsilons=[4.],
                              geometry_replicates=1 if pilot else 20, intrinsic_rank=4,
                              residual_thickness=thickness if family == "near_low_rank" else None,
                              cluster_spread=thickness if family == "clustered" else None,
                              cluster_count=4 if family == "clustered" else None,
                              families=["representatives"] if family == "clustered" else ["pca"],
                              ranks=[0, 1, 4, 16], counts=[1, 2, 4, 16],
                              cross_comparison=d == 32 and thickness == 0.02,
                              scaled_changing_instance=False,
                              diagnostics=d == 32 and family == "near_low_rank" and thickness == 0.02))
    for s in range(1 if pilot else 3):
        cases.append(dict(study_id="cube_diagnostics", geometry_family="frozen_cube",
                          n=20, d=50, epsilons=[8.], geometry_replicates=1,
                          cube_configuration=s, families=[], diagnostics=True,
                          scaled_changing_instance=False, diagnostics_only=True))
    return cases


PROJECTION_EPSILONS = [4., 16., 64., 256.]
PROJECTION_GRID = [0, 1, 2, 3, 4, 5, 6, 8, 16, 32, 64]


def projection_design(pilot: bool = False) -> list[dict]:
    """Opt-in projection-approximation epsilon sweep; not part of the default suite.

    Dense candidates around the true structure (rank/count 4). Mechanism noise
    is coupled across epsilon (same unit-rate draws scaled by 1/epsilon).
    The pilot adds a numerical epsilon=1e9 limit check against the noiseless proxy.
    """
    cases = []
    for d in [128, 512]:
        for family, thickness in [("unstructured", 0.), ("low_rank", 0.),
                                  ("near_low_rank", 0.02), ("near_low_rank", 0.2),
                                  ("clustered", 0.02), ("clustered", 0.2)]:
            clustered = family == "clustered"
            cases.append(dict(study_id="projection_epsilon", geometry_family=family,
                              n=128, d=d, epsilons=PROJECTION_EPSILONS+([1e9] if pilot else []),
                              geometry_replicates=1 if pilot else 20, intrinsic_rank=4,
                              residual_thickness=thickness if family == "near_low_rank" else None,
                              cluster_spread=thickness if clustered else None,
                              cluster_count=4 if clustered else None,
                              families=["representatives"] if clustered else ["pca"],
                              ranks=PROJECTION_GRID, counts=[v for v in PROJECTION_GRID if v],
                              scaled_changing_instance=False, diagnostics=False,
                              coupled_epsilon=True))
    return cases


CUBE_APPROXIMATION_EPSILONS = [4., 16., 64., 256.]


def cube_approximation_design(pilot: bool = False, n: int = 20, d: int = 50) -> list[dict]:
    """Opt-in PCA/representative sweeps on the frozen uniform-cube masters.

    Every integer rank 0..n-1 and count 1..n; each master configuration s is its own
    case (never pooled). New labeled noise, coupled across epsilon.
    """
    return [dict(study_id="cube_approximation", geometry_family="frozen_cube", n=n, d=d,
                 epsilons=CUBE_APPROXIMATION_EPSILONS, geometry_replicates=1, cube_configuration=s,
                 families=["pca", "representatives"], ranks=list(range(n)), counts=list(range(1, n+1)),
                 scaled_changing_instance=False, diagnostics=False, coupled_epsilon=True)
            for s in range(1 if pilot else 3)]


LOWER_BOUND_N = [16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768]  # n=1024 dropped by request
LOWER_BOUND_D = [128, 192, 256, 384, 512, 768, 1024]


def lower_bound_design(pilot: bool = False) -> list[dict]:
    """Opt-in Theorem 31 (Section 7) separation instances: Psi_2 versus Xi and the other local mechanisms.

    n-sweep: n = d. d-sweep: fixed n with the same construction zero-padded into R^d
    (public affine rank n-1, folded dimension n for every d). L = n/(2 epsilon), so
    epsilon * excess is epsilon-invariant in distribution; epsilon = 1 only.
    """
    ns = [16, 32, 64] if pilot else LOWER_BOUND_N
    fixed_n, ds = (16, [16, 32, 64]) if pilot else (128, LOWER_BOUND_D)
    base = dict(study_id="psi_lower_bound", geometry_family="two_population", epsilons=[1.],
                geometry_replicates=1, families=[], scaled_changing_instance=True, diagnostics=False)
    return ([dict(base, n=n, d=n, sweep="n_equals_d") for n in ns]
            + [dict(base, n=fixed_n, d=d, sweep="d_zero_padded", zero_padded=True) for d in ds])


INTERCHANGEABILITY_FAMILIES = {"permuted_two_population_jitter", "permuted_two_population_original",
                               "permuted_unique_simplex", "permuted_interchangeable_types"}


def interchangeability_design(pilot: bool = False) -> list[dict]:
    """Opt-in interchangeability/complementary-regime study (Sept 25 request), epsilon = 1.

    Panel 1: permuted two-population family with Gaussian jitter on both sides (sigma > 0,
    independent geometries) plus the deterministic original family (sigma = 0, labels only).
    Panel 2: unique-optimum scaled simplex, labels only. Panel 3: g interchangeable public types.
    Deterministic-up-to-labels cases use ONE geometry (labels are not independent point clouds).
    Only endpoint and public-selected PCA/representative candidates are simulated.
    """
    ns = [32, 64] if pilot else [32, 64, 128, 256]
    G = 2 if pilot else 20
    base = dict(study_id="interchangeability", epsilons=[1.], families=["representatives", "pca"],
                ranks=[0], counts=[1], scaled_changing_instance=False, diagnostics=False)
    cases = []
    for n in ns:
        for sigma in [.1, 1.]:
            cases.append(dict(base, panel="robust_two_population", geometry_family="permuted_two_population_jitter",
                              n=n, d=n, L=n/2, jitter_sigma=sigma, geometry_replicates=G))
        cases.append(dict(base, panel="robust_two_population", geometry_family="permuted_two_population_original",
                          n=n, d=n, L=n/2, jitter_sigma=0., geometry_replicates=1))
        cases.append(dict(base, panel="unique_simplex", geometry_family="permuted_unique_simplex",
                          n=n, d=n, simplex_height=n/4, geometry_replicates=1))
    n3 = 32 if pilot else 128
    for g in [2, 4, 8, 16]:
        for tau in [0., .02, .1]:
            cases.append(dict(base, panel="interchangeable_types", geometry_family="permuted_interchangeable_types",
                              n=n3, d=n3, type_count=g, jitter_tau=tau, geometry_replicates=1 if tau == 0 else G))
    return cases


def _interchangeability_geometry(case, n, d, rng, meta):
    """Returns Q, R, public labels, private labels (row-aligned after independent shuffles)."""
    family = case["geometry_family"]
    pq, pr = rng.permutation(n), rng.permutation(n)
    if family.startswith("permuted_two_population"):
        L, sigma = case["L"], case["jitter_sigma"]
        right = np.arange(n) >= n//2           # public: False = -L e1 (left), True = +L e1
        Q = np.zeros((n, d))
        Q[:, 0] = np.where(right, L, -L)
        B = np.arange(n) >= n//2               # private: False = A=(L/2)e1+2L e2, True = B=.3L e1
        R = np.zeros((n, d))
        R[~B, 0], R[~B, 1], R[B, 0] = .5*L, 2*L, .3*L
        if sigma > 0:
            Q += sigma*rng.normal(size=(n, d))
            R += sigma*rng.normal(size=(n, d))
            rank, jitter = min(n-1, d), "independent N(0, sigma^2 I_d) on both Q and R"
        else:
            delta = L/n**3
            Q[:, :n] += delta*np.eye(n)        # tiny public perturbation delta e_i before shuffling
            meta.update(public_perturbation=delta)
            rank, jitter = n-1, "none; public delta e_i perturbation, delta=L/n^3"
        Q, R, qlab, rlab = Q[pq], R[pr], right[pq], B[pr]
        meta.update(analytic_affine_rank=rank, jitter=jitter, private_A=[.5, 2.], private_B=[.3, 0.],
                    scale_convention="public centers +/- L e1, L=n/2; A=(L/2)e1+2L e2, B=.3L e1",
                    group_label_public="True = +L e1 (right)", group_label_private="True = B",
                    wrong_cluster_definition="fraction of tasks whose matched driver violates A-left/B-right")
    elif family == "permuted_unique_simplex":
        h = case["simplex_height"]
        H = h*np.eye(n, d)/np.sqrt(2)
        Q, R, qlab, rlab = H[pq], H[pr], pq, pr
        partner = np.argsort(pr)[pq]           # R row matched to Q row i in the unique optimum
        meta.update(analytic_affine_rank=n-1, true_partner=partner.tolist(),
                    scale_convention="q_i=r_i=h e_i/sqrt(2), h=n/4 (pairwise distance h)",
                    wrong_cluster_definition="fraction of tasks not matched to their unique true partner")
    elif family == "permuted_interchangeable_types":
        g, tau = case["type_count"], case["jitter_tau"]
        if n % g:
            raise ValueError("type count must divide n")
        types = np.arange(n) % g
        centers = (n/4)/np.sqrt(2)*np.eye(g, d)
        sd = tau*n/(4*np.sqrt(d))
        Q = centers[types] + sd*rng.normal(size=(n, d))
        R = centers[types] + sd*rng.normal(size=(n, d))
        Q, R, qlab, rlab = Q[pq], R[pr], types[pq], types[pr]
        meta.update(analytic_affine_rank=min(n-1, d) if tau > 0 else g-1, jitter_sd=sd,
                    copies_per_type=n//g,
                    scale_convention="type centers (n/4)/sqrt(2) e_t, t<g; jitter sd tau*n/(4 sqrt d)",
                    wrong_cluster_definition="fraction of tasks matched to a driver of a different type")
    else:
        raise ValueError(family)
    randomized = case["geometry_replicates"] > 1
    meta.update(geometry_randomized=randomized, label_permutation_randomized=True,
                replication_unit="independent geometry" if randomized else "single geometry; labels permuted only",
                intrinsic_rank=None)
    return Q, R, qlab, rlab


DIMENSION_FAMILIES = {"gaussian_full": None, "gaussian_rank4": 4}


def dimension_tradeoff_design(pilot: bool = False) -> list[dict]:
    """Opt-in n/d tradeoff (Sept 25 global numerics request).

    Q and R iid N(0, I_r/r) in r active coordinates (r=d full, r=4 rank-4), zero padded to d.
    No separate centering/normalization. Rank-4 active arrays are shared across d at fixed
    (n, replicate); mechanism noise is not paired across d. Noise coupled across epsilon.
    Only endpoint and public-selected PCA/representative candidates are simulated.
    """
    ns, ds, G = ([16, 64], [8, 128], 2) if pilot else ([16, 32, 64, 128], [8, 32, 128, 512], 20)
    return [dict(study_id="dimension_tradeoff", geometry_family=family, n=n, d=d, epsilons=[16., 64.],
                 geometry_replicates=G, active_rank=r, families=["pca", "representatives"], ranks=[0], counts=[1],
                 scaled_changing_instance=False, diagnostics=False, coupled_epsilon=True)
            for family, r in DIMENSION_FAMILIES.items() for n in ns for d in ds]


def _dimension_geometry(case, spec, meta):
    n, d, r = case["n"], case["d"], case["active_rank"] or case["d"]
    if r > d:
        raise ValueError("active rank exceeds ambient dimension")
    # Rank-4: the master seed excludes d, so the same active arrays are embedded for every d.
    master = {k: v for k, v in spec.items() if k != "d"} if case["active_rank"] else spec
    master_id = stable_id(dict(master, role="dimension_tradeoff_master"))
    rng = np.random.default_rng(int(master_id[:16], 16))
    Qa, Ra = rng.normal(size=(n, r))/np.sqrt(r), rng.normal(size=(n, r))/np.sqrt(r)
    Q, R = np.zeros((n, d)), np.zeros((n, d))
    Q[:, :r], R[:, :r] = Qa, Ra
    meta.update(active_rank=r, master_id=master_id, master_seed=int(master_id[:16], 16),
                embedding="active coordinates 0..r-1, zero padded to d" if r < d else "full ambient",
                master_shared_across_d=bool(case["active_rank"]), noise_paired_across_d=False,
                analytic_affine_rank=min(n-1, r), intrinsic_rank=r if r < d else None,
                expected_folded_dimension=min(n, r+1, d),
                scale_convention="Q,R iid N(0,I_r/r) in active coords; population E||x||^2=1; no centering or rescaling",
                public_scale=1., private_rms_norm=float(np.sqrt(np.mean(np.sum(R**2, axis=1)))),
                public_rms_norm=float(np.sqrt(np.mean(np.sum(Q**2, axis=1)))), geometry_randomized=True)
    return Q, R


def instance_units(cases: list[dict]) -> list[tuple[int, int]]:
    """Deterministic (case index, replicate) list; shards own whole instances."""
    return [(i, r) for i, case in enumerate(cases) for r in range(case["geometry_replicates"])]


def geometry(case: dict, replicate: int, epsilon: float, root_seed: int):
    spec = {k: v for k, v in case.items() if k not in {
        "epsilons", "geometry_replicates", "families", "ranks", "counts", "diagnostics",
        "secondary_pca", "cross_comparison", "coupled_epsilon"}}
    if case["scaled_changing_instance"]:
        spec["geometry_epsilon"] = epsilon
    spec.update(replicate=replicate, root_seed=root_seed)
    iid = stable_id(spec)
    seed = int(iid[:16], 16)
    rng = np.random.default_rng(seed)
    n, d = case["n"], case["d"]
    family = case["geometry_family"]
    meta = dict(spec, instance_id=iid, instance_seed=seed, geometry_replicate=replicate,
                geometry_randomized=family != "two_population",
                configuration_label=stable_id({k: v for k, v in spec.items() if k != "replicate"}),
                fixed_instance_sweep=not case["scaled_changing_instance"])
    clusters = populations = None
    if family == "frozen_cube":
        from pathlib import Path
        from private_matching.config import load_bundle
        from private_matching.instances import load_instance
        cfg = load_bundle([Path(__file__).resolve().parents[2]/"configs/production.yaml"], profile="production").scientific
        s = case["cube_configuration"]
        inst = load_instance(cfg, s, n, d)
        Q, R = inst.Q, inst.R
        meta.update(instance_id=f"cube:seed{cfg.root_seed}:s{s}:n{n}:d{d}",
                    instance_seed=None, root_seed=cfg.root_seed, geometry_replicate=s,
                    configuration_label=f"cube_configuration_{s+1}",
                    analytic_affine_rank=min(n-1, d), scale_convention="frozen Uniform([0,1]^d) master slice")
    elif family in DIMENSION_FAMILIES:
        Q, R = _dimension_geometry(case, spec, meta)
    elif family in INTERCHANGEABILITY_FAMILIES:
        Q, R, clusters, populations = _interchangeability_geometry(case, n, d, rng, meta)
    elif family == "two_population":
        padded = bool(case.get("zero_padded"))
        if (d < n if padded else n != d) or n < 4 or n % 2:
            raise ValueError("Appendix construction requires even n=d >=4 (or d >= n when zero_padded)")
        L = n / (2 * epsilon) if case["scaled_changing_instance"] else case["L"]
        delta = L / n**3
        clusters = np.arange(n) >= n // 2
        # zero_padded: the n=d construction embedded in R^d via zero trailing coordinates.
        Q = np.zeros((n, d))
        Q[:, :n] = delta * np.eye(n)
        Q[:, 0] += np.where(clusters, L, -L)  # row 0 is mathematical i=1
        R = np.zeros((n, d))
        R[:n//2, 0], R[n//2:, 0] = .5 * L, .3 * L
        R[:n//2, 1] = case.get("private_A_height", 2.) * L
        R[n//2:, 1] = case.get("private_B_height", 0.) * L
        # Target grouping is defined from ideal-center costs, including changed geometry.
        contrast = np.hypot(R[:, 0] + L, R[:, 1]) - np.hypot(R[:, 0] - L, R[:, 1])
        populations = np.zeros(n, dtype=bool)
        populations[np.argsort(contrast, kind="stable")[n//2:]] = True
        meta.update(L=L, public_perturbation=delta, private_A_axis=.5, private_B_axis=.3,
                    private_A_height=case.get("private_A_height", 2.),
                    private_B_height=case.get("private_B_height", 0.),
                    intrinsic_rank=n-1, analytic_affine_rank=n-1,
                    scale_convention="cluster centers +/- L; delta=L/n^3",
                    wrong_cluster_definition="ideal_center_optimal_population_grouping")
    else:
        r = min(case.get("intrinsic_rank", 4), d-1, n-1)
        Q = rng.normal(size=(n, d))
        if family in {"low_rank", "near_low_rank"}:
            Q[:, r:] *= case.get("residual_thickness") or 0.
            # thickness is relative RMS in complement versus signal block.
            if family == "near_low_rank":
                Q[:, r:] *= np.sqrt(r / (d-r))
        elif family == "clustered":
            count = case["cluster_count"]
            centers = rng.normal(size=(count, d)) / np.sqrt(d)
            Q = centers[np.arange(n) % count] + case["cluster_spread"] * Q / np.sqrt(d)
        Q -= Q.mean(axis=0)
        scale = np.sqrt(np.mean(np.sum(Q**2, axis=1)))
        Q /= scale
        R = rng.normal(size=(n, d)) / np.sqrt(d)
        # Explicit heterogeneous perpendicular displacement even for exactly low-rank Q.
        R[:n//2, r:] *= .5
        R[n//2:, r:] *= 4.
        meta.update(scale_convention="public centered RMS radius exactly 1",
                    public_scale=1., pre_normalization_rms=float(scale),
                    private_near_multiplier=.5, private_far_multiplier=4.,
                    intrinsic_rank=r if family in {"low_rank", "near_low_rank"} else None,
                    analytic_affine_rank=r if family == "low_rank" else min(n-1, d))
    meta["public_rms_radius"] = float(np.sqrt(np.mean(np.sum((Q-Q.mean(axis=0))**2, axis=1))))
    meta.setdefault("public_scale", meta["public_rms_radius"])
    return Q, R, meta, clusters, populations
