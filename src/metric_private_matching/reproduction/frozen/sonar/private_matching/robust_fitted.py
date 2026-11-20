"""Opt-in `robust_fitted` study: a restricted local matching family fitted on public data only.

Channel family: p(a|x) proportional to exp(-beta ||h(x)-h(q_a)||_2), beta = eps*tau/2, tau <= 1,
with h a 1-Lipschitz public map (identity or orthogonal projection onto the top-2 public PCA
directions). Score and log normalizer are beta-Lipschitz, so the channel is eps-metric private on
all of R^d; no mesh, clipping or rounding is involved. The server runs Hungarian on D[:, a_j].

The (channel, decoder) pair is SELECTED from a prespecified finite family by the worst mean true
unsquared matching regret over PUBLIC synthetic configurations. This is empirical finite-family
selection: not the free finite-element mechanism of the research note, not a jointly optimized
local protocol, not a minimax optimum and not a continuum regret certificate.
Fits take only (Q, epsilon, public key, phase); evaluation R is generated after all fits are saved.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy
from scipy.special import logsumexp

from private_matching.assignment import (pairwise_euclidean, exact_assignment, assignment_cost,
                                         optimal_cost, random_permutation_baseline)
from private_matching.followup import source_digest
from private_matching.login_guard import refuse_login_compute
from private_matching.proxies import PublicProxyCatalog
from private_matching.results import atomic_write_json, read_json
from private_matching.study_geometry import stable_id

VERSION = "robust-fitted-v1"
STUDY_ID = "robust_fitted"
ROOT_SEED = 20260926
RESULTS_SUBDIR = "robust_fitted_20260926"
FAMILIES = ("unstructured", "rank2", "clustered")
REGIMES = ("near", "crowded", "shifted")
FEATURES = ("identity", "pca2")
TAUS = (1.0, 0.5)
DECODERS = ("direct", "posterior_nearQ", "posterior_broad", "posterior_mixture")
TRAIN_KINDS = ("near", "crowded", "broad", "shifted")
BASELINES = ("folded", "ambient", "phi_infinity", "anchor")
SAMPLE_KEYS = ("excess", "output_cost", "oscillation_original", "oscillation_proxy", "realization_bound",
               "proxy_regret", "wrong_cluster_fraction", "clipped_radius_fraction", "mean_noise_norm",
               "max_reconstruction_error", "agreement_with_proxy_optimum", "near_minus_far_task_radius")
# Every numeric constant is part of the fingerprinted design. RMS values are in units of the
# public RMS radius (1 after normalization); a row perturbation with RMS s has E||noise||^2 = s^2.
PARAMETERS = dict(
    public_cluster_count=4, public_cluster_rms=.05,
    eval_near_rms=.2, eval_crowded_rms=.1, eval_shift_radii=3., eval_shift_rms=.7,
    train_near_rms=.3, train_crowded_rms=.1, train_broad_rms=1.5, train_shift=2.5, train_shift_rms=.6,
    prior_near_per_point=40, prior_near_rms=.25, prior_broad_count=2000, prior_broad_rms=2.,
    prior_shift_directions=8, prior_shift_per_direction=250, prior_shift=2.5, prior_shift_rms=.6,
)


def seed_of(*parts) -> int:
    return int(stable_id([VERSION, *parts])[:16], 16)


def gaussian_rms(rng, shape, rms):
    """Isotropic Gaussian rows with E||row||^2 = rms^2."""
    return rng.normal(size=shape)*(rms/np.sqrt(shape[-1]))


def unit(v):
    return v/np.linalg.norm(v)


def rms_radius(Q):
    return float(np.sqrt(np.mean(np.sum((Q-Q.mean(axis=0))**2, axis=1))))


def public_Q(family, n, d=16, root_seed=ROOT_SEED, P=PARAMETERS):
    """One fixed publicly seeded configuration per (family, n, d): centered, RMS radius 1."""
    rng = np.random.default_rng(seed_of("public_Q", family, n, d, root_seed))
    if family == "unstructured":
        Q = rng.normal(size=(n, d))
    elif family == "rank2":
        Q = np.zeros((n, d))
        Q[:, :2] = rng.normal(size=(n, 2))
    elif family == "clustered":
        count = P["public_cluster_count"]
        centers = rng.normal(size=(count, d))/np.sqrt(d)
        Q = centers[np.arange(n) % count]+gaussian_rms(rng, (n, d), P["public_cluster_rms"])
    else:
        raise ValueError(family)
    Q = Q-Q.mean(axis=0)
    return Q/rms_radius(Q)


def pca_basis(Q, rank=2):
    _, _, vt = np.linalg.svd(Q-Q.mean(axis=0), full_matrices=False)
    return np.ascontiguousarray(vt[:rank].T)


@dataclass(frozen=True)
class Channel:
    """Softmax over negative public feature distances to fixed prototypes (labels)."""
    feature: str
    tau: float
    epsilon: float
    prototypes: np.ndarray
    center: np.ndarray
    basis: np.ndarray | None

    @property
    def beta(self):
        return self.epsilon*self.tau/2

    def features(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X if self.feature == "identity" else (X-self.center) @ self.basis

    def log_probabilities(self, X):
        logits = -self.beta*pairwise_euclidean(self.features(X), self.features(self.prototypes))
        return logits-logsumexp(logits, axis=1, keepdims=True)

    def sample(self, X, uniforms):
        """Inverse-CDF labels; uniforms has shape (..., len(X))."""
        cdf = np.cumsum(np.exp(self.log_probabilities(X)), axis=1)
        labels = (np.asarray(uniforms)[..., None] > cdf).sum(axis=-1)
        return np.minimum(labels, len(self.prototypes)-1)


def make_channel(Q, feature, tau, epsilon):
    if not 0 < tau <= 1:
        raise ValueError("tau must lie in (0, 1] for eps-metric privacy")
    basis = pca_basis(Q, 2) if feature == "pca2" else None
    if feature not in FEATURES:
        raise ValueError(feature)
    return Channel(feature, float(tau), float(epsilon), np.array(Q, dtype=np.float64), Q.mean(axis=0), basis)


def quadrature(kind, Q, rng, P=PARAMETERS):
    """Public prior sample points and log weights (sum of weights = 1). Uses only Q and a public seed."""
    n, d = Q.shape
    if kind == "nearQ":
        per = P["prior_near_per_point"]
        X = np.repeat(Q, per, axis=0)+gaussian_rms(rng, (n*per, d), P["prior_near_rms"])
    elif kind == "broad":
        X = Q.mean(axis=0)+gaussian_rms(rng, (P["prior_broad_count"], d), P["prior_broad_rms"])
    elif kind == "shifted":
        per = P["prior_shift_per_direction"]
        directions = np.array([unit(v) for v in rng.normal(size=(P["prior_shift_directions"], d))])
        X = np.repeat(Q.mean(axis=0)+P["prior_shift"]*directions, per, axis=0)
        X = X+gaussian_rms(rng, X.shape, P["prior_shift_rms"])
    elif kind == "mixture":
        parts = [quadrature(k, Q, rng, P) for k in ("nearQ", "broad", "shifted")]
        X = np.concatenate([p[0] for p in parts])
        return X, np.concatenate([p[1]+np.log(1/3) for p in parts])
    else:
        raise ValueError(kind)
    return X, np.full(len(X), -np.log(len(X)))


def direct_decoder(Q, channel):
    return pairwise_euclidean(Q, channel.prototypes)


def posterior_decoder(Q, channel, X, log_w):
    """D[i,a] = sum_b w_b p(a|x_b)||q_i-x_b|| / sum_b w_b p(a|x_b), normalized in log space."""
    joint = log_w[:, None]+channel.log_probabilities(X)
    W = np.exp(joint-logsumexp(joint, axis=0, keepdims=True))
    return pairwise_euclidean(Q, X) @ W


def candidate_specs():
    return [dict(index=i, feature=f, tau=t, decoder=dec, candidate_id=f"{f}:tau{t:g}:{dec}")
            for i, (f, t, dec) in enumerate((f, t, dec) for f in FEATURES for t in TAUS for dec in DECODERS)]


def build_candidates(Q, epsilon, quadrature_seed, P=PARAMETERS):
    """All 16 (channel, decoder) pairs; decoders use public quadrature only."""
    rng = np.random.default_rng(quadrature_seed)
    priors = {kind: quadrature(kind, Q, rng, P) for kind in ("nearQ", "broad", "mixture")}
    channels, decoders = {}, []
    for spec in candidate_specs():
        key = (spec["feature"], spec["tau"])
        channel = channels.setdefault(key, make_channel(Q, spec["feature"], spec["tau"], epsilon))
        if spec["decoder"] == "direct":
            D = direct_decoder(Q, channel)
        else:
            D = posterior_decoder(Q, channel, *priors[spec["decoder"].removeprefix("posterior_")])
        decoders.append(D)
    return channels, np.array(decoders)


def synthetic_configuration(kind, Q, rng, P=PARAMETERS):
    """Public hypothetical private-side configuration of n clients (training/validation only)."""
    n, d = Q.shape
    if kind == "near":
        return Q[rng.permutation(n)]+gaussian_rms(rng, (n, d), P["train_near_rms"])
    if kind == "crowded":
        return Q[rng.integers(n, size=n)]+gaussian_rms(rng, (n, d), P["train_crowded_rms"])
    if kind == "broad":
        return Q.mean(axis=0)+gaussian_rms(rng, (n, d), P["train_broad_rms"])
    if kind == "shifted":
        return Q.mean(axis=0)+P["train_shift"]*unit(rng.normal(size=d))+gaussian_rms(rng, (n, d), P["train_shift_rms"])
    raise ValueError(kind)


def matching_regrets(C, optimum, D, labels):
    """True-cost regret of Hungarian on D[:, labels[k]] for each draw k."""
    out = np.empty(len(labels))
    for k, a in enumerate(labels):
        out[k] = assignment_cost(C, exact_assignment(D[:, a]))-optimum
    return out


def score_candidates(Q, channels, decoders, configurations, uniforms):
    """scores[c, s] = mean true matching regret over common label uniforms (fit only)."""
    specs = candidate_specs()
    scores = np.empty((len(specs), len(configurations)))
    for s, R in enumerate(configurations):
        C = pairwise_euclidean(Q, R)
        optimum = optimal_cost(C)[1]
        labels = {key: ch.sample(R, uniforms[s]) for key, ch in channels.items()}
        for spec in specs:
            scores[spec["index"], s] = matching_regrets(C, optimum, decoders[spec["index"]],
                                                        labels[(spec["feature"], spec["tau"])]).mean()
    return scores


def fit(Q, epsilon, fit_key, phase, K_train, K_validation, per_kind=3, root_seed=ROOT_SEED, P=PARAMETERS):
    """Select (channel, decoder) from public data only. There is deliberately no R argument."""
    started = time.monotonic()
    n = len(Q)
    seeds = {role: seed_of("fit", fit_key, phase, root_seed, role) for role in
             ["decoder_quadrature", "train_configurations", "train_uniforms",
              "validation_configurations", "validation_uniforms"]}
    channels, decoders = build_candidates(Q, epsilon, seeds["decoder_quadrature"], P)
    result = dict(fit_key=fit_key, epsilon=float(epsilon), phase=phase, seeds=seeds, K_train=K_train,
                  K_validation=K_validation, configurations_per_kind=per_kind, train_kinds=list(TRAIN_KINDS),
                  candidates=candidate_specs(), selection_rule="argmin over candidates of max over training "
                  "configurations of mean true unsquared matching regret; first candidate on ties",
                  uniforms="common across candidates within fit only; inverse-CDF label sampling")
    arrays = dict(Q=Q, decoders=decoders)
    for split, K in [("train", K_train), ("validation", K_validation)]:
        rng = np.random.default_rng(seeds[f"{split}_configurations"])
        configs = np.array([synthetic_configuration(kind, Q, rng, P) for kind in TRAIN_KINDS for _ in range(per_kind)])
        uniforms = np.random.default_rng(seeds[f"{split}_uniforms"]).random((len(configs), K, n))
        scores = score_candidates(Q, channels, decoders, configs, uniforms)
        arrays[f"{split}_configurations"], arrays[f"{split}_scores"] = configs, scores
        result[f"{split}_scores"] = scores.tolist()
        result[f"{split}_worst"] = scores.max(axis=1).tolist()
    worst = np.asarray(result["train_worst"])
    selected = int(np.argmin(worst))
    validation_worst = np.asarray(result["validation_worst"])
    result.update(selected_index=selected, selected=candidate_specs()[selected],
                  selected_train_worst=float(worst[selected]),
                  selected_validation_worst=float(validation_worst[selected]),
                  selected_validation_rank=int(np.sum(validation_worst < validation_worst[selected])),
                  pca2_basis=pca_basis(Q, 2).tolist(), fit_seconds=time.monotonic()-started,
                  restriction="restricted analytic softmax channel family and compact Hungarian decoder; "
                  "empirical finite-family selection, not a minimax optimum or continuum certificate")
    return result, arrays


def load_or_fit(directory, fingerprint, family, n, epsilon, manifest):
    fit_id = f"{family}_n{n}_eps{epsilon:g}"
    path = directory/"fits"/f"{fit_id}.json"
    Q = public_Q(family, n, manifest["d"], manifest["root_seed"], manifest["parameters"])
    if path.exists():
        record = read_json(path)
        if record.get("fingerprint") != fingerprint:
            raise ValueError(f"fit from different fingerprint: {path}")
        decoders = np.load(path.with_suffix(".npz"))["decoders"]
        return record, decoders
    record, arrays = fit(Q, epsilon, f"{family}:n{n}:d{manifest['d']}", manifest["phase"], manifest["K_train"],
                         manifest["K_validation"], manifest["train_per_kind"], manifest["root_seed"],
                         manifest["parameters"])
    record.update(fingerprint=fingerprint, fit_id=fit_id, family=family, n=n, d=manifest["d"],
                  fit_path=str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path.with_suffix(".npz"), **arrays)
    atomic_write_json(path, record)
    print(f"fit {fit_id}: selected {record['selected']['candidate_id']} train worst "
          f"{record['selected_train_worst']:.4f} validation worst {record['selected_validation_worst']:.4f} "
          f"({record['fit_seconds']:.1f}s)", flush=True)
    return record, arrays["decoders"]


def farthest_pair(Q):
    distances = pairwise_euclidean(Q, Q)
    i, j = np.unravel_index(int(np.argmax(distances)), distances.shape)
    return int(min(i, j)), int(max(i, j))


def shift_direction(family, n, d, root_seed):
    """Fixed public evaluation direction, seeded independently of every fit seed and phase."""
    return unit(np.random.default_rng(seed_of("evaluation_shift_direction", family, n, d, root_seed)).normal(size=d))


def evaluation_R(regime, Q, family, replicate, phase, root_seed=ROOT_SEED, P=PARAMETERS):
    """Held-out private configuration; never passed to fitting."""
    n, d = Q.shape
    rng = np.random.default_rng(seed_of("evaluation_R", family, n, d, regime, replicate, phase, root_seed))
    if regime == "near":
        return Q[rng.permutation(n)]+gaussian_rms(rng, (n, d), P["eval_near_rms"])
    if regime == "crowded":
        pair = np.array(farthest_pair(Q))
        return Q[pair[(rng.random(n) < .5).astype(int)]]+gaussian_rms(rng, (n, d), P["eval_crowded_rms"])
    if regime == "shifted":
        center = Q.mean(axis=0)+P["eval_shift_radii"]*rms_radius(Q)*shift_direction(family, n, d, root_seed)
        return center+gaussian_rms(rng, (n, d), P["eval_shift_rms"])
    raise ValueError(regime)


def design(pilot=False, **overrides):
    value = dict(version=VERSION, study_id=STUDY_ID, root_seed=ROOT_SEED, d=16, ns=[25, 50],
                 families=list(FAMILIES), epsilons=[4., 16., 64.], regimes=list(REGIMES),
                 replicates=2 if pilot else 20, K=5 if pilot else 50, K_train=10 if pilot else 30,
                 K_validation=10 if pilot else 30, train_per_kind=3, phase="pilot" if pilot else "production",
                 parameters=PARAMETERS, baselines=list(BASELINES), candidates=candidate_specs())
    value.update(overrides)
    return value


def identity(pilot=False, **overrides):
    value = dict(design(pilot, **overrides), source_digest=source_digest(), numpy=np.__version__,
                 scipy=scipy.__version__)
    return value, stable_id(value)


def fitted_family_rows(Q, R, C, optimum, opt_assignment, decoders, fit_record, eps, K, iid, meta):
    """Selected arm plus two ablations; they share label uniforms (recorded coupling group)."""
    n = len(Q)
    specs = candidate_specs()
    selected = fit_record["selected_index"]
    arms = [("robust_fitted", selected, "selected"),
            ("robust_fitted:direct_decoder", selected-selected % len(DECODERS), "ablation"),
            ("robust_fitted:default_unfitted", 0, "ablation")]
    coupling = stable_id([iid, eps, "fitted_family_labels"])
    uniforms = np.random.default_rng(int(coupling[:16], 16)).random((K, n))
    cache, rows = {}, []
    for arm, index, role in arms:
        spec = specs[index]
        reused = index in cache
        if not reused:
            start = time.monotonic()
            channel = make_channel(Q, spec["feature"], spec["tau"], eps)
            labels = channel.sample(R, uniforms)
            D = decoders[index]
            samples = {key: [] for key in SAMPLE_KEYS}
            for a in labels:
                C_hat = D[:, a]
                assignment = exact_assignment(C_hat)
                cost = assignment_cost(C, assignment)
                regret = cost-optimum
                osc = float(np.ptp(C_hat-C, axis=0).sum())
                tolerance = 1e-8*max(1., abs(optimum), abs(cost), osc)
                if regret < -tolerance or not np.isfinite(regret) or regret > osc+tolerance:
                    raise FloatingPointError("fitted arm regret outside [0, oscillation]")
                values = dict(excess=max(0., regret), output_cost=cost, oscillation_original=osc,
                              max_reconstruction_error=float(np.max(np.abs(C_hat-C))),
                              agreement_with_proxy_optimum=float(np.mean(assignment == opt_assignment)))
                for key in SAMPLE_KEYS:
                    samples[key].append(values.get(key))
            cache[index] = (arm, samples, time.monotonic()-start)
        source, samples, seconds = cache[index]
        row = dict(meta, arm_id=arm, arm_label=arm, base_mechanism="robust_fitted_softmax",
                   requested_base_mechanism="robust_fitted_softmax", decoder=f"hungarian_compact_{spec['decoder']}",
                   approximation_family="exact", privacy_model="local", epsilon=eps, epsilon_y=0., epsilon_r=0.,
                   epsilon_other=eps, channel_privacy_rate=eps*spec["tau"], channel_feature_map=spec["feature"],
                   channel_tau=spec["tau"], channel_beta=eps*spec["tau"]/2, decoder_family=spec["decoder"],
                   candidate_id=spec["candidate_id"], candidate_index=index, label_alphabet_size=n,
                   coordinate_noise_dimension=None, coordinate_noise_norm=None, expected_noise_norm=None,
                   scalar_noise_count=0, scalar_noise_scale=None, report_dimension=1, no_release=False,
                   noise_stream="robust_fitted_label_uniforms", coupling_group=coupling,
                   coupling_method="same per-client label uniforms (inverse CDF) across fitted-family arms",
                   D_proxy=0., D_proxy_per_point=0., mechanism_certificate_uncapped=None,
                   universal_cap=n*meta["public_diameter"], mechanism_certificate_capped=None,
                   selected_branch=fit_record["selected"]["candidate_id"] if role == "selected" else None,
                   selection_origin="public_synthetic_regret_fit" if role == "selected" else None,
                   candidate_set="16 prespecified (feature, tau, decoder) candidates" if role == "selected" else None,
                   operating_point=role, fit_id=fit_record["fit_id"], fit_path=fit_record["fit_path"],
                   fit_train_worst_mean_regret=fit_record["train_worst"][index],
                   fit_validation_worst_mean_regret=fit_record["validation_worst"][index],
                   fit_selected_candidate_id=fit_record["selected"]["candidate_id"],
                   fit_selection_objective=f"max over {len(TRAIN_KINDS)*fit_record['configurations_per_kind']} public "
                   "synthetic configurations of mean true matching regret (empirical, not minimax)",
                   K_train=fit_record["K_train"], mechanism_restriction="restricted analytic softmax channel + compact "
                   "Hungarian decoder; not the free finite-element mechanism",
                   source_arm_id=source if reused else None,
                   provenance="derived_reused" if reused else "simulated",
                   reuse_reason="same candidate index as an earlier fitted-family arm" if reused else None,
                   status="complete", pointwise_checked=True, pointwise_ok=True, runtime_seconds=seconds,
                   failure_reason=None, radius_budget_fraction=None, ablation=None if role == "selected" else arm)
        rows.append(dict(metadata=row, samples=samples))
    return rows


def baseline_rows(Q, R, C, optimum, fixed_cost, catalog, eps, K, iid, meta):
    """Existing local baselines through study.evaluate; ambient reuses full Psi when m=d, as run_case does."""
    from private_matching.study import evaluate
    geom, d = catalog.exact, Q.shape[1]
    rows, cache = [], {}
    for base in BASELINES:
        key = "folded" if base == "ambient" and geom.fold.m == d and catalog.rank == geom.fold.k else base
        reused = key in cache
        if not reused:
            cache[key] = (base, evaluate(Q, R, C, optimum, fixed_cost, catalog, key, geom, eps, K, iid))
        source, result = cache[key]
        row = dict(meta, **result["metadata"])
        row.update(arm_id=base, arm_label=base, requested_base_mechanism=base, approximation_family="exact",
                   source_arm_id=source if reused else None,
                   provenance="derived_reused" if reused else result["metadata"]["provenance"],
                   operating_point="baseline",
                   reuse_reason="canonical endpoint/isotropic equivalence (m=d)" if reused else None)
        row["D_proxy_per_point"] = row["D_proxy"]/len(Q)
        rows.append(dict(metadata=row, samples=result["samples"]))
    return rows


def unit_path(directory, family, n, regime, replicate, eps):
    return directory/"units"/f"{family}_n{n}_{regime}_r{replicate:02d}_eps{eps:g}.json"


def validate_unit(record, fingerprint, K, path):
    if record.get("fingerprint") != fingerprint or record.get("status") != "complete":
        raise ValueError(f"incompatible checkpoint {path}")
    if len(record["rows"]) != 3+len(BASELINES):
        raise ValueError(f"unexpected arm count in {path}")
    for row in record["rows"]:
        x = np.asarray(row["samples"]["excess"], dtype=float)
        if len(x) != K or not np.all(np.isfinite(x)) or np.any(x < 0) or not row["metadata"]["pointwise_ok"] \
                or any(len(v) != K for v in row["samples"].values()):
            raise ValueError(f"invalid checkpoint samples {path}")


def run_unit(manifest, directory, family, n, regime, replicate, eps, Q, catalog, fit_record, decoders):
    fp, K, phase = manifest["fingerprint"], manifest["K"], manifest["phase"]
    path = unit_path(directory, family, n, regime, replicate, eps)
    if path.exists():
        record = read_json(path)
        validate_unit(record, fp, K, path)
        return record
    started = time.monotonic()
    R = evaluation_R(regime, Q, family, replicate, phase, manifest["root_seed"], manifest["parameters"])
    spec = dict(version=VERSION, family=family, n=n, d=manifest["d"], regime=regime, replicate=replicate,
                phase=phase, root_seed=manifest["root_seed"])
    iid = stable_id(spec)
    C = pairwise_euclidean(Q, R)
    opt_assignment, optimum = optimal_cost(C)
    fixed_cost = assignment_cost(C, np.arange(n))
    random_cost = random_permutation_baseline(C)
    fold = catalog.exact.fold
    meta = dict(experiment_version=VERSION, fingerprint=fp, run_phase=phase, study_id=STUDY_ID,
                geometry_family=family, private_regime=regime, n=n, d=manifest["d"], instance_id=iid,
                instance_seed=seed_of("evaluation_R", family, n, manifest["d"], regime, replicate, phase,
                                      manifest["root_seed"]),
                root_seed=manifest["root_seed"], geometry_replicate=replicate, replicate=replicate,
                geometry_randomized=True, public_Q_fixed=True,
                replication_unit="independent private R draw; public Q fixed per (family, n)",
                configuration_label=f"{family}:n{n}:{regime}", fixed_instance_sweep=True,
                scaled_changing_instance=False, public_affine_rank=fold.k, numerical_rank_tolerance=fold.rank_tol,
                pca_rank_tolerance=catalog.tol, analytic_affine_rank=2 if family == "rank2" else min(n-1, manifest["d"]),
                intrinsic_rank=2 if family == "rank2" else None,
                cluster_count=manifest["parameters"]["public_cluster_count"] if family == "clustered" else None,
                cluster_spread=manifest["parameters"]["public_cluster_rms"] if family == "clustered" else None,
                exact_fold_residual=catalog.exact.D_proxy,
                rank_status="approximate_rank_truncation" if fold.k < (2 if family == "rank2" else min(n-1, manifest["d"]))
                else "expected_rank_retained", preprocessing="numerical_projection_residual_recorded",
                scale_convention="public Q centered, RMS radius 1; regime RMS values in these units",
                public_scale=1., public_rms_radius=rms_radius(Q), public_diameter=catalog.diameter, OPT=optimum,
                fixed_matching_cost=fixed_cost, expected_random_cost=random_cost, random_minus_OPT=random_cost-optimum,
                K_requested=K, matching_tie_rule="scipy_linear_sum_assignment_fixed_input_order",
                numpy_version=np.__version__, scipy_version=scipy.__version__)
    rows = fitted_family_rows(Q, R, C, optimum, opt_assignment, decoders, fit_record, eps, K, iid, meta)
    rows += baseline_rows(Q, R, C, optimum, fixed_cost, catalog, eps, K, iid, meta)
    record = dict(fingerprint=fp, status="complete", rows=rows, public_Q=Q.tolist(), private_R=R.tolist(),
                  fit_id=fit_record["fit_id"], elapsed_seconds=time.monotonic()-started)
    validate_unit(record, fp, K, path)
    atomic_write_json(path, record)
    return record


def results_directory(root, fingerprint, phase):
    return Path(root)/RESULTS_SUBDIR/phase/fingerprint


def run_robust_fitted(root, pilot=False, **overrides):
    refuse_login_compute("run the robust_fitted study")
    ident, fp = identity(pilot, **overrides)
    manifest = dict(ident, fingerprint=fp)
    directory = results_directory(root, fp, manifest["phase"])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/"manifest.json").exists():
        if read_json(directory/"manifest.json").get("fingerprint") != fp:
            raise ValueError(f"manifest from different fingerprint in {directory}")
    else:
        atomic_write_json(directory/"manifest.json", manifest)
    print(f"robust_fitted {manifest['phase']} -> {directory} (resolved {directory.resolve()})", flush=True)
    timings = dict(fits=[], units=[])
    # Stage 1: every fit is completed and saved before any evaluation R is generated.
    fits = {}
    for family in manifest["families"]:
        for n in manifest["ns"]:
            for eps in manifest["epsilons"]:
                started = time.monotonic()
                fits[family, n, eps] = load_or_fit(directory, fp, family, n, eps, manifest)
                timings["fits"].append(dict(family=family, n=n, epsilon=eps, seconds=time.monotonic()-started))
    # Stage 2: held-out evaluation, loading fits read-only.
    for family in manifest["families"]:
        for n in manifest["ns"]:
            Q = public_Q(family, n, manifest["d"], manifest["root_seed"], manifest["parameters"])
            catalog = PublicProxyCatalog(Q)
            for regime in manifest["regimes"]:
                block = time.monotonic()
                for replicate in range(manifest["replicates"]):
                    for eps in manifest["epsilons"]:
                        started = time.monotonic()
                        run_unit(manifest, directory, family, n, regime, replicate, eps, Q, catalog, *fits[family, n, eps])
                        timings["units"].append(dict(family=family, n=n, regime=regime, replicate=replicate,
                                                     epsilon=eps, seconds=time.monotonic()-started))
                print(f"{family} n={n} {regime}: {manifest['replicates']} replicates x "
                      f"{len(manifest['epsilons'])} eps in {time.monotonic()-block:.1f}s", flush=True)
    timings.update(fit_seconds=sum(t["seconds"] for t in timings["fits"]),
                   unit_seconds=sum(t["seconds"] for t in timings["units"]))
    atomic_write_json(directory/"timings.json", timings)
    finalize_robust_fitted(directory)
    return directory


def finalize_robust_fitted(directory):
    """Validate all checkpoints (no simulation), export the canonical table, summaries and figures."""
    refuse_login_compute("finalize the robust_fitted study")
    directory = Path(directory)
    manifest = read_json(directory/"manifest.json")
    count = 0
    for family in manifest["families"]:
        for n in manifest["ns"]:
            for regime in manifest["regimes"]:
                for replicate in range(manifest["replicates"]):
                    for eps in manifest["epsilons"]:
                        path = unit_path(directory, family, n, regime, replicate, eps)
                        validate_unit(read_json(path), manifest["fingerprint"], manifest["K"], path)
                        count += 1
    print(f"Validated {count} robust_fitted unit checkpoints in {directory}", flush=True)
    from private_matching.analysis_table import export_table
    from private_matching.robust_fitted_plots import make_report
    output = directory/"handoff"
    export_table([directory], output)
    make_report(directory, output)
    return output
