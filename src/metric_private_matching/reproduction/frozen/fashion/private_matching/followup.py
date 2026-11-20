"""Extended uniform-cube study, isolated from the frozen three-arm result schema."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import multiprocessing
import os
import platform
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import yaml

from private_matching.anchor import anchor_client, anchor_server, centered_client, optimized_split
from private_matching.assignment import assignment_cost, exact_assignment, pairwise_euclidean, optimal_cost, random_permutation_baseline
from private_matching.config import ConfigError, ScientificConfig, epsilon_key, load_bundle, parse_grid
from private_matching.instances import generate_master_pair, slice_instance
from private_matching.login_guard import refuse_login_compute
from private_matching.mechanisms import ambient_report, folded_report, phi_infinity_report, server_scores
from private_matching.proxies import PublicProxyCatalog, ProxyGeometry, endpoint_grid
from private_matching.results import atomic_write_json, read_json
from private_matching.seeds import float64_to_uint32_pair, make_generator, noise_generator

VERSION = "anchor-uniform-cube-v1"
MAIN_ARMS = ("ambient", "phi_infinity", "folded", "anchor", "centered", "fixed", "auto",
             "anchor_pca_best", "folded_pca_best", "anchor_representatives_best", "folded_representatives_best")


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


@dataclass(frozen=True)
class FollowupConfig:
    base: ScientificConfig
    name: str
    epsilon_values: tuple[float, ...]
    comparison_settings: tuple[tuple[int, int], ...]
    approximation_epsilons: tuple[float, ...]
    approximation_step: int = 5
    exponential_max_n: int = 8

    def epsilons_for(self, n: int, d: int) -> tuple[float, ...]:
        extra = self.epsilon_values + self.approximation_epsilons if (n, d) in self.comparison_settings else ()
        return tuple(sorted(set(self.base.epsilons + extra)))

    def settings(self) -> list[tuple[int, int]]:
        return sorted(set(self.base.settings()) | set(self.comparison_settings))

    def identity(self) -> dict[str, Any]:
        return {"version": VERSION, "base": self.base.__dict__, "name": self.name,
                "epsilon_values": self.epsilon_values, "comparison_settings": self.comparison_settings,
                "approximation_epsilons": self.approximation_epsilons,
                "approximation_step": self.approximation_step, "exponential_max_n": self.exponential_max_n,
                "main_arms": MAIN_ARMS, "selection": "public_certificate_only",
                "followup_noise_seed_scheme": "PCG64_SeedSequence_v2_SHA256_arm_128bit",
                "anchor_server_rule": "private_anchor_distance_inner_product",
                "representatives": "nested_farthest_first_lowest_index",
                "centered_sampler": "uniform_projected_lower_facets_cone",
                "tie_rule": "first_anchor_index_and_fixed_scipy_solver_input_order"}


def load_followup(path: str | Path) -> FollowupConfig:
    path = Path(path).resolve()
    data = yaml.safe_load(path.read_text())
    allowed = {"name", "base_config", "epsilon_values", "comparison_settings", "approximation_epsilons",
               "approximation_step", "exponential_max_n"}
    if not isinstance(data, dict) or set(data) - allowed:
        raise ConfigError("invalid follow-up mapping or unknown keys")
    base = load_bundle([path.parent / data["base_config"]], profile="production").scientific
    raw_eps = data["epsilon_values"]
    eps = tuple(float(x) for x in (parse_grid(raw_eps, name="epsilon_values") if isinstance(raw_eps, dict) else raw_eps))
    approx_eps = tuple(float(x) for x in data.get("approximation_epsilons", base.epsilons))
    settings = tuple(tuple(int(x) for x in setting) for setting in data["comparison_settings"])
    cfg = FollowupConfig(base, str(data["name"]), eps, settings, approx_eps,
                         int(data.get("approximation_step", 5)), int(data.get("exponential_max_n", 8)))
    validate_followup(cfg)
    return cfg


def validate_followup(cfg: FollowupConfig) -> None:
    if not cfg.name or Path(cfg.name).name != cfg.name or cfg.name in {".", ".."}:
        raise ConfigError("name must be a single directory component")
    for values in (cfg.epsilon_values, cfg.approximation_epsilons):
        if not values or len(set(values)) != len(values) or any(not np.isfinite(e) or e <= 0 for e in values):
            raise ConfigError("epsilon grids must be nonempty, unique, positive and finite")
    if cfg.approximation_step != 5:
        raise ConfigError("approximation sweeps use linear step 5 plus endpoints")
    if not 0 <= cfg.exponential_max_n <= 8:
        raise ConfigError("exact exponential enumeration is capped at n=8 (0 disables)")
    if not cfg.comparison_settings or len(set(cfg.comparison_settings)) != len(cfg.comparison_settings):
        raise ConfigError("comparison settings must be nonempty and unique")
    for setting in cfg.comparison_settings:
        if len(setting) != 2:
            raise ConfigError("each comparison setting must contain n,d")
        n, d = setting
        if not (2 <= n <= cfg.base.master_n and 1 <= d <= min(500, cfg.base.master_d)):
            raise ConfigError("comparison setting exceeds the frozen master/evaluated dimensions")


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def fingerprint(cfg: FollowupConfig) -> str:
    # Version the numerical environment too: resume cannot silently mix solver laws.
    payload = {"scientific": cfg.identity(), "source_sha256": source_digest(),
               "numpy": np.__version__, "scipy": scipy.__version__}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def result_directory(root: str | Path, cfg: FollowupConfig, fp: str) -> Path:
    return Path(root).resolve() / cfg.name / fp[:16]


def epsilon_path(directory: Path, s: int, n: int, d: int, eps: float) -> Path:
    return directory / "units" / f"s{s}_n{n}_d{d}_eps{epsilon_key(eps)}.json"


def extended_rng(cfg: ScientificConfig, s: int, n: int, d: int, key: str,
                 epsilon: float, k: int) -> np.random.Generator:
    if key in {"ambient", "folded", "phi_infinity"}:
        return noise_generator(cfg.root_seed, s, n, d, key, epsilon, k)
    digest = hashlib.sha256(key.encode()).digest()
    tags = tuple(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4))
    return make_generator(cfg.root_seed, (2, 3, s, n, d, *tags, *float64_to_uint32_pair(epsilon), k))


@lru_cache(maxsize=3)
def _masters(cfg: ScientificConfig, s: int) -> tuple[np.ndarray, np.ndarray]:
    return generate_master_pair(cfg, s)


@dataclass(frozen=True)
class Arm:
    name: str
    base: str
    family: str = "exact"
    value: int = -1
    selection_bound: float | None = None


def arms_for(cfg: FollowupConfig, catalog: PublicProxyCatalog, n: int, d: int, eps: float) -> list[Arm]:
    arms = [Arm(name, name) for name in ("ambient", "phi_infinity", "folded", "anchor", "centered", "fixed")]
    exact = catalog.exact
    auto_bounds = [exact.expected_bound(n, eps, False), exact.expected_bound(n, eps, True), n * catalog.diameter]
    choice = ("folded", "anchor", "fixed")[int(np.argmin(auto_bounds))]
    arms.append(Arm("auto", choice, selection_bound=float(min(auto_bounds))))
    for family in ("pca", "representatives"):
        for base in ("anchor", "folded"):
            value, bound = catalog.select(family, eps, base == "anchor")
            arms.append(Arm(f"{base}_{family}_best", base, family, value, bound))
    if n <= cfg.exponential_max_n:
        arms.append(Arm("exponential", "exponential"))
    if (n, d) in cfg.comparison_settings and eps in cfg.approximation_epsilons:
        for family, maximum in (("pca", catalog.rank), ("representatives", n)):
            for value in endpoint_grid(maximum, zero=family == "pca", step=cfg.approximation_step):
                for base in ("anchor", "folded"):
                    arms.append(Arm(f"{base}_{family}_{value}", base, family, value))
    return arms


def canonical_key(arm: Arm, n: int) -> str:
    if arm.family == "exact" or (arm.family == "representatives" and arm.value == n):
        return arm.base
    return f"{arm.base}:{arm.family}:{arm.value}"


def validate_record(record: dict, cfg: FollowupConfig, fp: str, s: int, n: int, d: int,
                    eps: float, names: set[str] | None = None) -> None:
    if (record.get("fingerprint"), record.get("status"), record.get("s"), record.get("n"),
        record.get("d"), record.get("epsilon"), record.get("K")) != (fp, "complete", s, n, d, eps, cfg.base.noise_repetitions_K):
        raise ValueError("incompatible/incomplete follow-up checkpoint")
    rows = record.get("arms", {})
    if names is not None and set(rows) != names:
        raise ValueError("checkpoint arm set differs from the requested experiment")
    if not set(MAIN_ARMS).issubset(rows):
        raise ValueError("checkpoint is missing primary curves")
    for row in rows.values():
        values = np.asarray(row["excess"])
        if (len(values) != cfg.base.noise_repetitions_K or not np.all(np.isfinite(values))
                or np.any(values < 0) or not row["pointwise_ok"]):
            raise ValueError("checkpoint has invalid samples/certificates")


def evaluate_arm(cfg: FollowupConfig, s: int, n: int, d: int, eps: float,
                 arm: Arm, geom: ProxyGeometry, Q: np.ndarray, R: np.ndarray,
                 C: np.ndarray, optimum: float, catalog: PublicProxyCatalog,
                 private_cache: dict, permutation_costs: np.ndarray | None,
                 progress=log) -> dict:
    start = time.monotonic()
    key = canonical_key(arm, n)
    base = arm.base
    no_release = base == "fixed" or (base in {"anchor", "folded"} and geom.anchor.B == 0)
    K = cfg.base.noise_repetitions_K
    meta: dict[str, Any] = {"base_mechanism": base, "family": arm.family,
                           "value": arm.value, "D_proxy": geom.D_proxy if base in {"anchor", "folded"} else 0.0,
                           "m": geom.fold.m, "secant_pair_count": len(geom.anchor.lengths),
                           "secant_count_kind": "nonzero_pairs_upper_bound_on_distinct_directions",
                           "B_directional": geom.anchor.B, "epsilon_y": 0.0, "epsilon_r": 0.0,
                           "epsilon_other": 0.0, "privacy_model": "local", "pointwise_ok": True,
                           "pointwise_checked": base != "exponential",
                           "max_pointwise_residual": 0.0, "clipped_radius_fraction": 0.0,
                           "random_stream": key, "no_release": no_release}
    if base == "fixed":
        bound, dimension = n * catalog.diameter, 0
    elif base in {"anchor", "folded"}:
        bound = geom.expected_bound(n, eps, base == "anchor")
        dimension = 0 if no_release else geom.fold.m + int(base == "anchor")
        if not no_release:
            if base == "anchor":
                meta["epsilon_y"], meta["epsilon_r"] = optimized_split(geom.anchor.B, eps)
            else:
                meta["epsilon_y"] = eps
    elif base == "ambient":
        bound, dimension = 2 * n * d / eps, d
        meta["epsilon_y"] = eps
    elif base == "phi_infinity":
        bound, dimension = 2 * n * n / eps, n
        meta["epsilon_other"] = eps
    elif base == "centered":
        bound, dimension = 2 * n * (n - 1) / eps, n - 1
        meta["epsilon_other"] = eps
    elif base == "exponential":
        bound, dimension = 2 * (math.lgamma(n + 1) + 1) / eps, None
        meta["privacy_model"], meta["epsilon_other"] = "central", eps
    else:
        raise ValueError(f"unknown mechanism {base}")
    meta.update(expected_bound=float(bound), report_dimension=dimension,
                anchor_bound_improves_folded=bool(geom.expected_bound(n, eps, True) < geom.expected_bound(n, eps, False)))
    progress(f"s={s+1} n={n} d={d} eps={eps:g} arm={arm.name} "
             f"dim={dimension} D_proxy={meta['D_proxy']:.5g} B={geom.anchor.B:.4g} "
             f"split=({meta['epsilon_y']:.4g},{meta['epsilon_r']:.4g}) bound={bound:.5g} "
             f"anchor_improves_folded={meta['anchor_bound_improves_folded']} START")
    folded_inputs = None
    if base in {"anchor", "folded"} and not no_release:
        geometry_key = (geom.family, geom.value)
        if geometry_key not in private_cache:
            private_cache[geometry_key] = geom.fold.fold_private(R)
        folded_inputs = private_cache[geometry_key]
    if base == "exponential":
        if permutation_costs is None:
            raise ValueError("missing small-n enumeration")
        weights = np.exp(-eps / 2 * (permutation_costs - permutation_costs.min()))
        probabilities = weights / weights.sum()
        meta["exact_expected_excess"] = float(probabilities @ (permutation_costs - optimum))
    values, clipped = [], 0
    last_progress = start
    for k in range(1 if no_release else K):
        rng = extended_rng(cfg.base, s, n, d, key, eps, k)
        if no_release:
            excess = assignment_cost(C, np.arange(n)) - optimum
            pointwise = bound
        elif base == "exponential":
            excess = float(permutation_costs[rng.choice(len(probabilities), p=probabilities)] - optimum)
            pointwise = None  # expectation bound, not a realization-wise guarantee
        else:
            if base == "anchor":
                message, W, eta = anchor_client(folded_inputs, geom.anchor, eps, rng)
                C_hat = anchor_server(message, geom.anchor.p)[geom.row_map]
                pointwise = 2 * geom.D_proxy + float((4 * geom.anchor.max_projection(W) + np.abs(eta)).sum())
                clipped += int(np.sum(message.radius == 0))
            elif base == "folded":
                report = folded_report(folded_inputs, eps, rng)
                C_hat = server_scores(report, Q_fold=geom.anchor.p)[geom.row_map]
                pointwise = 2 * geom.D_proxy + 2 * float(report.radii.sum())
            elif base == "ambient":
                report = ambient_report(R, eps, rng)
                C_hat = server_scores(report, Q=Q)
                pointwise = 2 * float(report.radii.sum())
            elif base == "phi_infinity":
                report = phi_infinity_report(n, n, eps, rng)
                # This addition is the client's release; the server receives C_hat only.
                C_hat = C + report.cost_noise
                pointwise = 2 * float(report.radii.sum())
            else:  # centered-distance client, server uses only its reported columns
                C_hat, radii = centered_client(C, eps, rng)
                pointwise = 2 * float(radii.sum())
            excess = assignment_cost(C, exact_assignment(C_hat)) - optimum
            transfer = float(np.ptp(C_hat - C, axis=0).sum())
            if excess > transfer + 1e-8 * max(1, abs(optimum), transfer):
                raise FloatingPointError(f"cost-range transfer failed for {key}")
        tolerance = 1e-8 * max(1, abs(optimum), abs(excess))
        if not np.isfinite(excess) or excess < -tolerance:
            raise FloatingPointError(f"invalid regret {excess} for {key}")
        if pointwise is not None:
            residual = float(excess - pointwise)
            meta["max_pointwise_residual"] = max(meta["max_pointwise_residual"], residual)
            if not np.isfinite(pointwise) or residual > tolerance:
                raise FloatingPointError(f"realization-wise certificate failed for {key}: {residual}")
        values.append(float(max(0, excess)))
        if time.monotonic() - last_progress >= 20:
            progress(f"s={s+1} n={n} d={d} eps={eps:g} {arm.name}: repetition {k+1}/{K}")
            last_progress = time.monotonic()
    meta["excess"] = values * K if no_release else values
    meta["clipped_radius_fraction"] = clipped / (n * K)
    meta["elapsed_seconds"] = time.monotonic() - start
    progress(f"s={s+1} n={n} d={d} eps={eps:g} {arm.name} DONE "
             f"mean_regret={np.mean(meta['excess']):.6g} elapsed={meta['elapsed_seconds']:.2f}s")
    return meta


def run_instance(cfg: FollowupConfig, directory: Path, fp: str, s: int, n: int, d: int,
                 progress=log) -> dict:
    refuse_login_compute("run a follow-up instance")
    start = time.monotonic()
    progress(f"INSTANCE START configuration={s+1} n={n} d={d} pid={os.getpid()}")
    inst = slice_instance(*_masters(cfg.base, s), s, n, d)
    C = pairwise_euclidean(inst.Q, inst.R)
    _, optimum = optimal_cost(C)
    baseline = random_permutation_baseline(C)
    catalog = PublicProxyCatalog(inst.Q)
    permutation_costs = None
    if n <= cfg.exponential_max_n:
        perms = np.asarray(list(itertools.permutations(range(n))), dtype=np.int64)
        permutation_costs = C[np.arange(n)[None], perms].sum(axis=1)
    private_cache: dict = {}
    written = skipped = 0
    for eps in cfg.epsilons_for(n, d):
        path = epsilon_path(directory, s, n, d, eps)
        arms = arms_for(cfg, catalog, n, d, eps)
        if path.exists():
            validate_record(read_json(path), cfg, fp, s, n, d, eps, {a.name for a in arms})
            skipped += 1
            progress(f"RESUME configuration={s+1} n={n} d={d} eps={eps:g} complete")
            continue
        results, cache = {}, {}
        for arm in arms:
            key = canonical_key(arm, n)
            if key not in cache:
                geom = catalog.exact if arm.family == "exact" else catalog.geometry(arm.family, arm.value)
                cache[key] = evaluate_arm(cfg, s, n, d, eps, arm, geom, inst.Q, inst.R, C,
                                          optimum, catalog, private_cache, permutation_costs, progress)
            results[arm.name] = dict(cache[key], family=arm.family, value=arm.value,
                                     selection_bound=arm.selection_bound,
                                     selected_branch=arm.base, main_curve=arm.name in MAIN_ARMS or arm.name == "exponential")
        record = {"fingerprint": fp, "status": "complete", "s": s, "n": n, "d": d,
                  "epsilon": eps, "K": cfg.base.noise_repetitions_K, "OPT": optimum,
                  "random_baseline": baseline, "gap": baseline - optimum,
                  "public_rank": catalog.rank, "m_fold": catalog.exact.fold.m, "arms": results}
        validate_record(record, cfg, fp, s, n, d, eps, {a.name for a in arms})
        atomic_write_json(path, record)
        written += 1
        progress(f"CHECKPOINT {path.name} ({len(results)} curves; {len(cache)} distinct evaluations)")
    duration = time.monotonic() - start
    progress(f"INSTANCE DONE configuration={s+1} n={n} d={d} elapsed={duration:.1f}s wrote={written} resumed={skipped}")
    return {"s": s, "n": n, "d": d, "elapsed_seconds": duration, "written": written, "skipped": skipped}


def run_followup(cfg: FollowupConfig, directory: Path, fp: str, workers: int) -> list[dict]:
    refuse_login_compute("run the follow-up experiment")
    if workers < 1:
        raise ValueError("workers must be positive")
    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    available = min(int(os.environ.get("SLURM_CPUS_PER_TASK", affinity)), affinity)
    if workers > available:
        raise ValueError(f"workers={workers} exceeds allocated CPUs={available}")
    # Spawned workers import numerical libraries after reading these caps.
    # The Slurm entry point sets them before the parent's imports as well.
    from private_matching.thread_limits import apply_thread_limits
    apply_thread_limits(1, overwrite=True)
    directory.mkdir(parents=True, exist_ok=True)
    tasks = [(s, n, d) for s in range(cfg.base.configurations_S) for n, d in cfg.settings()]
    # Submit likely long jobs first to reduce the last-worker tail.
    tasks.sort(key=lambda t: t[1]**3 * len(cfg.epsilons_for(t[1], t[2])) *
               (3 if (t[1], t[2]) in cfg.comparison_settings else 1), reverse=True)
    manifest = {"fingerprint": fp, "scientific": cfg.identity(), "source_sha256": source_digest(),
                "numpy": np.__version__, "scipy": scipy.__version__, "python": platform.python_version(),
                "platform": platform.platform(), "workers": workers, "numerical_threads_per_worker": 1,
                "instance_count": len(tasks), "epsilon_checkpoints": sum(len(cfg.epsilons_for(n, d)) for _, n, d in tasks)}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists() and read_json(manifest_path).get("fingerprint") != fp:
        raise ValueError("manifest mismatch; refusing to mix experiments")
    atomic_write_json(manifest_path, manifest)
    log(f"FOLLOWUP START instances={len(tasks)} epsilon_checkpoints={manifest['epsilon_checkpoints']} "
        f"S={cfg.base.configurations_S} K={cfg.base.noise_repetitions_K} workers={workers} results={directory}")
    outcomes = []
    started = time.monotonic()
    if workers == 1:
        for task in tasks:
            outcomes.append(run_instance(cfg, directory, fp, *task))
            log(f"PROGRESS {len(outcomes)}/{len(tasks)} instances elapsed={(time.monotonic()-started)/60:.1f}min")
    else:
        # Spawn avoids inheriting initialized BLAS thread pools through fork.
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            pending = {executor.submit(run_instance, cfg, directory, fp, *task): task for task in tasks}
            while pending:
                done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        outcomes.append(future.result())
                    except BaseException:
                        # Cancel queued work on a scientific/numerical failure. Active
                        # instances finish their atomic checkpoints before shutdown.
                        for queued in pending:
                            queued.cancel()
                        executor.shutdown(wait=True, cancel_futures=True)
                        raise
                    del pending[future]
                elapsed = time.monotonic() - started
                log(f"PROGRESS {len(outcomes)}/{len(tasks)} instances; {len(pending)} remaining; "
                    f"elapsed={elapsed/60:.1f}min (workers print active arms/repetitions)")
    atomic_write_json(directory / "timings.json", {"instances": outcomes, "wall_seconds": time.monotonic()-started})
    return outcomes


def load_complete_records(cfg: FollowupConfig, directory: Path, fp: str) -> list[dict]:
    records = []
    for s in range(cfg.base.configurations_S):
        for n, d in cfg.settings():
            for eps in cfg.epsilons_for(n, d):
                path = epsilon_path(directory, s, n, d, eps)
                if not path.exists():
                    raise ValueError(f"experiment incomplete: missing {path}")
                record = read_json(path)
                validate_record(record, cfg, fp, s, n, d, eps)
                # Require every requested approximation endpoint/interior, not just main arms.
                expected = set(MAIN_ARMS)
                if n <= cfg.exponential_max_n:
                    expected.add("exponential")
                if (n, d) in cfg.comparison_settings and eps in cfg.approximation_epsilons:
                    for family, maximum in (("pca", record["public_rank"]), ("representatives", n)):
                        for value in endpoint_grid(maximum, zero=family == "pca", step=cfg.approximation_step):
                            expected.update(f"{base}_{family}_{value}" for base in ("anchor", "folded"))
                if set(record["arms"]) != expected:
                    raise ValueError(f"incomplete arm set: {path}")
                records.append(record)
    return records
