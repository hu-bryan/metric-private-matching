"""Frozen epsilon-8 normalization comparison on the 15 existing benchmark samples."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from scipy.spatial.distance import cdist

from . import clustering_benchmark as cb
from .corrected_projection import CorrectedProjection
from .integral_mechanisms import Geometry, bound, evaluate_draw, solve, unit_channels
from .integral_study import atomic_json, save_npz, seed
from .login_guard import refuse_login_compute
from .mechanisms import sample_euclidean_laplace

NAME = "clustering_normalized_v1"
NAMESPACE = "clustering-normalized-epsilon8-20261114-v1"
BASE = Path("/ix1/arahimian/brh314/metric_private_covariate_matching/results")
ARCHIVES = [(BASE/"clustering_benchmark_v1/production/975562db8834d0cbf8bb0683", "aggregate"),
            (BASE/"clustering_benchmark_extension_v1/production/44f7a46b3e1370d516dc5819", "real")]
CORRECTED = BASE/"corrected_projection_benchmark_v1/production/9964f52f3872b546b734e5f7"
SCALED = ("uci/wine", "uci/wdbc", "uci/statlog", "uci/glass", "uci/ecoli", "uci/yeast", "uci/ionosphere", "uci/sonar")
# Zero-based feature indices and the two fixed source encodings.
BINARY = {"uci/ecoli": {2: (.48, 1.), 3: (.5, 1.)},
          "uci/yeast": {4: (.5, 1.)}, "uci/ionosphere": {0: (0., 1.)}}
ARMS = {1: (*cb.ARMS[1], "corrected_projection"), 2: cb.ARMS[2]}
ARM_KEYS = [(p, arm) for p in (1, 2) for arm in ARMS[p]]
FIELDS = ("total_cost", "total_regret", "row_oscillation", "realization_bound")


def hardware():
    model = next(x.split(":", 1)[1].strip() for x in Path("/proc/cpuinfo").read_text().splitlines()
                 if x.startswith("model name"))
    return dict(host=platform.node(), cpu_model=model, constraint=os.environ.get("PM_NORM_CONSTRAINT", "rome"),
                job_id=os.environ.get("SLURM_JOB_ID"), array_job=os.environ.get("SLURM_ARRAY_JOB_ID"),
                array_task=os.environ.get("SLURM_ARRAY_TASK_ID"))


def runtime():
    versions = cb.environment()
    versions["numpy_cpu_features"] = {k: bool(v) for k, v in np._core._multiarray_umath.__cpu_features__.items()}
    versions["thread_environment"] = {k: os.environ.get(k) for k in
        ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    return versions


def source_hashes():
    package = Path(__file__).parent
    return {p.name: cb.sha256(p) for p in sorted(package.glob("*.py"))}


def fit_normalization(dataset, raw):
    """Canonical pooled rows, float64 ddof=0, binary affine maps, final RMS=1."""
    raw = np.asarray(raw, dtype=np.float64)
    assert raw.ndim == 2 and np.isfinite(raw).all()
    offset = np.zeros(raw.shape[1]); scale = np.ones(raw.shape[1])
    constant = np.zeros(raw.shape[1], dtype=bool)
    if dataset in SCALED:
        mean = raw.mean(axis=0)
        variance = np.mean((raw-mean)**2, axis=0)
        # The float64 constant-feature error bound used by StandardScaler.
        eps = np.finfo(float).eps
        constant = variance <= len(raw)*eps*variance + (len(raw)*mean*eps)**2
        offset = mean.copy(); scale = np.sqrt(variance)
        scale[constant] = 1.
        for j, (low, high) in BINARY.get(dataset, {}).items():
            if not np.all(np.isclose(raw[:, j], low, atol=1e-12, rtol=0) |
                          np.isclose(raw[:, j], high, atol=1e-12, rtol=0)):
                raise ValueError(f"Unexpected binary encoding: {dataset} column {j+1}")
            offset[j], scale[j] = low, high-low
    feature = (raw-offset)/scale
    center = feature.mean(axis=0)
    radius = float(np.sqrt(np.mean(np.sum((feature-center)**2, axis=1))))
    if radius <= 0:
        raise ValueError("Zero pooled RMS radius")
    normalized = (feature-center)/radius
    assert np.allclose(normalized.mean(axis=0), 0., atol=2e-12)
    assert np.isclose(np.mean(np.sum(normalized**2, axis=1)), 1., atol=2e-12)
    return dict(raw=raw, normalized=normalized, feature_offset=offset, feature_scale=scale,
                feature_constant=constant, pooled_center=center, pooled_radius=np.array(radius))


def design(pilot=False):
    return dict(phase="pilot" if pilot else "production", epsilon=[8.], n=64,
        parents=1 if pilot else 20, partitions=1 if pilot else 5, K=3 if pilot else 50,
        normalization="pooled feature zscore ddof0 with fixed binary maps, then pooled RMS one; unscaled controls retain exact historical inputs",
        scaled_datasets=list(SCALED), binary_maps={d: {str(j): list(pair) for j, pair in columns.items()} for d, columns in BINARY.items()},
        constant_policy="variance <= n*eps*variance + (n*mean*eps)^2: scale=1, retain coordinate; log masks",
        parent_order="ascending original source row ID, shared by all five partitions",
        arms={str(p): list(arms) for p, arms in ARMS.items()}, seed_namespace=NAMESPACE,
        privacy_scope="fixed normalized input coordinates only; offline Q/R construction is outside the private mechanism",
        inference="noise mean -> five partition means -> 20 parent means; pointwise Student-t 95%",
        unchanged_policy="reuse exact historical epsilon8 K50 reports, geometry and corrected-projection outcomes; explicit source provenance",
        pilot_policy="same selected source inputs, separate pilot namespace; never pool pilot draws with production",
        hardware_constraint=os.environ.get("PM_NORM_CONSTRAINT", "rome"))


def origin_units(cfg):
    result, archives = [], []
    for archive, panel in ARCHIVES:
        manifest = json.loads((archive/"manifest.json").read_text())
        source = Path(manifest["source_data"].replace("/vast/", "/ix1/", 1))
        if not source.exists():
            source = Path(manifest["source_data"])
        archives.append(dict(directory=str(archive), panel=panel, fingerprint=manifest["fingerprint"],
                             manifest_sha256=cb.sha256(archive/"manifest.json")))
        for old in cb.all_units(manifest["design"]):
            if not any(m["panel"] == panel for m in old["memberships"]):
                continue
            if old["parent"] >= cfg["parents"] or old["partition"] >= cfg["partitions"]:
                continue
            stem = archive/"units"/old["instance_id"]
            meta = json.loads(Path(str(stem)+".json").read_text())
            corrected = CORRECTED/"units"/old["instance_id"]
            cm = json.loads(Path(str(corrected)+".json").read_text())
            identifier = "normalized_"+old["dataset"].replace("/", "_")+f'_parent{old["parent"]:02d}_part{old["partition"]}'
            result.append(dict(instance_id=identifier, dataset=old["dataset"], parent=old["parent"], partition=old["partition"],
                n=64, d=old["d"], changed=old["dataset"] in SCALED, origin_id=old["instance_id"],
                archive=str(archive), source_fingerprint=manifest["fingerprint"],
                source_file=str(source/(old["dataset"]+".data.gz")),
                source_sha256=manifest["sources"]["files"][old["dataset"]+".data.gz"]["sha256"],
                metadata_sha256=cb.sha256(Path(str(stem)+".json")),
                geometry_sha256=meta["files"][old["instance_id"]+".geometry.npz"]["sha256"],
                outcome_sha256=meta["files"][old["instance_id"]+".npz"]["sha256"],
                corrected_sha256=cm["files"][old["instance_id"]+".npz"]))
    assert len({u["dataset"] for u in result}) == 15
    assert len(result) == 15*cfg["parents"]*cfg["partitions"]
    return sorted(result, key=lambda u: (u["dataset"], u["parent"], u["partition"])), archives


def prepare(root, pilot=False):
    refuse_login_compute("prepare normalized benchmarks")
    cfg = design(pilot)
    units, archives = origin_units(cfg)
    payload = dict(kind=NAME, design=cfg, units=units, archives=archives,
        corrected_archive=str(CORRECTED), corrected_manifest_sha256=cb.sha256(CORRECTED/"manifest.json"),
        source_sha256=source_hashes(), versions=runtime())
    fp = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    payload["fingerprint"] = fp
    directory = Path(root)/NAME/cfg["phase"]/fp
    for sub in ("units", "inputs", "stage_records", "analysis_handoff"):
        (directory/sub).mkdir(parents=True, exist_ok=True)
    if (directory/"manifest.json").exists():
        assert json.loads((directory/"manifest.json").read_text()) == payload
    atomic_json(directory/"manifest.json", payload)
    last_dataset, raw = None, None
    input_registry = {}
    for u in units:
        key = parent_key(u)
        if key in input_registry:
            continue
        if u["dataset"] != last_dataset:
            assert cb.sha256(Path(u["source_file"])) == u["source_sha256"]
            raw = np.loadtxt(u["source_file"])
            last_dataset = u["dataset"]
            source_txt = Path(u["source_file"].replace(".data.gz", ".txt"))
            if source_txt.exists():
                shutil.copyfile(source_txt, directory/"inputs"/(u["dataset"].replace("/", "_")+".txt"))
        origin = Path(u["archive"])/"units"/u["origin_id"]
        with np.load(str(origin)+".geometry.npz") as saved:
            ids = np.sort(np.r_[saved["q_source_ids"], saved["r_source_ids"]])
        assert len(ids) == len(np.unique(ids)) == 128
        arrays = fit_normalization(u["dataset"], np.ascontiguousarray(raw[ids]))
        path = directory/"inputs"/(key+".npz")
        save_npz(path, source_ids=ids, **arrays)
        input_registry[key] = dict(file="inputs/"+path.name, sha256=cb.sha256(path),
                                   source_file=u["source_file"], source_sha256=u["source_sha256"])
    atomic_json(directory/"input_registry.json", input_registry)
    atomic_json(directory/"stage_records/prepare.json", dict(hardware=hardware(), versions=runtime(), status="passed"))
    print("Resolved normalized benchmark results:", directory, flush=True)
    return directory, payload


def parent_key(u):
    return u["dataset"].replace("/", "_")+f'_parent{u["parent"]:02d}'


def verify_runtime(manifest):
    if source_hashes() != manifest["source_sha256"] or runtime() != manifest["versions"]:
        raise RuntimeError("Frozen normalized-study source/runtime/CPU-dispatch drift")
    if hardware()["constraint"] != manifest["design"]["hardware_constraint"]:
        raise RuntimeError("Hardware constraint differs from manifest")


def prepared_unit(directory, manifest, u):
    registry = json.loads((directory/"input_registry.json").read_text())[parent_key(u)]
    path = directory/registry["file"]
    assert cb.sha256(path) == registry["sha256"]
    origin = Path(u["archive"])/"units"/u["origin_id"]
    assert cb.sha256(Path(str(origin)+".geometry.npz")) == u["geometry_sha256"]
    with np.load(str(origin)+".geometry.npz") as old, np.load(path) as parent:
        qi, ri = old["q_source_ids"], old["r_source_ids"]
        ids = parent["source_ids"]
        assert np.array_equal(np.sort(np.r_[qi, ri]), ids)
        qidx, ridx = np.searchsorted(ids, qi), np.searchsorted(ids, ri)
        if u["changed"]:
            Q, R = parent["normalized"][qidx], parent["normalized"][ridx]
        else:
            Q, R = old["Q"].copy(), old["R"].copy()
            assert np.allclose(Q, parent["normalized"][qidx], rtol=1e-12, atol=1e-12)
            assert np.allclose(R, parent["normalized"][ridx], rtol=1e-12, atol=1e-12)
        record = dict(q_source_ids=qi.copy(), r_source_ids=ri.copy(),
            q_stored_labels=old["q_stored_labels"].copy(), r_stored_labels=old["r_stored_labels"].copy())
    return Q, R, record


def valid_checkpoint(directory, manifest, u):
    path = directory/"units"/(u["instance_id"]+".json")
    if not path.exists():
        return False
    meta = json.loads(path.read_text())
    assert meta["fingerprint"] == manifest["fingerprint"] and meta["unit"] == u
    for filename, checksum in meta["files"].items():
        assert cb.sha256(directory/"units"/filename) == checksum
    return True


def corrected_result(g, folded, channels, epsilon, C, optimum):
    decoded, cert, anchors, ey, es = g.decode_folded(folded, channels[0], channels[1], epsilon)
    pi = solve(decoded)
    cost = float(C[np.arange(len(C)), pi].sum())
    osc = np.ptp(decoded-C, axis=1)
    tolerance = 1e-8*max(1., cost, float(cert.sum()))
    assert np.all(osc <= cert+1e-8*np.maximum(1., cert))
    assert -tolerance <= cost-optimum <= cert.sum()+tolerance
    return dict(permutation=pi, total_cost=cost, total_regret=cost-optimum,
                row_oscillation=float(osc.sum()), realization_bound=float(cert.sum()))


def run_unit(directory, manifest, u, force=False):
    if not force and valid_checkpoint(directory, manifest, u):
        return
    start = time.perf_counter()
    K = manifest["design"]["K"]
    Q, R, record = prepared_unit(directory, manifest, u)
    geo = Geometry.build(R, Q)
    g = CorrectedProjection.build(Q)
    C = {1: geo.C1, 2: geo.C2}
    optimum_pi = np.stack([solve(C[p]) for p in (1, 2)]).astype(np.uint16)
    optima = {p: float(C[p][np.arange(64), optimum_pi[p-1]].sum()) for p in (1, 2)}
    random = {p: float(C[p].sum()/64) for p in (1, 2)}
    gaps = {p: cb.gap_info(optima[p], random[p]) for p in (1, 2)}
    assert all(gaps[p]["status"] == "ok" for p in (1, 2))
    arrays = {name: np.empty((15, K)) for name in FIELDS}
    arrays["permutations"] = np.empty((15, K, 64), dtype=np.uint16)
    arrays["optimum_permutations"] = optimum_pi
    reuse, certs = {}, {}
    for p, arm in ARM_KEYS:
        if arm == "corrected_projection":
            ey, es = g.split(8.)
            uncapped = 64*(2*np.sqrt(g.anchor.B)+1)**2/8
            certs[f"{p}:{arm}"] = dict(bound_total=min(64*geo.diameter, uncapped), bound_uncapped=uncapped,
                bound_cap=64*geo.diameter, epsilon_y=ey, epsilon_s=es, L_infinity=1., L_one=64.)
        else:
            cert = bound(geo, arm, p, 8., "benchmark")
            certs[f"{p}:{arm}"] = {k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in cert.items()}
    if u["changed"]:
        folded = g.transform(R)
        assert np.allclose(cdist(folded, g.anchor.p), C[1], rtol=1e-12, atol=1e-10)
        for draw in range(K):
            def rng(channel):
                return np.random.default_rng(seed(NAMESPACE, manifest["design"]["phase"], u["instance_id"], draw, channel))
            channels, _ = unit_channels(geo, rng)
            result = evaluate_draw(geo, channels, 8., "benchmark", optima)
            W = sample_euclidean_laplace(64, g.dimension, 1., rng("corrected_coordinate"))[0]
            scalar = rng("corrected_scalar").laplace(size=64)
            result[1, "corrected_projection"] = corrected_result(g, folded, (W, scalar), 8., C[1], optima[1])
            for ia, key in enumerate(ARM_KEYS):
                item = result[key]
                arrays["permutations"][ia, draw] = item["permutation"]
                for field in FIELDS:
                    arrays[field][ia, draw] = item[field]
                if draw == 0:
                    reuse[f"{key[0]}:{key[1]}"] = dict(source_arm=item.get("source_arm", ""), reason=item.get("reuse_reason", ""))
        imported = False
    else:
        origin = Path(u["archive"])/"units"/u["origin_id"]
        assert cb.sha256(Path(str(origin)+".npz")) == u["outcome_sha256"]
        oldmeta = json.loads(Path(str(origin)+".json").read_text())
        oldmanifest = json.loads((Path(u["archive"])/"manifest.json").read_text())
        ie = oldmanifest["design"]["epsilon"].index(8.)
        with np.load(str(origin)+".npz") as old:
            for ia, (p, arm) in enumerate(ARM_KEYS):
                if arm == "corrected_projection":
                    continue
                oldindex = cb.ARMS[p].index(arm)
                for field in (*FIELDS, "permutations"):
                    arrays[field][ia] = old[field][p-1, oldindex, ie, :K]
                reuse[f"{p}:{arm}"] = dict(source_arm=arm, reason="unchanged normalized input; exact historical epsilon8 subset",
                    fingerprint=u["source_fingerprint"], instance_id=u["origin_id"], epsilon_index=ie)
        cpath = CORRECTED/"units"/(u["origin_id"]+".npz")
        assert cb.sha256(cpath) == u["corrected_sha256"]
        cm = json.loads((CORRECTED/"manifest.json").read_text())
        ci = cm["design"]["epsilon"].index(8.)
        ia = ARM_KEYS.index((1, "corrected_projection"))
        with np.load(cpath) as old:
            for field in (*FIELDS, "permutations"):
                arrays[field][ia] = old[field][ci, :K]
        reuse["1:corrected_projection"] = dict(source_arm="corrected_projection", reason="unchanged normalized input; exact historical epsilon8 subset",
            fingerprint=cm["fingerprint"], instance_id=u["origin_id"], epsilon_index=ci)
        imported = True
    assert np.all(np.sort(arrays["permutations"], axis=-1) == np.arange(64))
    for ia, (p, arm) in enumerate(ARM_KEYS):
        costs = C[p][np.arange(64), arrays["permutations"][ia]].sum(axis=-1)
        np.testing.assert_allclose(costs, arrays["total_cost"][ia], rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(costs-optima[p], arrays["total_regret"][ia], rtol=1e-12, atol=1e-10)
        assert np.all(arrays["total_regret"][ia] <= arrays["realization_bound"][ia]+1e-7)
    stem = directory/"units"/u["instance_id"]
    save_npz(str(stem)+".geometry.npz", Q=Q, R=R, **record)
    save_npz(str(stem)+".npz", **arrays)
    meta = dict(fingerprint=manifest["fingerprint"], unit=u, K=K, epsilon=8., imported=imported,
        optima={str(p): optima[p] for p in (1, 2)}, random_expected={str(p): random[p] for p in (1, 2)},
        gaps={str(p): gaps[p] for p in (1, 2)}, public=geo.public_metadata(),
        diagnostics={"Q": cb.spectrum(Q), "R": cb.spectrum(R)}, bounds=certs, reuse=reuse,
        corrected_dimension=g.dimension, corrected_identity=g.identity, corrected_directional_coefficient=g.anchor.B,
        hardware=hardware(), elapsed_seconds=time.perf_counter()-start,
        actual_private_solves=0 if imported else K*sum(not reuse[f"{p}:{a}"]["source_arm"] for p, a in ARM_KEYS),
        files={Path(str(stem)+suffix).name: cb.sha256(Path(str(stem)+suffix)) for suffix in (".geometry.npz", ".npz")})
    atomic_json(str(stem)+".json", meta)
    print(f'COMPLETE {u["instance_id"]} {meta["elapsed_seconds"]:.3f}s imported={imported}', flush=True)


def work_groups(manifest):
    keys = sorted({parent_key(u) for u in manifest["units"]})
    return [[u for u in manifest["units"] if parent_key(u) == key] for key in keys]


def run_study(root, pilot=False, stage="all", shard_index=None, shard_count=None, directory=None):
    refuse_login_compute("run normalized benchmark study")
    if stage in ("prepare", "all"):
        directory, manifest = prepare(root, pilot)
    else:
        directory = Path(directory)
        manifest = json.loads((directory/"manifest.json").read_text())
    verify_runtime(manifest)
    if stage in ("shard", "all"):
        index, count = shard_index or 0, shard_count or 1
        assert 0 <= index < count
        for group in work_groups(manifest)[index::count]:
            for u in group:
                run_unit(directory, manifest, u)
        atomic_json(directory/"stage_records"/f"shard_{index:03d}.json", dict(status="passed", hardware=hardware(), versions=runtime()))
    if stage in ("finalize", "all"):
        from .clustering_normalized_report import export
        export(directory, directory/"analysis_handoff")
    return directory
