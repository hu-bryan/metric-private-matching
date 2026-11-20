"""Reproducible four-mechanism clustering sweeps on normalized benchmark inputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.spatial.distance import cdist, pdist

from . import clustering_benchmark as cb
from . import clustering_normalized as previous
from .integral_mechanisms import Geometry, solve, supnorm_unit
from .integral_study import atomic_json, save_npz, seed
from .login_guard import refuse_login_compute
from .mechanisms import sample_euclidean_laplace

NAME = "clustering_manuscript_v1"
NAMESPACE = "clustering-manuscript-20261114-v1"
BASE = Path("/ix1/arahimian/brh314/metric_private_covariate_matching/results")
LEGACY = BASE/"clustering_normalized_v1/production/cc3b9d8426900d5ada15862c"
BANKNOTE = BASE/"clustering_manuscript_sources_20261114"
BANKNOTE_SHA = "d0539aaed2139ba7a587b3e34fb345ce503ff7d5d33dbf9912d8e195ce425cb9"
DATASETS = {"graves/ring_noisy": (1050, 2), "wut/circles": (4000, 2),
    "fcps/chainlink": (1000, 3), "other/iris": (150, 4), "uci/banknote": (1372, 4),
    "uci/ecoli": (336, 7), "uci/yeast": (1484, 8), "uci/glass": (214, 9),
    "uci/wine": (178, 13), "uci/statlog": (2310, 19), "uci/wdbc": (569, 30),
    "uci/ionosphere": (351, 34), "uci/sonar": (208, 60), "sipu/worms_64": (105000, 64),
    "mnist/digits10k": (10000, 784), "mnist/fashion10k": (10000, 784)}
ARMS = ("gumbel", "original_points", "projected_representation", "distance_vectors")
FIELDS = ("total_cost", "total_regret", "row_oscillation")
DIMENSIONS = tuple(range(784, 3, -25))+(4,)
SIZES = tuple(range(200, 19, -20))
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def hardware():
    result = previous.hardware()
    result["constraint"] = os.environ.get("PM_MANUSCRIPT_CONSTRAINT", "rome")
    return result


def runtime():
    return previous.runtime()


def source_hashes():
    return previous.source_hashes()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def design(pilot=False):
    settings = []
    def add(panel, dataset, n, d):
        settings.append(dict(setting_id=f'{panel}_{dataset.replace("/", "_")}_n{n}_d{d}',
            panel=panel, dataset=dataset, n=n, d=d,
            epsilon=[.25, 8., 16.] if pilot and panel == "epsilon" else
                    ([float(i)/4 for i in range(1, 65)] if panel == "epsilon" else [8.]),
            objectives=[1, 2] if panel == "epsilon" else [1]))
    for dataset, (rows, d) in DATASETS.items():
        assert rows >= 128
        add("epsilon", dataset, 64, d)
    for d in ((784, 4) if pilot else DIMENSIONS):
        add("dimension", "mnist/digits10k", 400, d)
    for n in ((200, 20) if pilot else SIZES):
        add("sample_size", "uci/banknote", n, 4)
    return dict(phase="pilot" if pilot else "production", sampling="without replacement; disjoint Q/R source rows",
        parents=1 if pilot else 20, partitions=1 if pilot else 5, K=3 if pilot else 50,
        settings=settings, arms=list(ARMS), seed_namespace=NAMESPACE,
        normalization="feature scaling as normalized study; pooled RMS one in each actual configuration",
        banknote_n_feature_fit="once on 400-record master; fixed feature weights over n and partitions",
        pixel_order="historical publicly seeded permutation, nested prefixes",
        privacy_metric="sum_record_euclidean_in_normalized_input_space",
        privacy_scope="offline input construction excluded; Gumbel final matching, other arms local reports",
        randomness="independent channels, coupled unit noise across epsilon/objectives; identity endpoints copied",
        inference="K noise -> five partitions -> 20 parents; pointwise Student-t 95%",
        hardware_constraint=os.environ.get("PM_MANUSCRIPT_CONSTRAINT", "rome"))


def groups(cfg):
    return [dict(setting=s, parent=parent, group_id=s["setting_id"]+f'_parent{parent:02d}')
            for s in cfg["settings"] for parent in range(cfg["parents"])]


def master_key(setting, parent):
    return setting["panel"]+"_"+setting["dataset"].replace("/", "_")+f'_parent{parent:02d}'


def feature_transform(dataset, raw):
    if dataset == "uci/banknote":
        # Wine's rule is precisely the all-continuous population z-score rule.
        fitted = previous.fit_normalization("uci/wine", raw)
    else:
        fitted = previous.fit_normalization(dataset, raw)
    return {k: fitted[k] for k in ("feature_offset", "feature_scale", "feature_constant")}


def source_inventory():
    records = {}
    for archive, _ in previous.ARCHIVES:
        manifest = json.loads((archive/"manifest.json").read_text())
        root = Path(manifest["source_data"].replace("/vast/", "/ix1/", 1))
        for dataset in DATASETS:
            if dataset == "uci/banknote" or dataset in records:
                continue
            name = dataset+".data.gz"
            if name in manifest["sources"]["files"]:
                records[dataset] = dict(data=str(root/name), labels=str(root/(dataset+".labels0.gz")),
                    sha256=manifest["sources"]["files"][name]["sha256"],
                    label_sha256=cb.sha256(root/(dataset+".labels0.gz")),
                    source_manifest=str(archive/"manifest.json"), commit=cb.COMMIT)
    records["uci/banknote"] = dict(data=str(BANKNOTE/"data_banknote_authentication.txt"),
        sha256=BANKNOTE_SHA, source_url="https://archive.ics.uci.edu/static/public/267/banknote+authentication.zip",
        doi="10.24432/C55P57", license="CC BY 4.0", label_column=4)
    assert set(records) == set(DATASETS)
    return records


def load_source(record, dataset):
    path = Path(record["data"])
    assert cb.sha256(path) == record["sha256"]
    if dataset == "uci/banknote":
        data = np.loadtxt(path, delimiter=",")
        raw, labels = data[:, :4], data[:, 4].astype(int)
    else:
        assert cb.sha256(Path(record["labels"])) == record["label_sha256"]
        raw, labels = np.loadtxt(path), np.loadtxt(record["labels"], dtype=int)
    assert raw.shape == DATASETS[dataset] and labels.shape == (len(raw),)
    assert np.isfinite(raw).all()
    return raw, labels


def prepare(root, pilot=False):
    refuse_login_compute("prepare manuscript benchmark sweeps")
    cfg = design(pilot)
    payload = dict(kind=NAME, design=cfg, sources=source_inventory(),
        source_sha256=source_hashes(), versions=runtime(), pixel_order=cb.pixel_order().tolist(),
        legacy_sampling=dict(directory=str(LEGACY), manifest_sha256=cb.sha256(LEGACY/"manifest.json"),
                             registry_sha256=cb.sha256(LEGACY/"input_registry.json")))
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]
    payload["fingerprint"] = fingerprint
    root = Path(root)/NAME/cfg["phase"]/fingerprint
    for name in ("masters", "units", "chunks", "stage_records", "analysis_handoff"):
        (root/name).mkdir(parents=True, exist_ok=True)
    if (root/"manifest.json").exists():
        assert json.loads((root/"manifest.json").read_text()) == payload
    atomic_json(root/"manifest.json", payload)
    registry = {}
    for dataset, source in payload["sources"].items():
        raw, labels = load_source(source, dataset)
        for g in groups(cfg):
            s, parent = g["setting"], g["parent"]
            if s["dataset"] != dataset:
                continue
            key = master_key(s, parent)
            if key in registry:
                continue
            count = 800 if s["panel"] == "dimension" else (400 if s["panel"] == "sample_size" else 2*s["n"])
            rng = np.random.default_rng(seed(NAMESPACE, cfg["phase"], key, "source_parent"))
            assert count <= len(raw)
            ids = np.sort(rng.choice(len(raw), count, replace=False))
            legacy_name = dataset.replace("/", "_")+f'_parent{parent:02d}'
            imported_sampling = s["panel"] == "epsilon" and dataset != "uci/banknote"
            if imported_sampling:
                legacy_registry = json.loads((LEGACY/"input_registry.json").read_text())[legacy_name]
                old_parent = LEGACY/legacy_registry["file"]
                assert cb.sha256(old_parent) == legacy_registry["sha256"]
                with np.load(old_parent) as saved:
                    ids = saved["source_ids"].copy()
                    np.testing.assert_array_equal(raw[ids], saved["raw"])
            selected = np.ascontiguousarray(raw[ids])
            fit = feature_transform(dataset, selected)
            orders = np.stack([np.random.default_rng(seed(NAMESPACE, cfg["phase"], key, part, "partition")).permutation(count)
                               for part in range(cfg["partitions"])])
            if imported_sampling:
                for part in range(cfg["partitions"]):
                    path = LEGACY/"units"/("normalized_"+legacy_name+f'_part{part}.geometry.npz')
                    with np.load(path) as saved:
                        orders[part] = np.r_[np.searchsorted(ids, saved["q_source_ids"]),
                                            np.searchsorted(ids, saved["r_source_ids"])]
                    assert np.array_equal(np.sort(orders[part]), np.arange(count))
            path = root/"masters"/(key+".npz")
            save_npz(path, raw=selected, source_ids=ids, labels=labels[ids], orders=orders, **fit)
            registry[key] = dict(file="masters/"+path.name, sha256=cb.sha256(path),
                dataset=dataset, parent=parent, occurrence_count=count,
                unique_source_count=len(np.unique(ids)), sampled_with_replacement=False,
                imported_source_rows_and_partitions=imported_sampling)
    atomic_json(root/"master_registry.json", registry)
    atomic_json(root/"stage_records/prepare.json", dict(status="passed", hardware=hardware()))
    print("Resolved manuscript benchmark results:", root, flush=True)
    return root, payload


def verify_runtime(manifest):
    if source_hashes() != manifest["source_sha256"] or runtime() != manifest["versions"]:
        raise RuntimeError("Frozen manuscript-study source/runtime/CPU-dispatch drift")
    assert hardware()["constraint"] == manifest["design"]["hardware_constraint"]


def build_input(root, manifest, group, part):
    s, parent = group["setting"], group["parent"]
    key = master_key(s, parent)
    registry = json.loads((Path(root)/"master_registry.json").read_text())[key]
    path = Path(root)/registry["file"]
    assert cb.sha256(path) == registry["sha256"]
    with np.load(path) as master:
        raw, order = master["raw"], master["orders"][part]
        half = len(order)//2
        qi, ri = order[:half][:s["n"]], order[half:][:s["n"]]
        assert len(qi) == len(ri) == s["n"] and len(np.intersect1d(qi, ri)) == 0
        features = np.array(manifest["pixel_order"][:s["d"]] if s["panel"] == "dimension" else np.arange(s["d"]))
        selected = np.sort(np.r_[qi, ri])
        feature = (raw-master["feature_offset"])/master["feature_scale"]
        values = feature[selected][:, features]
        center = values.mean(axis=0)
        radius = float(np.sqrt(np.mean(np.sum((values-center)**2, axis=1))))
        denominator = radius if radius > 0 else 1.
        q = (feature[qi][:, features]-center)/denominator
        r = (feature[ri][:, features]-center)/denominator
        qids, rids = master["source_ids"][qi], master["source_ids"][ri]
        assert len(np.unique(np.r_[qids, rids])) == 2*s["n"]
        arrays = dict(q_occurrence_ids=qi, r_occurrence_ids=ri, q_source_ids=qids, r_source_ids=rids,
            q_labels=master["labels"][qi], r_labels=master["labels"][ri], feature_ids=features,
            pooled_center=center, pooled_radius=np.array(radius))
        meta = dict(master_key=key, master_file=registry["file"], master_sha256=registry["sha256"],
            sampled_with_replacement=registry["sampled_with_replacement"],
            q_unique_source_count=len(np.unique(qids)), r_unique_source_count=len(np.unique(rids)),
            source_overlap_count=len(np.intersect1d(qids, rids)),
            normalized_q_sha256=array_sha(q), normalized_r_sha256=array_sha(r), zero_radius=radius == 0)
    assert np.allclose(np.r_[q, r].mean(axis=0), 0, atol=2e-12)
    assert np.isclose(np.mean(np.sum(np.r_[q, r]**2, axis=1)), 0. if radius == 0 else 1., atol=2e-12)
    return q, r, arrays, meta


def load_input(root, meta):
    """Reconstruct archived normalized inputs without their original source files."""
    with np.load(Path(root)/meta["input"]["master_file"]) as master, np.load(Path(root)/"units"/(meta["instance_id"]+".geometry.npz")) as geom:
        feature = (master["raw"]-master["feature_offset"])/master["feature_scale"]
        columns = geom["feature_ids"]
        denominator = float(geom["pooled_radius"]) or 1.
        q = (feature[geom["q_occurrence_ids"]][:, columns]-geom["pooled_center"])/denominator
        r = (feature[geom["r_occurrence_ids"]][:, columns]-geom["pooled_center"])/denominator
    assert array_sha(q) == meta["input"]["normalized_q_sha256"]
    assert array_sha(r) == meta["input"]["normalized_r_sha256"]
    return q, r


def draw_channels(geo, cfg, instance_id, draw):
    n, d = geo.X.shape
    def rng(channel):
        return np.random.default_rng(seed(NAMESPACE, cfg["phase"], instance_id, draw, channel))
    channels = dict(original_points=sample_euclidean_laplace(n, d, 1., rng("original_points"))[0],
        gumbel=rng("gumbel").gumbel(size=(n, n)), distance_vectors=supnorm_unit(n, rng("distance_vectors")))
    if min(geo.rank+1, d) < d:
        channels["projected_representation"] = sample_euclidean_laplace(n, geo.rank+1, 1., rng("projected_representation"))[0]
    return channels


def cached_reports(geo, channels):
    """Cache quadratic coefficients; full-dimensional reports are regenerated by seeds."""
    xc, yc = geo.X-geo.center, geo.Y-geo.center
    cache = {}
    for arm, x, y in [("original_points", xc, yc),
                       ("projected_representation", geo.fold_X, geo.fold_Y)]:
        if arm not in channels:
            continue
        w = channels[arm]
        cross_public = -2*w@y.T
        cache[arm] = (cross_public+2*np.sum(w*x, axis=1)[:, None],
                      np.sum(w*w, axis=1)[:, None], cross_public)
    return cache


def reconstructed_cost(geo, channels, cache, arm, p, epsilon):
    xc = geo.X-geo.center
    base = geo.C1 if p == 1 else geo.C2-np.sum(xc*xc, axis=1)[:, None]
    if arm in ("original_points", "projected_representation"):
        actual = "original_points" if arm not in cache else arm
        cross, squared_norm, public_cross = cache[actual]
        if p == 1:
            result = np.sqrt(np.maximum(0., geo.C2+cross/epsilon+squared_norm/epsilon**2))
        else:
            result = base+public_cross/epsilon
    elif arm == "distance_vectors":
        L = 1. if p == 1 else 2*geo.radius
        result = base+L*channels[arm]/epsilon
    elif arm == "gumbel":
        L = 2. if p == 1 else 2*geo.diameter
        result = base-L*channels[arm]/epsilon
    else:
        raise ValueError(arm)
    return result, base


def expected_bound(geo, arm, p, epsilon):
    n, d = geo.X.shape
    s = min(geo.rank+1, d)
    if p == 1:
        numerator = {"original_points": 2*n*d, "projected_representation": 2*n*s,
                     "distance_vectors": 2*n*n, "gumbel": 2*n*np.log(n)}[arm]
        return min(n*geo.diameter, numerator/epsilon)
    a = lambda m: min(m, 2*np.sqrt(m*np.log(n)))
    numerator = {"original_points": 2*n*geo.diameter*a(d),
                 "projected_representation": 2*n*geo.diameter*a(s),
                 "distance_vectors": 4*n*n*geo.radius, "gumbel": 2*n*geo.diameter*np.log(n)}[arm]
    return numerator/epsilon


def run_unit(root, manifest, group, part, force=False):
    s, cfg = group["setting"], manifest["design"]
    instance = group["group_id"]+f"_part{part}"
    root = Path(root); stem = root/"units"/instance
    marker = Path(str(stem)+".json")
    if marker.exists() and not force:
        meta = json.loads(marker.read_text())
        assert meta["fingerprint"] == manifest["fingerprint"]
        for filename, checksum in meta["files"].items():
            assert cb.sha256(root/"units"/filename) == checksum
        return meta
    started = time.perf_counter()
    Q, R, geometry_arrays, inputs = build_input(root, manifest, group, part)
    geo = Geometry.build(R, Q)
    objectives, budgets, K = s["objectives"], s["epsilon"], cfg["K"]
    shape = (len(objectives), 4, len(budgets), K)
    arrays = {field: np.empty(shape) for field in FIELDS}
    arrays["permutations"] = np.empty((*shape, s["n"]), dtype=np.uint16)
    optimum_pi = np.stack([solve(geo.C1 if p == 1 else geo.C2) for p in objectives]).astype(np.uint16)
    optimum = np.array([(geo.C1 if p == 1 else geo.C2)[np.arange(s["n"]), optimum_pi[ip]].sum()
                        for ip, p in enumerate(objectives)])
    random = np.array([(geo.C1 if p == 1 else geo.C2).sum()/s["n"] for p in objectives])
    arrays.update(optimum_permutations=optimum_pi, optimum=optimum, random_expected=random)
    aliases = min(geo.rank+1, s["d"]) == s["d"]
    solves = 0
    for draw in range(K):
        channels = draw_channels(geo, cfg, instance, draw)
        cache = cached_reports(geo, channels)
        for ip, p in enumerate(objectives):
            true = geo.C1 if p == 1 else geo.C2
            for ie, epsilon in enumerate(budgets):
                for ia, arm in enumerate(ARMS):
                    if arm == "projected_representation" and aliases:
                        for field in (*FIELDS, "permutations"):
                            arrays[field][ip, ia, ie, draw] = arrays[field][ip, 1, ie, draw]
                        continue
                    noisy, baseline = reconstructed_cost(geo, channels, cache, arm, p, epsilon)
                    pi = solve(noisy); solves += 1
                    cost = float(true[np.arange(s["n"]), pi].sum())
                    regret = cost-float(optimum[ip])
                    osc = float(np.ptp(noisy-baseline, axis=1).sum())
                    tolerance = 1e-8*max(1., abs(cost), osc)
                    assert -tolerance <= regret <= osc+tolerance
                    arrays["permutations"][ip, ia, ie, draw] = pi
                    arrays["total_cost"][ip, ia, ie, draw] = cost
                    arrays["total_regret"][ip, ia, ie, draw] = max(0., regret)
                    arrays["row_oscillation"][ip, ia, ie, draw] = osc
    arrays["expected_bound"] = np.array([[[expected_bound(geo, arm, p, eps) for eps in budgets]
                                          for arm in ARMS] for p in objectives])
    save_npz(str(stem)+".geometry.npz", **geometry_arrays)
    save_npz(str(stem)+".npz", **arrays)
    meta = dict(fingerprint=manifest["fingerprint"], instance_id=instance, setting=s,
        parent=group["parent"], partition=part, K=K, input=inputs, alias_projection_to_original=aliases,
        public=geo.public_metadata(), diagnostics={"Q": cb.spectrum(Q), "R": cb.spectrum(R)},
        pooled_diameter=float(pdist(np.r_[Q, R]).max()), actual_assignment_solves=solves,
        elapsed_seconds=time.perf_counter()-started, hardware=hardware(),
        files={Path(str(stem)+ext).name: cb.sha256(Path(str(stem)+ext)) for ext in (".npz", ".geometry.npz")})
    atomic_json(marker, meta)
    print(f"COMPLETE {instance} {meta['elapsed_seconds']:.2f}s solves={solves}", flush=True)
    return meta


def run_group(root, manifest, group):
    metas = [run_unit(root, manifest, group, part) for part in range(manifest["design"]["partitions"])]
    # Each owner exports its own audited, compressed draw chunk.
    from .clustering_manuscript_report import export_group
    export_group(Path(root), manifest, group, metas)


def run_study(results_root, pilot=False, stage="all", shard_index=None, shard_count=None,
              directory=None):
    refuse_login_compute("run manuscript benchmark sweeps")
    if stage in ("all", "prepare"):
        root, manifest = prepare(results_root, pilot)
    else:
        root = Path(directory); manifest = json.loads((root/"manifest.json").read_text())
    verify_runtime(manifest)
    if stage in ("all", "shard"):
        index, count = shard_index or 0, shard_count or 1
        for group in groups(manifest["design"])[index::count]:
            run_group(root, manifest, group)
        atomic_json(root/"stage_records"/f"shard_{index:04d}.json", dict(status="passed", hardware=hardware()))
    if stage in ("all", "finalize"):
        from .clustering_manuscript_report import export
        export(root, root/"analysis_handoff")
    return root
