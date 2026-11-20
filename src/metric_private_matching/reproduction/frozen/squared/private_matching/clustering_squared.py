"""Squared-cost benchmark using the public-span report in math_drafts.tex."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd

from . import clustering_manuscript as base
from .clustering_benchmark import sha256
from .clustering_manuscript_report import export_group, summarized
from .integral_mechanisms import Geometry, solve, supnorm_unit
from .integral_study import atomic_json, save_npz, seed
from .login_guard import refuse_login_compute
from .mechanisms import sample_euclidean_laplace

NAME = "clustering_squared_v1"
NAMESPACE = "clustering-public-span-squared-20261114-v1"
REFERENCE = base.BASE / "clustering_manuscript_v1/production/26612223b86b05d1a5cf1ff9"


def design(pilot=False):
    cfg = base.design(pilot)
    cfg["settings"] = [dict(s, objectives=[2]) for s in cfg["settings"] if s["panel"] == "epsilon"]
    cfg.update(seed_namespace=NAMESPACE, projected_report="public affine span; r-dimensional radial Laplace",
               randomness="fresh independent mechanism channels; common unit noise across epsilon; full-rank alias",
               reference_sampling=str(REFERENCE))
    return cfg


def prepare(results_root, pilot=False):
    refuse_login_compute("prepare squared-cost benchmark")
    cfg = design(pilot)
    old = json.loads((REFERENCE / "manifest.json").read_text())
    manifest = dict(kind=NAME, design=cfg, source_sha256=base.source_hashes(), versions=base.runtime(),
                    sources=old["sources"], pixel_order=old["pixel_order"],
                    reference_manifest_sha256=sha256(REFERENCE / "manifest.json"))
    fingerprint = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:24]
    manifest["fingerprint"] = fingerprint
    root = Path(results_root) / NAME / cfg["phase"] / fingerprint
    for directory in ("masters", "units", "chunks", "stage_records", "analysis_handoff"):
        (root / directory).mkdir(parents=True, exist_ok=True)
    if (root / "manifest.json").exists():
        assert json.loads((root / "manifest.json").read_text()) == manifest
    atomic_json(root / "manifest.json", manifest)
    source = json.loads((REFERENCE / "master_registry.json").read_text())
    registry = {}
    for group in base.groups(cfg):
        key = base.master_key(group["setting"], group["parent"])
        entry = source[key]
        assert sha256(REFERENCE / entry["file"]) == entry["sha256"]
        shutil.copyfile(REFERENCE / entry["file"], root / entry["file"])
        registry[key] = dict(entry, input_provenance=str(REFERENCE / entry["file"]))
    atomic_json(root / "master_registry.json", registry)
    print("Resolved squared benchmark results:", root, flush=True)
    return root, manifest


def channels(geo, cfg, instance, draw):
    n, d = geo.X.shape
    def rng(arm):
        return np.random.default_rng(seed(NAMESPACE, cfg["phase"], instance, draw, arm))
    result = dict(original_points=sample_euclidean_laplace(n, d, 1., rng("original_points"))[0],
                  gumbel=rng("gumbel").gumbel(size=(n, n)),
                  distance_vectors=supnorm_unit(n, rng("distance_vectors")))
    if 0 < geo.rank < d:
        result["projected_representation"] = sample_euclidean_laplace(n, geo.rank, 1., rng("projected_representation"))[0]
    return result


def coefficients(geo, noise):
    result = dict(original_points=-2 * noise["original_points"] @ (geo.Y-geo.center).T,
                  gumbel=-2 * geo.diameter * noise["gumbel"],
                  distance_vectors=2 * geo.radius * noise["distance_vectors"])
    result["projected_representation"] = (
        result["original_points"] if geo.rank == geo.X.shape[1] else
        (-2 * noise["projected_representation"] @ geo.V.T if geo.rank else np.zeros_like(geo.C2)))
    return result


def bound(geo, arm, epsilon):
    if arm != "projected_representation":
        return base.expected_bound(geo, arm, 2, epsilon)
    n = len(geo.Y)
    a = min(geo.rank, 2*np.sqrt(geo.rank*np.log(n)))
    return 2*n*geo.diameter*a/epsilon


def run_unit(root, manifest, group, part, force=False):
    root = Path(root)
    setting, cfg = group["setting"], manifest["design"]
    instance = group["group_id"]+f"_part{part}"
    stem = root / "units" / instance
    marker = Path(str(stem)+".json")
    if marker.exists() and not force:
        meta = json.loads(marker.read_text())
        assert meta["fingerprint"] == manifest["fingerprint"]
        for filename, checksum in meta["files"].items():
            assert sha256(root/"units"/filename) == checksum
        return meta
    started = time.perf_counter()
    Q, R, geometry_arrays, inputs = base.build_input(root, manifest, group, part)
    geo = Geometry.build(R, Q)
    n, d = R.shape
    budgets, K = setting["epsilon"], cfg["K"]
    shape = (1, 4, len(budgets), K)
    arrays = {field: np.empty(shape) for field in base.FIELDS}
    arrays["permutations"] = np.empty((*shape, n), dtype=np.uint16)
    optimum_pi = solve(geo.C2).astype(np.uint16)
    optimum = float(geo.C2[np.arange(n), optimum_pi].sum())
    arrays.update(optimum_permutations=optimum_pi[None, :], optimum=np.array([optimum]),
                  random_expected=np.array([geo.C2.sum()/n]))
    baseline = geo.C2-np.sum((R-geo.center)**2, axis=1)[:, None]
    aliases = geo.rank == d
    solves = 0
    for draw in range(K):
        noise = channels(geo, cfg, instance, draw)
        coefficient = coefficients(geo, noise)
        for ie, epsilon in enumerate(budgets):
            for ia, arm in enumerate(base.ARMS):
                if arm == "projected_representation" and aliases:
                    for field in (*base.FIELDS, "permutations"):
                        arrays[field][0, ia, ie, draw] = arrays[field][0, 1, ie, draw]
                    continue
                noisy = baseline+coefficient[arm]/epsilon
                pi = solve(noisy)
                solves += 1
                cost = float(geo.C2[np.arange(n), pi].sum())
                osc = float(np.ptp(coefficient[arm]/epsilon, axis=1).sum())
                assert -1e-8 <= cost-optimum <= osc+1e-7
                arrays["permutations"][0, ia, ie, draw] = pi
                arrays["total_cost"][0, ia, ie, draw] = cost
                arrays["total_regret"][0, ia, ie, draw] = max(0., cost-optimum)
                arrays["row_oscillation"][0, ia, ie, draw] = osc
    arrays["expected_bound"] = np.array([[[bound(geo, arm, eps) for eps in budgets] for arm in base.ARMS]])
    save_npz(str(stem)+".geometry.npz", **geometry_arrays)
    save_npz(str(stem)+".npz", **arrays)
    meta = dict(fingerprint=manifest["fingerprint"], instance_id=instance, setting=setting,
        parent=group["parent"], partition=part, K=K, input=inputs, alias_projection_to_original=aliases,
        projected_noise_dimension=geo.rank, projected_report="public_affine_span",
        public=geo.public_metadata(), diagnostics={"Q": base.cb.spectrum(Q), "R": base.cb.spectrum(R)},
        pooled_diameter=float(base.pdist(np.r_[Q,R]).max()), actual_assignment_solves=solves,
        elapsed_seconds=time.perf_counter()-started, hardware=base.hardware(),
        files={Path(str(stem)+ext).name: sha256(Path(str(stem)+ext)) for ext in (".npz", ".geometry.npz")})
    atomic_json(marker, meta)
    return meta


def run_group(root, manifest, group):
    metas = [run_unit(root, manifest, group, part) for part in range(manifest["design"]["partitions"])]
    export_group(root, manifest, group, metas)
    print("COMPLETE", group["group_id"], flush=True)


def export(root, output):
    refuse_login_compute("aggregate squared-cost benchmark")
    root, output = Path(root), Path(output)
    manifest = json.loads((root/"manifest.json").read_text())
    base.verify_runtime(manifest)
    parents, geometries, index = [], [], []
    for group in base.groups(manifest["design"]):
        stem = root/"chunks"/group["group_id"]
        meta = json.loads(Path(str(stem)+".json").read_text())
        assert meta["fingerprint"] == manifest["fingerprint"] and meta["status"] == "passed"
        for name, checksum in meta["files"].items():
            assert sha256(root/"chunks"/name) == checksum
        parents.append(pd.read_csv(str(stem)+".parents.csv.gz"))
        geometries.append(pd.read_csv(str(stem)+".geometry.csv.gz"))
        index.append(dict(group_id=group["group_id"], draw_rows=meta["draw_rows"],
            actual_assignment_solves=meta["actual_assignment_solves"], file="../chunks/"+stem.name+".draws.csv.gz"))
    parent = pd.concat(parents, ignore_index=True)
    summary = summarized(parent)
    assert (summary.mean_total_regret_parent_count == manifest["design"]["parents"]).all()
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output/"summary.csv", index=False)
    parent.to_csv(output/"parent_summaries.csv.gz", index=False)
    pd.concat(geometries, ignore_index=True).to_csv(output/"geometry.csv.gz", index=False)
    pd.DataFrame(index).to_csv(output/"table_index.csv", index=False)
    for filename in ("manifest.json", "master_registry.json"):
        shutil.copyfile(root/filename, output/filename)
    verification = dict(status="passed", geometries=len(pd.concat(geometries)), chunk_groups=len(index),
        canonical_draw_rows=sum(x["draw_rows"] for x in index),
        actual_assignment_solves=sum(x["actual_assignment_solves"] for x in index),
        all_saved_permutations_and_costs_audited=True, all_normalizers_audited=True,
        projected_noise_dimension="public affine rank r", summary_sha256=sha256(output/"summary.csv"))
    atomic_json(output/"verification.json", verification)
    return verification


def run_study(results_root, pilot=False, stage="all", shard_index=None, shard_count=None, directory=None):
    refuse_login_compute("run squared-cost benchmark")
    if stage in ("all", "prepare"):
        root, manifest = prepare(results_root, pilot)
    else:
        root = Path(directory)
        manifest = json.loads((root/"manifest.json").read_text())
    base.verify_runtime(manifest)
    if stage in ("all", "shard"):
        for group in base.groups(manifest["design"])[shard_index or 0::shard_count or 1]:
            run_group(root, manifest, group)
    if stage in ("all", "finalize"):
        export(root, root/"analysis_handoff")
    return root
