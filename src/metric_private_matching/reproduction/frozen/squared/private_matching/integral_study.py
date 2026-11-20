"""Versioned, resumable integral matching study. One owner per geometry."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import time

import numpy as np
import scipy

from private_matching.integral_mechanisms import ARMS, Geometry, bound, evaluate_draw, solve, unit_channels
from private_matching.login_guard import refuse_login_compute

NAME = "integral_matching_comparison_v1"
FAMILIES = ("U", "G", "A01", "A20", "M01", "M04", "M10")
FLOAT_FIELDS = ("total_cost", "total_regret", "row_oscillation", "realization_bound", "runtime_seconds", "solve_seconds")
RAW_FIELDS = ("study", "fingerprint", "phase", "geometry_id", "setting_id", "panel", "family", "n", "d", "tau", "geometry_rep", "public_rank", "public_diameter", "public_radius", "fold_dimension", "objective_p", "mechanism", "epsilon", "noise_draw", "noise_coupling_id", "total_cost", "total_regret", "optimum_total_cost", "uniform_expected_total_cost", "bound_total", "bound_uncapped", "bound_cap", "bound_id", "epsilon_y", "epsilon_s", "L_infinity", "L_one", "row_oscillation", "realization_bound", "runtime_seconds", "solve_seconds", "source_arm", "reuse_reason", "privacy_output", "permutation_file", "permutation_array", "permutation_index", "geometry_file")


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    os.replace(temp, path)


def seed(*keys):
    return int.from_bytes(hashlib.sha256(json.dumps(keys).encode()).digest()[:16], "little")


def design(pilot=False):
    settings = []
    def add(panel, family, n, d, tau=None):
        settings.append(dict(setting_id=f"{panel}_{family}_n{n}_d{d}"+(f"_t{tau:g}" if tau else ""), panel=panel, family=family, n=n, d=d, tau=tau))
    if pilot:
        for d in (32, 512):
            for family in ("U", "G", "A01", "M01", "M04", "M10"):
                add("A", family, 64, d)
            for family in ("G", "M04"):
                add("B", family, 256, d)
        add("C", "G", 64, 512, .5)
    else:
        for family in FAMILIES:
            for d in (8, 32, 128, 512):
                add("A", family, 64, d)
        for family in ("G", "M04"):
            for n in (16, 32, 128, 256):
                for d in (32, 512):
                    add("B", family, n, d)
            for d in (8, 32, 128, 512):
                for tau in (.05, .2, .5):
                    add("C", family, 64, d, tau)
    return dict(schema=NAME, phase="pilot" if pilot else "production", epsilon=[.5, 4., 8.] if pilot else (np.arange(1,17)/2).tolist(), K=5 if pilot else 50, geometries=3 if pilot else 20, settings=settings, seed_namespace="integral-matching-20261110-v1", objective="sum", privacy_metric="sum_record_euclidean", arms={str(k): list(v) for k,v in ARMS.items()})


def resolved(results_root, pilot):
    cfg = design(pilot)
    hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("integral_study.py", "integral_mechanisms.py", "mechanisms.py")}
    versions = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__)
    fp = hashlib.sha256(json.dumps(dict(design=cfg, source=hashes, versions=versions), sort_keys=True).encode()).hexdigest()[:24]
    return Path(results_root)/NAME/cfg["phase"]/fp, dict(kind=NAME, fingerprint=fp, design=cfg, source_sha256=hashes, versions=versions)


def units(cfg):
    return [dict(s, geometry_rep=g, geometry_id=f'{s["setting_id"]}_g{g:02d}') for s in cfg["settings"] for g in range(cfg["geometries"])]


def population(family, n, d, rng):
    if family == "U":
        return rng.uniform(-np.sqrt(3/d), np.sqrt(3/d), (n,d)), np.full(n,-1)
    z = rng.normal(size=(n,d))
    if family == "G":
        return z/np.sqrt(d), np.full(n,-1)
    if family.startswith("A"):
        alpha = {"A01": .01, "A20": .2}[family]
        return z*np.sqrt(np.r_[np.full(4,(1-alpha)/4),np.full(d-4,alpha/(d-4))]), np.full(n,-1)
    s = {"M01": .1, "M04": .4, "M10": 1.}[family]
    j = rng.integers(4,size=n)
    centers = np.zeros((4,d)); centers[0,0]=1; centers[1,0]=-1; centers[2,1]=1; centers[3,1]=-1
    return (centers[j]+s*np.array([.5,1,1.5,2])[j,None]*z/np.sqrt(d))/np.sqrt(1+15*s*s/8), j


def make_geometry(cfg, unit):
    key = (cfg["seed_namespace"], cfg["phase"], unit["geometry_id"])
    rng = np.random.default_rng(seed(*key,"geometry"))
    n,d,f = unit["n"],unit["d"],unit["family"]
    Y, labels_y = population(f,n,d,rng)
    if unit["tau"] is None:
        X, labels_x = population(f,n,d,rng); planted = np.full(n,-1)
    else:
        planted = rng.permutation(n)
        X = Y[planted]+unit["tau"]*rng.normal(size=(n,d))/np.sqrt(d)
        labels_x = labels_y[planted]
    O,R = np.linalg.qr(rng.normal(size=(d,d)))
    O *= np.where(np.diag(R)<0,-1.,1.)[None,:]
    return Geometry.build(X@O,Y@O), dict(rotation=O, labels_x=labels_x, labels_y=labels_y, planted=planted)


def save_npz(path, **arrays):
    tmp = Path(str(path)+".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp,path)


def valid_checkpoint(directory, unit, fp):
    stem = directory/"units"/unit["geometry_id"]
    marker = Path(str(stem)+".json")
    if not marker.exists():
        return False
    meta = json.loads(marker.read_text())
    if meta["fingerprint"] != fp or meta["geometry_id"] != unit["geometry_id"]:
        raise RuntimeError(f"checkpoint identity conflict {marker}")
    for suffix in (".npz", ".csv.gz", ".geometry.npz"):
        p = Path(str(stem)+suffix)
        if not p.exists() or p.stat().st_size != meta["files"][suffix]["bytes"]:
            raise RuntimeError(f"incomplete committed checkpoint {p}")
    return True


def write_raw(path, meta, arrays, cfg):
    """Headerless gzip member, concatenated into the canonical table at export."""
    tmp = Path(str(path)+".tmp")
    unit = meta["unit"]
    common = dict(study=NAME, fingerprint=meta["fingerprint"], phase=cfg["phase"], **{k:unit[k] for k in ("geometry_id","setting_id","panel","family","n","d","tau","geometry_rep")}, **{k:meta["public"][k] for k in ("public_rank","public_diameter","public_radius","fold_dimension")}, permutation_file=f'units/{unit["geometry_id"]}.npz', geometry_file=f'units/{unit["geometry_id"]}.geometry.npz')
    def clean(row):
        return {k:("" if isinstance(v,float) and not np.isfinite(v) else v) for k,v in row.items()}
    with gzip.open(tmp,"wt",newline="",compresslevel=4) as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_FIELDS)
        for ip,p in enumerate((1,2)):
            base = dict(common, objective_p=p, optimum_total_cost=meta["optima"][str(p)], uniform_expected_total_cost=meta["random_expected"][str(p)])
            writer.writerow(dict(base, mechanism="OPT", total_cost=meta["optima"][str(p)],total_regret=0., permutation_array="optimum_permutations",permutation_index=str(ip), privacy_output="NONPRIVATE_REFERENCE"))
            for draw in range(cfg["K"]):
                writer.writerow(dict(base,mechanism="uniform_random",noise_draw=draw,noise_coupling_id=f'{cfg["phase"]}:{unit["geometry_id"]}:random:{draw}',total_cost=arrays["random_cost"][ip,draw],total_regret=arrays["random_cost"][ip,draw]-meta["optima"][str(p)],permutation_array="random_permutations",permutation_index=str(draw),privacy_output="input_independent"))
            for ia,arm in enumerate(ARMS[p]):
                reuse = meta["reuse"][f"{p}:{arm}"]
                for ie,epsilon in enumerate(cfg["epsilon"]):
                    cert = meta["bounds"][f"{p}:{arm}"][ie]
                    for draw in range(cfg["K"]):
                        channel = reuse["source_arm"] or arm
                        row = dict(base,mechanism=arm,epsilon=epsilon,noise_draw=draw,noise_coupling_id=f'{cfg["phase"]}:{unit["geometry_id"]}:unit:{draw}:{channel}',**cert,**reuse,privacy_output="final_matching_only" if arm=="gumbel" else "local_report_and_matching",permutation_array="permutations",permutation_index=f"{ip},{ia},{ie},{draw}")
                        row.update({f:float(arrays[f][ip,ia,ie,draw]) for f in FLOAT_FIELDS})
                        writer.writerow(clean(row))
    os.replace(tmp,path)


def run_unit(directory, manifest, unit):
    if valid_checkpoint(directory,unit,manifest["fingerprint"]):
        print("REUSE",unit["geometry_id"],flush=True); return
    started = time.perf_counter()
    cfg = manifest["design"]; n,d = unit["n"],unit["d"]
    geo, extra = make_geometry(cfg,unit)
    shape = (2,7,len(cfg["epsilon"]),cfg["K"])
    arrays = {field:np.empty(shape) for field in FLOAT_FIELDS}
    arrays["permutations"] = np.empty((*shape,n),dtype=np.uint16)
    arrays["optimum_permutations"] = np.stack([solve(geo.C1),solve(geo.C2)]).astype(np.uint16)
    optima = {p:float((geo.C1 if p==1 else geo.C2)[np.arange(n),arrays["optimum_permutations"][ip]].sum()) for ip,p in enumerate((1,2))}
    rng = np.random.default_rng(seed(cfg["seed_namespace"],cfg["phase"],unit["geometry_id"],"random"))
    arrays["random_permutations"] = np.stack([rng.permutation(n) for _ in range(cfg["K"])]).astype(np.uint16)
    arrays["random_cost"] = np.stack([C[np.arange(n)[None,:],arrays["random_permutations"]].sum(axis=1) for C in (geo.C1,geo.C2)])
    prep_seconds = time.perf_counter()-started
    meta = dict(fingerprint=manifest["fingerprint"],geometry_id=unit["geometry_id"],unit=unit,public=geo.public_metadata(),optima={str(p):v for p,v in optima.items()},random_expected={str(p):float(C.sum()/n) for p,C in ((1,geo.C1),(2,geo.C2))},bounds={},reuse={},preparation_seconds=prep_seconds,channel_generation_seconds={},host=socket.gethostname(),slurm_job_id=os.environ.get("SLURM_JOB_ID"),slurm_array_task_id=os.environ.get("SLURM_ARRAY_TASK_ID"),seed_namespace=cfg["seed_namespace"])
    if unit["tau"] is not None:
        meta["planted_cost"]={str(p):float(C[np.arange(n),extra["planted"]].sum()) for p,C in ((1,geo.C1),(2,geo.C2))}
    for draw in range(cfg["K"]):
        def rng_for(channel):
            return np.random.default_rng(seed(cfg["seed_namespace"],cfg["phase"],unit["geometry_id"],"unit",draw,channel))
        channels,timings = unit_channels(geo,rng_for)
        for channel,seconds in timings.items():
            meta["channel_generation_seconds"][channel]=meta["channel_generation_seconds"].get(channel,0.)+seconds
        for ie,epsilon in enumerate(cfg["epsilon"]):
            results=evaluate_draw(geo,channels,epsilon,unit["family"],optima)
            for ip,p in enumerate((1,2)):
                for ia,arm in enumerate(ARMS[p]):
                    item=results[p,arm]
                    for field in FLOAT_FIELDS:
                        arrays[field][ip,ia,ie,draw]=item[field]
                    arrays["permutations"][ip,ia,ie,draw]=item["permutation"]
                    if draw==0:
                        meta["reuse"][f"{p}:{arm}"]={key:item[key] for key in ("source_arm","reuse_reason")}
                        cert = bound(geo,arm,p,epsilon,unit["family"])
                        meta["bounds"].setdefault(f"{p}:{arm}",[]).append({key:(None if isinstance(value,float) and not np.isfinite(value) else value) for key,value in cert.items()})
    if not np.all(np.diff(np.sort(arrays["permutations"],axis=-1),axis=-1)==1):
        raise RuntimeError("nonintegral or incomplete permutation")
    # Recompute selected costs independently from the saved permutations.
    for ip,C in enumerate((geo.C1,geo.C2)):
        rebuilt=C[np.arange(n),arrays["permutations"][ip]].sum(axis=-1)
        if not np.allclose(rebuilt,arrays["total_cost"][ip],rtol=1e-12,atol=1e-10):
            raise RuntimeError("saved permutation/cost mismatch")
    stem=directory/"units"/unit["geometry_id"]
    save_npz(Path(str(stem)+".geometry.npz"),**geo.arrays(),**extra)
    save_npz(Path(str(stem)+".npz"),**arrays)
    write_raw(Path(str(stem)+".csv.gz"),meta,arrays,cfg)
    meta["elapsed_seconds"]=time.perf_counter()-started
    meta["raw_rows"]=2*(1+cfg["K"]+7*len(cfg["epsilon"])*cfg["K"])
    meta["actual_solves"]=2+sum(cfg["K"]*len(cfg["epsilon"]) for v in meta["reuse"].values() if not v["source_arm"])
    meta["files"]={}
    for suffix in (".npz",".geometry.npz",".csv.gz"):
        file=Path(str(stem)+suffix)
        meta["files"][suffix]=dict(bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    atomic_json(Path(str(stem)+".json"),meta)
    print(f'COMPLETE {unit["geometry_id"]} {meta["elapsed_seconds"]:.3f}s rows={meta["raw_rows"]}',flush=True)


def run_study(results_root, pilot=False, stage="all", shard_index=None, shard_count=None):
    refuse_login_compute("run the integral matching study")
    directory,manifest=resolved(results_root,pilot)
    cfg=manifest["design"]
    print(f"Resolved integral matching results: {directory}",flush=True)
    if stage in ("prepare","all"):
        directory.mkdir(parents=True,exist_ok=True)
        (directory/"units").mkdir(exist_ok=True)
        path=directory/"manifest.json"
        if path.exists() and json.loads(path.read_text())!=manifest:
            raise RuntimeError("manifest conflict")
        atomic_json(path,manifest)
        atomic_json(directory/"expected_units.json",units(cfg))
    if not (directory/"manifest.json").exists():
        raise RuntimeError("prepare stage required")
    if stage in ("all","shard"):
        index=0 if shard_index is None else shard_index
        count=1 if shard_count is None else shard_count
        if not 0<=index<count:
            raise ValueError("invalid shard")
        # Largest first, greedy balanced deterministic bins. Whole geometries
        # retain their independent draws and are never divided across writers.
        bins=[[] for _ in range(count)]; loads=np.zeros(count)
        ordered=sorted(units(cfg),key=lambda u:u["n"]**3+4*u["n"]**2*u["d"],reverse=True)
        for unit in ordered:
            target=int(loads.argmin()); bins[target].append(unit)
            loads[target]+=unit["n"]**3+4*unit["n"]**2*unit["d"]
        for unit in bins[index]:
            run_unit(directory,manifest,unit)
    if stage in ("all","finalize"):
        from private_matching.integral_report import finalize
        finalize(directory)
    return directory
