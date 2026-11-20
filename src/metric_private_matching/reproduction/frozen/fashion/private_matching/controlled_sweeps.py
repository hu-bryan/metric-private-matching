"""Fixed-distance ambient sweeps and matched-circle cardinality sweeps."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist
from . import clustering_manuscript as manuscript
from .clustering_benchmark import sha256, spectrum
from .clustering_normalized import runtime, source_hashes
from .integral_mechanisms import Geometry, solve, supnorm_unit
from .integral_study import atomic_json, save_npz, seed
from .login_guard import refuse_login_compute
from .mechanisms import sample_euclidean_laplace

NAME = "controlled_circle_sweeps_v1"
NAMESPACE = "controlled-circle-20261114-v1"
ARMS = manuscript.ARMS
DIMENSIONS = (2,3,4,5,6,8,10,12,16,24,32,48,64,96,128)
SIZES = (4,6,8,12,16,20,24,32,40,48,56,64,96,128,192,256,384,512)


def design(pilot=False):
    ds = (2,3,4,8,128) if pilot else DIMENSIONS
    ns = (4,16,56,64,512) if pilot else SIZES
    pairs = [(64,3)]+[(64,d) for d in ds if d != 3]+[(n,4) for n in ns if n != 64]
    settings = [dict(setting_id=f"n{n}_d{d}", n=n, d=d,
                     panels=(["dimension"] if n == 64 else [])+(["sample_size"] if d == 4 else [])) for n,d in pairs]
    return dict(phase="pilot" if pilot else "production", geometries=2 if pilot else 20,
                K=3 if pilot else 50, epsilon=[4.,8.], settings=settings, dimensions=list(ds), sizes=list(ns),
                namespace=NAMESPACE, intrinsic_rank=2, normalization="unit circle; exact antipodal pairs; no dimension-dependent refit",
                objective="sum unsquared Euclidean", privacy_metric="sum-record Euclidean",
                sampling="synthetic unique points; private cloud is independently permuted public cloud; OPT=0",
                construction="antipodal stratified angles with uniform [-0.2,0.2] bin jitter; same jitter master over n; n sets not nested",
                coupling="shared geometry across n/d; same unit noise across epsilon; dimension-invariant arms copied from n64_d3",
                inference="K conditional noise means, then Student-t95% over independent geometry masters",
                hardware_constraint="rome")


def units(cfg):
    return [dict(setting=s, rep=rep, unit_id=s["setting_id"]+f"_g{rep:02d}")
            for s in cfg["settings"] for rep in range(cfg["geometries"])]


def master_arrays(cfg, rep):
    rng = np.random.default_rng(seed(NAMESPACE, cfg["phase"], rep, "geometry"))
    jitter = rng.uniform(-.2,.2,256)
    phase = float(rng.uniform(0,2*np.pi))
    arrays = dict(jitter=jitter, phase=np.array(phase))
    for n in sorted(set(cfg["sizes"]+[64])):
        m = n//2
        angles = np.pi*(np.arange(m)+.5+jitter[:m])/m+phase
        half = np.c_[np.cos(angles), np.sin(angles)]
        arrays[f"points_{n}"] = np.r_[half,-half]
        for side in ("q", "r"):
            arrays[f"{side}_order_{n}"] = np.random.default_rng(seed(NAMESPACE,cfg["phase"],rep,n,side)).permutation(n)
    return arrays


def prepare(results_root, pilot=False):
    refuse_login_compute("prepare controlled sweeps")
    cfg = design(pilot)
    payload = dict(kind=NAME, design=cfg, source_sha256=source_hashes(), versions=runtime())
    fp = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:24]
    payload["fingerprint"] = fp
    root = Path(results_root)/NAME/cfg["phase"]/fp
    for directory in ("masters","units","analysis_handoff"):
        (root/directory).mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists():
        assert json.loads((root/"manifest.json").read_text()) == payload
    atomic_json(root/"manifest.json",payload)
    registry = {}
    for rep in range(cfg["geometries"]):
        path = root/"masters"/f"geometry_{rep:02d}.npz"
        save_npz(path,**master_arrays(cfg,rep))
        registry[str(rep)] = dict(file="masters/"+path.name,sha256=sha256(path))
    atomic_json(root/"master_registry.json",registry)
    print("Resolved controlled sweep results:",root,flush=True)
    return root,payload


def verify_runtime(manifest):
    assert source_hashes() == manifest["source_sha256"], "frozen source drift"
    assert runtime() == manifest["versions"], "runtime/CPU/thread drift"


def inputs(root, unit):
    n,d = unit["setting"]["n"],unit["setting"]["d"]
    record = json.loads((Path(root)/"master_registry.json").read_text())[str(unit["rep"])]
    assert sha256(Path(root)/record["file"]) == record["sha256"]
    with np.load(Path(root)/record["file"]) as saved:
        points = saved[f"points_{n}"]
        Q = np.pad(points[saved[f"q_order_{n}"]],((0,0),(0,d-2)))
        R = np.pad(points[saved[f"r_order_{n}"]],((0,0),(0,d-2)))
    return Q,R


def channels(g, cfg, unit_id, draw):
    n,d = g.X.shape
    rng = lambda arm: np.random.default_rng(seed(NAMESPACE,cfg["phase"],unit_id,draw,arm))
    values = dict(original_points=sample_euclidean_laplace(n,d,1.,rng("original_points"))[0],
                  gumbel=rng("gumbel").gumbel(size=(n,n)),distance_vectors=supnorm_unit(n,rng("distance_vectors")))
    if min(g.rank+1,d)<d:
        values["projected_representation"] = sample_euclidean_laplace(n,g.rank+1,1.,rng("projected_representation"))[0]
    return values


def run_unit(root, manifest, unit, force=False):
    root = Path(root); cfg=manifest["design"]; setting=unit["setting"]
    stem = root/"units"/unit["unit_id"]
    if stem.with_suffix(".json").exists() and not force:
        meta = json.loads(stem.with_suffix(".json").read_text())
        assert meta["fingerprint"] == manifest["fingerprint"] and sha256(stem.with_suffix(".npz")) == meta["sha256"]
        return meta
    Q,R=inputs(root,unit); n,d=Q.shape; g=Geometry.build(R,Q)
    C=cdist(R,Q); optimum_pi=solve(C); optimum=float(C[np.arange(n),optimum_pi].sum())
    assert optimum == 0 and g.rank == 2
    assert np.allclose(np.r_[Q,R].mean(axis=0),0,atol=1e-14)
    assert np.allclose(np.sum(Q*Q,axis=1),1,atol=1e-14)
    assert np.isclose(g.diameter,2,atol=1e-14)
    shape=(len(cfg["epsilon"]),4,cfg["K"])
    arrays={name:np.empty(shape) for name in ("total_cost","regret","row_oscillation")}
    arrays["permutations"]=np.empty((*shape,n),dtype=np.uint16)
    arrays.update(optimum=np.array(optimum),RAND=np.array(C.sum()/n),optimum_permutation=optimum_pi.astype(np.uint16))
    aliases={}
    reference = n==64 and d!=3
    if reference:
        path=root/"units"/f"n64_d3_g{unit['rep']:02d}.npz"
        refmeta=json.loads(path.with_suffix(".json").read_text())
        assert sha256(path)==refmeta["sha256"]
        with np.load(path) as z:
            ref={name:z[name] for name in z.files}
        rq,rr=inputs(root,dict(unit,setting=dict(setting,d=3)))
        np.testing.assert_array_equal(C,cdist(rr,rq))
        for arm in ("gumbel","distance_vectors")+(("projected_representation",) if d>=3 else ()):
            ia=ARMS.index(arm); source=1 if arm=="projected_representation" else ia
            for name in ("total_cost","regret","row_oscillation","permutations"):
                arrays[name][:,ia]=ref[name][:,source]
            aliases[arm]=dict(unit=path.stem,arm=ARMS[source],reason="same costs/calibration" if source==ia else "isometric full 3D projected report")
    solves=0
    for draw in range(cfg["K"]):
        noise=channels(g,cfg,unit["unit_id"],draw)
        cache=manuscript.cached_reports(g,noise)
        for ie,epsilon in enumerate(cfg["epsilon"]):
            for ia,arm in enumerate(ARMS):
                if arm in aliases:
                    continue
                if arm=="projected_representation" and min(g.rank+1,d)==d:
                    for name in ("total_cost","regret","row_oscillation","permutations"):
                        arrays[name][ie,ia,draw]=arrays[name][ie,1,draw]
                    aliases.setdefault("original_endpoint",dict(arm="original_points",reason="full dimension endpoint"))
                    continue
                noisy,baseline=manuscript.reconstructed_cost(g,noise,cache,arm,1,epsilon)
                pi=solve(noisy); solves+=1
                cost=float(C[np.arange(n),pi].sum())
                osc=float(np.ptp(noisy-baseline,axis=1).sum())
                assert cost-optimum <= osc+1e-8*max(1.,osc)
                arrays["permutations"][ie,ia,draw]=pi
                arrays["total_cost"][ie,ia,draw]=cost
                arrays["regret"][ie,ia,draw]=cost-optimum
                arrays["row_oscillation"][ie,ia,draw]=osc
    factor=np.array([np.log(n),d,min(g.rank+1,d),n])
    arrays["bound_factor"]=factor
    arrays["bound_uncapped"]=np.array([2*n*factor/e for e in cfg["epsilon"]])
    arrays["bound"]=np.minimum(n*g.diameter,arrays["bound_uncapped"])
    save_npz(stem.with_suffix(".npz"),**arrays)
    meta=dict(fingerprint=manifest["fingerprint"],unit=unit,sha256=sha256(stem.with_suffix(".npz")),aliases=aliases,
              actual_assignment_solves=solves,public=g.public_metadata(),effective_rank=spectrum(Q)["effective_rank"],
              Q_sha256=manuscript.array_sha(Q),R_sha256=manuscript.array_sha(R),hardware=manuscript.hardware())
    atomic_json(stem.with_suffix(".json"),meta)
    print("COMPLETE",unit["unit_id"],"solves",solves,flush=True)
    return meta


def run_study(results_root,pilot=False,stage="all",shard_index=None,shard_count=None,directory=None):
    refuse_login_compute("controlled sweeps")
    if stage in ("all","prepare"):
        root,manifest=prepare(results_root,pilot)
    else:
        root=Path(directory);manifest=json.loads((root/"manifest.json").read_text())
    verify_runtime(manifest)
    if stage in ("all","shard"):
        for unit in units(manifest["design"])[(shard_index or 0)::(shard_count or 1)]:
            run_unit(root,manifest,unit)
    if stage in ("all","finalize"):
        from .controlled_sweeps_report import export
        export(root,root/"analysis_handoff")
    return root
