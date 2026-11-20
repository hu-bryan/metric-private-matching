"""Normalized Fashion-MNIST Haar-flag dimension study with archived raw inputs."""
from __future__ import annotations
import hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from . import clustering_manuscript as old
from .clustering_benchmark import sha256
from .clustering_normalized import runtime,source_hashes
from .integral_mechanisms import Geometry,solve
from .integral_study import atomic_json,save_npz,seed
from .login_guard import refuse_login_compute
from .mnist_random_projection import haar_frame,verify_runtime,chain_input

NAME="fashion_dimension_v1"
CALIBRATED_NAME="fashion_dimension_epsilon20_v1"
REFERENCE=Path("/ix1/arahimian/brh314/metric_private_covariate_matching/results/fashion_dimension_v1/production/f9ff7281c5575e912d29b9af")
PILOT_REFERENCE=Path("/vast/arahimian/brh314/metric_private_covariate_matching/results/fashion_dimension_v1/pilot/21a792581597c3acd342df59")
NAMESPACE="fashion-normalized-full-haar-dimension-20261114-v1"
SOURCE=Path("/ix1/arahimian/brh314/metric_private_covariate_matching/results/internal_projection_bound_search_20261114/validation/mnist_fashion10k_n800")
ARMS=old.ARMS
DIMENSIONS=sorted(set(range(204,3,-5))|set(range(1,13))|{254,304,404,604,784},reverse=True)

def design(pilot=False,calibrated=False,epsilon=None):
    cfg=dict(phase="pilot" if pilot else "production",parents=1 if pilot else 20,
        chains=1 if pilot else 5,partitions=1 if pilot else 5,K=3 if pilot else 50,
        n=800,D=784,epsilon=8.,dimensions=[784,104,4] if pilot else DIMENSIONS,
        blocks=1 if pilot else 4,namespace=NAMESPACE,normalize_projection=True,
        sampling="1600 distinct archived Fashion-MNIST source rows per parent; five saved disjoint 800/800 partitions",
        projection="full 784x784 positive-diagonal Gaussian QR Haar flags; nested prefixes, equivalent to successive independent orthogonal reductions",
        scaling="each pooled projected cloud centered and divided by its actual RMS; intermediate scalar rescalings cancel at next RMS normalization",
        privacy_metric="sum-record Euclidean in each fixed RMS-one projected space; offline benchmark construction",
        reuse="raw masters and partition orders from prior geometry-only validation, with explicit hashes; all flags and mechanism draws fresh; no pilot draws imported",
        inference="noise -> partitions -> chains -> twenty parent means; pointwise Student-t95%",
        hardware_constraint="genoa",display_max_dimension=104)
    budget=float(epsilon) if epsilon is not None else (20. if calibrated else 8.)
    if not np.isfinite(budget) or budget<=0:raise ValueError("epsilon must be finite and positive")
    cfg["epsilon"]=budget
    if calibrated or budget!=8.:
        cfg.update(geometry_reference=str(PILOT_REFERENCE if pilot else REFERENCE),
            reuse=f"identical epsilon-8 raw masters, transforms, partitions and full Haar flags; same per-instance unit-noise seeds explicitly coupled across budgets; all epsilon-{budget:g} assignments newly solved",
            calibration="exact saved OPT/RAND; report nonvacuity at the requested budget without requiring it at every dimension")
    return cfg

def tasks(cfg):
    return ([dict(kind="basis",parent=p,chain=c) for p in range(cfg["parents"]) for c in range(cfg["chains"])]+
        [dict(kind="block",parent=p,chain=c,block=b) for p in range(cfg["parents"]) for c in range(cfg["chains"]) for b in range(cfg["blocks"])])

def prepare(results_root,pilot=False,calibrated=False,epsilon=None):
    refuse_login_compute("prepare Fashion dimension study")
    cfg=design(pilot,calibrated,epsilon)
    sources={str(p):dict(file=str(SOURCE/f"parent{p:02d}.npz"),sha256=sha256(SOURCE/f"parent{p:02d}.npz")) for p in range(cfg["parents"])}
    kind=NAME if cfg["epsilon"]==8. else "fashion_dimension_epsilon"+format(cfg["epsilon"],"g").replace(".","p")+"_v1"
    payload=dict(kind=kind,design=cfg,source_sha256=source_hashes(),versions=runtime(),source=sources)
    if cfg.get("geometry_reference"):payload["reference_manifest_sha256"]=sha256(Path(cfg["geometry_reference"])/"manifest.json")
    fp=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:24];payload["fingerprint"]=fp
    root=Path(results_root)/kind/cfg["phase"]/fp
    for name in ("masters","bases","units","chunks","analysis_handoff"):(root/name).mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists():assert json.loads((root/"manifest.json").read_text())==payload
    atomic_json(root/"manifest.json",payload);registry={}
    for p,record in sources.items():
        with np.load(record["file"]) as z:
            raw=z["raw"];v=(raw-z["feature_offset"])/z["feature_scale"]
            center=v.mean(axis=0);radius=np.sqrt(np.mean(np.sum((v-center)**2,axis=1)))
            Z=(v-center)/radius;np.testing.assert_allclose(Z,z["Z"],rtol=2e-14,atol=2e-14)
            path=root/"masters"/f"parent{int(p):02d}.npz"
            save_npz(path,raw=raw,Z=Z,source_ids=z["source_ids"],orders=z["orders"],
                feature_offset=z["feature_offset"],feature_scale=z["feature_scale"],
                original_center=center,original_radius=np.array(radius),pixel_order=np.arange(784))
        if cfg.get("geometry_reference"):
            reference=Path(cfg["geometry_reference"])/"masters"/path.name
            with np.load(path) as actual,np.load(reference) as expected:
                assert set(actual.files)==set(expected.files)
                for key in actual.files:np.testing.assert_array_equal(actual[key],expected[key])
            shutil.copyfile(reference,path)
        registry[p]=dict(file="masters/"+path.name,sha256=sha256(path),source=record)
    atomic_json(root/"master_registry.json",registry)
    print("Resolved Fashion dimension results:",root,flush=True);return root,payload

def make_basis(root,manifest,parent,chain,regenerate=False):
    cfg=manifest["design"];root=Path(root);stem=root/"bases"/f"p{parent:02d}_c{chain}"
    record=json.loads((root/"master_registry.json").read_text())[str(parent)]
    assert sha256(root/record["file"])==record["sha256"]
    with np.load(root/record["file"]) as z:Z=z["Z"]
    if cfg.get("geometry_reference") and not regenerate:
        reference=Path(cfg["geometry_reference"])/"bases"/stem.name
        refmeta=json.loads(reference.with_suffix(".json").read_text())
        assert sha256(reference.with_suffix(".npz"))==refmeta["sha256"]
        assert record["sha256"]==refmeta["master_sha256"]
        shutil.copyfile(reference.with_suffix(".npz"),stem.with_suffix(".npz"))
        meta=dict(refmeta,fingerprint=manifest["fingerprint"],reference_basis=str(reference.with_suffix(".npz")))
        atomic_json(stem.with_suffix(".json"),meta);return meta
    O=haar_frame(784,784,np.random.default_rng(seed(NAMESPACE,cfg["phase"],parent,chain,"basis")))
    error=float(np.max(np.abs(O.T@O-np.eye(784))));assert error<2e-12
    projected=Z@O;center=projected.mean(axis=0);W=projected-center
    radii=np.array([np.sqrt(np.mean(np.sum(W[:,:d]**2,axis=1))) for d in cfg["dimensions"]])
    assert np.all(radii>0)
    save_npz(stem.with_suffix(".npz"),O=O,W=W,projected_center=center,radii=radii)
    meta=dict(fingerprint=manifest["fingerprint"],parent=parent,chain=chain,sha256=sha256(stem.with_suffix(".npz")),
        master_sha256=record["sha256"],orthogonality_error=error)
    atomic_json(stem.with_suffix(".json"),meta);return meta

def inputs(data,cfg,d,part):
    radius=float(data["radii"][cfg["dimensions"].index(d)])
    qi,ri=np.split(data["orders"][part],2);n=cfg["n"]
    assert len(qi)==len(ri)==n and len(np.intersect1d(data["ids"][qi],data["ids"][ri]))==0
    Q=data["W"][qi,:d]/radius;R=data["W"][ri,:d]/radius
    assert np.allclose(np.r_[Q,R].mean(axis=0),0,atol=2e-12)
    assert np.isclose(np.mean(np.sum(np.r_[Q,R]**2,axis=1)),1.,atol=2e-12)
    return Q,R,radius

def draw_channels(g,cfg,instance,draw):
    n,d=g.X.shape
    rng=lambda arm:np.random.default_rng(seed(NAMESPACE,cfg["phase"],instance,draw,arm))
    channels=dict(original_points=old.sample_euclidean_laplace(n,d,1.,rng("original_points"))[0],
        gumbel=rng("gumbel").gumbel(size=(n,n)),distance_vectors=old.supnorm_unit(n,rng("distance_vectors")))
    s=min(g.rank+1,d)
    if s<d:channels["projected_representation"]=old.sample_euclidean_laplace(n,s,1.,rng("projected_representation"))[0]
    return channels

def run_instance(root,manifest,parent,chain,d,part,data=None,force=False):
    cfg=manifest["design"];root=Path(root);instance=f"p{parent:02d}_c{chain}_d{d:03d}_part{part}";stem=root/"units"/instance
    if stem.with_suffix(".json").exists() and not force:
        meta=json.loads(stem.with_suffix(".json").read_text());assert meta["fingerprint"]==manifest["fingerprint"] and sha256(stem.with_suffix(".npz"))==meta["sha256"];return meta
    started=time.perf_counter();data=data or chain_input(root,manifest,parent,chain)
    Q,R,radius=inputs(data,cfg,d,part);g=Geometry.build(R,Q);n=cfg["n"];s=min(g.rank+1,d);alias=s==d
    optimum_pi=solve(g.C1);optimum=float(g.C1[np.arange(n),optimum_pi].sum());random=float(g.C1.sum()/n)
    arrays={k:np.empty((4,cfg["K"])) for k in ("total_cost","regret","row_oscillation")}
    arrays["permutations"]=np.empty((4,cfg["K"],n),dtype=np.uint16)
    for draw in range(cfg["K"]):
        channels=draw_channels(g,cfg,instance,draw);cache=old.cached_reports(g,channels)
        for ia,arm in enumerate(ARMS):
            if arm=="projected_representation" and alias:
                for field in arrays:arrays[field][ia,draw]=arrays[field][1,draw]
                continue
            noisy,base=old.reconstructed_cost(g,channels,cache,arm,1,cfg["epsilon"])
            pi=solve(noisy);cost=float(g.C1[np.arange(n),pi].sum());osc=float(np.ptp(noisy-base,axis=1).sum())
            assert -1e-8<=cost-optimum<=osc+1e-7*max(1.,osc)
            arrays["total_cost"][ia,draw]=cost;arrays["regret"][ia,draw]=max(0.,cost-optimum)
            arrays["row_oscillation"][ia,draw]=osc;arrays["permutations"][ia,draw]=pi
    arrays.update(optimum=np.array(optimum),RAND=np.array(random),optimum_permutation=optimum_pi.astype(np.uint16))
    save_npz(stem.with_suffix(".npz"),**arrays);lam=g.singular**2
    meta=dict(fingerprint=manifest["fingerprint"],instance_id=instance,parent=parent,chain=chain,d=d,partition=part,n=n,K=cfg["K"],epsilon=cfg["epsilon"],sha256=sha256(stem.with_suffix(".npz")),
        basis_sha256=data["basis_sha256"],Q_sha256=old.array_sha(Q),R_sha256=old.array_sha(R),projected_RMS=radius,
        expected_isometry_multiplier=float(np.sqrt(784/d)),prefinal_RMS=float(radius*np.sqrt(784/d)),retained_energy=radius**2,
        input_RMS=1.,total_operator_norm=1/radius,public_rank=g.rank,fold_dimension=s,effective_rank=float(lam.sum()**2/(lam@lam)),
        alias_projection_to_original=alias,actual_assignment_solves=(3 if alias else 4)*cfg["K"],elapsed_seconds=time.perf_counter()-started)
    if cfg.get("geometry_reference"):
        reference=Path(cfg["geometry_reference"])/"units"/(instance+".json")
        refmeta=json.loads(reference.read_text())
        assert meta["Q_sha256"]==refmeta["Q_sha256"] and meta["R_sha256"]==refmeta["R_sha256"]
        meta.update(reference_unit_sha256=refmeta["sha256"],reference_unit=str(reference.with_suffix(".npz")),unit_noise_coupled_to_epsilon8=True)
    atomic_json(stem.with_suffix(".json"),meta);return meta

def run_block(root,manifest,parent,chain,block):
    cfg=manifest["design"];data=chain_input(root,manifest,parent,chain)
    dimensions=np.array_split(cfg["dimensions"],cfg["blocks"])[block].tolist()
    metas=[run_instance(root,manifest,parent,chain,d,p,data) for d in dimensions for p in range(cfg["partitions"])]
    from .fashion_dimension_report import export_block
    export_block(Path(root),manifest,parent,chain,block,metas,data)
    print("COMPLETE BLOCK",parent,chain,block,len(metas),flush=True)

def run_study(results_root,pilot=False,stage="all",shard_index=None,shard_count=None,directory=None,calibrated=False,epsilon=None):
    refuse_login_compute("Fashion dimension study")
    if stage in ("all","prepare"):root,manifest=prepare(results_root,pilot,calibrated,epsilon)
    else:root=Path(directory);manifest=json.loads((root/"manifest.json").read_text())
    verify_runtime(manifest)
    if stage in ("all","shard"):
        for task in tasks(manifest["design"])[(shard_index or 0)::(shard_count or 1)]:
            if task["kind"]=="basis":make_basis(root,manifest,task["parent"],task["chain"])
            else:run_block(root,manifest,task["parent"],task["chain"],task["block"])
    if stage in ("all","finalize"):
        from .fashion_dimension_report import export
        export(root,root/"analysis_handoff")
    return root
