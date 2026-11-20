"""Five independent Haar flags for the low-dimensional MNIST sweep."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import shutil, time
import numpy as np
from . import clustering_manuscript as old
from .clustering_benchmark import sha256
from .clustering_normalized import runtime, source_hashes
from .integral_mechanisms import Geometry, solve
from .integral_study import atomic_json, save_npz, seed
from .login_guard import refuse_login_compute

NAME="mnist_random_projection_v1"
NAMESPACE="mnist-haar-flags-20261114-v1"
UNNORMALIZED_NAME="mnist_random_projection_unnormalized_v1"
UNNORMALIZED_NAMESPACE="mnist-haar-no-rescaling-20261114-v1"
REFERENCE=Path("/vast/arahimian/brh314/metric_private_covariate_matching/results/mnist_random_projection_v1/production/5fab1ee69f374dd87793b765")
SOURCE=Path("/vast/arahimian/brh314/metric_private_covariate_matching/results/clustering_manuscript_v1/production/26612223b86b05d1a5cf1ff9")
DIMENSIONS=tuple(range(204,3,-5))
ARMS=old.ARMS


def design(pilot=False,unnormalized=False):
    cfg=dict(phase="pilot" if pilot else "production",parents=1 if pilot else 20,
        chains=2 if pilot else 5,partitions=1 if pilot else 5,K=3 if pilot else 50,n=400,D=784,epsilon=8.,
        dimensions=[204,104,4] if pilot else list(DIMENSIONS),blocks=1 if pilot else 2,
        sampling="same archived 800 distinct source rows and five disjoint 400/400 partitions per parent",
        projection="independent thin Gaussian QR with positive diagonal; first d columns of one 784x204 Haar frame per parent/chain",
        scaling="center projected pooled inputs and divide by actual pooled RMS; expected isometry factor sqrt(784/d) cancels with final RMS",
        privacy_metric="sum-record Euclidean in each fixed normalized projected space; offline input construction",
        namespace=NAMESPACE,inference="noise -> partitions -> chains -> independent parents; pointwise Student-t95%",
        unused_dimensions="Haar-flag equivalence to nested reductions 784,779,...,204,...,4; do not simulate unplotted d>204",
        hardware_constraint="rome")
    if unnormalized:
        cfg.update(normalize_projection=False,namespace=UNNORMALIZED_NAMESPACE,
            basis_namespace=NAMESPACE,
            scaling="retain original RMS-one 784-dimensional master; centered nested orthogonal prefixes only; no post-projection RMS or expected-isometry multiplier",
            privacy_metric="sum-record Euclidean in fixed original-master units; orthogonal projection is nonexpansive",
            normalized_reference=str(REFERENCE),
            reuse="identical production masters, source IDs, partitions and Haar frames; fresh independent mechanism-noise namespace; no imported draws")
    return cfg


def tasks(cfg):
    bases=[dict(kind="basis",parent=p,chain=c) for p in range(cfg["parents"]) for c in range(cfg["chains"])]
    return bases+[dict(kind="block",parent=p,chain=c,block=b) for p in range(cfg["parents"])
                 for c in range(cfg["chains"]) for b in range(cfg["blocks"])]


def source_records(cfg):
    registry=json.loads((SOURCE/"master_registry.json").read_text())
    return {str(p):registry[f"dimension_mnist_digits10k_parent{p:02d}"] for p in range(cfg["parents"])}


def prepare(results_root,pilot=False,unnormalized=False):
    refuse_login_compute("prepare random-projection MNIST")
    cfg=design(pilot,unnormalized)
    payload=dict(kind=UNNORMALIZED_NAME if unnormalized else NAME,design=cfg,source_sha256=source_hashes(),versions=runtime(),
                 source=dict(directory=str(SOURCE),manifest_sha256=sha256(SOURCE/"manifest.json"),masters=source_records(cfg)))
    if unnormalized:
        payload["source"]["normalized_reference"]=dict(directory=str(REFERENCE),
            manifest_sha256=sha256(REFERENCE/"manifest.json"),
            verification_sha256=sha256(REFERENCE/"analysis_handoff/verification.json"))
    fp=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:24];payload["fingerprint"]=fp
    root=Path(results_root)/payload["kind"]/cfg["phase"]/fp
    for name in ("masters","bases","units","chunks","analysis_handoff"):(root/name).mkdir(parents=True,exist_ok=True)
    if (root/"manifest.json").exists():assert json.loads((root/"manifest.json").read_text())==payload
    atomic_json(root/"manifest.json",payload)
    order=np.array(json.loads((SOURCE/"manifest.json").read_text())["pixel_order"])
    registry={}
    for parent,record in payload["source"]["masters"].items():
        source=SOURCE/record["file"];assert sha256(source)==record["sha256"]
        with np.load(source) as z:
            raw=z["raw"];feature=(raw-z["feature_offset"])/z["feature_scale"]
            values=feature[:,order];center=values.mean(axis=0)
            radius=float(np.sqrt(np.mean(np.sum((values-center)**2,axis=1))))
            normalized=(values-center)/radius
            path=root/"masters"/f"parent{int(parent):02d}.npz"
            save_npz(path,raw=raw,Z=normalized,source_ids=z["source_ids"],orders=z["orders"],
                     feature_offset=z["feature_offset"],feature_scale=z["feature_scale"],
                     original_center=center,original_radius=np.array(radius),pixel_order=order)
        if unnormalized:
            reference_registry=json.loads((REFERENCE/"master_registry.json").read_text())
            reference=REFERENCE/reference_registry[parent]["file"]
            assert sha256(reference)==reference_registry[parent]["sha256"]
            with np.load(path) as actual,np.load(reference) as expected:
                assert set(actual.files)==set(expected.files)
                for key in actual.files:np.testing.assert_array_equal(actual[key],expected[key])
            shutil.copyfile(reference,path)
        registry[parent]=dict(file="masters/"+path.name,sha256=sha256(path))
    atomic_json(root/"master_registry.json",registry)
    print("Resolved MNIST random-projection results:",root,flush=True)
    return root,payload


def verify_runtime(manifest):
    assert source_hashes()==manifest["source_sha256"],"frozen source drift"
    assert runtime()==manifest["versions"],"runtime/CPU/thread drift"


def haar_frame(D,k,rng):
    Q,R=np.linalg.qr(rng.normal(size=(D,k)),mode="reduced")
    Q*=np.where(np.diag(R)<0,-1.,1.)[None,:]
    return Q


def make_basis(root,manifest,parent,chain,verify_reference=True):
    cfg=manifest["design"];root=Path(root);stem=root/"bases"/f"p{parent:02d}_c{chain}"
    if stem.with_suffix(".json").exists():
        meta=json.loads(stem.with_suffix(".json").read_text());assert meta["fingerprint"]==manifest["fingerprint"]
        assert sha256(stem.with_suffix(".npz"))==meta["sha256"];return meta
    registry=json.loads((root/"master_registry.json").read_text())[str(parent)]
    assert sha256(root/registry["file"])==registry["sha256"]
    with np.load(root/registry["file"]) as z:Z=z["Z"]
    O=haar_frame(784,204,np.random.default_rng(seed(cfg.get("basis_namespace",NAMESPACE),cfg["phase"],parent,chain,"basis")))
    error=float(np.max(np.abs(O.T@O-np.eye(204))));assert error<2e-12
    transformed=Z@O;center=transformed.mean(axis=0);W=transformed-center
    radii=np.array([np.sqrt(np.mean(np.sum(W[:,:d]**2,axis=1))) for d in cfg["dimensions"]])
    assert np.all(radii>0)
    reference_basis=None
    if verify_reference and not cfg.get("normalize_projection",True) and cfg["phase"]=="production":
        reference=REFERENCE/"bases"/stem.name
        reference_meta=json.loads(reference.with_suffix(".json").read_text())
        assert sha256(reference.with_suffix(".npz"))==reference_meta["sha256"]
        with np.load(reference.with_suffix(".npz")) as z:
            for key,actual in dict(O=O,W=W,projected_center=center,radii=radii).items():
                np.testing.assert_array_equal(actual,z[key])
        reference_basis=dict(file=str(reference.with_suffix(".npz")),sha256=reference_meta["sha256"])
    save_npz(stem.with_suffix(".npz"),O=O,W=W,projected_center=center,radii=radii)
    meta=dict(fingerprint=manifest["fingerprint"],parent=parent,chain=chain,sha256=sha256(stem.with_suffix(".npz")),
              master_sha256=registry["sha256"],orthogonality_error=error)
    if reference_basis is not None:meta["normalized_reference_basis"]=reference_basis
    atomic_json(stem.with_suffix(".json"),meta);return meta


def chain_input(root,manifest,parent,chain):
    root=Path(root);stem=root/"bases"/f"p{parent:02d}_c{chain}"
    meta=json.loads(stem.with_suffix(".json").read_text())
    assert meta["fingerprint"]==manifest["fingerprint"] and sha256(stem.with_suffix(".npz"))==meta["sha256"]
    record=json.loads((root/"master_registry.json").read_text())[str(parent)]
    assert sha256(root/record["file"])==record["sha256"]
    with np.load(root/record["file"]) as z:orders=z["orders"];ids=z["source_ids"]
    with np.load(stem.with_suffix(".npz")) as z:W=z["W"];radii=z["radii"]
    return dict(W=W,radii=radii,orders=orders,ids=ids,basis_sha256=meta["sha256"])


def inputs(data,cfg,d,part):
    radius=float(data["radii"][cfg["dimensions"].index(d)])
    qi,ri=np.split(data["orders"][part],2)
    assert len(qi)==len(ri)==400 and len(np.intersect1d(data["ids"][qi],data["ids"][ri]))==0
    divisor=radius if cfg.get("normalize_projection",True) else 1.
    Q=data["W"][qi,:d]/divisor;R=data["W"][ri,:d]/divisor
    assert np.allclose(np.r_[Q,R].mean(axis=0),0,atol=2e-12)
    assert np.isclose(np.mean(np.sum(np.r_[Q,R]**2,axis=1)),(radius/divisor)**2,atol=2e-12)
    return Q,R,radius


def draw_channels(g,cfg,instance,draw):
    n,d=g.X.shape
    rng=lambda arm:np.random.default_rng(seed(cfg.get("namespace",NAMESPACE),cfg["phase"],instance,draw,arm))
    return dict(original_points=old.sample_euclidean_laplace(n,d,1.,rng("original_points"))[0],
                gumbel=rng("gumbel").gumbel(size=(n,n)),distance_vectors=old.supnorm_unit(n,rng("distance_vectors")))


def run_instance(root,manifest,parent,chain,d,part,data=None,force=False):
    cfg=manifest["design"];root=Path(root);instance=f"p{parent:02d}_c{chain}_d{d:03d}_part{part}"
    stem=root/"units"/instance
    if stem.with_suffix(".json").exists() and not force:
        meta=json.loads(stem.with_suffix(".json").read_text())
        assert meta["fingerprint"]==manifest["fingerprint"] and sha256(stem.with_suffix(".npz"))==meta["sha256"]
        return meta
    started=time.perf_counter();data=data or chain_input(root,manifest,parent,chain)
    Q,R,radius=inputs(data,cfg,d,part);g=Geometry.build(R,Q);n=400
    assert g.rank==d,"Generic random projections below n must retain full public affine rank"
    optimum_pi=solve(g.C1);optimum=float(g.C1[np.arange(n),optimum_pi].sum());random=float(g.C1.sum()/n)
    arrays={k:np.empty((4,cfg["K"])) for k in ("total_cost","regret","row_oscillation")}
    arrays["permutations"]=np.empty((4,cfg["K"],n),dtype=np.uint16)
    for draw in range(cfg["K"]):
        channels=draw_channels(g,cfg,instance,draw);cache=old.cached_reports(g,channels)
        for ia,arm in enumerate(ARMS):
            if arm=="projected_representation":
                for field in arrays:arrays[field][ia,draw]=arrays[field][1,draw]
                continue
            noisy,base=old.reconstructed_cost(g,channels,cache,arm,1,cfg["epsilon"])
            pi=solve(noisy);cost=float(g.C1[np.arange(n),pi].sum());osc=float(np.ptp(noisy-base,axis=1).sum())
            assert -1e-8<=cost-optimum<=osc+1e-7*max(1.,osc)
            arrays["total_cost"][ia,draw]=cost;arrays["regret"][ia,draw]=max(0.,cost-optimum)
            arrays["row_oscillation"][ia,draw]=osc;arrays["permutations"][ia,draw]=pi
    arrays.update(optimum=np.array(optimum),RAND=np.array(random),optimum_permutation=optimum_pi.astype(np.uint16))
    save_npz(stem.with_suffix(".npz"),**arrays)
    lam=g.singular**2;effective=float(lam.sum()**2/(lam@lam))
    meta=dict(fingerprint=manifest["fingerprint"],instance_id=instance,parent=parent,chain=chain,d=d,partition=part,
        n=n,K=cfg["K"],epsilon=cfg["epsilon"],sha256=sha256(stem.with_suffix(".npz")),
        basis_sha256=data["basis_sha256"],Q_sha256=old.array_sha(Q),R_sha256=old.array_sha(R),
        projected_RMS=radius,expected_isometry_multiplier=float(np.sqrt(784/d)),
        prefinal_RMS=float(radius*np.sqrt(784/d)),retained_energy=radius**2,
        input_RMS=1. if cfg.get("normalize_projection",True) else radius,
        total_operator_norm=1/radius if cfg.get("normalize_projection",True) else 1.,
        public_rank=d,fold_dimension=d,effective_rank=effective,alias_projection_to_original=True,
        actual_assignment_solves=3*cfg["K"],elapsed_seconds=time.perf_counter()-started)
    atomic_json(stem.with_suffix(".json"),meta);return meta


def run_block(root,manifest,parent,chain,block):
    cfg=manifest["design"];data=chain_input(root,manifest,parent,chain)
    dimensions=np.array_split(cfg["dimensions"],cfg["blocks"])[block].tolist()
    metas=[run_instance(root,manifest,parent,chain,d,part,data) for d in dimensions for part in range(cfg["partitions"])]
    from .mnist_random_projection_report import export_block
    export_block(Path(root),manifest,parent,chain,block,metas,data)
    print("COMPLETE BLOCK",parent,chain,block,len(metas),flush=True)


def run_study(results_root,pilot=False,stage="all",shard_index=None,shard_count=None,directory=None,unnormalized=False):
    refuse_login_compute("MNIST random projection study")
    if stage in ("all","prepare"):root,manifest=prepare(results_root,pilot,unnormalized)
    else:root=Path(directory);manifest=json.loads((root/"manifest.json").read_text())
    verify_runtime(manifest)
    if stage in ("all","shard"):
        for task in tasks(manifest["design"])[(shard_index or 0)::(shard_count or 1)]:
            if task["kind"]=="basis":make_basis(root,manifest,task["parent"],task["chain"])
            else:run_block(root,manifest,task["parent"],task["chain"],task["block"])
    if stage in ("all","finalize"):
        from .mnist_random_projection_report import export
        export(root,root/"analysis_handoff")
    return root
