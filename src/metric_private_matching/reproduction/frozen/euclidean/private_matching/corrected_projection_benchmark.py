"""Incremental, separately fingerprinted exact-paper arm on 15 saved benchmarks.

Historical executions are read-only. There are 20 parents x 5 partitions x
50 noise draws x 7 epsilon alternatives per dataset. Only the new mechanism
is sampled; baseline conditional means are imported with explicit provenance.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from . import clustering_benchmark as cb
from .clustering_benchmark_report import summarize
from .corrected_projection import CorrectedProjection
from .integral_mechanisms import solve
from .integral_study import atomic_json, save_npz, seed
from .login_guard import refuse_login_compute
from .mechanisms import sample_euclidean_laplace

NAME = "corrected_projection_benchmark_v1"
ARM = "corrected_projection"
NAMESPACE = "paper-corrected-projection-20261113-v1"
ARCHIVES = (
    (Path("/ix1/arahimian/brh314/metric_private_covariate_matching/results/clustering_benchmark_v1/production/975562db8834d0cbf8bb0683"), "aggregate"),
    (Path("/ix1/arahimian/brh314/metric_private_covariate_matching/results/clustering_benchmark_extension_v1/production/44f7a46b3e1370d516dc5819"), "real"),
)
EPSILONS = [.25,.5,1.,2.,4.,8.,16.]


def source_hashes():
    package = Path(__file__).parent
    return {p.name: cb.sha256(p) for p in sorted(package.glob("*.py"))}


def design(pilot=False):
    units, sources = [], []
    for directory,panel in ARCHIVES:
        manifest = json.loads((directory/"manifest.json").read_text())
        handoff = directory/"analysis_handoff"
        source = dict(directory=str(directory), panel=panel, fingerprint=manifest["fingerprint"],
                      manifest_sha256=cb.sha256(directory/"manifest.json"),
                      summary_sha256=cb.sha256(handoff/"summary.csv"),
                      parent_sha256=cb.sha256(handoff/"parent_summaries.csv.gz"))
        sources.append(source)
        for u in cb.all_units(manifest["design"]):
            if not any(m["panel"] == panel for m in u["memberships"]):
                continue
            if pilot and (u["parent"] != 0 or u["partition"] != 0):
                continue
            assert u["n"] == 64
            path = directory/"units"/(u["instance_id"]+".json")
            meta = json.loads(path.read_text())
            units.append(dict(**u, archive=str(directory), source_fingerprint=manifest["fingerprint"],
                              panel=panel, source_metadata_sha256=cb.sha256(path),
                              geometry_sha256=meta["files"][u["instance_id"]+".geometry.npz"]["sha256"]))
    assert len({u["dataset"] for u in units}) == 15
    assert len(units) == (15 if pilot else 1500)
    return dict(phase="pilot" if pilot else "production", parents=1 if pilot else 20,
                partitions=1 if pilot else 5,K=3 if pilot else 50,
                epsilon=[.25,4.,8.,16.] if pilot else EPSILONS,
                mechanism=ARM, objective_p=1,units=units,sources=sources,
                noise="full-m Euclidean Laplace plus independent scalar Laplace",
                representation="identity if m=d; otherwise public span plus private perpendicular norm",
                split="b_Q=min(m,sqrt(2*m*log(2*N))); epsilon_y:epsilon_s=2*sqrt(b_Q):1; N=all nonzero public pairs",
                composition="adaptive epsilon_y+epsilon_s=epsilon per private record; joint records use sum metric",
                coupling="unit noise reused across alternative epsilon simulations; no joint privacy claim across epsilon",
                inference="noise mean per instance; five partitions per parent; Student-t intervals over 20 parent means",
                normalization="exact saved standardized Q/R; no raw-data preprocessing privacy claim")


def verify_runtime(manifest):
    if source_hashes() != manifest["source_sha256"] or cb.environment() != manifest["versions"]:
        raise RuntimeError("frozen source/runtime drift")
    for s in manifest["design"]["sources"]:
        root = Path(s["directory"])
        for path,key in [(root/"manifest.json","manifest_sha256"),
                         (root/"analysis_handoff/summary.csv","summary_sha256"),
                         (root/"analysis_handoff/parent_summaries.csv.gz","parent_sha256")]:
            if cb.sha256(path) != s[key]:
                raise RuntimeError("historical source changed: "+str(path))


def noise_for(unit,phase,draw,m):
    rng = np.random.default_rng(seed(NAMESPACE,phase,unit["instance_id"],draw,"coordinate"))
    scalar_rng = np.random.default_rng(seed(NAMESPACE,phase,unit["instance_id"],draw,"scalar"))
    return sample_euclidean_laplace(unit["n"],m,1.,rng)[0],scalar_rng.laplace(size=unit["n"])


def valid_checkpoint(directory,u,fp):
    stem = directory/"units"/u["instance_id"]
    path = Path(str(stem)+".json")
    if not path.exists():
        return False
    meta = json.loads(path.read_text())
    if meta["fingerprint"] != fp:
        raise RuntimeError("checkpoint fingerprint conflict")
    for name,checksum in meta["files"].items():
        if cb.sha256(stem.parent/name) != checksum:
            raise RuntimeError("checkpoint checksum conflict")
    return True


def run_unit(directory,manifest,u):
    if valid_checkpoint(directory,u,manifest["fingerprint"]):
        return
    start = time.perf_counter()
    cfg = manifest["design"]
    origin = Path(u["archive"])/"units"/u["instance_id"]
    if cb.sha256(Path(str(origin)+".json")) != u["source_metadata_sha256"] or cb.sha256(Path(str(origin)+".geometry.npz")) != u["geometry_sha256"]:
        raise RuntimeError("historical geometry/metadata drift")
    oldmeta = json.loads(Path(str(origin)+".json").read_text())
    with np.load(str(origin)+".geometry.npz") as old:
        Q,R = old["Q"],old["R"]
        selected = {key:old[key] for key in ("q_source_ids","r_source_ids","feature_ids","pooled_center","source_radius")}
    g = CorrectedProjection.build(Q)
    assert g.rank == oldmeta["public"]["public_rank"]
    C = cdist(R,Q)
    optimum = float(C[np.arange(len(R)),solve(C)].sum())
    rand = float(C.sum()/len(Q)); gap = rand-optimum
    assert np.isclose(optimum,oldmeta["optima"]["1"],rtol=1e-12,atol=1e-10)
    assert np.isclose(rand,oldmeta["random_expected"]["1"],rtol=1e-12,atol=1e-10)
    assert gap > 64*np.finfo(float).eps*max(1.,abs(optimum),abs(rand))
    folded = g.transform(R)
    terminal_error = float(np.abs(cdist(folded,g.anchor.p)-C).max())
    assert terminal_error <= 1e-10*max(1.,C.max())
    shape = (len(cfg["epsilon"]),cfg["K"])
    arrays = {key:np.empty(shape) for key in ("total_cost","total_regret","score","row_oscillation","realization_bound")}
    arrays["permutations"] = np.empty((*shape,len(Q)),dtype=np.uint16)
    for draw in range(cfg["K"]):
        W,eta = noise_for(u,cfg["phase"],draw,g.dimension)
        for ie,eps in enumerate(cfg["epsilon"]):
            decoded,secant,_,_,_ = g.decode_folded(folded,W,eta,eps)
            pi = solve(decoded)
            cost = float(C[np.arange(len(R)),pi].sum()); regret = cost-optimum
            osc = np.ptp(decoded-C,axis=1)
            tol = 1e-8*max(1.,cost,secant.sum())
            assert regret >= -tol and regret <= osc.sum()+tol
            assert np.all(osc <= secant+1e-8*np.maximum(1.,secant))
            for key,value in dict(total_cost=cost,total_regret=regret,score=regret/gap,
                                  row_oscillation=osc.sum(),realization_bound=secant.sum()).items():
                arrays[key][ie,draw] = value
            arrays["permutations"][ie,draw] = pi
    stem = directory/"units"/u["instance_id"]
    save_npz(str(stem)+".npz",Q=Q,R=R,**selected,**arrays)
    pair_count = len(g.anchor.lengths)
    meta = dict(fingerprint=manifest["fingerprint"],unit=u,mechanism=ARM,objective_p=1,
                optimum=optimum,random_expected=rand,gap=gap,public_rank=g.rank,
                representation_dimension=g.dimension,identity=g.identity,
                rank_tolerance=g.tolerance,public_projection_residual=g.residual,
                terminal_distance_max_error=terminal_error,nonzero_public_pair_count=pair_count,
                directional_coefficient=g.anchor.B,
                budgets=[dict(epsilon=eps,epsilon_y=g.split(eps)[0],epsilon_s=g.split(eps)[1],
                              bound_total=min(len(Q)*oldmeta["public"]["public_diameter"],
                                              len(Q)*(2*np.sqrt(g.anchor.B)+1)**2/eps)) for eps in cfg["epsilon"]],
                source_normalization=oldmeta["normalization"],source_radius=oldmeta["source_radius"],
                diagnostics=oldmeta["diagnostics"],public=oldmeta["public"],
                files={stem.name+".npz":cb.sha256(Path(str(stem)+".npz"))},
                job_id=os.environ.get("SLURM_JOB_ID"),host=socket.gethostname(),
                elapsed_seconds=time.perf_counter()-start,private_assignment_solves=cfg["K"]*len(cfg["epsilon"]))
    atomic_json(str(stem)+".json",meta)
    print(f"COMPLETE {u['instance_id']} {meta['elapsed_seconds']:.3f}s",flush=True)


def export(directory,output):
    refuse_login_compute("export corrected-projection benchmarks")
    directory,output = Path(directory),Path(output)
    manifest = json.loads((directory/"manifest.json").read_text())
    verify_runtime(manifest)
    cfg = manifest["design"]
    output.mkdir(parents=True,exist_ok=True)
    (output/"units").mkdir(exist_ok=True)
    instances,metadata = [],[]
    raw_path = output/"analysis_table.csv.gz"
    raw_count = 0
    with gzip.open(str(raw_path)+".tmp","wt",newline="") as stream:
        writer = None
        for u in cfg["units"]:
            assert valid_checkpoint(directory,u,manifest["fingerprint"]), u["instance_id"]
            stem = directory/"units"/u["instance_id"]
            meta = json.loads(Path(str(stem)+".json").read_text())
            metadata.append(meta)
            for suffix in (".npz",".json"):
                destination = output/"units"/(u["instance_id"]+suffix)
                if not destination.exists():
                    # Hard links preserve a standalone handoff without storing
                    # checkpoint bytes twice; copying supports other filesystems.
                    try:
                        os.link(Path(str(stem)+suffix),destination)
                    except OSError:
                        shutil.copyfile(Path(str(stem)+suffix),destination)
            with np.load(str(stem)+".npz") as saved:
                C = cdist(saved["R"],saved["Q"])
                pi = saved["permutations"]
                assert np.all(np.sort(pi,axis=-1) == np.arange(u["n"]))
                rebuilt = C[np.arange(u["n"]),pi].sum(axis=-1)
                np.testing.assert_allclose(rebuilt,saved["total_cost"],rtol=1e-12,atol=1e-10)
                np.testing.assert_allclose(saved["total_regret"],rebuilt-meta["optimum"],rtol=1e-12,atol=1e-10)
                np.testing.assert_allclose(saved["score"],saved["total_regret"]/meta["gap"],rtol=1e-12,atol=1e-10)
                for ie,eps in enumerate(cfg["epsilon"]):
                    common = {key:u[key] for key in ("panel","setting_id","dataset","condition","n","d","parent","partition","instance_id","source_fingerprint")}
                    common.update(fingerprint=manifest["fingerprint"],mechanism=ARM,objective_p=1,epsilon=eps,
                                  optimum=meta["optimum"],random_expected=meta["random_expected"],gap=meta["gap"],
                                  epsilon_y=meta["budgets"][ie]["epsilon_y"],epsilon_s=meta["budgets"][ie]["epsilon_s"],
                                  representation_dimension=meta["representation_dimension"],
                                  outcome_file="units/"+u["instance_id"]+".npz",epsilon_index=ie)
                    instances.append(dict(common,score=float(saved["score"][ie].mean()),
                                          mean_total_regret=float(saved["total_regret"][ie].mean()),
                                          mean_total_cost=float(saved["total_cost"][ie].mean()),
                                          bound_total=meta["budgets"][ie]["bound_total"],K=cfg["K"],
                                          q_rank=meta["public_rank"],
                                          q_rank_ratio=meta["diagnostics"]["Q"]["effective_rank_over_rank"]))
                    for draw in range(cfg["K"]):
                        record = dict(common,draw=draw,**{key:float(saved[key][ie,draw]) for key in
                                      ("total_cost","total_regret","score","row_oscillation","realization_bound")})
                        if writer is None:
                            writer = csv.DictWriter(stream,fieldnames=list(record)); writer.writeheader()
                        writer.writerow(record); raw_count += 1
    os.replace(str(raw_path)+".tmp",raw_path)
    df = pd.DataFrame(instances)
    df.to_csv(output/"instance_summaries.csv.gz",index=False)
    keys = ["panel","setting_id","dataset","condition","n","d","objective_p","mechanism","epsilon"]
    metrics = ["score","mean_total_regret","mean_total_cost","bound_total","q_rank","q_rank_ratio"]
    parent = df.groupby(keys+["parent"])[metrics].mean().reset_index()
    counts = df.groupby(keys+["parent"]).size()
    assert (counts == cfg["partitions"]).all()
    parent["partition_count"] = counts.values
    parent.to_csv(output/"parent_summaries.csv.gz",index=False)
    summary = None
    for value in metrics:
        part = summarize(parent,keys,value)
        summary = part if summary is None else summary.merge(part,on=keys,validate="one_to_one")
    summary["source_fingerprint"] = manifest["fingerprint"]
    summary["source_summary"] = str(output/"summary.csv")
    summary.to_csv(output/"summary.csv",index=False)
    # Preserve old source identities while assembling comparisons. These rows
    # are existing conditional means, never inferred or pooled private draws.
    oldframes,oldparents = [],[]
    for source in cfg["sources"]:
        oldout = Path(source["directory"])/"analysis_handoff"
        evidence = output/"baseline_sources"/source["fingerprint"]
        evidence.mkdir(parents=True,exist_ok=True)
        for filename in ("summary.csv","parent_summaries.csv.gz","manifest.json"):
            shutil.copyfile(oldout/filename,evidence/filename)
        frame = pd.read_csv(oldout/"summary.csv")
        frame = frame[(frame.panel == source["panel"]) & (frame.objective_p == 1)].copy()
        frame["source_fingerprint"] = source["fingerprint"]
        frame["source_summary"] = str(oldout/"summary.csv")
        oldframes.append(frame)
        pp = pd.read_csv(oldout/"parent_summaries.csv.gz")
        pp = pp[(pp.panel == source["panel"]) & (pp.objective_p == 1)].copy()
        pp["source_fingerprint"] = source["fingerprint"]
        oldparents.append(pp)
    combined = pd.concat([*oldframes,summary],ignore_index=True)
    combined.to_csv(output/"comparison_summary.csv",index=False)
    comparisons = []
    oldparent = pd.concat(oldparents,ignore_index=True)
    pairkeys = ["dataset","epsilon","parent"]
    for arm in oldparent.mechanism.unique():
        pair = parent.merge(oldparent[oldparent.mechanism == arm][pairkeys+["score"]],on=pairkeys,suffixes=("_new","_old"),validate="one_to_one")
        pair["score_difference"] = pair.score_new-pair.score_old
        pair["comparison"] = ARM+" minus "+arm
        comparisons.append(summarize(pair,["dataset","epsilon","comparison"],"score_difference"))
    pd.concat(comparisons,ignore_index=True).to_csv(output/"paired_comparisons.csv",index=False)
    actual_rows = 0
    with gzip.open(raw_path,"rt") as stream:
        actual_rows = sum(1 for _ in stream)-1
    assert actual_rows == raw_count == len(cfg["units"])*cfg["K"]*len(cfg["epsilon"])
    assert len(summary) == 15*len(cfg["epsilon"])
    atomic_json(output/"manifest.json",manifest)
    validation = dict(status="passed",fingerprint=manifest["fingerprint"],datasets=15,instances=len(cfg["units"]),
                      actual_canonical_rows=actual_rows,private_assignment_solves=actual_rows,
                      every_saved_permutation_and_true_cost_verified=True,
                      partition_counts_verified=True,noise_partition_parent_aggregation=True,
                      historical_sources_unchanged=True,analysis_table_sha256=cb.sha256(raw_path),
                      hardware_hosts=sorted({m["host"] for m in metadata}),
                      job_ids=sorted({m["job_id"] for m in metadata}))
    atomic_json(output/"verification.json",validation)
    audit(directory,manifest,output)
    # Freeze implementation, tests, and plotting/launch tools for replay.
    snapshot = output/"source_snapshot"
    snapshot.mkdir(exist_ok=True)
    package_root = Path(__file__).parent
    shutil.copytree(package_root,snapshot/"src/private_matching",dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
    atomic_json(directory/"COMPLETE.json",validation)
    print(json.dumps(validation),flush=True)
    return output


def audit(directory,manifest,output):
    """Independent literal squared-distance reconstruction, all 15 datasets."""
    cfg = manifest["design"]; checked = 0; worst = 0.
    for u in cfg["units"]:
        if u["parent"] != 0 or u["partition"] != 0:
            continue
        with np.load(directory/"units"/(u["instance_id"]+".npz")) as saved:
            Q,R = saved["Q"],saved["R"]
            g = CorrectedProjection.build(Q); folded = g.transform(R)
            W,eta = noise_for(u,cfg["phase"],0,g.dimension)
            for ie,eps in enumerate(cfg["epsilon"]):
                ey,es = g.split(eps)
                sq = cdist(folded+W/ey,g.anchor.p)**2
                a = sq.argmin(axis=1)
                T = np.maximum(0.,np.linalg.norm(folded-g.anchor.p[a],axis=1)+eta/es)
                literal = np.sqrt(T[:,None]**2+np.maximum(0.,sq-sq[np.arange(u["n"]),a,None]))
                stable,_,aa,_,_ = g.decode_draw(R,W,eta,eps)
                assert np.array_equal(a,aa)
                np.testing.assert_allclose(literal,stable,rtol=2e-9,atol=2e-8)
                np.testing.assert_array_equal(solve(stable),saved["permutations"][ie,0])
                # Literal subtraction can round ties differently; compare its
                # objective against the stored optimum for the decoded costs.
                pi = solve(literal)
                stored = saved["permutations"][ie,0]
                np.testing.assert_allclose(literal[np.arange(u["n"]),pi].sum(),literal[np.arange(u["n"]),stored].sum(),rtol=1e-10,atol=1e-8)
                worst = max(worst,float(np.abs(literal-stable).max())); checked += 1
    assert checked == 15*len(cfg["epsilon"])
    atomic_json(output/"independent_decoder_audit.json",dict(status="passed",draws=checked,
                datasets=15,literal_formula="sqrt(T^2+cdist(Y,p)^2-cdist(Y,p_anchor)^2)",
                max_absolute_decoder_difference=worst,production_and_audit_not_pooled=True))


def run_study(root,pilot=False,stage="all",shard_index=None,shard_count=None,directory=None):
    refuse_login_compute("run exact corrected-projection benchmark")
    if stage in ("prepare","all"):
        cfg = design(pilot)
        identity = dict(kind=NAME,design=cfg,source_sha256=source_hashes(),versions=cb.environment())
        fp = hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
        directory = Path(root)/NAME/cfg["phase"]/fp
        directory.mkdir(parents=True,exist_ok=True); (directory/"units").mkdir(exist_ok=True)
        manifest = dict(identity,fingerprint=fp)
        atomic_json(directory/"manifest.json",manifest)
        print("Resolved corrected projection results: "+str(directory),flush=True)
        if stage == "prepare":
            return directory
    else:
        if directory is None:
            raise ValueError("shard/finalize require --directory")
        directory = Path(directory)
        manifest = json.loads((directory/"manifest.json").read_text())
    verify_runtime(manifest)
    if stage in ("all","shard"):
        if stage == "shard" and not (shard_count and shard_index is not None and 0 <= shard_index < shard_count):
            raise ValueError("valid shard index/count required")
        for i,u in enumerate(manifest["design"]["units"]):
            if stage == "all" or i%shard_count == shard_index:
                run_unit(directory,manifest,u)
    if stage in ("all","finalize"):
        export(directory,directory/"analysis_handoff")
    return directory
