"""Parallel cost audits and hierarchical summaries for nested MNIST projections."""
import csv,gzip,json,os
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import t
from . import fashion_dimension as study
from .clustering_benchmark import sha256
from .clustering_manuscript_report import COLORS,LABELS,STYLES
from .integral_mechanisms import solve
from .integral_study import atomic_json,seed
from .login_guard import refuse_login_compute


def export_block(root,manifest,parent,chain,block,metas,data):
    cfg=manifest["design"];stem=root/"chunks"/f"p{parent:02d}_c{chain}_b{block}"
    rows=[];geo=[];solves=0;count=0
    path=Path(str(stem)+".draws.csv.gz")
    with gzip.open(str(path)+".tmp","wt",newline="",compresslevel=1) as stream:
        writer=csv.writer(stream);writer.writerow(["instance_id","mechanism","draw","total_cost","regret","row_oscillation"])
        for meta in metas:
            assert meta["fingerprint"]==manifest["fingerprint"]
            result=root/"units"/(meta["instance_id"]+".npz");assert sha256(result)==meta["sha256"]
            Q,R,radius=study.inputs(data,cfg,meta["d"],meta["partition"])
            assert study.old.array_sha(Q)==meta["Q_sha256"] and study.old.array_sha(R)==meta["R_sha256"]
            C=cdist(R,Q);pi0=solve(C);opt=float(C[np.arange(cfg["n"]),pi0].sum());rand=float(C.sum()/cfg["n"])
            if not cfg.get("normalize_projection",True) and cfg["phase"]=="production":
                reference=study.REFERENCE/"units"/meta["instance_id"]
                refmeta=json.loads(reference.with_suffix(".json").read_text())
                assert sha256(reference.with_suffix(".npz"))==refmeta["sha256"]
                assert study.old.array_sha(Q/radius)==refmeta["Q_sha256"]
                assert study.old.array_sha(R/radius)==refmeta["R_sha256"]
                with np.load(reference.with_suffix(".npz")) as ref:
                    np.testing.assert_allclose([opt,rand],radius*np.array([ref["optimum"],ref["RAND"]]),rtol=2e-12,atol=1e-9)
                    np.testing.assert_allclose(C[np.arange(cfg["n"]),ref["optimum_permutation"]].sum(),opt,rtol=2e-12,atol=1e-9)
                meta["normalized_reference_unit_sha256"]=refmeta["sha256"]
            with np.load(result) as z:a={k:z[k] for k in z.files}
            assert np.all(np.sort(a["permutations"],axis=-1)==np.arange(cfg["n"]))
            costs=C[np.arange(cfg["n"]),a["permutations"]].sum(axis=-1)
            np.testing.assert_allclose(costs,a["total_cost"],rtol=1e-11,atol=1e-9)
            np.testing.assert_allclose(costs-opt,a["regret"],rtol=1e-11,atol=1e-9)
            np.testing.assert_allclose([opt,rand],[a["optimum"],a["RAND"]],atol=1e-9)
            if cfg.get("geometry_reference"):
                assert sha256(meta["reference_unit"])==meta["reference_unit_sha256"]
                with np.load(meta["reference_unit"]) as ref:
                    for key in ("optimum","RAND","optimum_permutation"):np.testing.assert_array_equal(a[key],ref[key])
            assert np.all(a["regret"]<=a["row_oscillation"]+1e-6)
            if meta["alias_projection_to_original"]:
                for field in ("regret","total_cost","row_oscillation","permutations"):
                    np.testing.assert_array_equal(a[field][1],a[field][2])
            geo.append(meta)
            for ia,arm in enumerate(study.ARMS):
                rows.append(dict(parent=parent,chain=chain,d=meta["d"],partition=meta["partition"],mechanism=arm,
                    regret=float(a["regret"][ia].mean()),regret_per_pair=float(a["regret"][ia].mean()/cfg["n"]),
                    gap_fraction=float(a["regret"][ia].mean()/(rand-opt)),OPT=opt,RAND_gap=rand-opt,
                    public_rank=meta["public_rank"],effective_rank=meta["effective_rank"],
                    projected_RMS=radius,retained_energy=radius**2,total_operator_norm=meta["total_operator_norm"],
                    input_RMS=meta.get("input_RMS",1.)))
                writer.writerows((meta["instance_id"],arm,k,a["total_cost"][ia,k],a["regret"][ia,k],a["row_oscillation"][ia,k]) for k in range(cfg["K"]))
                count+=cfg["K"]
            solves+=meta["actual_assignment_solves"]
    os.replace(str(path)+".tmp",path)
    with gzip.open(path,"rb") as stream:actual=sum(block.count(b"\n") for block in iter(lambda:stream.read(8<<20),b""))-1
    assert actual==count
    pd.DataFrame(rows).to_csv(str(stem)+".instances.csv.gz",index=False)
    atomic_json(str(stem)+".geometry.json",geo)
    files={p.name:sha256(p) for p in root.glob("chunks/"+stem.name+".*") if p.suffix!=".tmp" and p.name!=stem.name+".json"}
    atomic_json(str(stem)+".json",dict(status="passed",fingerprint=manifest["fingerprint"],parent=parent,chain=chain,block=block,
        instances=len(metas),draw_rows=count,actual_assignment_solves=solves,files=files))


def export(root,output):
    refuse_login_compute("aggregate MNIST random projections")
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/"manifest.json").read_text());study.verify_runtime(manifest);cfg=manifest["design"]
    index=[];frames=[];geometries=[];draws=solves=instances=0
    for task in study.tasks(cfg):
        if task["kind"]=="basis":
            p,c=task["parent"],task["chain"];stem=root/"bases"/f"p{p:02d}_c{c}"
            meta=json.loads(stem.with_suffix(".json").read_text());assert sha256(stem.with_suffix(".npz"))==meta["sha256"]
            O=study.haar_frame(784,784,np.random.default_rng(seed(cfg.get("basis_namespace",study.NAMESPACE),cfg["phase"],p,c,"basis")))
            record=json.loads((root/"master_registry.json").read_text())[str(p)]
            assert sha256(root/record["file"])==record["sha256"]
            with np.load(root/record["file"]) as z:
                Z=z["Z"];v=np.ascontiguousarray(((z["raw"]-z["feature_offset"])/z["feature_scale"])[:,z["pixel_order"]])
                center0=v.mean(axis=0);r0=np.sqrt(np.mean(np.sum((v-center0)**2,axis=1)))
                np.testing.assert_array_equal(Z,(v-center0)/r0)
                np.testing.assert_array_equal(center0,z["original_center"])
                np.testing.assert_array_equal(r0,z["original_radius"])
            projected=Z@O;center=projected.mean(axis=0)
            with np.load(stem.with_suffix(".npz")) as z:
                W=z["W"];np.testing.assert_array_equal(O,z["O"]);np.testing.assert_array_equal(projected-center,W)
                for d,r in zip(cfg["dimensions"],z["radii"]):np.testing.assert_allclose(r,np.sqrt(np.mean(np.sum(W[:,:d]**2,axis=1))))
            continue
        stem=root/"chunks"/f"p{task['parent']:02d}_c{task['chain']}_b{task['block']}"
        meta=json.loads(Path(str(stem)+".json").read_text());assert meta["status"]=="passed" and meta["fingerprint"]==manifest["fingerprint"]
        for name,checksum in meta["files"].items():
            assert sha256(root/"chunks"/name)==checksum;index.append(dict(file="../chunks/"+name,sha256=checksum))
        frames.append(pd.read_csv(str(stem)+".instances.csv.gz"))
        geometries.extend(json.loads(Path(str(stem)+".geometry.json").read_text()))
        draws+=meta["draw_rows"];solves+=meta["actual_assignment_solves"];instances+=meta["instances"]
    frame=pd.concat(frames,ignore_index=True)
    metrics=["regret","regret_per_pair","gap_fraction","OPT","RAND_gap","public_rank","effective_rank","projected_RMS","retained_energy","total_operator_norm","input_RMS"]
    # Neither partitions nor projection chains count as independent parents.
    chain=frame.groupby(["parent","chain","d","mechanism"])[metrics].mean().reset_index()
    assert (frame.groupby(["parent","chain","d","mechanism"]).size()==cfg["partitions"]).all()
    parent=chain.groupby(["parent","d","mechanism"])[metrics].mean().reset_index()
    assert (chain.groupby(["parent","d","mechanism"]).size()==cfg["chains"]).all()
    rows=[]
    for (d,arm),group in parent.groupby(["d","mechanism"]):
        assert len(group)==cfg["parents"]
        row=dict(d=d,mechanism=arm,parents=len(group))
        for metric in metrics:
            mean=float(group[metric].mean());se=float(group[metric].std(ddof=1)/np.sqrt(len(group))) if len(group)>1 else 0.
            half=float(t.ppf(.975,len(group)-1)*se) if len(group)>1 else 0.
            row[metric]=mean;row[metric+"_low"]=mean-half;row[metric+"_high"]=mean+half
        rows.append(row)
    summary=pd.DataFrame(rows)
    frame.to_csv(output/"instance_summary.csv.gz",index=False);chain.to_csv(output/"chain_summary.csv",index=False)
    parent.to_csv(output/"parent_summary.csv",index=False);summary.to_csv(output/"summary.csv",index=False)
    pd.DataFrame(geometries).to_csv(output/"geometry.csv.gz",index=False);pd.DataFrame(index).to_csv(output/"table_index.csv",index=False)
    expected=cfg["parents"]*cfg["chains"]*len(cfg["dimensions"])*cfg["partitions"]
    assert instances==expected and draws==expected*4*cfg["K"] 
    verification=dict(status="passed",fingerprint=manifest["fingerprint"],instances=instances,draw_labels=draws,new_assignment_solves=solves,
        source_inputs_basis_flags_scales_regenerated=True,all_costs_OPT_RAND_permutations_and_aliases_audited=True,
        hierarchy="noise -> partitions -> chains -> parents",summary_sha256=sha256(output/"summary.csv"))
    if cfg.get("geometry_reference"):
        margins=frame[frame.mechanism=="gumbel"].RAND_gap-2*cfg["n"]*np.minimum(frame[frame.mechanism=="gumbel"].d,np.log(cfg["n"]))/cfg["epsilon"]
        verification.update(geometry_reference=cfg["geometry_reference"],all_inputs_OPT_RAND_bitwise_identical_to_epsilon8=True,
            unit_noise_coupled_across_budgets=True,all_geometries_selector_bound_below_RAND_gap=bool((margins>0).all()),
            geometries_selector_bound_below_RAND_gap=int((margins>0).sum()),minimum_selector_margin=float(margins.min()))
    if not cfg.get("normalize_projection",True):
        verification.update(no_post_projection_rescaling=True,operator_norm_one=True,
            normalized_reference_geometry_checks=sum("normalized_reference_unit_sha256" in g for g in geometries),
            fresh_noise_namespace=cfg["namespace"],imported_production_draws=0)
        assert np.all(frame.total_operator_norm==1.)
        if cfg["phase"]=="production":assert verification["normalized_reference_geometry_checks"]==expected
    atomic_json(output/"verification.json",verification)
    atomic_json(output/"data_dictionary.json",dict(
        privacy=cfg["privacy_metric"]+"; epsilon="+str(cfg["epsilon"]),
        draw_tables="chunks/*.draws.csv.gz, indexed by table_index.csv; one row per instance/arm/draw",
        units="units/*.npz: arm x draw arrays total_cost/regret/row_oscillation; arm x draw x n permutations; optimum, RAND, optimum_permutation",
        arm_order=list(study.ARMS),
        inputs="masters: raw source rows, IDs, all partitions, feature transform, full784 normalized Z; bases: Haar O, centered Z@O, all radii",
        reconstruction="select orders[partition][:n] and orders[partition][n:]; take "+("W[:,:d]/projected_RMS" if cfg.get("normalize_projection",True) else "W[:,:d] without rescaling; full784 master remains RMS-one"),
        noise="seed(namespace,phase,instance_id,draw,arm); complete source and runtime archived",
        hierarchy="K noise -> 5 partitions -> 5 chains -> 20 independent parents; Student-t95% pointwise",
        endpoint_alias="projection equals original when public affine rank=d; copied scientific arrays, three new private assignment solves per draw"))
    for filename in ("manifest.json","master_registry.json"):shutil.copyfile(root/filename,output/filename)
    for name in ("masters","bases","units","chunks","source_snapshot"):
        if (root/name).exists() and not (output/name).exists():(output/name).symlink_to("../"+name,target_is_directory=True)
    figure(summary,output,cfg)
    print(json.dumps(verification),flush=True)


def figure(summary,output,cfg):
    directory=output/"figures";directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"pdf.fonttype":42,"axes.spines.top":False,"axes.spines.right":False})
    fig,ax=plt.subplots(figsize=(9,4.6));fig.subplots_adjust(left=.1,right=.98,top=.91,bottom=.25)
    for arm in study.ARMS:
        v=summary[summary.mechanism==arm].sort_values("d")
        ax.plot(v.d,v.regret,color=COLORS[arm],ls=STYLES[arm],lw=1.5,label=LABELS[arm])
        ax.fill_between(v.d,v.regret_low,v.regret_high,color=COLORS[arm],alpha=.14,lw=0)
    v=summary[summary.mechanism=="gumbel"].sort_values("d")
    ax.plot(v.d,v.RAND_gap,color=".4",ls=":",lw=1.6,label="RAND − OPT")
    x=np.unique(np.r_[np.linspace(1,cfg["display_max_dimension"],400),np.log(cfg["n"])])
    y=2*cfg["n"]*np.minimum(x,np.log(cfg["n"]))/cfg["epsilon"]
    ax.plot(x,y,color="black",ls="-.",lw=1.4,label=r"$2n\min\{d,\log n\}/\varepsilon$")
    ax.set_xlim(0,cfg["display_max_dimension"]);ax.set_ylim(bottom=0)
    if cfg["display_max_dimension"]<=104:ax.set_xticks([0,20,40,60,80,100])
    ax.set_xlabel("Projected dimension d");ax.set_ylabel("E[u] − OPT")
    ax.set_title("Fashion-MNIST")
    ax.axvline(np.log(cfg["n"]),color=".6",ls=":",lw=.8);ax.grid(axis="y",alpha=.17);ax.set_axisbelow(True)
    fig.legend(*ax.get_legend_handles_labels(),loc="lower center",ncol=3,frameon=False,bbox_to_anchor=(.52,.005),fontsize=9)
    for suffix in ("png","pdf"):fig.savefig(directory/("fashion_dimension_dimension."+suffix),dpi=200,bbox_inches="tight")
    plt.close(fig)
