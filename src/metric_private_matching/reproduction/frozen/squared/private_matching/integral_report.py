"""Streaming numerical handoff and figures for integral matching checkpoints."""
from __future__ import annotations

import csv
import gzip
import hashlib
import html
import json
import os
import re
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.stats import t

from private_matching.integral_mechanisms import ARMS
from private_matching.integral_study import NAME, RAW_FIELDS, atomic_json, units, valid_checkpoint
from private_matching.login_guard import refuse_login_compute

LABELS = dict(gumbel="Gumbel assignment",terminal_fold="Terminal fold",anchor="Anchor",span_radial="Public-span radial",coordinate_allocated="Coordinate allocation",ambient_radial="Point radial Laplace",ambient_iid="Point iid Laplace",cost_infinity="Cost infinity-norm noise",cost_iid="Cost iid Laplace")
COLORS = dict(gumbel="#0072B2",terminal_fold="#009E73",anchor="#CC79A7",span_radial="#009E73",coordinate_allocated="#CC79A7",ambient_radial="#D55E00",ambient_iid="#E69F00",cost_infinity="#56B4E9",cost_iid="#666666")
PAIRS = {1:(("gumbel","ambient_radial"),("gumbel","cost_infinity"),("anchor","terminal_fold")),2:(("gumbel","ambient_radial"),("gumbel","cost_infinity"),("coordinate_allocated","span_radial"))}


def write_csv(path, rows, fields=None):
    rows=iter(rows)
    first=next(rows,None)
    if first is None:
        raise RuntimeError(f"no rows for {path}")
    fields=list(first) if fields is None else fields
    path=Path(path); tmp=Path(str(path)+".tmp")
    opener=gzip.open if path.suffix==".gz" else open
    with opener(tmp,"wt",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields)
        writer.writeheader(); writer.writerow(first); writer.writerows(rows)
    os.replace(tmp,path)


def interval(values):
    x=np.asarray(values,float); mean=float(x.mean())
    se=float(x.std(ddof=1)/np.sqrt(len(x))) if len(x)>1 else 0.
    half=float(t.ppf(.975,len(x)-1))*se if len(x)>1 else 0.
    return mean,se,mean-half,mean+half


def validate(directory):
    manifest=json.loads((directory/"manifest.json").read_text())
    if manifest.get("kind")!=NAME:
        raise RuntimeError("wrong study manifest")
    expected=units(manifest["design"])
    missing=[u["geometry_id"] for u in expected if not valid_checkpoint(directory,u,manifest["fingerprint"])]
    if missing:
        raise RuntimeError(f"incomplete study: {len(missing)} missing geometries, first {missing[:5]}")
    if len(list((directory/"units").glob("*.json")))!=len(expected):
        raise RuntimeError("unexpected extra geometry checkpoints")
    return manifest,expected


def export(directory, output):
    refuse_login_compute("export the integral matching numerical handoff")
    directory,output=Path(directory),Path(output)
    manifest,expected=validate(directory)
    output.mkdir(parents=True,exist_ok=True)
    cfg=manifest["design"]
    grouped={}; paired={}; runtime=[]
    summary_path=output/"geometry_summaries.csv.gz"
    raw_count=0; actual_solves=0
    start=time.perf_counter()
    # Streaming output: no materialization of the 15-million-row master table.
    master=output/"analysis_table.csv.gz"
    temp=Path(str(master)+".tmp")
    with gzip.open(temp,"wt",newline="") as handle:
        csv.writer(handle).writerow(RAW_FIELDS)
    summary_temp=Path(str(summary_path)+".tmp")
    with temp.open("ab") as merged, gzip.open(summary_temp,"wt",newline="") as summary:
        writer=None
        for number,unit in enumerate(expected):
            stem=directory/"units"/unit["geometry_id"]
            meta=json.loads(Path(str(stem)+".json").read_text())
            # Compressed members are copied byte for byte after validating each
            # source checksum. Standard gzip/CSV readers support concatenation.
            for suffix,record in meta["files"].items():
                source=Path(str(stem)+suffix)
                if hashlib.sha256(source.read_bytes()).hexdigest()!=record["sha256"]:
                    raise RuntimeError(f"checksum mismatch {source}")
            with Path(str(stem)+".csv.gz").open("rb") as source:
                shutil.copyfileobj(source,merged,1024*1024)
            raw_count+=meta["raw_rows"]; actual_solves+=meta["actual_solves"]
            with np.load(Path(str(stem)+".npz")) as saved:
                # NPZ access decompresses the entire named array on every
                # lookup. Cache only these four small numerical arrays once;
                # the large permutation array stays on disk during export.
                arrays={key:saved[key] for key in ("total_cost","total_regret","runtime_seconds","solve_seconds")}
                for ip,p in enumerate((1,2)):
                    optimum=meta["optima"][str(p)]
                    random_mean=meta["random_expected"][str(p)]
                    for ia,arm in enumerate(ARMS[p]):
                        for ie,epsilon in enumerate(cfg["epsilon"]):
                            cost=arrays["total_cost"][ip,ia,ie]
                            regret=arrays["total_regret"][ip,ia,ie]
                            if len(cost)!=cfg["K"] or not np.all(np.isfinite(cost)) or not np.allclose(cost-optimum,regret,atol=1e-9):
                                raise RuntimeError("checkpoint numerical validation failed")
                            cert=meta["bounds"][f"{p}:{arm}"][ie]
                            planted=meta.get("planted_cost",{}).get(str(p))
                            row=dict(fingerprint=manifest["fingerprint"],phase=cfg["phase"],**unit,objective_p=p,mechanism=arm,epsilon=epsilon,K=cfg["K"],total_cost_mean=float(cost.mean()),total_regret_mean=float(regret.mean()),noise_sd=float(regret.std(ddof=1)),noise_se=float(regret.std(ddof=1)/np.sqrt(cfg["K"])),regret_q05=float(np.quantile(regret,.05)),regret_q50=float(np.quantile(regret,.5)),regret_q95=float(np.quantile(regret,.95)),optimum_total_cost=optimum,uniform_expected_total_cost=random_mean,uniform_expected_regret=random_mean-optimum,planted_total_cost=planted,planted_excess=None if planted is None else planted-optimum,bound_total=cert["bound_total"],bound_uncapped=cert["bound_uncapped"],bound_cap=cert["bound_cap"],runtime_seconds_mean=float(arrays["runtime_seconds"][ip,ia,ie].mean()),solve_seconds_mean=float(arrays["solve_seconds"][ip,ia,ie].mean()),**meta["public"],**meta["reuse"][f"{p}:{arm}"])
                            if writer is None:
                                writer=csv.DictWriter(summary,fieldnames=list(row)); writer.writeheader()
                            writer.writerow(row)
                            key=(unit["setting_id"],p,arm,epsilon)
                            grouped.setdefault(key,[]).append(row)
                    for left,right in PAIRS[p]:
                        il,ir=ARMS[p].index(left),ARMS[p].index(right)
                        for ie,epsilon in enumerate(cfg["epsilon"]):
                            difference=float(arrays["total_regret"][ip,il,ie].mean()-arrays["total_regret"][ip,ir,ie].mean())
                            key=(unit["setting_id"],p,left,right,epsilon)
                            paired.setdefault(key,[]).append(difference)
            runtime.append(dict(**unit,phase=cfg["phase"],elapsed_seconds=meta["elapsed_seconds"],preparation_seconds=meta["preparation_seconds"],channel_generation_seconds=sum(meta["channel_generation_seconds"].values()),raw_rows=meta["raw_rows"],actual_solves=meta["actual_solves"],slurm_job_id=meta["slurm_job_id"],slurm_array_task_id=meta["slurm_array_task_id"],host=meta["host"]))
            if (number+1)%100==0:
                print(f"Export {number+1}/{len(expected)} geometries",flush=True)
    os.replace(temp,master); os.replace(summary_temp,summary_path)
    expected_rows=len(expected)*2*(1+cfg["K"]+7*len(cfg["epsilon"])*cfg["K"])
    if raw_count!=expected_rows:
        raise RuntimeError("raw row count mismatch")
    # Full decompression checks CRCs, concatenated-member support and row count.
    with gzip.open(master,"rb") as handle:
        lines=sum(chunk.count(b"\n") for chunk in iter(lambda:handle.read(4*1024*1024),b""))
    if lines!=raw_count+1:
        raise RuntimeError(f"master row verification failed: {lines-1} != {raw_count}")
    summaries=[]
    for key,rows in sorted(grouped.items()):
        first=rows[0]
        result={k:first[k] for k in ("fingerprint","phase","setting_id","panel","family","n","d","tau","objective_p","mechanism","epsilon","source_arm","reuse_reason")}
        result.update(geometries=len(rows),K=cfg["K"])
        for source,target in (("total_regret_mean","total_regret"),("total_cost_mean","total_cost"),("optimum_total_cost","optimum_total_cost"),("uniform_expected_regret","uniform_expected_regret")):
            mean,se,lo,hi=interval([row[source] for row in rows])
            result.update({target+"_mean":mean,target+"_se":se,target+"_ci_low":lo,target+"_ci_high":hi})
        for field in ("bound_total","bound_uncapped","runtime_seconds_mean","solve_seconds_mean","public_rank","public_diameter","coordinate_width_certificate","noise_se"):
            result[field]=float(np.mean([row[field] for row in rows]))
        for field in ("bound_cap","planted_total_cost","planted_excess"):
            result[field]=None if first[field] is None else float(np.mean([row[field] for row in rows]))
        result["bound_to_regret"]=None if result["total_regret_mean"]<=0 else result["bound_total"]/result["total_regret_mean"]
        summaries.append(result)
    write_csv(output/"configuration_summary.csv",summaries)
    pairrows=[]
    for key,values in sorted(paired.items()):
        setting,p,left,right,epsilon=key
        mean,se,lo,hi=interval(values)
        setting_info=next(s for s in cfg["settings"] if s["setting_id"]==setting)
        pairrows.append(dict(**setting_info,phase=cfg["phase"],objective_p=p,left=left,right=right,epsilon=epsilon,geometries=len(values),mean_total_regret_difference=mean,se=se,ci_low=lo,ci_high=hi,interval_method="Student_t_over_independent_geometry_mean_differences",negative_favors_left=True))
    write_csv(output/"paired_comparisons.csv",pairrows)
    write_csv(output/"runtime_by_geometry.csv",runtime)
    write_csv(output/"selected_epsilon_table.csv",(r for r in summaries if r["epsilon"] in (1.,2.,4.,8.)))
    readable_tables(output,summaries)
    dictionary=dict(study=NAME,fingerprint=manifest["fingerprint"],phase=cfg["phase"],raw_fields=list(RAW_FIELDS),objective="SUM of original pairwise distances^p, never divided by n",epsilon_units="inverse distance; Euclidean sum-record metric",intervals="Pointwise 95% Student t across independent geometry conditional-noise means; not simultaneous",uniform_reference="Exact conditional expectation sum(C)/n; this is an expected SUM, not a per-pair objective. Sampled permutations also saved.",coupling="Channel-specific common unit draws across epsilon and p. Different channel seeds are independent. Endpoint source_arm/reuse_reason explicitly identify copied draws. Geometry pairing is used for cross-mechanism comparisons; numerical draw indices alone never establish a coupling.",guarantees="Expected TOTAL regret bounds, conditional on public geometry; dashed curves average those bounds with the same geometry weights as solid empirical means. Realization_bound is separately checked for every draw.",runtime="solve_seconds: assignment solve. runtime_seconds: score construction, solve, scoring and stability checks, excluding common geometry preparation and unit-noise generation/projection. Copied endpoints consume zero incremental solve time. Whole geometry and shared generation timings separately saved.",families={"U":"rotated cube, RMS radius 1","G":"isotropic Gaussian, trace covariance 1","A01":"four principal variances, residual trace .01","A20":"four principal variances, residual trace .20","M01":"heteroscedastic four-component mixture, scale .1","M04":"heteroscedastic four-component mixture, scale .4","M10":"heteroscedastic four-component mixture, scale 1"},seed_recipe="SHA256(JSON tuple) first 16 bytes interpreted little endian; namespace, phase, geometry ID, unit, draw, channel. See frozen integral_study.py. Random matching has separate random channel.",permutation_lookup="permutation_file relative to fingerprint directory; permutation_array and comma-separated permutation_index select saved uint16 complete bijection. Geometry coordinates/PCA/rotation/labels are saved in geometry_file.",baseline_calibration="Point iid scale sqrt(d)/epsilon. Cost iid scale L_one/epsilon, cost infinity-norm joint density exp(-epsilon ||e||inf/L_infinity), gamma radial dimension n.")
    atomic_json(output/"data_dictionary.json",dictionary)
    totals=dict(fingerprint=manifest["fingerprint"],phase=cfg["phase"],settings=len(cfg["settings"]),geometries=len(expected),K=cfg["K"],epsilon=cfg["epsilon"],private_arm_rows=len(expected)*14*len(cfg["epsilon"])*cfg["K"],raw_rows=raw_count,actual_assignment_solves=actual_solves,all_draws_integral=True,all_saved_costs_reconstructed_in_worker=True,all_source_checksums_verified=True,master_decompressed_row_count_verified=True,simulation_worker_seconds=sum(r["elapsed_seconds"] for r in runtime),export_seconds=time.perf_counter()-start,master_bytes=master.stat().st_size,master_sha256=hashlib.sha256(master.read_bytes()).hexdigest(),report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(output/"verification.json",totals)
    return manifest,summaries,pairrows,runtime,totals


def readable_tables(output,summaries):
    tables=output/"tables";tables.mkdir(exist_ok=True)
    selected=[r for r in summaries if r["panel"]=="A" and r["epsilon"] in (1.,2.,4.,8.)]
    for family in sorted({r["family"] for r in selected}):
        for p in (1,2):
            rows=[r for r in selected if r["family"]==family and r["objective_p"]==p]
            text=[f"# {family}, objective p={p}","","All costs and regrets are sums; smaller regret is better. Intervals are pointwise 95% geometry intervals.","","| d | epsilon | Mechanism | Total cost | Total regret [95% CI] | OPT cost | Random cost | Bound |","|---:|---:|---|---:|---:|---:|---:|---:|"]
            for r in sorted(rows,key=lambda r:(r["d"],r["epsilon"],r["mechanism"])):
                text.append(f"| {r['d']} | {r['epsilon']:g} | {LABELS[r['mechanism']]} | {r['total_cost_mean']:.4f} | {r['total_regret_mean']:.4f} [{r['total_regret_ci_low']:.4f}, {r['total_regret_ci_high']:.4f}] | {r['optimum_total_cost_mean']:.4f} | {r['optimum_total_cost_mean']+r['uniform_expected_regret_mean']:.4f} | {r['bound_total']:.4f} |")
            (tables/f"T1_{family}_p{p}.md").write_text("\n".join(text)+"\n")
    wide={}
    for r in summaries:
        if r["panel"] not in ("A","B") or r["epsilon"] not in (.5,2.,4.,8.):continue
        key=(r["setting_id"],r["objective_p"],r["epsilon"])
        row=wide.setdefault(key,{k:r[k] for k in ("panel","family","n","d","objective_p","epsilon","public_rank","public_diameter","coordinate_width_certificate")})
        row[r["mechanism"]]=r["total_regret_mean"]
    allarms=sorted(set(ARMS[1]+ARMS[2]))
    fields=["panel","family","n","d","objective_p","epsilon","public_rank","public_diameter","coordinate_width_certificate"]+allarms
    write_csv(tables/"T3_scaling_wide.csv",(wide[k] for k in sorted(wide)),fields)
    write_csv(tables/"T2_guarantees.csv",({k:r[k] for k in ("setting_id","panel","family","n","d","tau","objective_p","mechanism","epsilon","total_regret_mean","bound_uncapped","bound_cap","bound_total","bound_to_regret")} for r in summaries))


def plot_all(output, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.2,"savefig.dpi":160})
    figures=output/"figures"; figures.mkdir(exist_ok=True)
    figure_manifest=[]
    # Save exact plotted input rows separately from the presentation.
    write_csv(figures/"plot_input.csv",summaries)
    def save(fig,name,caption):
        fig.savefig(figures/(name+".png"),bbox_inches="tight")
        fig.savefig(figures/(name+".pdf"),bbox_inches="tight")
        plt.close(fig)
        figure_manifest.append(dict(file=name+".png",pdf=name+".pdf",caption=caption))
    def line(ax,rows,arm,xfield="epsilon",detail=True):
        rows=sorted(rows,key=lambda r:r[xfield]); x=[r[xfield] for r in rows]
        if not rows:return
        y=np.array([r["total_regret_mean"] for r in rows]); lo=np.array([r["total_regret_ci_low"] for r in rows]); hi=np.array([r["total_regret_ci_high"] for r in rows])
        ax.plot(x,y,color=COLORS[arm],label=LABELS[arm],lw=1.7)
        ax.fill_between(x,np.maximum(0.,lo),hi,color=COLORS[arm],alpha=.1)
        ax.plot(x,[r["bound_total"] for r in rows],color=COLORS[arm],ls="--",lw=1.1)
        if detail:
            # Clearly named empirical-detail view; companion full view retains
            # every bound. No expectation certificate is silently truncated.
            ax.set_ylim(0.,max(ax.get_ylim()[1] if getattr(ax,"_detail_set",False) else 0.,float(hi.max())*1.2,1e-8))
            ax._detail_set=True
    A=[r for r in summaries if r["panel"]=="A"]
    for family in sorted({r["family"] for r in A}):
        for p in (1,2):
            chosen=[r for r in A if r["family"]==family and r["objective_p"]==p]
            for group,arms in (("methods",ARMS[p][:3]+("ambient_radial",)),("baselines",("gumbel",)+ARMS[p][3:])):
                for detail in (True,False):
                    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
                    for ax,d in zip(axes.flat,sorted({r["d"] for r in chosen})):
                        subset=[r for r in chosen if r["d"]==d]
                        for arm in arms:line(ax,[r for r in subset if r["mechanism"]==arm],arm,detail=detail)
                        random=next(r["uniform_expected_regret_mean"] for r in subset)
                        ax.axhline(random,color="black",ls=":",label="Uniform random (exact expectation)")
                        ax.axhline(0,color="black",lw=.7,label="OPT")
                        if detail:ax.set_ylim(0,max(ax.get_ylim()[1],random*1.15))
                        ax.set_title(f"{family}: n=64, d={d}, p={p}"); ax.set_xlabel(r"$\epsilon$"); ax.set_ylabel("Total regret (sum)")
                        if not detail:ax.set_yscale("symlog",linthresh=max(random*.03,.01))
                    handles,labels=axes.flat[0].get_legend_handles_labels()
                    fig.legend(handles,labels,loc="lower center",ncol=3,bbox_to_anchor=(.5,-.02))
                    fig.suptitle("Solid: empirical mean; dashed: expected-regret certificate"+(" · empirical detail (bounds may extend above view)" if detail else " · full bound range"))
                    fig.tight_layout(rect=(0,.08,1,.95))
                    save(fig,f"P1_{family}_p{p}_{group}_{'detail' if detail else 'full'}",f"{family}, p={p}, {group}: epsilon curves across dimension. Shading: pointwise 95% CI across geometry means.")
    # Dimension and sample-size comparisons hold epsilon fixed within each line.
    for panel,xfield,title in (("A","d","Dimension"),("B","n","Sample size")):
        panelrows=[r for r in summaries if r["panel"]==panel and r["epsilon"] in (.5,2.,4.,8.)]
        for family in sorted({r["family"] for r in panelrows}):
            for p in (1,2):
                subset=[r for r in panelrows if r["family"]==family and r["objective_p"]==p]
                dimensions=[None] if panel=="A" else sorted({r["d"] for r in subset})
                for dimension in dimensions:
                    fig,axes=plt.subplots(2,2,figsize=(12,8))
                    for ax,epsilon in zip(axes.flat,(.5,2.,4.,8.)):
                        rows=[r for r in subset if r["epsilon"]==epsilon and (dimension is None or r["d"]==dimension)]
                        for arm in ARMS[p]:line(ax,[r for r in rows if r["mechanism"]==arm],arm,xfield,detail=False)
                        ref=sorted([r for r in rows if r["mechanism"]=="gumbel"],key=lambda r:r[xfield])
                        ax.plot([r[xfield] for r in ref],[r["uniform_expected_regret_mean"] for r in ref],"k:",label="Uniform random")
                        ax.set(xlabel=xfield,ylabel="Total regret (sum)",title=f"{family}, p={p}, epsilon={epsilon}"+(f", d={dimension}" if dimension else ""))
                        ax.set_yscale("symlog",linthresh=1.)
                    h,l=axes.flat[0].get_legend_handles_labels(); fig.legend(h,l,loc="lower center",ncol=4,bbox_to_anchor=(.5,-.01)); fig.tight_layout(rect=(0,.07,1,1))
                    save(fig,f"P{2 if panel=='A' else 3}_{family}_p{p}"+(f"_d{dimension}" if dimension else ""),f"{title} at fixed epsilon, including all expected bounds (dashed).")
    # Distribution-shape comparisons at a fixed n,d and common budgets.
    if any(r["family"]=="A20" for r in A):
        for p in (1,2):
            fig,axes=plt.subplots(1,2,figsize=(13,5))
            for ax,families in zip(axes,(("G","A20","A01"),("M01","M04","M10"))):
                for ia,arm in enumerate(ARMS[p]):
                    ys=[]; bs=[]
                    for family in families:
                        r=next(r for r in A if r["family"]==family and r["d"]==512 and r["objective_p"]==p and r["epsilon"]==4. and r["mechanism"]==arm)
                        ys.append(r["total_regret_mean"]);bs.append(r["bound_total"])
                    ax.plot(families,ys,color=COLORS[arm],marker="o",label=LABELS[arm]);ax.plot(families,bs,color=COLORS[arm],ls="--")
                ax.set_yscale("symlog",linthresh=1.);ax.set_ylabel("Total regret (sum)");ax.set_title(f"n=64, d=512, epsilon=4, p={p}")
            h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc="lower center",ncol=4,bbox_to_anchor=(.5,-.08));fig.tight_layout(rect=(0,.08,1,1));save(fig,f"P4_shape_p{p}","Full-rank anisotropy and heteroscedastic mixture overlap; dashed bounds retain their full range.")
    C=[r for r in summaries if r["panel"]=="C" and r["epsilon"]==8.]
    for family in sorted({r["family"] for r in C}):
        for p in (1,2):
            fig,axes=plt.subplots(2,2,figsize=(12,8))
            for ax,d in zip(axes.flat,sorted({r["d"] for r in C})):
                rows=[r for r in C if r["family"]==family and r["objective_p"]==p and r["d"]==d]
                for arm in ARMS[p]:line(ax,[dict(r,tau=float(r["tau"])) for r in rows if r["mechanism"]==arm],arm,"tau",detail=False)
                ax.set_yscale("symlog",linthresh=1.);ax.set(xlabel="Near-partner noise tau",ylabel="Total regret (sum)",title=f"{family}, d={d}, p={p}, epsilon=8")
            h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc="lower center",ncol=4,bbox_to_anchor=(.5,-.01));fig.tight_layout(rect=(0,.07,1,1));save(fig,f"P5_partner_{family}_p{p}","Near-partner experiment; the optimum is solved independently of the planted pairing.")
    for p in (1,2):
        selected=[r for r in A if r["family"]=="G" and r["d"]==512 and r["objective_p"]==p]
        if not selected:continue
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for arm in ARMS[p]:
            rows=sorted([r for r in selected if r["mechanism"]==arm],key=lambda r:r["epsilon"])
            axes[0].plot([r["epsilon"] for r in rows],[r["bound_to_regret"] for r in rows],color=COLORS[arm],label=LABELS[arm])
            axes[1].plot([r["epsilon"] for r in rows],[1000*r["solve_seconds_mean"] for r in rows],color=COLORS[arm],label=LABELS[arm])
        axes[0].set(xlabel="epsilon",ylabel="Expected bound / empirical mean regret",yscale="log",title=f"Bound tightness, n=64,d=512,p={p}")
        axes[1].set(xlabel="epsilon",ylabel="Assignment time (milliseconds)",title="Incremental solves; reused endpoint is zero")
        h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc="lower center",ncol=4,bbox_to_anchor=(.5,-.14));fig.tight_layout(rect=(0,.08,1,1));save(fig,f"P6_P7_bound_runtime_p{p}","Bound tightness and measured assignment-solver time; shared preprocessing and noise generation are tabulated separately.")
    atomic_json(figures/"figure_manifest.json",figure_manifest)
    return figure_manifest


def plot_completion(output,summaries,runtime):
    """Additional fixed-budget views and whole-suite diagnostics, without draws."""
    import matplotlib.pyplot as plt
    figures=output/"figures"; records=[]
    def save(fig,name,caption):
        fig.savefig(figures/(name+".png"),bbox_inches="tight",dpi=160)
        fig.savefig(figures/(name+".pdf"),bbox_inches="tight")
        plt.close(fig);records.append(dict(file=name+".png",pdf=name+".pdf",caption=caption))
    def curves(ax,rows,arms,xfield,detail):
        ymax=1e-8
        for arm in arms:
            selected=sorted([r for r in rows if r["mechanism"]==arm],key=lambda r:r[xfield])
            if not selected:continue
            x=[r[xfield] for r in selected]
            y=[r["total_regret_mean"] for r in selected]
            low=np.maximum(0.,[r["total_regret_ci_low"] for r in selected]);high=[r["total_regret_ci_high"] for r in selected]
            ax.plot(x,y,color=COLORS[arm],lw=1.6,marker=".",label=LABELS[arm]);ax.fill_between(x,low,high,color=COLORS[arm],alpha=.1)
            ax.plot(x,[r["bound_total"] for r in selected],color=COLORS[arm],ls="--",lw=1.)
            ymax=max(ymax,max(high))
        references=sorted([r for r in rows if r["mechanism"]=="gumbel"],key=lambda r:r[xfield])
        if references:
            ax.plot([r[xfield] for r in references],[r["uniform_expected_regret_mean"] for r in references],"k:",label="Uniform random expectation")
            ymax=max(ymax,max(r["uniform_expected_regret_mean"] for r in references))
            if references[0]["planted_excess"] is not None:
                ax.plot([r[xfield] for r in references],[r["planted_excess"] for r in references],color="#888888",ls="-.",label="Planted excess over OPT (nonprivate)")
        ax.axhline(0,color="black",lw=.7,label="OPT")
        if detail:ax.set_ylim(0,ymax*1.15)
        else:ax.set_yscale("symlog",linthresh=max(ymax*.02,.01))
        ax.set_ylabel("Total regret (sum)")
    def legend(fig,axes,detail):
        h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc="lower center",ncol=3,bbox_to_anchor=(.5,-.025))
        fig.suptitle("Solid means and 95% intervals; dashed expected bounds · "+("empirical detail; bounds can extend above view" if detail else "full bound range"))
        fig.tight_layout(rect=(0,.10,1,.94))
    for panel,xfield in (("A","d"),("B","n")):
        rows=[r for r in summaries if r["panel"]==panel]
        for family in sorted({r["family"] for r in rows}):
            for p in (1,2):
                selected=[r for r in rows if r["family"]==family and r["objective_p"]==p]
                ds=[None] if panel=="A" else sorted({r["d"] for r in selected})
                for d in ds:
                    groups=(("methods",ARMS[p][:3]+("ambient_radial",)),("baselines",("gumbel",)+ARMS[p][3:])) if panel=="A" else (("all",ARMS[p]),)
                    for group,arms in groups:
                        fig,axes=plt.subplots(2,2,figsize=(12,8))
                        for ax,epsilon in zip(axes.flat,(.5,2.,4.,8.)):
                            subset=[r for r in selected if r["epsilon"]==epsilon and (d is None or r["d"]==d)]
                            curves(ax,subset,arms,xfield,True)
                            ax.set(xlabel=xfield,title=f"{family}, p={p}, epsilon={epsilon}"+(f", d={d}" if d else ""))
                        legend(fig,axes,True)
                        save(fig,f"P{2 if panel=='A' else 3}_detail_{family}_p{p}_{group}"+(f"_d{d}" if d else ""),"Scaling with fixed epsilon. Companion full-range plots show bounds that exceed the empirical-detail view.")
    for p in (1,2):
        for d in (32,512):
            for epsilon in (2.,8.):
                for detail in (False,True):
                    fig,axes=plt.subplots(1,2,figsize=(13,5))
                    for ax,families in zip(axes,(("G","A20","A01"),("M01","M04","M10"))):
                        subset=[r for r in summaries if r["panel"]=="A" and r["d"]==d and r["epsilon"]==epsilon and r["objective_p"]==p and r["family"] in families]
                        # Categorical order is prescribed rather than alphabetical.
                        subset=[dict(r,shape_index=families.index(r["family"])) for r in subset]
                        curves(ax,subset,ARMS[p],"shape_index",detail)
                        ax.set_xticks(range(3),families);ax.set_title(f"d={d}, p={p}, epsilon={epsilon}")
                    legend(fig,axes,detail)
                    save(fig,f"P4_shape_d{d}_p{p}_e{epsilon:g}_{'detail' if detail else 'full'}","Prespecified anisotropy and mixture-overlap comparisons at n=64; known population RMS radius one.")
    C=[r for r in summaries if r["panel"]=="C"]
    for family in sorted({r["family"] for r in C}):
        for p in (1,2):
            for epsilon in (.5,2.,4.,8.):
                for detail in (False,True):
                    fig,axes=plt.subplots(2,2,figsize=(12,8))
                    for ax,d in zip(axes.flat,sorted({r["d"] for r in C})):
                        rows=[dict(r,tau=float(r["tau"])) for r in C if r["family"]==family and r["objective_p"]==p and r["epsilon"]==epsilon and r["d"]==d]
                        curves(ax,rows,ARMS[p],"tau",detail)
                        ax.set(xlabel="Near-partner noise tau",title=f"{family}, d={d}, p={p}, epsilon={epsilon}")
                    legend(fig,axes,detail)
                    save(fig,f"P5_partner_{family}_p{p}_e{epsilon:g}_{'detail' if detail else 'full'}","Noisy partners: OPT is solved independently; grey dash-dot curve shows the planted pairing's actual excess over OPT.")
    for p in (1,2):
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for arm in ARMS[p]:
            rows=[r for r in summaries if r["objective_p"]==p and r["mechanism"]==arm and r["total_regret_mean"]>0 and r["bound_total"]>0]
            axes[0].scatter([r["total_regret_mean"] for r in rows],[r["bound_total"] for r in rows],s=6,alpha=.35,color=COLORS[arm],label=LABELS[arm])
            axes[1].hist([np.log10(r["bound_to_regret"]) for r in rows],bins=25,histtype="step",color=COLORS[arm],label=LABELS[arm])
        axes[0].set(xlabel="Empirical mean total regret",ylabel="Expected bound",xscale="log",yscale="log",title=f"All settings and budgets, p={p}")
        limits=axes[0].get_xlim();axes[0].plot(limits,limits,"k:")
        axes[1].set(xlabel="log10(bound / empirical mean total regret)",ylabel="Number of configurations",title="Zero denominators omitted explicitly")
        h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc="lower center",ncol=4,bbox_to_anchor=(.5,-.09));fig.tight_layout(rect=(0,.07,1,1));save(fig,f"P6_all_settings_p{p}","Expected-bound tightness across every setting and budget; no claim of a pointwise ceiling on noisy empirical estimates.")
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,d in zip(axes,(32,512)):
            for arm in ARMS[p]:
                rows=sorted([r for r in summaries if r["panel"]=="B" and r["family"]=="G" and r["d"]==d and r["objective_p"]==p and r["epsilon"]==4. and r["mechanism"]==arm],key=lambda r:r["n"])
                ax.plot([r["n"] for r in rows],[1000*r["runtime_seconds_mean"] for r in rows],color=COLORS[arm],marker=".",label=LABELS[arm])
            ax.set(xlabel="n",ylabel="Incremental mechanism time (ms/draw)",title=f"G, d={d}, epsilon=4, p={p}")
        h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc="lower center",ncol=4,bbox_to_anchor=(.5,-.09));fig.tight_layout(rect=(0,.07,1,1));save(fig,f"P7_n_scaling_p{p}","Score construction, assignment and checks; shared preprocessing and noise generation excluded and shown separately.")
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for panel,xfield,ax in (("A","d",axes[0]),("B","n",axes[1])):
        for family in ("G","M04"):
            for d in ([None] if panel=="A" else [32,512]):
                rows=[r for r in runtime if r["panel"]==panel and r["family"]==family and (d is None or r["d"]==d)]
                x=sorted({r[xfield] for r in rows})
                y=[1000*np.mean([r["preparation_seconds"] for r in rows if r[xfield]==v]) for v in x]
                ax.plot(x,y,marker=".",label=family+(f", d={d}" if d else ""))
        ax.set(xlabel=xfield,ylabel="Shared geometry preparation (ms)");ax.legend()
    fig.tight_layout();save(fig,"P7_shared_setup","One-time geometry preparation, public PCA and exact OPT solves; never attributed repeatedly to individual arms.")
    return records


def basic_html(text):
    """Render this report's simple Markdown without an optional dependency."""
    def inline(value):
        value=html.escape(value)
        value=re.sub(r'!\[([^]]*)\]\(([^)]+)\)',r'<img alt="\1" src="\2">',value)
        value=re.sub(r'(?<!!)\[([^]]*)\]\(([^)]+)\)',r'<a href="\2">\1</a>',value)
        value=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',value)
        return re.sub(r'`([^`]+)`',r'<code>\1</code>',value)
    parts=[];table=False
    for line in text.splitlines():
        if line.startswith("|"):
            if re.match(r'^\|[\s:|\-]+$',line):continue
            first=not table
            if first:parts.append("<table>");table=True
            tag="th" if first else "td"
            parts.append("<tr>"+"".join(f"<{tag}>{inline(cell.strip())}</{tag}>" for cell in line.strip("|").split("|"))+"</tr>")
            continue
        if table:parts.append("</table>");table=False
        if not line:continue
        match=re.match(r'^(#{1,6}) (.*)',line)
        if match:parts.append(f"<h{len(match[1])}>{inline(match[2])}</h{len(match[1])}>")
        else:parts.append("<p>"+inline(line.removeprefix("- "))+"</p>")
    if table:parts.append("</table>")
    return "\n".join(parts)


def report(output,manifest,summaries,pairs,runtime,totals,figures):
    cfg=manifest["design"]
    lines=["# Integral matching experiment results", "",f"Completed {cfg['phase']} run, fingerprint `{manifest['fingerprint']}`.","",f"The run contains **{totals['settings']} settings, {totals['geometries']} independent geometries, {cfg['K']} noise draws per mechanism and budget, {len(cfg['epsilon'])} epsilon values, and {totals['actual_assignment_solves']:,} actual assignment solves**. The canonical table contains {totals['raw_rows']:,} rows, including explicit endpoint aliases and reference draws.","","Every cost and regret below is a **sum over the n matched pairs**. Epsilon stays fixed in comparisons across dimension, sample size and distributions. The two objective panels use Euclidean and squared Euclidean costs separately.","","## Numerical record","","- [Master draw table](analysis_table.csv.gz)","- [Configuration means and 95% intervals](configuration_summary.csv)","- [Geometry summaries and noise variation](geometry_summaries.csv.gz)","- [Prespecified paired comparisons](paired_comparisons.csv)","- [Selected epsilon table](selected_epsilon_table.csv)","- [Runtime and job IDs](runtime_by_geometry.csv)","- [Verification counts and checksum](verification.json)","- [Data dictionary](data_dictionary.json)","- [Figure inputs](figures/plot_input.csv)","","The master table names the saved complete permutation and geometry for every draw. All checkpoint source checksums and the master decompressed row count were verified. Workers independently reconstructed costs from every saved matching, checked integrality, and checked realized regret bounds, including the anchor secant certificate.","","## Design and interpretation","",f"Budgets: {', '.join(str(e) for e in cfg['epsilon'])}. The populations have known RMS radius one; they are not rescaled using the realized private sample. U is a rotated cube, G an isotropic Gaussian, A01/A20 full-rank anisotropic Gaussians, and M01/M04/M10 four-component mixtures with unequal component variances and increasing overlap.","","Panel A holds n=64 and changes dimension. Panel B changes n at d=32 and d=512. Panel C introduces noisy partners and still solves OPT independently. The public sample span has rank at most min(d,n−1), even for a full-rank generating population.","","Gumbel protects the released final matching under a trusted curator. The representation and cost-noise mechanisms also protect their local reports. These output guarantees differ and are labelled in the master table. All mechanisms use the same pure Euclidean metric-privacy parameter.","","Intervals are pointwise 95% Student t intervals across independent geometry means, each averaging K noise draws. They describe variation across both fresh geometry and mechanism noise. Cross-mechanism comparisons pair the same geometry; draw indices alone are not treated as proof of a shared noise draw. Pilot and production are separate fingerprints and are never pooled.","","The uniform-random reference line uses its exact conditional expected total cost, sum(C)/n. The division by n here is the probability that a particular edge is used by a random permutation; the resulting quantity remains a sum of n costs. Sampled random permutations are also retained.","","Dashed curves are proved expected-total-regret bounds in the matching mechanism's color. Solid curves are empirical means. These bounds concern expectations, not a ceiling on each draw or a simultaneous confidence band. Empirical-detail figures identify when bounds extend above the view, and matching full-range figures show all bounds. Squared-cost bounds have no finite domain cap for the unbounded Gaussian populations.","","## Readable results at epsilon 8","","The table below uses the main dimension panel. OPT and random columns are mean total costs; every mechanism column reports mean total regret with its pointwise 95% interval.","","| Family | d | p | OPT total cost | Random total cost | Mechanism | Total regret [95% CI] | Expected bound |","|---|---:|---:|---:|---:|---|---:|---:|"]
    for row in summaries:
        if row["panel"]=="A" and row["epsilon"]==8. and row["d"] in (32,512):
            lines.append(f"| {row['family']} | {row['d']} | {row['objective_p']} | {row['optimum_total_cost_mean']:.4f} | {row['optimum_total_cost_mean']+row['uniform_expected_regret_mean']:.4f} | {LABELS[row['mechanism']]} | {row['total_regret_mean']:.4f} [{row['total_regret_ci_low']:.4f}, {row['total_regret_ci_high']:.4f}] | {row['bound_total']:.4f} |")
    lines += ["","## Prespecified comparisons","","Differences are left mechanism minus right mechanism in total regret; negative favors the left. These examples use G, n=64, d=512, epsilon=8. They are descriptive pointwise comparisons without multiplicity adjustment.","","| p | Left − right | Mean difference | 95% interval |","|---:|---|---:|---:|"]
    for row in pairs:
        if row["panel"]=="A" and row["family"]=="G" and row["d"]==512 and row["epsilon"]==8.:
            lines.append(f"| {row['objective_p']} | {LABELS[row['left']]} − {LABELS[row['right']]} | {row['mean_total_regret_difference']:.4f} | [{row['ci_low']:.4f}, {row['ci_high']:.4f}] |")
    lines += ["","## Computation","",f"Measured worker time summed across geometries: **{totals['simulation_worker_seconds']/3600:.3f} CPU-hours**; export time {totals['export_seconds']:.1f} seconds. Worker totals include geometry preparation, noise generation, all assignments, verification, checkpoint compression and per-geometry CSV writing. The canonical master is {totals['master_bytes']/1e6:.1f} MB compressed. Per-geometry Slurm job IDs and hostnames are retained in the runtime table.","","Runtime per mechanism distinguishes assignment solves from score construction and stability checks. Public setup, common noise generation and reusable projections are measured separately. Full-span/full-fold endpoints reuse the equivalent ambient sample, with zero incremental solve time; they are not counted as fresh independent observations.","","## Figures","",f"{len(figures)} figure panels are available in PNG and PDF. All plotted numerical inputs are saved separately.",""]
    for figure in figures:
        lines += [f"### {figure['file'][:-4]}","",figure["caption"],"",f"![{figure['caption']}](figures/{figure['file']})","",f"[PDF](figures/{figure['pdf']})",""]
    text="\n".join(lines)
    (output/"report.md").write_text(text)
    try:
        import markdown
        body=markdown.markdown(text,extensions=["tables"])
    except ImportError:
        body=basic_html(text)
    (output/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>Integral matching results</title><style>body{max-width:1400px;margin:2em auto;font:16px sans-serif;padding:1em}img{max-width:100%}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.35em}pre{white-space:pre-wrap}</style>'+body)


def finalize(directory):
    refuse_login_compute("finalize integral matching figures and report")
    directory=Path(directory)
    output=directory/"analysis_handoff"
    manifest,summaries,pairs,runtime,totals=export(directory,output)
    figures=plot_all(output,summaries)
    figures.extend(plot_completion(output,summaries,runtime))
    atomic_json(output/"figures/figure_manifest.json",figures)
    report(output,manifest,summaries,pairs,runtime,totals,figures)
    atomic_json(directory/"COMPLETE.json",dict(fingerprint=manifest["fingerprint"],verification=totals,figures=len(figures),report="analysis_handoff/report.md"))
    print(f"COMPLETE REPORT: {output/'report.md'}",flush=True)
