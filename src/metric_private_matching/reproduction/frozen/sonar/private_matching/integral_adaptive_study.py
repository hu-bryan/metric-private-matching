"""Fresh screening/confirmation experiments for integral adaptive mechanisms."""
from __future__ import annotations
import csv,gzip,hashlib,json,os,platform,time
from pathlib import Path
import numpy as np
import scipy
from private_matching.integral_adaptive import (ARMS,BASE_ARMS,public_plan,unit_noise,candidate,total_cost,portfolio,guard,bound_for,public_surrogate_choice,robust_choice,regret_certificate)
from private_matching.integral_mechanisms import Geometry,solve
from private_matching.integral_study import atomic_json,seed,make_geometry,save_npz
from private_matching.integral_report import interval,write_csv
from private_matching.login_guard import refuse_login_compute

NAME="integral_adaptive_v1"
ALL_ARMS=ARMS+("oracle_NONPRIVATE",)
FIELDS=("fingerprint","phase","setting_id","geometry_id","geometry_rep","family","mode","n","d","epsilon","mechanism","draw","total_cost","total_regret","optimum_total_cost","random_expected_total_cost","expected_bound","no_worse_random_cap_valid","source_channel","source_arm","selected","guard_claim","guard_accepted","guard_false_claim","guard_harmful_accept","guard_harm_bound","guard_proposal_improvement","selection_excess_bound","conditional_expected_cost","candidate_count","cert_upper_regret","cert_covered","cert_below_rand","cert_noise_scale","privacy_output")
AUX=("cert_upper_regret","cert_covered","cert_below_rand","guard_accepted","guard_false_claim","guard_harmful_accept")


def design(pilot):
    settings=[]
    dims=(8,32,512) if pilot else (8,32,128,512)
    for family in ("U","G","A01","A20","M01","M04","M10"):
        for d in dims:
            settings.append(dict(family=family,n=64,d=d,mode="iid",tau=None))
    for family in ("G","M04"):
        for d in (32,512):settings.append(dict(family=family,n=64,d=d,mode="paired",tau=.05))
    for d in (32,512):
        for mode in ("private_scale10","private_weak_spikes"):
            settings.append(dict(family="A01",n=64,d=d,mode=mode,tau=None))
    for s in settings:s["setting_id"]=f'{s["family"]}_n{s["n"]}_d{s["d"]}_{s["mode"]}'
    return dict(kind=NAME,phase="screening" if pilot else "confirmation",settings=settings,
                geometries=3 if pilot else 20,K=10 if pilot else 50,
                epsilon=[2.,4.,8.] if pilot else (np.arange(1,17)/2).tolist(),
                seed_namespace="integral-adaptive-20261110-v1",arms=list(ALL_ARMS),
                raw_fields=list(FIELDS),
                objective="sum_squared_euclidean",private_metric="sum_record_euclidean",
                public_surrogate_draws=6,guard_alpha=.05,
                note="Fresh geometries and noise; no pooling with original suite or screening. Surrogates use only Y. Oracle is NONPRIVATE.")


def resolve(results_root,pilot):
    cfg=design(pilot)
    source={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("integral_adaptive.py","integral_adaptive_study.py","integral_mechanisms.py","integral_study.py","mechanisms.py")}
    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__)
    manifest=dict(kind=NAME,design=cfg,source_sha256=source,versions=versions)
    manifest["fingerprint"]=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()[:24]
    return Path(results_root)/NAME/cfg["phase"]/manifest["fingerprint"],manifest


def units(cfg):
    return [dict(s,geometry_rep=g,geometry_id=f'{s["setting_id"]}_g{g:02d}') for s in cfg["settings"] for g in range(cfg["geometries"])]


def geometry(cfg,unit):
    geo,extra=make_geometry(dict(cfg,phase=cfg["phase"]),unit)
    rng=np.random.default_rng(seed(cfg["seed_namespace"],cfg["phase"],unit["geometry_id"],"stress_private_X"))
    if unit["mode"]=="private_scale10":
        X=10*rng.normal(size=geo.X.shape)/np.sqrt(unit["d"])
        geo=Geometry.build(X,geo.Y)
    if unit["mode"]=="private_weak_spikes":
        X=rng.normal(size=geo.X.shape)/np.sqrt(unit["d"])
        X+=20*rng.normal(size=(unit["n"],4))@geo.B[:,-4:].T
        geo=Geometry.build(X,geo.Y)
    return geo,extra


def run_unit(root,manifest,u):
    stem=root/"units"/u["geometry_id"]
    marker=Path(str(stem)+".json")
    if marker.exists():
        meta=json.loads(marker.read_text())
        assert meta["fingerprint"]==manifest["fingerprint"]
        for suffix,record in meta["files"].items():assert Path(str(stem)+suffix).stat().st_size==record["bytes"]
        print("REUSE",u["geometry_id"],flush=True);return
    start=time.perf_counter();cfg=manifest["design"]
    geo,extra=geometry(cfg,u);plan=public_plan(geo);n=u["n"]
    optimum_pi=solve(geo.C2);optimum=total_cost(geo,optimum_pi);random=float(geo.C2.sum()/n)
    shape=(len(ALL_ARMS),len(cfg["epsilon"]),cfg["K"])
    costs=np.empty(shape);permutations=np.empty((*shape,n),np.uint16)
    auxiliary={field:np.full(shape,np.nan) for field in AUX}
    source_choices=[];policy_scores=[]
    for ie,eps in enumerate(cfg["epsilon"]):
        def public_rng(channel):return np.random.default_rng(seed(cfg["seed_namespace"],cfg["phase"],u["geometry_id"],"public_policy",eps,channel))
        chosen,values=public_surrogate_choice(geo,plan,eps,public_rng,K=cfg["public_surrogate_draws"])
        source_choices.append(chosen);policy_scores.append(values.tolist())
    tmp=Path(str(stem)+".csv.gz.tmp");rows=0;false_claims=0;guard_claims=0
    with gzip.open(tmp,"wt",newline="",compresslevel=4) as handle:
        writer=csv.DictWriter(handle,fieldnames=FIELDS)
        for k in range(cfg["K"]):
            def rng_for(channel):return np.random.default_rng(seed(cfg["seed_namespace"],cfg["phase"],u["geometry_id"],"unit",k,channel))
            noise=unit_noise(geo,plan,rng_for)
            for ie,eps in enumerate(cfg["epsilon"]):
                results={};notes={}
                for arm in BASE_ARMS:
                    pi,score=candidate(geo,plan,noise,eps,arm)
                    osc=float(np.ptp(score-geo.base2,axis=1).sum())
                    assert total_cost(geo,pi)-optimum<=osc+1e-7*max(1,osc)
                    results[arm]=pi;notes[arm]={}
                    alias=noise["block_alias"]
                    if alias and arm=="block_radial":notes[arm]["source_arm"]=alias
                    if alias and arm=="block_oas":notes[arm]["source_arm"]="span_oas" if alias=="span_radial" else "coordinate_oas"
                chosen=source_choices[ie]
                results["public_surrogate_choice"]=results[chosen].copy()
                notes["public_surrogate_choice"]=dict(source_arm=chosen,selected=chosen)
                robust=robust_choice(policy_scores[ie])
                results["robust_public_choice"]=results[robust].copy()
                notes["robust_public_choice"]=dict(source_arm=robust,selected=robust)
                for arm,fractions in (("portfolio_40_40_20",(.4,.4,.2)),("portfolio_30_30_40",(.3,.3,.4))):
                    results[arm],notes[arm]=portfolio(geo,plan,noise,eps,fractions,rng_for(f"{arm}:{eps}"))
                proposals={}
                for arm,kind in (("guard_gumbel","gumbel"),("guard_block_oas","block_oas"),("guard_public_choice",chosen),("guard_robust_choice",robust)):
                    if kind not in proposals:proposals[kind]=candidate(geo,plan,noise,.9*eps,kind)[0]
                    proposal=proposals[kind]
                    results[arm],notes[arm]=guard(geo,proposal,.1*eps,rng_for(f"{arm}:{eps}"),alpha=cfg["guard_alpha"])
                    notes[arm]["source_arm"]=kind
                for arm,kind in (("cert_gumbel","gumbel"),("cert_block_oas","block_oas"),("cert_public_choice",chosen),("cert_robust_choice",robust)):
                    results[arm]=proposals[kind].copy()
                    notes[arm]=dict(source_arm=kind,selected=kind,**regret_certificate(geo,proposals[kind],optimum,.1*eps,rng_for(f"{arm}:{eps}"),alpha=cfg["guard_alpha"]))
                best=min(BASE_ARMS,key=lambda arm:total_cost(geo,results[arm]))
                results["oracle_NONPRIVATE"]=results[best].copy();notes["oracle_NONPRIVATE"]=dict(selected=best,source_arm=best)
                for ia,arm in enumerate(ALL_ARMS):
                    pi=results[arm]
                    assert np.array_equal(np.sort(pi),np.arange(n))
                    value=total_cost(geo,pi)
                    assert value>=optimum-1e-8*max(1,abs(optimum))
                    costs[ia,ie,k]=value;permutations[ia,ie,k]=pi
                    source=notes[arm].get("source_arm",arm)
                    channel=("span" if source.startswith("span") else "coordinate" if source.startswith("coordinate") else "ambient" if source.startswith("ambient") else (noise["block_alias"] or "block") if source.startswith("block") else "gumbel" if source=="gumbel" else "joint:"+arm)
                    row=dict(fingerprint=manifest["fingerprint"],phase=cfg["phase"],**{key:u[key] for key in ("setting_id","geometry_id","geometry_rep","family","mode","n","d")},epsilon=eps,mechanism=arm,draw=k,total_cost=value,total_regret=max(0.,value-optimum),optimum_total_cost=optimum,random_expected_total_cost=random,expected_bound=bound_for(geo,plan,eps,arm),no_worse_random_cap_valid=arm in ("gumbel","span_radial","coordinate_allocated","block_radial","ambient_radial"),source_channel=f'{cfg["phase"]}:{u["geometry_id"]}:unit:{k}:{channel}',privacy_output="NONPRIVATE_DIAGNOSTIC" if arm=="oracle_NONPRIVATE" else ("final_matching_with_private_certificate" if arm.startswith("cert") else ("final_matching_only" if arm.startswith(("guard","portfolio","public","robust")) or arm=="gumbel" else "private_report_postprocessing")))
                    row.update(notes[arm]);writer.writerow(row);rows+=1
                    for field in AUX:
                        if field in notes[arm]:auxiliary[field][ia,ie,k]=notes[arm][field]
                    if arm.startswith("guard"):
                        false_claims+=notes[arm]["guard_false_claim"];guard_claims+=notes[arm]["guard_accepted"]
    os.replace(tmp,Path(str(stem)+".csv.gz"))
    save_npz(Path(str(stem)+".npz"),cost=costs,regret=costs-optimum,permutations=permutations,optimum_permutation=optimum_pi,**auxiliary)
    save_npz(Path(str(stem)+".geometry.npz"),**geo.arrays(),**extra)
    meta=dict(fingerprint=manifest["fingerprint"],unit=u,optimum=optimum,random_expected=random,public=geo.public_metadata(),blocks=plan.blocks,block_rates=plan.rates.tolist(),coordinate_rates=plan.coordinate_rates.tolist(),public_prior_variance=plan.prior_variance.tolist(),oas_shrinkage=plan.oas_shrinkage,block_coefficient=plan.coefficient,public_policy_choices=source_choices,public_policy_scores=policy_scores,elapsed_seconds=time.perf_counter()-start,rows=rows,guard_positive_claims=guard_claims,guard_false_claims=false_claims,slurm_job=os.environ.get("SLURM_JOB_ID"),files={})
    for suffix in (".csv.gz",".npz",".geometry.npz"):
        path=Path(str(stem)+suffix);meta["files"][suffix]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    atomic_json(marker,meta)
    print("COMPLETE",u["geometry_id"],round(meta["elapsed_seconds"],3),flush=True)


def export(root,out):
    refuse_login_compute("export adaptive integral experiments")
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    m=json.loads((root/"manifest.json").read_text());cfg=m["design"]
    arms=tuple(cfg["arms"])
    raw_fields=cfg.get("raw_fields",[field for field in FIELDS if not field.startswith("cert_")])
    grouped={};pairgroup={};metas=[];count=0
    target=out/"analysis_table.csv.gz";tmp=Path(str(target)+".tmp")
    with gzip.open(tmp,"wt",newline="") as h:csv.writer(h).writerow(raw_fields)
    with tmp.open("ab") as h:
        import shutil
        for u in units(cfg):
            stem=root/"units"/u["geometry_id"]
            meta=json.loads(Path(str(stem)+".json").read_text());assert meta["fingerprint"]==m["fingerprint"]
            for suffix,record in meta["files"].items():assert hashlib.sha256(Path(str(stem)+suffix).read_bytes()).hexdigest()==record["sha256"]
            with Path(str(stem)+".csv.gz").open("rb") as src:shutil.copyfileobj(src,h)
            metas.append(meta);count+=meta["rows"]
            with np.load(Path(str(stem)+".npz")) as z:
                regret=z["regret"];cost=z["cost"]
                aux={field:z[field] if field in z.files else np.full(regret.shape,np.nan) for field in AUX}
            means=regret.mean(axis=-1)
            for ia,arm in enumerate(arms):
                for ie,eps in enumerate(cfg["epsilon"]):
                    row=dict(**u,mechanism=arm,epsilon=eps,regret=float(means[ia,ie]),cost=float(cost[ia,ie].mean()),noise_se=float(regret[ia,ie].std(ddof=1)/np.sqrt(cfg["K"])),random_regret=meta["random_expected"]-meta["optimum"],optimum=meta["optimum"],public_policy_choice=meta["public_policy_choices"][ie],public_rank=meta["public"]["public_rank"],blocks=json.dumps(meta["blocks"]))
                    for field in AUX:
                        vals=aux[field][ia,ie]
                        row[field]=float(vals.mean()) if np.all(np.isfinite(vals)) else None
                    grouped.setdefault((u["setting_id"],arm,eps),[]).append(row)
            for left,right in (("block_radial","span_radial"),("block_oas","coordinate_allocated"),("block_oas","gumbel"),("public_surrogate_choice","gumbel"),("guard_public_choice","gumbel"),("portfolio_40_40_20","gumbel"),("portfolio_30_30_40","gumbel")):
                for ie,eps in enumerate(cfg["epsilon"]):pairgroup.setdefault((u["setting_id"],left,right,eps),[]).append(float(means[arms.index(left),ie]-means[arms.index(right),ie]))
    os.replace(tmp,target)
    with gzip.open(target,"rb") as h:lines=sum(chunk.count(b"\n") for chunk in iter(lambda:h.read(4*1024*1024),b""))
    assert lines==count+1
    summary=[]
    for key,rs in sorted(grouped.items()):
        first=rs[0]
        mean,se,lo,hi=interval([r["regret"] for r in rs])
        row={k:first[k] for k in ("setting_id","family","mode","n","d","mechanism","epsilon")}
        random=np.mean([r["random_regret"] for r in rs])
        row.update(phase=cfg["phase"],fingerprint=m["fingerprint"],geometries=len(rs),K=cfg["K"],total_regret_mean=mean,total_regret_se=se,ci_low=lo,ci_high=hi,total_cost_mean=float(np.mean([r["cost"] for r in rs])),random_regret=float(random),optimum_total_cost=float(np.mean([r["optimum"] for r in rs])),fraction_random_gap_removed=float(1-mean/random),public_rank=first["public_rank"])
        for field in AUX:row[field]=None if first[field] is None else float(np.mean([r[field] for r in rs]))
        summary.append(row)
    write_csv(out/"configuration_summary.csv",summary)
    def pairrows():
        for (setting,left,right,eps),values in sorted(pairgroup.items()):
            mean,se,lo,hi=interval(values)
            yield dict(setting_id=setting,left=left,right=right,epsilon=eps,geometries=len(values),mean_difference=mean,se=se,ci_low=lo,ci_high=hi)
    write_csv(out/"paired_comparisons.csv",pairrows())
    write_csv(out/"geometry_summary.csv",(r for rs in grouped.values() for r in rs))
    atomic_json(out/"public_plans.json",metas)
    atomic_json(out/"data_dictionary.json",dict(
        objective="All costs/regrets are SUMS of squared Euclidean distances; never divided by n.",
        phase=cfg["phase"],fingerprint=m["fingerprint"],arms=list(arms),raw_fields=raw_fields,
        replication="Each configuration mean averages independent geometry means, each from K private draws; pointwise Student t confidence intervals across geometries.",
        original_suite="No samples or geometries pooled with the original integral suite. Screening and confirmation use independent seed namespaces via phase.",
        calibration="Public Y only: centered PCA, interval-block partition minimizing the proved coefficient, optimal block L2 privacy budgets, and OAS covariance. No private X, labels, or true costs enter public choices.",
        block_noise="Density proportional to exp(-sum_b lambda_b*||z_b||2), orthogonal blocks, sum_b lambda_b^2=epsilon^2; retains all span coordinates and every assignment slot. Full radial/coordinate endpoints reuse their matching noise channels.",
        oas="Linear shrinkage using public OAS covariance and exact report noise covariance. Unbounded OAS has no asserted finite uniform utility certificate. Bounded OAS clips the correction to one noise standard deviation per coordinate and has a finite uniform regret bound.",
        public_choices="Six surrogate noise draws on one public Gaussian proxy geometry per epsilon. Standard choice uses all base arms; robust choice excludes unbounded OAS arms. A public prior remains an accuracy assumption, not a privacy assumption.",
        portfolios="40/40/20 or 30/30/40 percent of TOTAL epsilon to Gumbel proposal, one block report with free postprocessed decoders, and exponential selection. Includes RAND as an expected-cost action. This is not free selection among full-budget outputs.",
        guards="90% epsilon for proposal, 10% for Laplace validation of improvement over expected RAND cost; use R=max_j||y_j-mean(Y)||, b=2R/(.1epsilon), alpha=.05. Publish lower improvement = max(0,improvement+Lap(b)-b log(1/(2alpha))). Otherwise output independent uniform permutation. Expected excess cost over RAND at most alpha*b/e for any proposal.",
        guard_coverage="Probability of publishing a lower bound larger than true improvement is at most .05 per execution. This is not a conditional false-discovery-rate guarantee among accepted outputs.",
        certificates="90% epsilon for proposal and 10% for a private upper confidence bound on TOTAL regret: max(0,regret+Lap(b)+b log(1/(2alpha))), b=2D_Y/(.1epsilon), alpha=.05. Coverage at least .95 per execution. cert_below_rand compares this private bound with the NONPRIVATE research benchmark; that indicator is not a proposed public release.",
        expected_bound="Proved uniform expected-regret bounds only for indicated raw or bounded arms. no_worse_random_cap_valid records a separately derived baseline-domination fact for additive score noise; original bounds are preserved rather than silently capped.",
        coupling="Public plans shared across arms; channel-specific unit noise reused across epsilon. Raw and denoised decoders share reports. Guard/certificate arms reuse the same 90%-budget proposals but have independent validation channels. These are separate experimental mechanisms, not a claim of privacy for their joint release.",
        oracle="oracle_NONPRIVATE chooses the best realized base output using true costs. It is an explicitly nonprivate diagnostic, never a deployable selector.",
        privacy_scope="Portfolio/guard/public-choice/certificate outputs require a trusted curator; representation report privacy applies to the underlying report. Publishing all raw research records is not the privacy mechanism on real confidential input."))
    atomic_json(out/"verification.json",dict(fingerprint=m["fingerprint"],phase=cfg["phase"],geometries=len(metas),rows=count,all_source_checksums_verified=True,master_row_count_verified=True,all_permutations_integral=True,worker_seconds=sum(meta["elapsed_seconds"] for meta in metas),guard_positive_claims=sum(meta["guard_positive_claims"] for meta in metas),guard_false_claims=sum(meta["guard_false_claims"] for meta in metas)))
    return summary


def run_study(results_root,pilot,stage,shard_index=None,shard_count=None):
    refuse_login_compute("run adaptive integral experiments")
    root,m=resolve(results_root,pilot);cfg=m["design"]
    print("ADAPTIVE RESULTS:",root,flush=True)
    if stage in ("prepare","all"):
        (root/"units").mkdir(parents=True,exist_ok=True)
        atomic_json(root/"manifest.json",m);atomic_json(root/"expected_units.json",units(cfg))
    if stage in ("all","shard"):
        index=0 if shard_index is None else shard_index;count=1 if shard_count is None else shard_count
        assert 0<=index<count
        for j,u in enumerate(units(cfg)):
            if j%count==index:run_unit(root,m,u)
    if stage in ("all","finalize"):
        summary=export(root,root/"analysis_handoff")
        plot(root,summary)
        atomic_json(root/"COMPLETE.json",dict(fingerprint=m["fingerprint"],geometries=len(units(cfg)),phase=cfg["phase"]))
    return root


def plot(root,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out=root/"analysis_handoff";figdir=out/"figures";figdir.mkdir(exist_ok=True)
    colors=dict(gumbel="C0",coordinate_allocated="C1",block_radial="C2",block_bounded_oas="C3",robust_public_choice="C4",cert_robust_choice="C5",portfolio_40_40_20="C6")
    for mode in sorted({r["mode"] for r in rows}):
        fig,axes=plt.subplots(2,4,figsize=(16,8))
        for ax,family in zip(axes.flat,("U","G","A01","A20","M01","M04","M10")):
            for arm,color in colors.items():
                selected=sorted([r for r in rows if r["mode"]==mode and r["family"]==family and r["d"]==512 and r["mechanism"]==arm],key=lambda r:r["epsilon"])
                if not selected:continue
                x=[r["epsilon"] for r in selected];y=[r["total_regret_mean"] for r in selected]
                ax.plot(x,y,color=color,label=arm);ax.fill_between(x,[r["ci_low"] for r in selected],[r["ci_high"] for r in selected],color=color,alpha=.1)
            ref=[r for r in rows if r["mode"]==mode and r["family"]==family and r["d"]==512 and r["mechanism"]=="gumbel"]
            if ref:ax.axhline(ref[0]["random_regret"],color="black",ls=":",label="RAND")
            ax.set(title=f"{family}: {mode}, d=512",xlabel="epsilon",ylabel="Total squared regret (sum)")
        axes.flat[-1].axis("off")
        h,l=next((ax.get_legend_handles_labels() for ax in axes.flat if ax.get_legend_handles_labels()[0]),([],[]))
        fig.legend(h,l,loc="lower center",ncol=4);fig.tight_layout(rect=(0,.08,1,1))
        fig.savefig(figdir/f"{mode}_d512.png",dpi=160);fig.savefig(figdir/f"{mode}_d512.pdf");plt.close(fig)
    if any(r["mechanism"]=="cert_robust_choice" for r in rows):
        fig,axes=plt.subplots(2,4,figsize=(16,8))
        for ax,family in zip(axes.flat,("U","G","A01","A20","M01","M04","M10")):
            for arm,color in (("cert_gumbel","C0"),("cert_robust_choice","C3")):
                selected=sorted([r for r in rows if r["mode"]=="iid" and r["family"]==family and r["d"]==512 and r["mechanism"]==arm],key=lambda r:r["epsilon"])
                if not selected:continue
                x=[r["epsilon"] for r in selected]
                ax.plot(x,[r["total_regret_mean"] for r in selected],color=color,label=arm+" mean regret")
                ax.plot(x,[r["cert_upper_regret"] for r in selected],color=color,ls="--",label=arm+" mean private upper certificate")
                ax.axhline(selected[0]["random_regret"],color="black",ls=":")
            ax.set(title=family,xlabel="TOTAL epsilon",ylabel="Total squared regret / certificate")
        axes.flat[-1].axis("off");h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc="lower center",ncol=2)
        fig.suptitle("Dashed: mean of per-execution 95% private regret certificates; not an expected-regret theorem")
        fig.tight_layout(rect=(0,.08,1,.95));fig.savefig(figdir/"private_regret_certificates.png",dpi=160);fig.savefig(figdir/"private_regret_certificates.pdf");plt.close(fig)
