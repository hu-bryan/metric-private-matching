"""Audited handoff and illustrations distinguishing bounds from observed transitions."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import t
from . import controlled_sweeps as study
from .clustering_benchmark import sha256
from .clustering_manuscript_report import COLORS, LABELS, STYLES
from .integral_mechanisms import solve
from .integral_study import atomic_json
from .login_guard import refuse_login_compute


def interval(values):
    values=np.asarray(values,float); mean=float(values.mean())
    se=float(values.std(ddof=1)/np.sqrt(len(values)))
    half=float(t.ppf(.975,len(values)-1)*se)
    return mean,mean-half,mean+half


def export(root, output):
    refuse_login_compute("audit/render controlled sweeps")
    root,output=Path(root),Path(output); output.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((root/"manifest.json").read_text()); study.verify_runtime(manifest)
    cfg=manifest["design"]
    registry=json.loads((root/"master_registry.json").read_text())
    for rep in range(cfg["geometries"]):
        record=registry[str(rep)]
        assert sha256(root/record["file"])==record["sha256"]
        expected=study.master_arrays(cfg,rep)
        with np.load(root/record["file"]) as z:
            for name,value in expected.items():
                np.testing.assert_array_equal(value,z[name])
    instance_rows=[]; draw_chunks=[]; total_solves=0; files={}
    for unit in study.units(cfg):
        stem=root/"units"/unit["unit_id"]
        meta=json.loads(stem.with_suffix(".json").read_text())
        assert meta["fingerprint"]==manifest["fingerprint"] and meta["unit"]==unit
        assert sha256(stem.with_suffix(".npz"))==meta["sha256"]
        files["../units/"+stem.name+".npz"]=meta["sha256"]
        files["../units/"+stem.name+".json"]=sha256(stem.with_suffix(".json"))
        Q,R=study.inputs(root,unit); n,d=Q.shape
        assert study.manuscript.array_sha(Q)==meta["Q_sha256"] and study.manuscript.array_sha(R)==meta["R_sha256"]
        C=cdist(R,Q); opt=float(C[np.arange(n),solve(C)].sum()); random=float(C.sum()/n)
        with np.load(stem.with_suffix(".npz")) as z:
            a={name:z[name] for name in z.files}
        pi=a["permutations"]
        assert np.all(np.sort(pi,axis=-1)==np.arange(n)) and opt==float(a["optimum"])==0.
        costs=C[np.arange(n),pi].sum(axis=-1)
        np.testing.assert_allclose(costs,a["total_cost"],atol=1e-10,rtol=1e-12)
        np.testing.assert_allclose(costs-opt,a["regret"],atol=1e-10,rtol=1e-12)
        np.testing.assert_allclose(random,a["RAND"],atol=1e-10,rtol=1e-12)
        assert np.all(a["regret"]<=a["row_oscillation"]+1e-7)
        for arm,record in meta["aliases"].items():
            if arm=="original_endpoint":
                for field in ("regret","permutations","row_oscillation"):
                    np.testing.assert_array_equal(a[field][:,2],a[field][:,1])
                continue
            with np.load(root/"units"/(record["unit"]+".npz")) as ref:
                for field in ("regret","permutations","row_oscillation"):
                    np.testing.assert_array_equal(a[field][:,study.ARMS.index(arm)],ref[field][:,study.ARMS.index(record["arm"])])
        ie,ia,draw=np.indices(a["regret"].shape)
        draw_chunks.append(pd.DataFrame(dict(unit_id=unit["unit_id"],rep=unit["rep"],n=n,d=d,
            epsilon=np.asarray(cfg["epsilon"])[ie.ravel()],mechanism=np.asarray(study.ARMS)[ia.ravel()],draw=draw.ravel(),
            total_regret=a["regret"].ravel(),regret_per_pair=a["regret"].ravel()/n,
            fraction_of_random_gap=a["regret"].ravel()/(random-opt),row_oscillation=a["row_oscillation"].ravel())))
        factor=np.array([np.log(n),d,min(meta["public"]["public_rank"]+1,d),n])
        np.testing.assert_allclose(a["bound_factor"],factor)
        for ie,e in enumerate(cfg["epsilon"]):
            np.testing.assert_allclose(a["bound_uncapped"][ie],2*n*factor/e)
            np.testing.assert_allclose(a["bound"][ie],np.minimum(n*meta["public"]["public_diameter"],2*n*factor/e))
            for ia,arm in enumerate(study.ARMS):
                instance_rows.append(dict(setting_id=unit["setting"]["setting_id"],rep=unit["rep"],n=n,d=d,
                    panels=";".join(unit["setting"]["panels"]),epsilon=e,mechanism=arm,K=cfg["K"],
                    total_regret=float(a["regret"][ie,ia].mean()),regret_per_pair=float(a["regret"][ie,ia].mean()/n),
                    fraction_of_random_gap=float(a["regret"][ie,ia].mean()/(random-opt)),
                    RAND_per_pair=random/n,OPT=opt,public_rank=meta["public"]["public_rank"],
                    effective_rank=meta["effective_rank"],fold_dimension=min(3,d),
                    bound_factor=factor[ia],bound_per_pair=a["bound"][ie,ia]/n,
                    bound_uncapped_per_pair=a["bound_uncapped"][ie,ia]/n,
                    source_unit=meta["aliases"].get(arm,{}).get("unit","")))
        total_solves+=meta["actual_assignment_solves"]
    instances=pd.DataFrame(instance_rows); draws=pd.concat(draw_chunks,ignore_index=True)
    expected=len(study.units(cfg))*len(cfg["epsilon"])*4*cfg["K"]
    assert len(draws)==expected
    instances.to_csv(output/"instance_summary.csv",index=False)
    draws.to_csv(output/"draws.csv.gz",index=False)
    summary=[]; differences=[]
    keys=["setting_id","n","d","panels","epsilon","mechanism"]
    for key,frame in instances.groupby(keys):
        assert len(frame)==cfg["geometries"] and frame.rep.nunique()==cfg["geometries"]
        row=dict(zip(keys,key));row["geometry_count"]=len(frame)
        for metric in ("total_regret","regret_per_pair","fraction_of_random_gap","RAND_per_pair","effective_rank",
                       "public_rank","fold_dimension","bound_factor","bound_per_pair","bound_uncapped_per_pair"):
            row[metric],row[metric+"_low"],row[metric+"_high"]=interval(frame[metric])
        summary.append(row)
    keys=["setting_id","n","d","panels","epsilon"]
    for key,frame in instances.groupby(keys):
        pivot=frame.pivot(index="rep",columns="mechanism",values="regret_per_pair")
        for arm in ("original_points","projected_representation"):
            mean,low,high=interval(pivot[arm]-pivot.gumbel)
            differences.append(dict(zip(keys,key),mechanism=arm,minus="gumbel",mean=mean,low=low,high=high))
    summary=pd.DataFrame(summary);paired=pd.DataFrame(differences)
    summary.to_csv(output/"summary.csv",index=False);paired.to_csv(output/"paired_differences.csv",index=False)
    atomic_json(output/"unit_checksums.json",files)
    verification=dict(status="passed",fingerprint=manifest["fingerprint"],geometries=cfg["geometries"],
        configurations=len(cfg["settings"]),units=len(study.units(cfg)),draw_labels=len(draws),
        actual_assignment_solves=total_solves,all_costs_permutations_aliases_inputs_and_bounds_audited=True,
        all_master_inputs_regenerated=True,summary_sha256=sha256(output/"summary.csv"),
        draws_sha256=sha256(output/"draws.csv.gz"),pilot_draws_excluded=True)
    atomic_json(output/"verification.json",verification)
    for filename in ("manifest.json","master_registry.json"):
        shutil.copyfile(root/filename,output/filename)
    for name in ("masters","units","source_snapshot"):
        if (root/name).exists() and not (output/name).exists():
            (output/name).symlink_to("../"+name,target_is_directory=True)
    atomic_json(output/"data_dictionary.json",dict(draws="one epsilon/arm/noise label; row order joins permutations via unit_id and epsilon/arm/draw indices",
        permutation="units/<unit_id>.npz permutations[epsilon_index,arm_index,draw,private_row]",
        inputs="masters/geometry_XX.npz has points_n, q_order_n, r_order_n; pad exactly d-2 zero coordinates",
        inference="K noise average then 20 independent jitter geometries; pointwise Student-t95%; paired differences share geometry",
        seed="SHA256 JSON tuple(namespace,phase,unit_id,draw,channel), first16bytes little-endian, PCG64",
        arms=list(study.ARMS),epsilon=cfg["epsilon"],reuse="unit JSON aliases name exact source; invariant arms are not independent reruns",
        metric="sum-record Euclidean in unit-radius coordinates; matched-cloud diagnostic, not real-record resampling"))
    figures(summary,root,output)
    report(manifest,summary,paired,verification,output)
    print(json.dumps(verification),flush=True)


def figures(summary,root,output):
    directory=output/"figures";directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10,"font.family":"DejaVu Sans","pdf.fonttype":42,
                        "axes.spines.top":False,"axes.spines.right":False})
    low,mid,high="#dcebf9","#e9e0e7","#f6d5d5"
    def save(fig,name):
        for suffix in ("png","pdf"):
            fig.savefig(directory/(name+"."+suffix),dpi=200,bbox_inches="tight")
        plt.close(fig)
    def frame(panel,e):
        return summary[(summary.panels.str.contains(panel)) & (summary.epsilon==e)]
    def decorate(ax,panel,wide=False):
        if panel=="dimension":
            limit=128 if wide else 16
            ax.set_xlim(2,limit);ax.set_xlabel("Ambient dimension d")
            ax.axvspan(2,np.log(64),color=low,zorder=0)
            ax.axvspan(np.log(64),min(64,limit),color=mid,zorder=0)
            if wide: ax.axvspan(64,128,color=high,zorder=0);ax.axvline(64,color=".5",ls=":",lw=1)
            ax.axvline(np.log(64),color=".4",ls=":",lw=1)
            ax.set_xticks([2,4,6,8,10,12,16] if not wide else [2,32,64,96,128])
        else:
            ax.set_xscale("log",base=2);ax.set_xlim(4,512);ax.set_xlabel("Sample size n per side")
            ax.set_xticks([4,8,16,32,64,128,256,512],["4","8","16","32","64","128","256","512"])
            ax.axvspan(4,np.exp(4),color=mid,zorder=0);ax.axvspan(np.exp(4),512,color=low,zorder=0)
            ax.axvline(np.exp(3),color=".5",ls="--",lw=1);ax.axvline(np.exp(4),color=".4",ls=":",lw=1)
        ax.grid(axis="y",alpha=.18);ax.set_axisbelow(True)
    def empirical(ax,panel,e,wide=False):
        f=frame(panel,e);column="d" if panel=="dimension" else "n"
        handles=[]
        for arm in study.ARMS:
            v=f[f.mechanism==arm].sort_values(column)
            line,=ax.plot(v[column],v.regret_per_pair,color=COLORS[arm],ls=STYLES[arm],lw=1.5,label=LABELS[arm]);handles.append(line)
            ax.fill_between(v[column],v.regret_per_pair_low,v.regret_per_pair_high,color=COLORS[arm],alpha=.15,lw=0)
        v=f[f.mechanism=="gumbel"].sort_values(column)
        line,=ax.plot(v[column],v.RAND_per_pair,color=".35",ls="--",lw=1.2,label="(RAND − OPT)/n");handles.append(line)
        decorate(ax,panel,wide);ax.set_ylim(bottom=0);ax.set_ylabel("(E[u] − OPT) / n")
        return handles
    fig,axes=plt.subplots(2,2,figsize=(12,7.5));fig.subplots_adjust(hspace=.46,wspace=.23,bottom=.16,top=.92)
    for j,panel in enumerate(("dimension","sample_size")):
        handles=empirical(axes[0,j],panel,8.)
        axes[0,j].set_title("Fixed geometry · n=64 · ε=8" if j==0 else "Matched circles · d=4 · ε=8")
        f=frame(panel,8.);column="d" if j==0 else "n"
        for arm in study.ARMS:
            v=f[f.mechanism==arm].sort_values(column)
            axes[1,j].plot(v[column],v.bound_factor,color=COLORS[arm],ls=STYLES[arm],lw=1.5)
        decorate(axes[1,j],panel);axes[1,j].set_yscale("log");axes[1,j].set_ylabel("Uncapped bound factor εB / (2n)")
        axes[1,j].set_title("Analytic upper-bound factors")
    fig.legend(handles=handles,loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.51,.01))
    save(fig,"controlled_transitions")
    fig,axes=plt.subplots(1,2,figsize=(12,3.8));fig.subplots_adjust(bottom=.26,wspace=.23)
    for j,panel in enumerate(("dimension","sample_size")):
        handles=empirical(axes[j],panel,4.);axes[j].set_title("Fixed geometry · ε=4" if j==0 else "Matched circles · ε=4")
    fig.legend(handles=handles,loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.51,-.015))
    save(fig,"epsilon4_sensitivity")
    fig,axes=plt.subplots(1,2,figsize=(12,3.8));fig.subplots_adjust(bottom=.26,wspace=.23)
    for ax,e in zip(axes,(4.,8.)):
        handles=empirical(ax,"dimension",e,True);ax.set_title(f"Full dimension range · ε={e:g}")
    fig.legend(handles=handles,loc="lower center",ncol=5,frameon=False,bbox_to_anchor=(.51,-.015))
    save(fig,"dimension_full_range")
    fig,axes=plt.subplots(1,4,figsize=(13,3.1))
    with np.load(root/"masters/geometry_00.npz") as master:
        for ax,n,title in zip(axes,(64,4,16,64),("Fixed cloud; append zeros","n=4","n=16","n=64")):
            points=master[f"points_{n}"] if f"points_{n}" in master else master["points_64"]
            ax.plot(np.cos(np.linspace(0,2*np.pi,200)),np.sin(np.linspace(0,2*np.pi,200)),color=".8",lw=.8)
            ax.scatter(points[:,0],points[:,1],s=18,color=COLORS["original_points"])
            ax.scatter(points[:,0],points[:,1],s=42,facecolors="none",edgecolors=COLORS["gumbel"],lw=.7)
            ax.set_aspect("equal");ax.set_xlim(-1.3,1.3);ax.set_ylim(-1.3,1.3);ax.axis("off");ax.set_title(title)
        axes[0].text(.5,-.02,"(x, y) → (x, y, 0, …, 0)",transform=axes[0].transAxes,ha="center",fontsize=10)
    fig.subplots_adjust(wspace=.22,bottom=.16)
    save(fig,"controlled_setup")


def report(manifest,summary,paired,verification,output):
    cfg=manifest["design"]
    text=["# Controlled dimension and sample-size transitions","",
        "These experiments replace changing-pixel MNIST sweeps with a controlled geometric diagnostic. The main result shows how ambient dimension hurts original-point perturbation at fixed distances, and how increasing the number of distinct matching alternatives changes the preferred mechanism. Theoretical crossings are shown alongside measured errors, without claiming that upper-bound crossings predict exact empirical crossings.","",
        "![Controlled geometric setup](figures/controlled_setup.png)","",
        "Filled green public points and blue private rings coincide, with independently permuted row identities. Both sides contain the same unique points, so OPT=0 exactly. This is a deliberately matched synthetic cloud, not disjoint-source MNIST or a claim that real private records equal public records. All four mechanisms still receive their usual inputs and noise; no noiseless diagnostic arm is substituted.","",
        "**Construction.** Each of 20 independent geometry masters supplies stratified circle angles with uniform bin jitter in [−0.2,0.2]. Antipodal pairs give radius one, public diameter two, affine rank two and effective rank close to two. The d-sweep fixes n=64 and the exact same points/row order, appending zeros in dimensions 2,3,4,5,6,8,10,12,16,24,32,48,64,96,128. No normalization or distances change with d. The n-sweep fixes ambient d=4 and uses n=4,6,8,12,16,20,24,32,40,48,56,64,96,128,192,256,384,512. Increasing n refines the same circle with unique points; its sets are paired through a jitter master but are not nested. OPT remains zero, diameter two and scale one; RAND/n changes modestly with discretization and is drawn explicitly.","",
        "**Privacy and estimation.** Epsilon is fixed in these unit-radius Euclidean coordinates: 8 for the main comparison and 4 for sensitivity. Privacy uses the sum of recordwise Euclidean displacements, without additional n/d budget scaling. Gumbel releases the curator's matching; other arms support private local reports. K=50 noise draws are averaged within each geometry, then across 20 independent geometry means. Bands are pointwise 95% Student-t intervals; paired mechanism differences use those same 20 geometry pairs. There are no five artificial partitions of an identical matching cloud. Main axes show per-match regret, so changes in n do not merely add more summed terms; total regret and the fraction of RAND−OPT are retained in the tables.","",
        "![Measured transitions and analytic bounds](figures/controlled_transitions.png)","",
        "Top: measured regret at ε=8; the d-axis is linear and zoomed to 2–16, while the n-axis uses log₂ spacing to display small and large counts together. Dashed RAND/n references expose residual task-scale changes. Bottom: the exact uncapped theoretical factors, with a logarithmic factor axis; these are analytic upper bounds, not fitted error curves. Blue backgrounds mark d≤ln(n), the blended color ln(n)<d<n, and rose d≥n in the full-range figure. Dotted lines mark d=ln(64)≈4.159 and n=e⁴≈54.60 (original points versus Gumbel bounds); the dashed vertical n=e³≈20.09 line marks projection versus Gumbel bounds. These are bound crossings, not fitted empirical change points.","",
        "For total regret B, the four uncapped bounds are **2nd/ε** (original points), **2ns/ε** (projection), **2n²/ε** (distance vectors), and **2n ln(n)/ε** (Gumbel), where s=min(rank(Q)+1,d). Thus the plotted factors εB/(2n) are d, s, n and ln(n). Per-match bounds equal twice these factors divided by ε. Every Euclidean bound also admits the common cap D_Q=2 per match; both capped and uncapped bounds are saved. RAND/n is a baseline, not a universal upper bound. The factor plot retains uncapped expressions so their comparisons remain visible when a certificate is otherwise vacuous.","",
        "**Rank and projection.** Intrinsic rank stays two while ambient d changes. Projection uses two span coordinates plus the full perpendicular-distance coordinate: s=3 for d≥3, with s=2 at the d=2 original-point endpoint. The d=2→3 change in its noise dimension is therefore intentional. For d≥3 its entire report law is isometric to the n=64,d=3 original-point report. Those realizations are reused exactly, as are dimension-invariant Gumbel and distance-vector assignments. Flat curves are thus scientific invariances with explicit provenance, not independent draws that happened to agree. The n=64,d=4 setting is shared by both sweeps and computed once.","",
        "![Full ambient-dimension range](figures/dimension_full_range.png)","",
        "The wider linear d-range reaches d=n=64 and d>n, displaying saturation and the original-point versus distance-vector bound-factor crossing. The low-d zoom above prevents the important ln(n) region from being compressed into the left edge.","",
        "![Epsilon-four sensitivity](figures/epsilon4_sensitivity.png)","",
        "The ε=4 run uses the same production geometries and coupled unit noise but its own calibration. It checks whether the qualitative order changes survive a second budget; it is not an independent geometry sample. Neither budget was selected after looking at production outcomes.","",
        "**Selected measured results (ε=8).** All values below are per-match regret.","",
        "| Panel | n | d | Original points | Projection | Gumbel | Distance vectors | RAND/n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for panel,ns,ds in [("dimension",64,d) for d in (2,4,6,8,16,64,128)]+[("sample_size",n,4) for n in (4,8,16,20,32,56,128,512)]:
        f=summary[(summary.n==ns)&(summary.d==ds)&(summary.epsilon==8.)].set_index("mechanism")
        if f.empty:continue
        values=[f.loc[a,"regret_per_pair"] for a in ("original_points","projected_representation","gumbel","distance_vectors")]
        text.append(f'| {panel} | {ns} | {ds} | '+" | ".join(f"{v:.4f}" for v in values)+f' | {f.loc["gumbel","RAND_per_pair"]:.4f} |')
    text += ["","**Paired evidence for observed order changes.** A negative difference below favors the named mechanism over Gumbel; a positive difference favors Gumbel. These are pointwise intervals on the tested grid, not simultaneous confidence sets for an exact crossover.","",
             "| Panel | Mechanism − Gumbel | Largest tested x favoring named arm | Smallest tested x favoring Gumbel |","|---|---|---|---|"]
    for panel in ("dimension","sample_size"):
        column="d" if panel=="dimension" else "n"
        for arm in ("original_points","projected_representation"):
            f=paired[(paired.panels.str.contains(panel))&(paired.epsilon==8.)&(paired.mechanism==arm)].sort_values(column)
            signs=[f'{r[column]:g}: {r["mean"]:.4f} [{r["low"]:.4f}, {r["high"]:.4f}]' for _,r in f.iterrows() if r.high<0]
            other=[f'{r[column]:g}: {r["mean"]:.4f} [{r["low"]:.4f}, {r["high"]:.4f}]' for _,r in f.iterrows() if r.low>0]
            text.append(f'| {panel} | {LABELS[arm]} | '+(signs[-1] if signs else "None")+" | "+(other[0] if other else "None")+" |")
    text += ["","**How to read the result.** The fixed-distance d-sweep isolates ambient-coordinate sensitivity: changes in pixel information, OPT, nearest-match geometry and RAND−OPT cannot create a hump. The n-sweep is a growing-alternative experiment on a common low-rank manifold; discretization still changes, so it does not claim to vary n while preserving every pairwise distance. Projection's rank-aware guarantee should be compared using s=3, rather than ambient d=4. A measured switch need not occur exactly at e³ or e⁴ because these compare upper bounds over arbitrary geometries. This controlled example illustrates a mechanism and a regime comparison; it does not establish a universal empirical transition or replace the real-data benchmarks.","",
        "**Exploration and reproducibility.** A separately seeded exploratory pilot compared line, circle, spherical and Gaussian matched clouds, dimensions 4/6, sample sizes 4–256, and ε=2,4,8,16 (K=20). The circle was chosen for its fixed-radius interpretation and visible order changes. The complete pilot table and script are archived; no exploratory draws enter production. A separate gate pilot checks replay and the largest production configuration before submission. Production was frozen before any production outcome was inspected.","",
        f'Fingerprint `{manifest["fingerprint"]}`: {verification["units"]} checkpoints, {verification["draw_labels"]:,} draw labels, {verification["actual_assignment_solves"]:,} new assignment solves. Every assignment/cost/OPT/RAND/alias/input/bound is audited; master inputs are independently regenerated from their seeds. Reference reuse is indexed in each unit JSON. Source, runtime binaries, numerical threads and CPU dispatch are pinned. Active files use /vast; durable files use /ix1; /ihome contains this report and links.',"",
        "[Summary and intervals](summary.csv) · [Paired differences](paired_differences.csv) · [Geometry-level means](instance_summary.csv) · [All draw labels](draws.csv.gz) · [Manifest](manifest.json) · [Data dictionary](data_dictionary.json) · [Checksums](unit_checksums.json) · [Verification](verification.json) · [Frozen source](source_snapshot/)","",
        "Rebuild with the archived environment and frozen source on a compatible CRC compute node: `python -m private_matching analysis-table --input RUN --output RUN/analysis_handoff`. Run stages through `study --study controlled_sweeps`; compute n64,d3 references before other d configurations. Archived Slurm scripts specify the ordering, threads and concurrency. Representative original-environment replay and durable-copy verification accompany publication.",""]
    (output/"report.md").write_text("\n".join(text))
