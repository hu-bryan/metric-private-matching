"""Publish the exact-paper incremental arm alongside immutable historical draws."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from . import clustering_benchmark as cb
from .clustering_benchmark_report import summarize
from .corrected_projection_benchmark import ARM, ARCHIVES, verify_runtime
from .integral_report import COLORS, LABELS, basic_html
from .integral_study import atomic_json

LABEL = dict(LABELS,anchor="Span anchor",corrected_projection="Corrected projection (paper)")
COLOR = dict(COLORS,corrected_projection="#7b3294")
ARMS = ["ambient_radial","ambient_iid","terminal_fold","anchor",ARM,"cost_infinity","cost_iid","gumbel"]
BEGIN = "<!-- exact-corrected-projection:start -->"
END = "<!-- exact-corrected-projection:end -->"


def replace_section(text,block):
    if BEGIN in text:
        return re.sub(re.escape(BEGIN)+r".*?"+re.escape(END),lambda _:block,text,flags=re.S)
    first_break = text.find("\n\n")
    return text[:first_break+2]+block+"\n\n"+text[first_break+2:]


def save_table(frame,path):
    values = frame.pivot(index="epsilon",columns="mechanism",values="score_mean")
    lines = ["# Main benchmark: unsquared Euclidean cost, n=64", "",
             "Mean normalized regret; 0 is OPT and 1 is RAND. Corrected projection uses the exact manuscript construction; Span anchor is the historical variant.","",
             "| epsilon | "+" | ".join(LABEL[a] for a in ARMS)+" |",
             "|---|"+"---:|"*len(ARMS)]
    for eps,row in values.sort_index().iterrows():
        lines.append(f"| {eps:g} | "+" | ".join(f"{row[a]:.3f}" for a in ARMS)+" |")
    path.write_text("\n".join(lines)+"\n")


def curve_panels(frame,figures,stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    datasets = sorted(frame.dataset.unique())
    rows = (len(datasets)+2)//3
    fig,axes = plt.subplots(rows,3,figsize=(14,3.4*rows+.8),squeeze=False)
    for ax,ds in zip(axes.flat,datasets):
        for arm in ARMS:
            points = frame[(frame.dataset == ds)&(frame.mechanism == arm)].sort_values("epsilon")
            ax.plot(points.epsilon,points.score_mean,label=LABEL[arm],color=COLOR[arm],
                    lw=2. if arm == ARM else 1.1,marker="o",markersize=3,
                    linestyle="--" if arm == "anchor" else "-")
            ax.fill_between(points.epsilon.to_numpy(float),points.score_low.to_numpy(float),
                            points.score_high.to_numpy(float),color=COLOR[arm],alpha=.10)
        ax.axhline(1,color="#777",ls=":",lw=.7); ax.axhline(0,color="#aaa",lw=.5)
        ax.set_xscale("log",base=2); ax.set_xticks([.25,1,4,16],labels=[".25","1","4","16"])
        ax.set_title(ds,fontsize=10); ax.set_xlabel("epsilon"); ax.set_ylabel("Normalized regret")
        ax.grid(alpha=.15)
    for ax in axes.flat[len(datasets):]: ax.axis("off")
    handles,labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=4,frameon=False,fontsize=9)
    fig.suptitle("Euclidean matching: exact corrected projection and historical mechanisms\nPointwise 95% intervals over 20 parent collections",fontsize=13)
    fig.tight_layout(rect=(0,.085 if rows <=3 else .055,1,.95 if rows <=3 else .97))
    for suffix in ("png","pdf"): fig.savefig(figures/(stem+"."+suffix),dpi=160,bbox_inches="tight")
    plt.close(fig)


def equal_dataset_curves(parent,figures,stem,title):
    import matplotlib.pyplot as plt
    grouped = parent.groupby(["parent","mechanism","epsilon"]).score.mean().reset_index()
    summary = summarize(grouped,["mechanism","epsilon"],"score")
    fig,ax = plt.subplots(figsize=(9,5.5))
    for arm in ARMS:
        p = summary[summary.mechanism == arm].sort_values("epsilon")
        ax.plot(p.epsilon,p.score_mean,label=LABEL[arm],color=COLOR[arm],marker="o",ms=3,lw=2 if arm == ARM else 1.2)
        ax.fill_between(p.epsilon.to_numpy(float),p.score_low.to_numpy(float),p.score_high.to_numpy(float),color=COLOR[arm],alpha=.10)
    ax.set_xscale("log",base=2); ax.set_xlabel("epsilon"); ax.set_ylabel("Mean normalized regret")
    ax.set_title(title); ax.axhline(1,color="#777",ls=":"); ax.grid(alpha=.15)
    ax.legend(ncol=2,fontsize=8);fig.tight_layout()
    for suffix in ("png","pdf"): fig.savefig(figures/(stem+"."+suffix),dpi=160,bbox_inches="tight")
    plt.close(fig)
    return summary


def extension_group_curves(extension,newparent):
    import matplotlib.pyplot as plt
    historical = pd.read_csv(extension/"separate_group_summary.csv")
    pp = newparent[newparent.panel == "real"].groupby(["parent","mechanism","epsilon"]).score.mean().reset_index()
    added = summarize(pp,["mechanism","epsilon"],"score")
    added["panel"],added["objective_p"] = "real",1
    summary = pd.concat([historical[historical.objective_p == 1],added],ignore_index=True)
    summary.to_csv(extension/"corrected_projection_group_summary.csv",index=False)
    fig,axes = plt.subplots(1,3,figsize=(14,5))
    for ax,panel in zip(axes,("real","g2mg","h2mg")):
        for arm in ARMS:
            p = summary[(summary.panel == panel)&(summary.mechanism == arm)].sort_values("epsilon")
            if p.empty: continue
            ax.plot(p.epsilon,p.score_mean,label=LABEL[arm],color=COLOR[arm],marker="o",ms=3,lw=2 if arm == ARM else 1.2)
            ax.fill_between(p.epsilon.to_numpy(float),p.score_low.to_numpy(float),p.score_high.to_numpy(float),color=COLOR[arm],alpha=.1)
        ax.set_xscale("log",base=2);ax.set_xlabel("epsilon");ax.set_ylabel("Mean normalized regret")
        ax.set_title(panel+" — equal dataset weights");ax.axhline(1,color="#777",ls=":");ax.grid(alpha=.15)
    handles,labels = axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=4,frameon=False,fontsize=8)
    fig.suptitle("Euclidean cost; corrected projection added to nine real datasets only")
    fig.tight_layout(rect=(0,.15,1,.94))
    for suffix in ("png","pdf"):
        fig.savefig(extension/"figures"/("separate_groups_p1."+suffix),dpi=160,bbox_inches="tight")
    plt.close(fig)


PRIVACY = r"""
The public-only representation is identity when $m=\min(k+1,d)=d$; otherwise
$f_Q(x)=(B^T(x-c),\|(I-BB^T)(x-c)\|_2)$, with $c,B$ determined by Q alone.
Orthogonal decomposition and the reverse triangle inequality prove
$\|f_Q(x)-f_Q(x')\|_2\le\|x-x'\|_2$.

The first release is $Y=f_Q(x)+Z$ with density proportional to
$\exp(-\epsilon_y\|Z\|_2)$ in the full m-dimensional representation.
Polar integration gives radius $H\sim\mathrm{Gamma}(m,\mathrm{scale}=1/\epsilon_y)$,
independent uniform direction, and coordinate variance $(m+1)/\epsilon_y^2$.
The triangle inequality bounds its log-density ratio by
$\epsilon_y\|x-x'\|_2$.

Conditional on any fixed first output Y, the nearest public anchor a(Y) is fixed.
The scalar $t(x)=\|f_Q(x)-p_a\|_2=\|x-q_a\|_2$ is 1-Lipschitz. Independently
draw $\eta\sim\mathrm{Laplace}(0,1/\epsilon_s)$ and release
$T=\max(0,t(x)+\eta)$. The unclipped scalar log-density ratio is bounded by
$\epsilon_s\|x-x'\|_2$; clipping is post-processing, including its atom at zero.
Adaptive sequential composition therefore gives
$(\epsilon_y+\epsilon_s)\|x-x'\|_2=\epsilon\|x-x'\|_2$ for the joint report.
Independent per-record reports compose into epsilon times the sum of record
displacements; decoding and assignment are post-processing. There is no extra
n or sqrt(m) factor in either noise scale.

For N nonzero unordered public pairs, the manuscript's experimental split uses
$b_Q=\min(m,\sqrt{2m\log(2N)})$ and
$\epsilon_y:\epsilon_s=2\sqrt{b_Q}:1$. It minimizes
$4b_Q/\epsilon_y+1/\epsilon_s$ subject to their sum being epsilon.
All these quantities use public Q only. Coincident public candidates require
no release and a fixed matching.

The decoder is exactly
$\widehat C_i=\sqrt{T^2+\|Y-p_i\|_2^2-\|Y-p_a\|_2^2}$.
The implementation evaluates the difference by an algebraically equivalent
inner-product expression to avoid subtracting large squared norms. It checks
each draw's secant certificate. An independent audit reconstructs the literal
squared-distance expression for all seven budgets on one saved instance from
each of the 15 datasets, and all stored permutations and true costs are verified.

Unit noise is coupled across alternative epsilon simulations for comparison;
these alternatives are not a jointly private release. Privacy concerns the
designated local messages and matching, not this research archive of public
benchmark inputs and evaluator diagnostics. Raw-data-dependent normalization
has no privacy guarantee; the saved standardized Q/R arrays are reused exactly.
"""


def block(summary,output,prefix=""):
    manifest = json.loads((output/"manifest.json").read_text())
    table = ["| Dataset | epsilon | Corrected projection | Span anchor | Terminal fold |",
             "|---|---:|---:|---:|---:|"]
    for ds in sorted(summary.dataset.unique()):
        for eps in (4.,8.):
            rows = summary[(summary.dataset == ds)&(summary.epsilon == eps)].set_index("mechanism")
            table.append(f"| {ds} | {eps:g} | {rows.loc[ARM,'score_mean']:.3f} | {rows.loc['anchor','score_mean']:.3f} | {rows.loc['terminal_fold','score_mean']:.3f} |")
    text = [BEGIN,"## Exact manuscript corrected projection: all 15 main benchmarks","",
            f"Added production run `{manifest['fingerprint']}`. The new arm implements the v1 manuscript's **Corrected Projected Representation** exactly, including full folded noise dimension and the experimental privacy split. The historical `anchor` arm is labeled **Span anchor** in updated plots. Both remain visible; their noise distributions differ.","",
            "The scope is the six original aggregate datasets and nine real-data additions, n=64. The exact saved standardized inputs, 20 parent collections, five partitions, 50 fresh draws per instance and all seven epsilon values are retained. This adds **525,000 private assignment solves on 1,500 instances**. Pilot and audit draws are excluded. Only unsquared Euclidean cost applies; squared-cost results, synthetic datasets and the old label/n/d sweeps retain their existing scope.","",
            "The updated main-dataset epsilon plots and reading tables compare eight arms. Historical numerical tables remain frozen and identify the seven earlier arms; the supplementary summary contains the new eighth arm. Earlier winner statements and geometry-range analyses concern those historical arms unless explicitly updated here.","",
            "The full regression suite passed **210 tests** on a two-CPU Turin allocation, including analytic noise moments, the full joint density ratio, adaptive composition and the clipping atom. The independent decoder audit and production realization checks also passed. [Test evidence]("+prefix+"corrected_projection/test_validation.txt).", "",
            "Scores below are mean normalized regret, with 0=OPT and 1=RAND. Noise is averaged within each instance, then over five partitions and 20 parents. The full summary contains pointwise 95% Student-t intervals; paired differences use the same parents, with independent mechanism noise.","",
            *table,"",
            f"[Exact-arm summary]({prefix}corrected_projection/summary.csv) · [Paired differences against every historical arm]({prefix}corrected_projection/paired_comparisons.csv) · [Canonical draw table]({prefix}corrected_projection/analysis_table.csv.gz) · [Independent decoder audit]({prefix}corrected_projection/independent_decoder_audit.json) · [Validation]({prefix}corrected_projection/verification.json)","",
            f"![All 15 datasets across epsilon]({prefix}figures/corrected_projection_epsilon_p1.png)","",
            f"![Overview at epsilon 4]({prefix}figures/benchmark_overview_epsilon4.png)","",
            f"![Overview at epsilon 8]({prefix}figures/benchmark_overview_epsilon8.png)","",
            f"![Dataset distribution at epsilon 4]({prefix}figures/benchmark_boxplot_epsilon4_15datasets.png)","",
            f"![Dataset distribution at epsilon 8]({prefix}figures/benchmark_boxplot_epsilon8_15datasets.png)","",
            "### Privacy calibration and implementation checks","",PRIVACY.strip(),"",END]
    return "\n".join(text)


def publish(directory,repo):
    directory,repo = Path(directory),Path(repo)
    output = directory/"analysis_handoff"
    manifest = json.loads((output/"manifest.json").read_text())
    verify_runtime(manifest)
    assert manifest["design"]["phase"] == "production"
    assert json.loads((output/"verification.json").read_text())["actual_canonical_rows"] == 525000
    assert json.loads((output/"independent_decoder_audit.json").read_text())["status"] == "passed"
    combined = pd.read_csv(output/"comparison_summary.csv")
    newparent = pd.read_csv(output/"parent_summaries.csv.gz")
    validation_log = Path(os.environ["PM_CORRECTED_VALIDATION_LOG"])
    validated = validation_log.read_text()
    assert "210 passed" in validated and "failed," not in validated
    research = repo/"research/private_ot_research_20261109"
    extension = research/"clustering_benchmark_extension_results"
    figures = extension/"figures"
    shutil.copyfile(output/"summary.csv",extension/"corrected_projection_summary.csv")
    for archive,panel in ARCHIVES:
        out = archive/"analysis_handoff"
        link = out/"corrected_projection"
        if link.is_symlink():
            assert link.resolve() == output.resolve()
        elif not link.exists():
            link.symlink_to(output,target_is_directory=True)
        else: raise RuntimeError("supplement destination exists")
        frame = combined[combined.panel == panel]
        stem = "aggregate_p1" if panel == "aggregate" else "real_epsilon_p1"
        curve_panels(frame,out/"figures",stem)
        for sid,part in frame.groupby("setting_id"):
            save_table(part,out/"tables"/(sid+"_p1.md"))
        oldparent = pd.read_csv(out/"parent_summaries.csv.gz")
        pp = pd.concat([oldparent[(oldparent.panel == panel)&(oldparent.objective_p == 1)],
                        newparent[newparent.panel == panel]],ignore_index=True)
        equal_dataset_curves(pp,out/"figures","equal_datasets_p1" if panel == "aggregate" else "corrected_real_equal_datasets_p1",
                             "Euclidean cost — equal weights for "+("six original datasets" if panel == "aggregate" else "nine real additions"))
    curve_panels(combined,figures,"corrected_projection_epsilon_p1")
    extension_group_curves(extension,newparent)
    # Freeze plotting programs before executing them; root/output options allow
    # replay from the archived snapshot without assuming its filesystem layout.
    source_root = Path(__file__).resolve().parents[2]
    tools = output/"source_snapshot/theory_checks"; tools.mkdir(parents=True,exist_ok=True)
    for name in ("plot_clustering_benchmark_overview.py","plot_clustering_benchmark_boxplot.py"):
        shutil.copyfile(source_root/"theory_checks"/name,tools/name)
    for name in ("tests","slurm"):
        if (source_root/name).exists():
            shutil.copytree(source_root/name,output/"source_snapshot"/name,dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
    for eps in ("4","8"):
        subprocess.run([sys.executable,str(tools/"plot_clustering_benchmark_overview.py"),"--epsilon",eps,"--research-root",str(research)],check=True)
        subprocess.run([sys.executable,str(tools/"plot_clustering_benchmark_boxplot.py"),"--epsilon",eps,"--output",str(figures)],check=True)
        data = pd.read_csv(figures/f"benchmark_overview_epsilon{eps}_data.csv")
        assert len(data) == 240 and (data.mechanism == ARM).sum() == 16
        boxes = pd.read_csv(figures/f"benchmark_boxplot_epsilon{eps}_15datasets_five_numbers.csv")
        assert len(boxes) == 15 and (boxes.mechanism == ARM).sum() == 1
    # The original report points to extension-hosted all-dataset presentation.
    for archive,panel in ARCHIVES:
        out = archive/"analysis_handoff"
        oldtext = (out/"report.md").read_text()
        if panel == "aggregate":
            # Use a direct companion link for figures, avoiding layout-dependent
            # traversal across the two independently versioned run trees.
            companion = out/"corrected_projection_figures"
            if not companion.exists(): companion.symlink_to(figures,target_is_directory=True)
            reportblock = block(combined,output).replace("](figures/", "](corrected_projection_figures/")
        else:
            reportblock = block(combined,output)
        updated = replace_section(oldtext,reportblock)
        (out/"report.md").write_text(updated);(out/"report.html").write_text(basic_html(updated))
        local = research/("clustering_benchmark_experiment_results.md" if panel == "aggregate" else "clustering_benchmark_extension_results.md")
        folder = "clustering_benchmark_results" if panel == "aggregate" else "clustering_benchmark_extension_results"
        localblock = re.sub(r"\]\(([^)]+)\)",lambda m:"]("+folder+"/"+m.group(1)+")",reportblock)
        local.write_text(replace_section(local.read_text(),localblock))
    supplement = "# Exact corrected projected representation\n\n"+block(combined,output).replace("](corrected_projection/", "](")
    (output/"report.md").write_text(supplement)
    (output/"report.html").write_text(basic_html(supplement))
    # Figures belong to the extension handoff; expose them within this supplement.
    if not (output/"figures").exists(): (output/"figures").symlink_to(figures,target_is_directory=True)
    shutil.copyfile(validation_log,output/"test_validation.txt")
    atomic_json(output/"test_validation.json",dict(status="passed",tests_passed=210,
                log_sha256=cb.sha256(output/"test_validation.txt"),allocation_cpus=2))
    atomic_json(output/"publication_verification.json",dict(status="passed",fingerprint=manifest["fingerprint"],
                datasets=15,epsilon4_and8_overviews_and_boxplots=True,
                overview_cells_per_epsilon=240,boxplot_boxes_per_epsilon=15,
                report_paths=[str(research/"clustering_benchmark_experiment_results.md"),str(research/"clustering_benchmark_extension_results.md")],
                historical_numerical_sources_unchanged=True))
    verify_runtime(manifest)
    print("Published exact corrected projection reports and figures",flush=True)


if __name__ == "__main__":
    publish(Path(sys.argv[1]),Path(sys.argv[2]))
