"""Render the selected manuscript figures/tables from audited aggregate results.

Run on CRC compute. Arguments: squared-cost handoff directory, output directory,
Euclidean handoff directory, Fashion-MNIST handoff directory.
The output is copied into the Overleaf project only after pulling that project.
"""
from pathlib import Path
import hashlib
import json
import sys
import socket
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


SQUARED, OUT, MAIN, FASHION = map(Path, sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
ARMS = ["gumbel", "original_points", "projected_representation", "distance_vectors"]
LABELS = ["Gumbel", "Original points", "Projected representation", "Distance vectors"]
COLORS = ["#4477AA", "#228833", "#AA3377", "#AA9922"]
STYLES = ["-", "-", "--", "-."]
DATASETS = ["graves/ring_noisy", "wut/circles", "fcps/chainlink", "other/iris", "uci/banknote",
            "uci/ecoli", "uci/yeast", "uci/glass", "uci/wine", "uci/statlog", "uci/wdbc",
            "uci/ionosphere", "uci/sonar", "sipu/worms_64", "mnist/digits10k", "mnist/fashion10k"]
NAMES = ["Ring noisy", "Circles", "Chainlink", "Iris", "Banknote", "Ecoli", "Yeast", "Glass",
         "Wine", "Statlog", "WDBC", "Ionosphere", "Sonar", "Worms-64", "MNIST", "Fashion-MNIST"]
NAME = dict(zip(DATASETS, NAMES))
plt.rcParams.update({"font.family":"serif", "font.serif":["STIXGeneral"], "mathtext.fontset":"stix",
    "font.size":8.5, "axes.titlesize":8.5, "axes.labelsize":8.5, "xtick.labelsize":7.5,
    "ytick.labelsize":7.5, "legend.fontsize":7.7, "legend.frameon":False,
    "axes.spines.top":False, "axes.spines.right":False, "pdf.fonttype":42, "ps.fonttype":42})
main = pd.read_csv(MAIN/"summary.csv")
square = pd.read_csv(SQUARED/"summary.csv")
fashion = pd.read_csv(FASHION/"summary.csv")
assert set(square.objective_p) == {2} and len(square) == 16*4*64



combined = pd.concat([main[main.objective_p == 1], square], ignore_index=True)
files = []

def save(fig, name):
    for ext in ("pdf", "png"):
        path = OUT/(name+"."+ext)
        fig.savefig(path, dpi=230, bbox_inches="tight", pad_inches=.035)
        files.append(path)
    plt.close(fig)

def method_handles():
    return [Line2D([], [], color=c, ls=s, lw=1.6, label=l) for c,s,l in zip(COLORS,STYLES,LABELS)]

def curve(ax, frame, x, scale=1.):
    for arm,c,s in zip(ARMS,COLORS,STYLES):
        part = frame[frame.mechanism == arm].sort_values(x)
        divisor = part[x].to_numpy() if scale == "n" else scale
        ax.plot(part[x], part.mean_total_regret_mean/divisor, color=c, ls=s, lw=1.25)
        ax.fill_between(part[x], part.mean_total_regret_low/divisor, part.mean_total_regret_high/divisor,
                        color=c, alpha=.13, lw=0)
    part = frame[frame.mechanism == "gumbel"].sort_values(x)
    divisor = part[x].to_numpy() if scale == "n" else scale
    ax.plot(part[x], part.random_minus_optimum_mean/divisor, color=".42", ls=":", lw=1.25)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=".93", lw=.5)

for p in (1,2):
    for page in (0,1):
        fig, axes = plt.subplots(4,2, figsize=(5.5,6.3))
        fig.subplots_adjust(left=.09,right=.99,bottom=.135,top=.97,wspace=.25,hspace=.50)
        for ax, dataset in zip(axes.flat, DATASETS[page*8:page*8+8]):
            part = combined[(combined.panel=="epsilon") & (combined.objective_p==p) & (combined.dataset==dataset)]
            assert len(part)==256
            curve(ax, part, "epsilon")
            ax.set_title(NAME[dataset]+r" ($d="+str(int(part.d.iloc[0]))+"$)",pad=3)
            ax.set_xlim(.25,16); ax.set_xticks([0,8,16])
        fig.supylabel(r"$\mathbb{E}[\mathrm{err}]$",x=.005,fontsize=9)
        fig.supxlabel(r"$\varepsilon$",y=.075,fontsize=9)
        handles = method_handles()+[Line2D([],[],color=".42",ls=":",label="RAND")]
        fig.legend(handles=handles,loc="lower center",ncol=3,columnspacing=1.7,handlelength=2.2)
        save(fig,f"numerical_epsilon_p{p}_{page+1}")

low, high = np.array(matplotlib.colors.to_rgb("#dcebf9")), np.array(matplotlib.colors.to_rgb("#f6d5d5"))
mix = (low+high)/2
for p in (1,2):
    # The two objectives are stacked in one manuscript figure, sharing
    # the legend above the first panel and the same physical axes width.
    fig, ax = plt.subplots(figsize=(5.5,2.85 if p == 1 else 2.45))
    fig.subplots_adjust(left=.08,right=.99,
                        bottom=.31 if p == 1 else .31*2.85/2.45,
                        top=.83 if p == 1 else .83*2.85/2.45)
    frame = combined[(combined.panel=="epsilon") & (combined.objective_p==p) & (combined.epsilon==8)]
    for j,dataset in enumerate(DATASETS):
        row = frame[frame.dataset==dataset].iloc[0]
        ax.axvspan(j-.5,j+.5,color=low if row.d<=np.log(64) else high if row.d>=64 else mix,zorder=0)
        ax.hlines(row.random_minus_optimum_mean,j-.43,j+.43,color=".42",ls=":",lw=1.1,zorder=2)
    for i,(arm,c) in enumerate(zip(ARMS,COLORS)):
        part = frame[frame.mechanism==arm].set_index("dataset").loc[DATASETS]
        mean = part.mean_total_regret_mean.to_numpy()
        err = np.stack([mean-part.mean_total_regret_low,part.mean_total_regret_high-mean])
        ax.bar(np.arange(16)+(i-1.5)*.18,mean,width=.18,color=c,yerr=err,
               error_kw={"elinewidth":.55,"ecolor":".2","capsize":.8},zorder=3)
    ax.set_xticks(range(16),NAMES,rotation=70,ha="right",fontsize=7.2)
    ax.set_xlim(-.5,15.5); ax.set_ylim(bottom=0)
    ax.set_ylabel(r"$\mathbb{E}[\mathrm{err}]$")
    handles = [Patch(color=c,label=l) for c,l in zip(COLORS,LABELS)]
    handles += [Line2D([],[],color=".42",ls=":",label="RAND")]
    if p == 1:
        fig.legend(handles=handles,loc="upper center",ncol=3,columnspacing=1.8,bbox_to_anchor=(.5,1.025))
    save(fig,f"numerical_bars_p{p}")

fig, axes = plt.subplots(1,2,figsize=(5.5,3.2))
fig.subplots_adjust(left=.08,right=.99,bottom=.29,top=.90,wspace=.33)
ax=axes[0]
for arm,c,s in zip(ARMS,COLORS,STYLES):
    part=fashion[(fashion.mechanism==arm)&(fashion.d<=104)].sort_values("d")
    ax.plot(part.d,part.regret,color=c,ls=s,lw=1.4)
    ax.fill_between(part.d,part.regret_low,part.regret_high,color=c,alpha=.13,lw=0)
part=fashion[(fashion.mechanism=="gumbel")&(fashion.d<=104)].sort_values("d")
ax.plot(part.d,part.RAND_gap,color=".42",ls=":",lw=1.3)
ax.plot(part.d,2*800*np.minimum(part.d,np.log(800))/16,color="black",ls="-.",lw=1.2)
ax.axvline(np.log(800),color=".65",ls=":",lw=.8)
ax.set(xlim=(0,104),ylim=(0,None),xlabel=r"Projected dimension $d$",ylabel=r"$\mathbb{E}[\mathrm{err}]$",title="(a) Fashion-MNIST")
ax.grid(axis="y",color=".93",lw=.5)
ax=axes[1]
part=main[main.panel=="sample_size"]
curve(ax,part,"n",scale="n")
grid=np.linspace(20,200,400)
ax.plot(grid,2*np.minimum(4,np.log(grid))/8,color="black",ls="-.",lw=1.2)
ax.axvline(np.exp(4),color=".65",ls=":",lw=.8)
ax.set(xlim=(20,200),xlabel=r"Matching size $n$",ylabel=r"$\mathbb{E}[\mathrm{err}]/n$",title="(b) Banknote")
ax.set_xticks([20,60,100,150,200])
handles=method_handles()+[Line2D([],[],color=".42",ls=":",label="RAND"),Line2D([],[],color="black",ls="-.",label=r"$2n\min\{d,\log n\}/\varepsilon$")]
fig.legend(handles=handles,loc="lower center",ncol=3,columnspacing=1.5,handlelength=2.4)
save(fig,"numerical_dimension_and_size")

def cell(mean, low, high):
    assert np.isclose(mean-low,high-mean,atol=2e-9)
    return f"${mean:.3f}\\,\\pm\\,{(high-low)/2:.3f}$"

tables=[]
for p in (1,2):
    for eps in (4,8,16):
        frame=combined[(combined.panel=="epsilon")&(combined.objective_p==p)&(combined.epsilon==eps)]
        objective="Euclidean" if p==1 else "Squared Euclidean"
        tables += [r"\begin{table}[p]",r"\centering\scriptsize",r"\setlength{\tabcolsep}{3pt}",
            f"\\caption{{{objective} excess matching cost at $\\eps={eps}$, $n=64$. Entries are means $\\pm$ pointwise 95\\% confidence half-widths over twenty parent collections. RAND denotes $\\mathrm{{RAND}}-\\mathrm{{OPT}}$.}}",
            f"\\label{{tab:exp-bench-p{p}-eps{eps}}}",r"\begin{tabular}{lrrrrr}",r"\toprule",
            r"Dataset & Gumbel & Original points & \shortstack{Projected\\representation} & Distance vectors & RAND\\",r"\midrule"]
        for dataset in DATASETS:
            part=frame[frame.dataset==dataset].set_index("mechanism")
            cells=[cell(part.loc[a,"mean_total_regret_mean"],part.loc[a,"mean_total_regret_low"],part.loc[a,"mean_total_regret_high"]) for a in ARMS]
            tables.append(NAME[dataset]+" & "+" & ".join(cells)+f" & {part.iloc[0].random_minus_optimum_mean:.3f}"+r"\\")
        tables += [r"\bottomrule",r"\end{tabular}",r"\end{table}",""]
tables += [r"\clearpage",r"{\scriptsize",r"\setlength{\tabcolsep}{3pt}",r"\begin{longtable}{rrrrrrr}",
    r"\caption{Fashion-MNIST dimension sweep, $n=800$, $\eps=16$. Mechanism entries are total excess-cost means $\pm$ pointwise 95\% confidence half-widths. RAND is $\mathrm{RAND}-\mathrm{OPT}$; $B=2n\min\{d,\log n\}/\eps$.}\label{tab:exp-fashion-values}\\",
    r"\toprule",r"$d$ & Gumbel & Original points & \shortstack{Projected\\representation} & Distance vectors & RAND & $B$\\",r"\midrule\endfirsthead",
    r"\toprule",r"$d$ & Gumbel & Original points & \shortstack{Projected\\representation} & Distance vectors & RAND & $B$\\",r"\midrule\endhead",
    r"\bottomrule\endfoot"]
for d in sorted(fashion.d.unique()):
    part=fashion[fashion.d==d].set_index("mechanism")
    cells=[cell(part.loc[a,"regret"],part.loc[a,"regret_low"],part.loc[a,"regret_high"]) for a in ARMS]
    tables.append(str(d)+" & "+" & ".join(cells)+f" & {part.iloc[0].RAND_gap:.3f} & {2*800*min(d,np.log(800))/16:.3f}"+r"\\")
tables += [r"\end{longtable}",r"}",r"\begin{table}[p]",r"\centering\scriptsize",r"\setlength{\tabcolsep}{3pt}",
    r"\caption{Banknote size sweep, $d=4$, $\eps=8$. Mechanism entries are per-pair excess-cost means $\pm$ pointwise 95\% confidence half-widths. RAND is $(\mathrm{RAND}-\mathrm{OPT})/n$; $B/n=2\min\{4,\log n\}/8$.}",
    r"\label{tab:exp-banknote-values}",r"\begin{tabular}{rrrrrrr}",r"\toprule",
    r"$n$ & Gumbel & Original points & \shortstack{Projected\\representation} & Distance vectors & RAND & $B/n$\\",r"\midrule"]
for n in sorted(main[main.panel=="sample_size"].n.unique()):
    part=main[(main.panel=="sample_size")&(main.n==n)].set_index("mechanism")
    cells=[cell(part.loc[a,"mean_total_regret_mean"]/n,part.loc[a,"mean_total_regret_low"]/n,part.loc[a,"mean_total_regret_high"]/n) for a in ARMS]
    tables.append(str(n)+" & "+" & ".join(cells)+f" & {part.iloc[0].random_minus_optimum_mean/n:.3f} & {2*min(4,np.log(n))/8:.3f}"+r"\\")
tables += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
table_text = "\n".join(tables)+"\n"
bank_start = table_text.index(r"\begin{table}[p]", table_text.index(r"\end{longtable}"))
bank = table_text[bank_start:].replace(r"\begin{table}[p]", r"\begin{table}[!htbp]", 1)
table_text = bank + "\n" + table_text[:bank_start]
(OUT/"experiment_numerical_tables.tex").write_text(table_text)
combined.to_csv(OUT/"numerical_experiment_summary.csv",index=False)
fashion[[c for c in fashion if not any(w in c for w in ("effective_rank", "public_rank"))]].to_csv(OUT/"numerical_fashion_summary.csv",index=False)
facts=[]
for p in (1,2):
    frame=combined[(combined.panel=="epsilon")&(combined.objective_p==p)&(combined.epsilon==8)]
    for dataset in DATASETS:
        part=frame[frame.dataset==dataset].set_index("mechanism")
        facts.append(dict(objective=p,dataset=NAME[dataset],means={a:float(part.loc[a,"mean_total_regret_mean"]) for a in ARMS}))
(OUT/"numerical_facts.json").write_text(json.dumps(facts,indent=2)+"\n")
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
provenance=dict(status="passed", source_files={str(p.resolve()):digest(p) for p in (MAIN/"summary.csv", SQUARED/"summary.csv", FASHION/"summary.csv", Path(__file__))},
    figures={p.name:digest(p) for p in files}, figure_pairs=len(files)//2,
    main_handoff=str(MAIN.resolve()), fashion_handoff=str(FASHION.resolve()), squared_handoff=str(SQUARED.resolve()),
    unchanged_scientific_means=True, intervals_from_audited_parent_means=True,
    methods=LABELS, RAND_style="gray dotted", bound_style="black dash-dot", numerical_rank_not_plotted=True)
(OUT/"numerical_figure_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")
print(json.dumps(provenance,indent=2))
