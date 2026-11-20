"""Summaries and uncluttered figures for the opt-in robust_fitted study (no simulation).

Main traces read only the canonical analysis table. Cross-replicate intervals are Student-t
over the conditional noise means of independent private R draws, conditional on each fixed Q.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from private_matching.results import read_json

MAIN = [("robust_fitted", "Fitted restricted family", "#2a78d6", "o", "-"),
        ("ambient", "Ambient", "#eb6834", "s", "--"),
        ("folded", "Projected (full Ψ)\n= ambient where m=d", "#1baf7a", "^", "-"),
        ("phi_infinity", "Φ∞ distance vector", "#eda100", "D", "-"),
        ("anchor", "Anchor (full)", "#e87ba4", "v", "-")]
ABLATION = [("robust_fitted", "Selected (channel + decoder)", "#2a78d6", "o", "-"),
            ("robust_fitted:direct_decoder", "Selected channel, direct decoder", "#eb6834", "s", "--"),
            ("robust_fitted:default_unfitted", "Default: identity, τ=1, direct", "#1baf7a", "^", "-")]
FAMILY_LABEL = {"unstructured": "Unstructured Gaussian", "rank2": "Exact rank 2 (zero padded)",
                "clustered": "Four tight clusters"}
REGIME_LABEL = {"near": "Near public partners (RMS 0.2)", "crowded": "Crowded at farthest public pair (RMS 0.1)",
                "shifted": "Shifted cloud (3 public RMS radii, RMS 0.7)"}


def summarize_table(table):
    """One row per (regime, family, n, epsilon, arm): mean of the conditional noise means."""
    rows = []
    keys = ["private_regime", "geometry_family", "n", "epsilon", "arm_id"]
    for key, group in table.groupby(keys, sort=True):
        x = group["mean_excess_cost"].to_numpy(float)
        count = len(x)
        mean = float(x.mean())
        half = float(t.ppf(.975, count-1)*x.std(ddof=1)/np.sqrt(count)) if count > 1 else np.nan
        n = key[2]
        rows.append(dict(zip(keys, key), replicates=count, K_per_replicate=int(group["K"].iloc[0]),
                         mean_regret=mean, ci95_low=mean-half, ci95_high=mean+half,
                         mean_regret_per_point=mean/n, ci95_low_per_point=(mean-half)/n,
                         ci95_high_per_point=(mean+half)/n,
                         mean_OPT=float(group["OPT"].mean()), mean_random_minus_OPT=float(group["random_minus_OPT"].mean()),
                         provenance=";".join(sorted(set(group["provenance"].astype(str)))),
                         source_arm_id=";".join(sorted(set(group["source_arm_id"].dropna().astype(str)))),
                         interval="Student-t 95% over replicate conditional noise means; Q fixed"))
    return pd.DataFrame(rows)


LINTHRESH = 1e-3  # per-point regret; the y axis is linear on [0, LINTHRESH] and logarithmic above
ZERO_NOTE = (f"y axis: symmetric log, linear on [0, {LINTHRESH:g}] and logarithmic above, so exact zeros are drawn at 0. "
             "An observed zero means no regret in any finite draw; it does not prove zero expected regret. "
             "Error bars: Student-t 95% over replicate conditional noise means, lower ends clipped at 0 for display "
             "only (untruncated values in summary_by_arm.csv).")


def _style(ax, top):
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=LINTHRESH, linscale=.5)
    ax.set_xticks([4, 16, 64])
    ax.set_xticklabels(["4", "16", "64"])
    ax.xaxis.set_minor_locator(NullLocator())
    decades = np.arange(np.log10(LINTHRESH), max(0, np.ceil(np.log10(max(top, LINTHRESH))))+1)
    ax.yaxis.set_major_locator(FixedLocator([0.]+list(10.**decades)))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"$10^{{{int(round(np.log10(v)))}}}$"))
    ax.set_ylim(bottom=0, top=10**decades[-1]*1.3)
    ax.grid(True, which="major", color="#e6e6e3", linewidth=.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, color="#8a8984", labelcolor="#52514e")


def _grid_figure(summary, regime, traces, title, path_stem, note):
    import matplotlib.pyplot as plt
    ns = sorted(summary["n"].unique())
    families = [f for f in FAMILY_LABEL if f in set(summary["geometry_family"])]
    fig, axes = plt.subplots(len(ns), len(families), figsize=(3.1*len(families)+1.9, 2.5*len(ns)+1.1),
                             squeeze=False, sharex=True)
    handles = {}
    # Small multiplicative x offsets keep the slim error bars of different arms apart.
    dodge = np.exp(np.linspace(-.07, .07, len(traces)))
    for r, n in enumerate(ns):
        for c, family in enumerate(families):
            ax = axes[r, c]
            part = summary[(summary.private_regime == regime) & (summary.geometry_family == family) & (summary.n == n)]
            top = 0.
            for (arm, label, color, marker, style), shift in zip(traces, dodge):
                data = part[part.arm_id == arm].sort_values("epsilon")
                # Ambient reusing the full-Psi samples (m=d) is the same law: the projected curve shows both.
                if data.empty or (arm == "ambient" and (data.provenance == "derived_reused").all()):
                    continue
                y = data["mean_regret_per_point"].to_numpy()
                lo = np.maximum(data["ci95_low_per_point"].to_numpy(), 0.)
                hi = data["ci95_high_per_point"].to_numpy()
                top = max(top, float(np.nanmax(hi)))
                x = data["epsilon"].to_numpy()*shift
                (line,) = ax.plot(x, y, color=color, marker=marker, markersize=4.5, linewidth=1.4, linestyle=style,
                                  markerfacecolor="white" if style == "--" else color, zorder=3)
                ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt="none", ecolor=color, elinewidth=.9, capsize=1.8,
                            alpha=.8, zorder=2)
                handles[label] = line
            _style(ax, top)
            if r == 0:
                ax.set_title(FAMILY_LABEL[family], fontsize=9, color="#0b0b0b")
            if c == 0:
                ax.set_ylabel(f"n = {n}\nmean regret per point", fontsize=8, color="#52514e")
            if r == len(ns)-1:
                ax.set_xlabel("ε (log scale)", fontsize=8, color="#52514e")
    fig.suptitle(title, fontsize=10, x=.02, ha="left", color="#0b0b0b")
    fig.legend(list(handles.values()), list(handles.keys()), loc="center left", bbox_to_anchor=(.82, .55),
               frameon=False, fontsize=8)
    import textwrap
    fig.text(.02, .01, "\n".join(textwrap.wrap(note, 190)), fontsize=6, color="#52514e", ha="left", va="bottom")
    fig.tight_layout(rect=(0, .1, .81, .95))
    for suffix in ["png", "pdf"]:
        fig.savefig(f"{path_stem}.{suffix}", dpi=200)
    plt.close(fig)


def geometry_figure(directory, output, manifest):
    import matplotlib.pyplot as plt
    from private_matching.robust_fitted import public_Q, evaluation_R
    ns, families = manifest["ns"], manifest["families"]
    fig, axes = plt.subplots(len(ns), len(families), figsize=(3*len(families), 2.8*len(ns)), squeeze=False)
    for r, n in enumerate(ns):
        for c, family in enumerate(families):
            Q = public_Q(family, n, manifest["d"], manifest["root_seed"], manifest["parameters"])
            centered = Q-Q.mean(axis=0)
            _, sv, vt = np.linalg.svd(centered, full_matrices=False)
            share = sv**2/np.sum(sv**2)
            xy = centered @ vt[:2].T
            ax = axes[r, c]
            ax.scatter(xy[:, 0], xy[:, 1], s=14, color="#2a78d6", edgecolor="white", linewidth=.5)
            ax.set_aspect("equal", adjustable="datalim")
            ax.set_xlabel(f"public PC1 ({share[0]:.0%} of variance)", fontsize=7, color="#52514e")
            ax.set_ylabel(f"public PC2 ({share[1]:.0%})", fontsize=7, color="#52514e")
            ax.set_title(f"{FAMILY_LABEL[family]}, n={n}", fontsize=8)
            ax.tick_params(labelsize=6)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
    fig.suptitle("Public Q (d=16) projected onto its own top two public principal directions", fontsize=9)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output/f"geometry_public_Q.{suffix}", dpi=200)
    plt.close(fig)
    # One illustrative replicate of each held-out regime (evaluation data; not used by fits).
    family, n = families[0], ns[0]
    Q = public_Q(family, n, manifest["d"], manifest["root_seed"], manifest["parameters"])
    _, _, vt = np.linalg.svd(Q-Q.mean(axis=0), full_matrices=False)
    fig, axes = plt.subplots(1, len(manifest["regimes"]), figsize=(3*len(manifest["regimes"]), 3), squeeze=False)
    for c, regime in enumerate(manifest["regimes"]):
        R = evaluation_R(regime, Q, family, 0, manifest["phase"], manifest["root_seed"], manifest["parameters"])
        ax = axes[0, c]
        ax.scatter(*((Q-Q.mean(axis=0)) @ vt[:2].T).T, s=12, color="#8a8984", label="public Q")
        ax.scatter(*((R-Q.mean(axis=0)) @ vt[:2].T).T, s=12, color="#eb6834", marker="x", label="private R")
        ax.set_title(REGIME_LABEL[regime], fontsize=7)
        ax.set_xlabel("public PC1", fontsize=7)
        ax.set_ylabel("public PC2", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6, frameon=False)
    fig.suptitle(f"{FAMILY_LABEL[family]}, n={n}, replicate 0: 2-D projection of 16-D points "
                 "(off-plane displacement, e.g. the shift, is not visible)", fontsize=7)
    fig.tight_layout()
    for suffix in ["png", "pdf"]:
        fig.savefig(output/f"geometry_regimes_example.{suffix}", dpi=200)
    plt.close(fig)


def selection_outputs(directory, output, manifest):
    import matplotlib.pyplot as plt
    records = [read_json(p) for p in sorted((directory/"fits").glob("*.json"))]
    rows = []
    for rec in records:
        for spec in rec["candidates"]:
            i = spec["index"]
            rows.append(dict(family=rec["family"], n=rec["n"], epsilon=rec["epsilon"], candidate_index=i,
                             candidate_id=spec["candidate_id"], selected=i == rec["selected_index"],
                             train_worst_mean_regret=rec["train_worst"][i],
                             validation_worst_mean_regret=rec["validation_worst"][i],
                             train_worst_per_point=rec["train_worst"][i]/rec["n"],
                             validation_worst_per_point=rec["validation_worst"][i]/rec["n"],
                             fit_seconds=rec["fit_seconds"], fit_path=rec["fit_path"]))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"fit_candidate_scores.csv", index=False)
    frame[frame.selected].to_csv(output/"fit_selected.csv", index=False)
    ns, families = manifest["ns"], manifest["families"]
    colors = {4.: "#2a78d6", 16.: "#eb6834", 64.: "#1baf7a"}
    fig, axes = plt.subplots(len(ns), len(families), figsize=(3.2*len(families)+1.4, 2.5*len(ns)+.6),
                             squeeze=False, sharex=True)
    for r, n in enumerate(ns):
        for c, family in enumerate(families):
            ax = axes[r, c]
            for eps, color in colors.items():
                part = frame[(frame.family == family) & (frame.n == n) & (frame.epsilon == eps)].sort_values("candidate_index")
                if part.empty:
                    continue
                ax.plot(part.candidate_index, part.train_worst_per_point, color=color, linewidth=1.2, marker="o",
                        markersize=3, label=f"ε={eps:g}")
                chosen = part[part.selected]
                ax.scatter(chosen.candidate_index, chosen.train_worst_per_point, s=70, facecolor="none",
                           edgecolor="#0b0b0b", linewidth=1.1, zorder=5)
            ax.set_yscale("log")
            ax.set_xticks([0, 4, 8, 12])
            ax.tick_params(labelsize=7)
            for side in ["top", "right"]:
                ax.spines[side].set_visible(False)
            if r == 0:
                ax.set_title(FAMILY_LABEL[family], fontsize=8)
            if c == 0:
                ax.set_ylabel(f"n={n}\ntrain worst regret / point", fontsize=7)
            if r == len(ns)-1:
                ax.set_xlabel("candidate index", fontsize=7)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(.85, .5), frameon=False, fontsize=7)
    fig.text(.01, .01, "Candidates 0-7 identity features, 8-15 public PCA-2; within each block of 4: direct, "
             "posterior near-Q, posterior broad, posterior mixture; blocks 0-3/8-11 τ=1, 4-7/12-15 τ=0.5. "
             "Circle: selected (public training data only).", fontsize=6, color="#52514e")
    fig.tight_layout(rect=(0, .05, .84, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(output/"figures"/f"fit_selection_diagnostics.{suffix}", dpi=200)
    plt.close(fig)
    return frame


def make_report(directory, output):
    import matplotlib
    matplotlib.use("Agg")
    directory, output = Path(directory), Path(output)
    manifest = read_json(directory/"manifest.json")
    table = pd.read_csv(output/"analysis_table.csv", low_memory=False)
    summary = summarize_table(table)
    summary.to_csv(output/"summary_by_arm.csv", index=False)
    figures = output/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    phase = manifest["phase"]
    reuse = ("Where m=d (unstructured, clustered) ambient reuses the full-Ψ samples exactly, so the projected curve "
             "represents both; ambient is drawn separately only for rank 2 (m=3<d).")
    for regime in manifest["regimes"]:
        common = (f"{phase}: {manifest['replicates']} independent private R draws per panel (one fixed public Q), "
                  f"K={manifest['K']} mechanism draws each. ")
        note = (common+reuse+" Fitted = restricted analytic softmax channel + compact Hungarian decoder selected on "
                "public synthetic data only; not the free finite-element mechanism, not a minimax optimum. "+ZERO_NOTE)
        _grid_figure(summary, regime, MAIN, f"Held-out regime: {REGIME_LABEL[regime]}",
                     figures/f"regret_{regime}", note)
        _grid_figure(summary, regime, ABLATION, f"Fitted-family ablation — {REGIME_LABEL[regime]}",
                     figures/f"ablation_{regime}", common+"Fitted-family arms share per-client label uniforms "
                     "(coupled, recorded coupling_group). "+ZERO_NOTE)
    geometry_figure(directory, figures, manifest)
    selection_outputs(directory, output, manifest)
    wide = summary.pivot_table(index=["private_regime", "geometry_family", "n", "epsilon"], columns="arm_id",
                               values="mean_regret_per_point").reset_index()
    for arm in ["ambient", "folded", "phi_infinity", "anchor"]:
        if arm in wide:
            wide[f"fitted_over_{arm}"] = wide["robust_fitted"]/wide[arm]
    wide.to_csv(output/"fitted_vs_baselines_per_point.csv", index=False)
    print(f"robust_fitted report: {output}", flush=True)
    return summary
