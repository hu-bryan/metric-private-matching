"""Summaries and figures for the opt-in continuous_decoder study (no simulation).

Paired decoder-minus-direct differences: per R, mean over the K SHARED reports of (decoder - folded)
regret; then Student-t 95% over the conditional R replicates (public Q fixed). Absolute figures use
per-arm intervals over the same replicate means. Rows with provenance `derived_reused` that alias
another plotted arm are omitted from figures (kept in the tables).
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from private_matching.results import read_json
from private_matching.robust_fitted_plots import FAMILY_LABEL, REGIME_LABEL, summarize_table, _style

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"
DIFF_TRACES = [("robust_interval_a0.1", "Randomized interval-minimax (α=0.1),\nconditional mean over its output law",
                "#2a78d6", "o", "-"),
               ("biascorr_l0.5", "Bias-corrected, λ=0.5", "#eb6834", "s", "--"),
               ("biascorr_l1", "Bias-corrected, λ=1", "#1baf7a", "^", ":")]
ABS_TRACES = [("folded", "Direct = full Ψ (shared reports)", "#1baf7a", "^", "-"),
              ("robust_interval_a0.1", "Randomized interval-minimax (α=0.1)", "#2a78d6", "o", "-"),
              ("ambient", "Ambient", "#eb6834", "s", "--"),
              ("phi_infinity", "Φ∞ distance vector", "#eda100", "D", "-"),
              ("anchor", "Anchor (full fold)", "#e87ba4", "v", "-")]
SPAN_TRACES = [("span_anchor", "Span-only anchor (k=2, new)", "#2a78d6", "o", "-"),
               ("anchor", "Original anchor (fold m=3)", "#eb6834", "s", "--"),
               ("folded", "Direct = full Ψ", "#1baf7a", "^", ":")]


def ci(x):
    x = np.asarray(x, dtype=float)
    mean = float(x.mean())
    half = float(t.ppf(.975, len(x)-1)*x.std(ddof=1)/np.sqrt(len(x))) if len(x) > 1 else np.nan
    return mean, mean-half, mean+half


def paired_differences(table):
    """decoder - folded, per replicate on shared reports; then t-interval over replicates."""
    keys = ["private_regime", "geometry_family", "n", "epsilon"]
    base = table[table.arm_id == "folded"].set_index(keys+["geometry_replicate"])["mean_excess_cost"]
    rows = []
    shared = table[table.report_source_arm_id.fillna("") == "folded"]
    for arm in sorted(set(shared.arm_id)):
        part = shared[shared.arm_id == arm].set_index(keys+["geometry_replicate"])
        diff = (part["mean_excess_cost"]-base.reindex(part.index)).dropna()
        if len(diff) != len(part):
            raise ValueError(f"unpaired rows for {arm}")
        for key, group in diff.groupby(level=list(range(len(keys)))):
            n = key[2]
            mean, lo, hi = ci(group.to_numpy())
            wins = int(np.sum(group.to_numpy() < 0))
            rows.append(dict(zip(keys, key), arm_id=arm, replicates=len(group), mean_diff=mean, ci95_low=lo,
                             ci95_high=hi, mean_diff_per_point=mean/n, ci95_low_per_point=lo/n,
                             ci95_high_per_point=hi/n, replicates_decoder_better=wins,
                             replicates_decoder_worse=int(np.sum(group.to_numpy() > 0)),
                             interval="paired: per-R mean over K shared reports of (decoder - direct); Student-t 95% "
                             "over conditional R replicates; Q fixed"))
    return pd.DataFrame(rows)


def _axes_grid(summary, traces):
    import matplotlib.pyplot as plt
    ns = sorted(summary["n"].unique())
    families = [f for f in FAMILY_LABEL if f in set(summary["geometry_family"])]
    fig, axes = plt.subplots(len(ns), len(families), figsize=(3.1*len(families)+2.2, 2.5*len(ns)+1.2),
                             squeeze=False, sharex=True)
    return fig, axes, ns, families


def _finish(fig, handles, title, note, stem, right=.79):
    fig.suptitle(title, fontsize=10, x=.02, ha="left", color=INK)
    fig.legend(list(handles.values()), list(handles.keys()), loc="center left", bbox_to_anchor=(right+.01, .55),
               frameon=False, fontsize=7.5)
    fig.text(.02, .01, "\n".join(textwrap.wrap(note, 185)), fontsize=6, color=MUTED, ha="left", va="bottom")
    fig.tight_layout(rect=(0, .11, right, .95))
    for suffix in ["png", "pdf"]:
        fig.savefig(f"{stem}.{suffix}", dpi=200, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _linear_style(ax):
    from matplotlib.ticker import NullLocator
    ax.set_xscale("log")
    ax.set_xticks([4, 16, 64])
    ax.set_xticklabels(["4", "16", "64"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.axhline(0, color=MUTED, linewidth=.9, zorder=1)
    ax.grid(True, which="major", color=GRID, linewidth=.6)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8, color="#8a8984", labelcolor=MUTED)


def difference_figure(diffs, regime, stem, note):
    fig, axes, ns, families = _axes_grid(diffs, DIFF_TRACES)
    handles = {}
    dodge = np.exp(np.linspace(-.06, .06, len(DIFF_TRACES)))
    for r, n in enumerate(ns):
        for c, family in enumerate(families):
            ax = axes[r, c]
            part = diffs[(diffs.private_regime == regime) & (diffs.geometry_family == family) & (diffs.n == n)]
            for (arm, label, color, marker, style), shift in zip(DIFF_TRACES, dodge):
                data = part[part.arm_id == arm].sort_values("epsilon")
                if data.empty:
                    continue
                x, y = data.epsilon.to_numpy()*shift, data.mean_diff_per_point.to_numpy()
                lo, hi = data.ci95_low_per_point.to_numpy(), data.ci95_high_per_point.to_numpy()
                (line,) = ax.plot(x, y, color=color, marker=marker, markersize=4.5, linewidth=1.4, linestyle=style,
                                  zorder=3)
                ax.errorbar(x, y, yerr=[y-lo, hi-y], fmt="none", ecolor=color, elinewidth=.9, capsize=1.8, alpha=.85)
                handles[label] = line
            _linear_style(ax)
            if r == 0:
                ax.set_title(FAMILY_LABEL[family], fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(f"n = {n}\n(decoder − direct) regret / point", fontsize=7.5, color=MUTED)
            if r == len(ns)-1:
                ax.set_xlabel("ε (log scale)", fontsize=8, color=MUTED)
    _finish(fig, handles, f"Decoder minus direct on identical reports — {REGIME_LABEL[regime]}  (below 0 = better)",
            note, stem)


def absolute_figure(summary, regime, traces, stem, title, note, families=None):
    fig, axes, ns, fams = _axes_grid(summary if families is None else summary[summary.geometry_family.isin(families)],
                                     traces)
    handles = {}
    dodge = np.exp(np.linspace(-.07, .07, len(traces)))
    for r, n in enumerate(ns):
        for c, family in enumerate(fams):
            ax = axes[r, c]
            part = summary[(summary.private_regime == regime) & (summary.geometry_family == family) & (summary.n == n)]
            top = 0.
            for (arm, label, color, marker, style), shift in zip(traces, dodge):
                data = part[part.arm_id == arm].sort_values("epsilon")
                if data.empty or (data.provenance == "derived_reused").all():
                    continue  # aliases (e.g. ambient = full Ψ when m=d) are drawn once
                y = data.mean_regret_per_point.to_numpy()
                lo = data.ci95_low_per_point.to_numpy()
                hi = data.ci95_high_per_point.to_numpy()
                top = max(top, float(np.nanmax(hi)))
                x = data.epsilon.to_numpy()*shift
                (line,) = ax.plot(x, y, color=color, marker=marker, markersize=4.5, linewidth=1.4, linestyle=style,
                                  markerfacecolor="white" if style == "--" else color, zorder=3)
                ax.errorbar(x, y, yerr=[y-np.maximum(lo, 0), hi-y], fmt="none", ecolor=color, elinewidth=.9,
                            capsize=1.8, alpha=.8)
                handles[label] = line
            _style(ax, top)
            if r == 0:
                ax.set_title(FAMILY_LABEL[family], fontsize=9, color=INK)
            if c == 0:
                ax.set_ylabel(f"n = {n}\nmean regret per point", fontsize=8, color=MUTED)
            if r == len(ns)-1:
                ax.set_xlabel("ε (log scale)", fontsize=8, color=MUTED)
    _finish(fig, handles, title, note, stem)


def span_figure(summary, stem, note):
    import matplotlib.pyplot as plt
    part = summary[summary.geometry_family == "rank2"]
    ns, regimes = sorted(part.n.unique()), [r for r in REGIME_LABEL if r in set(part.private_regime)]
    fig, axes = plt.subplots(len(ns), len(regimes), figsize=(3.1*len(regimes)+2.2, 2.5*len(ns)+1.2), squeeze=False,
                             sharex=True)
    handles = {}
    dodge = np.exp(np.linspace(-.06, .06, len(SPAN_TRACES)))
    for r, n in enumerate(ns):
        for c, regime in enumerate(regimes):
            ax = axes[r, c]
            cell = part[(part.n == n) & (part.private_regime == regime)]
            top = 0.
            for (arm, label, color, marker, style), shift in zip(SPAN_TRACES, dodge):
                data = cell[cell.arm_id == arm].sort_values("epsilon")
                if data.empty:
                    continue
                y, lo, hi = (data[k].to_numpy() for k in ["mean_regret_per_point", "ci95_low_per_point",
                                                            "ci95_high_per_point"])
                top = max(top, float(np.nanmax(hi)))
                x = data.epsilon.to_numpy()*shift
                (line,) = ax.plot(x, y, color=color, marker=marker, markersize=4.5, linewidth=1.4, linestyle=style,
                                  zorder=3)
                ax.errorbar(x, y, yerr=[y-np.maximum(lo, 0), hi-y], fmt="none", ecolor=color, elinewidth=.9,
                            capsize=1.8, alpha=.8)
                handles[label] = line
            _style(ax, top)
            if r == 0:
                ax.set_title(REGIME_LABEL[regime], fontsize=8, color=INK)
            if c == 0:
                ax.set_ylabel(f"n = {n}\nmean regret per point", fontsize=8, color=MUTED)
            if r == len(ns)-1:
                ax.set_xlabel("ε (log scale)", fontsize=8, color=MUTED)
    _finish(fig, handles, "Exact rank-2 public Q: span-only anchor vs original anchor", note, stem)


def robust_diagnostics(table, output):
    robust = table[table.decoder_family == "robust_interval"].copy()
    if robust.empty:
        return None
    keys = ["arm_id", "geometry_family", "n", "epsilon"]
    cols = ["mean_coverage_event", "mean_radius_event", "mean_solver_seconds", "mean_solver_exact",
            "mean_interval_gap", "mean_interval_upper", "mean_direct_interval_objective", "mean_support_size",
            "mean_bvn_terms", "mean_bvn_leftover", "mean_max_H_entry", "mean_confidence_radius",
            "interval_expected_certificate", "mean_excess_cost", "mean_sampled_output_excess"]
    robust["certificate_holds"] = robust.mean_excess_cost <= robust.interval_expected_certificate+1e-9
    diag = robust.groupby(keys)[cols].mean().reset_index()
    diag["certificate_holds_in_every_unit"] = robust.groupby(keys)["certificate_holds"].all().values
    diag["max_interval_gap"] = robust.groupby(keys)["mean_interval_gap"].max().values
    diag["min_coverage_event"] = robust.groupby(keys)["mean_coverage_event"].min().values
    diag.to_csv(output/"robust_solver_diagnostics.csv", index=False)
    import matplotlib.pyplot as plt
    panels = [("mean_coverage_event", "interval coverage rate\n(all n² costs in box)", False),
              ("mean_solver_seconds", "LP solve seconds / draw", True),
              ("ratio", "F(H) / F(direct)\n(interval worst-case regret)", False),
              ("mean_support_size", "nonzeros of H (n = identity support)", True)]
    diag["ratio"] = diag.mean_interval_upper/diag.mean_direct_interval_objective
    styles = {("unstructured", 25): ("#2a78d6", "o", "-"), ("unstructured", 50): ("#2a78d6", "o", "--"),
              ("rank2", 25): ("#eb6834", "s", "-"), ("rank2", 50): ("#eb6834", "s", "--"),
              ("clustered", 25): ("#1baf7a", "^", "-"), ("clustered", 50): ("#1baf7a", "^", "--")}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    handles = {}
    main = diag[diag.arm_id == "robust_interval_a0.1"] if "robust_interval_a0.1" in set(diag.arm_id) else diag
    for ax, (col, label, log) in zip(axes, panels):
        for (family, n), (color, marker, style) in styles.items():
            data = main[(main.geometry_family == family) & (main.n == n)].sort_values("epsilon")
            if data.empty:
                continue
            (line,) = ax.plot(data.epsilon, data[col], color=color, marker=marker, linestyle=style, linewidth=1.3,
                              markersize=4.5, markerfacecolor="white" if style == "--" else color)
            handles[f"{FAMILY_LABEL[family]}, n={n}"] = line
        from matplotlib.ticker import NullLocator, NullFormatter
        ax.set_xscale("log")
        ax.set_xticks([4, 16, 64])
        ax.set_xticklabels(["4", "16", "64"])
        ax.xaxis.set_minor_locator(NullLocator())
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_minor_formatter(NullFormatter())
        elif col != "ratio":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
        ax.set_title(label, fontsize=8, color=INK)
        ax.set_xlabel("ε", fontsize=8, color=MUTED)
        ax.grid(True, color=GRID, linewidth=.6)
        ax.tick_params(labelsize=7)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.legend(list(handles.values()), list(handles.keys()), loc="center left", bbox_to_anchor=(.86, .5),
               frameon=False, fontsize=7)
    fig.text(.01, .01, "Robust interval decoder (α=0.1) diagnostics averaged over regimes, replicates and draws. "
             "Coverage is the evaluator-only event that every true cost lies in [l,u] (target ≥ 1-α). "
             "F = interval worst-case regret; direct is feasible so F(H) ≤ F(direct) up to the recorded gap.",
             fontsize=6, color=MUTED)
    fig.tight_layout(rect=(0, .06, .85, 1))
    for suffix in ["png", "pdf"]:
        fig.savefig(output/"figures"/f"robust_solver_diagnostics.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return diag


def selection_outputs(directory, output):
    rows = []
    for path in sorted((directory/"fits").glob("*.json")):
        rec = read_json(path)
        for i, name in enumerate(rec["candidates"]):
            rows.append(dict(family=rec["family"], n=rec["n"], epsilon=rec["epsilon"], candidate=name,
                             selected=i == rec["selected_index"], train_worst_mean_regret=rec["train_worst"][i],
                             validation_worst_mean_regret=rec["validation_worst"][i], fit_seconds=rec["fit_seconds"],
                             fit_path=str(path)))
    frame = pd.DataFrame(rows)
    frame.to_csv(output/"fit_candidate_scores.csv", index=False)
    return frame


def verdicts(diffs):
    """Cells where the paired 95% interval lies entirely below / above zero (descriptive, unadjusted)."""
    out = []
    for arm, group in diffs.groupby("arm_id"):
        out.append(dict(arm_id=arm, cells=len(group), ci_below_zero=int(np.sum(group.ci95_high < 0)),
                        ci_above_zero=int(np.sum(group.ci95_low > 0)),
                        mean_below_zero=int(np.sum(group.mean_diff < 0)), mean_above_zero=int(np.sum(group.mean_diff > 0)),
                        exact_zero=int(np.sum(group.mean_diff == 0))))
    return pd.DataFrame(out)


def make_report(directory, output):
    import matplotlib
    matplotlib.use("Agg")
    directory, output = Path(directory), Path(output)
    manifest = read_json(directory/"manifest.json")
    table = pd.read_csv(output/"analysis_table.csv", low_memory=False)
    summary = summarize_table(table)
    summary.to_csv(output/"summary_by_arm.csv", index=False)
    diffs = paired_differences(table)
    diffs.to_csv(output/"paired_differences_vs_direct.csv", index=False)
    verdict = verdicts(diffs)
    verdict.to_csv(output/"paired_verdict_counts.csv", index=False)
    figures = output/"figures"
    figures.mkdir(parents=True, exist_ok=True)
    phase = manifest["phase"]
    common = (f"{phase}: one fixed public Q per panel; {manifest['replicates']} conditional private R configurations "
              f"(NOT independent Q), K={manifest['K']} report draws each. ")
    if manifest.get("robust_panel"):
        # Prespecified from pilot TIMING only: the robust arm was not run outside this panel.
        common += (f"Robust interval decoder run only for {', '.join(manifest['robust_panel'])} (compute budget, fixed "
                   "from pilot timing); panels without a robust trace were NOT evaluated for it. ")
    for regime in manifest["regimes"]:
        difference_figure(diffs, regime, figures/f"paired_difference_{regime}",
                          common+"All decoders post-process the identical full-fold reports of the direct (full Ψ) "
                          "baseline. Per R: mean over the K shared reports of (decoder − direct) regret per point; "
                          "bars: Student-t 95% over the R replicates. Robust = conditional expected cost of its "
                          "randomized output (a permutation sampled from its Birkhoff mixture). Linear axis; 0 = no change.")
        absolute_figure(summary, regime, ABS_TRACES, figures/f"absolute_{regime}",
                        f"Absolute regret — {REGIME_LABEL[regime]}",
                        common+"Ambient is omitted where it reuses the full-Ψ samples (m=d). Bars: Student-t 95% over "
                        "replicate means, lower ends clipped at 0 for display only (untruncated in summary_by_arm.csv). "
                        "y: symmetric log, linear on [0, 1e-3], so zeros are drawn at 0; an observed zero does not prove "
                        "zero expected regret.")
    if "rank2" in set(summary.geometry_family):
        span_figure(summary, figures/"span_anchor_rank2",
                    common+"Span-only anchor releases noise only in the 2-D public span plus one scalar radius; the "
                    "original anchor uses the 3-D fold (with height). Independent noise streams (not paired). Bars: "
                    "Student-t 95% over replicate means; y symmetric log, linear on [0, 1e-3].")
    robust_diagnostics(table, output)
    selection_outputs(directory, output)
    write_readme(directory, output, manifest, summary, diffs, verdict)
    print(f"continuous_decoder report: {output}", flush=True)
    return summary


def markdown_table(frame):
    """Plain markdown table (no optional tabulate dependency)."""
    def cell(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)
    head = "| " + " | ".join(map(str, frame.columns)) + " |"
    rule = "| " + " | ".join("---" for _ in frame.columns) + " |"
    return "\n".join([head, rule]+["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)])


def write_readme(directory, output, manifest, summary, diffs, verdict):
    timing = read_json(directory/"timings.json") if (directory/"timings.json").exists() else {}
    diag_path = output/"robust_solver_diagnostics.csv"
    lines = [f"# continuous_decoder handoff ({manifest['phase']}, fingerprint `{manifest['fingerprint']}`)", "",
             "Auto-generated from checkpoints; interpretation is in `theory_checks/claude_continuous_decoder_20260926/"
             "STATUS.md` and the study README section of RESEARCH_SUITE.md.", "",
             "## Files", "",
             "- `analysis_table.csv` + `data_dictionary.md`: canonical rows (one per unit x arm).",
             "- `summary_by_arm.csv`: per cell (regime, family, n, eps, arm) mean over R replicates, t-intervals.",
             "- `paired_differences_vs_direct.csv`: decoder − direct on identical reports (paired over R).",
             "- `paired_verdict_counts.csv`: count of cells whose paired 95% CI lies below/above zero (unadjusted).",
             "- `robust_solver_diagnostics.csv`: coverage, solver time, gaps, F(H) vs F(direct), randomization.",
             "- `fit_candidate_scores.csv`: public stress-set scores of the cheap decoders.",
             "- `figures/`: paired_difference_*, absolute_*, span_anchor_rank2, robust_solver_diagnostics (PNG/PDF).",
             "- Raw: `../units/*.json` (per-draw samples, Q, R, metadata) and `../units/*.npz` (reports, assignments, "
             "robust H, Birkhoff mixtures, sampled outputs).", "",
             f"Design: {manifest['replicates']} R x K={manifest['K']}; alphas {manifest['alphas']}; lambdas "
             f"{manifest['lambdas']}; robust panel {manifest['robust_panel'] or 'all (family, n)'}; solver "
             f"{manifest['robust_solver']}.", ""]
    if timing:
        lines += [f"Serial unit seconds {timing.get('serial_seconds', 0):.0f}, of which robust solver "
                  f"{timing.get('robust_solver_seconds', 0):.0f}.", ""]
    lines += ["## Paired verdict counts (decoder − direct; negative = better)", "", markdown_table(verdict), ""]
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        lines += ["## Robust solver diagnostics", "", markdown_table(diag), ""]
    lines += ["## Limitations", "",
              "- Fixed Q per panel; intervals are over 20 conditional R draws, not over public geometries.",
              "- The robust decoder is exactly optimal only for the interval-box relaxation over randomized "
              "matchings; the box contains non-Euclidean cost matrices. No minimax claim for the original problem.",
              "- The expected certificate estimate is a diagnostic on these scenarios, not a bound on untested R.",
              "- Cheap-arm selection uses a finite public stress set (heuristic).",
              "- Descriptive counts are not multiplicity-adjusted significance tests.", ""]
    (output/"README.md").write_text("\n".join(lines))
