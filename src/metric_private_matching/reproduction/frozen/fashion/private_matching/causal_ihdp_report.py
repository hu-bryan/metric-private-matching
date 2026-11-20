"""Figures, tables and REPORT.md for the IHDP private-matching experiment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from private_matching.causal_ihdp import (
    ARMS, COND_NP, COND_RANDOM, COND_SQ, KALLUS_COND1_MSE, MECH_LAMBDA, MECH_PHI, MECH_PSI2, MECH_XI,
    write_json,
)

# Reference categorical slots 1-3 (validated all-pairs, light surface); slot 4 only if Psi2 runs.
SERIES = {
    MECH_LAMBDA: ("#2a78d6", "o", r"$\Lambda_\varepsilon$"),
    MECH_PHI: ("#eb6834", "s", r"$\Phi_\infty$"),
    MECH_XI: ("#1baf7a", "^", r"$\Xi_\varepsilon$"),
    MECH_PSI2: ("#eda100", "D", r"$\Psi_2$"),
}
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
BASE_STYLE = {
    COND_SQ: dict(color=INK2, ls=":", lw=1.6, label="Cond. 1: non-private, squared Mahalanobis"),
    COND_NP: dict(color=INK, ls="-", lw=1.4, label=r"Cond. 2: non-private ($\varepsilon=\infty$)"),
    COND_RANDOM: dict(color=MUTED, ls="--", lw=1.4, label="Cond. 4: random matching"),
}
ARM_TITLE = {"A": "Arm A: treated private (n=94), controls public",
             "B": "Arm B: controls private (n=279), treated public"}
MECH_NAME = {MECH_LAMBDA: "Lambda_eps", MECH_PHI: "Phi_inf", MECH_XI: "Xi_eps", MECH_PSI2: "Psi_2",
             COND_SQ: "Cond 1 (non-private, squared)", COND_NP: "Cond 2 (non-private, eps=inf)",
             COND_RANDOM: "Cond 4 (random)"}


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "axes.titlesize": 10.5, "legend.frameon": False,
        "lines.linewidth": 2.0, "lines.markersize": 5,
    })
    return plt


def _save(fig, out: Path, stem: str) -> list[str]:
    paths = []
    for ext in ("png", "pdf"):
        p = out / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight")
        paths.append(str(p))
    return paths


def _mechs(summary: pd.DataFrame) -> list[str]:
    present = set(summary["mech"])
    return [m for m in (MECH_LAMBDA, MECH_PHI, MECH_XI, MECH_PSI2) if m in present]


def _baselines(summary: pd.DataFrame, arm: str) -> pd.DataFrame:
    return summary[(summary["arm"] == arm) & ~np.isfinite(summary["eps"])].set_index("mech")


def fig_metric_vs_eps(summary: pd.DataFrame, out: Path, metric: str, ylabel: str, stem: str,
                      title: str, band_frac: float | None = None) -> list[str]:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for ax, arm in zip(axes, ARMS):
        base = _baselines(summary, arm)
        for mech in _mechs(summary):
            g = summary[(summary["arm"] == arm) & (summary["mech"] == mech)].sort_values("eps")
            if g.empty:
                continue
            color, marker, label = SERIES[mech]
            ax.plot(g["eps"], g[metric], color=color, marker=marker, ms=6, label=label, zorder=3)
            if metric == "mse":
                ax.fill_between(g["eps"], g["mse"] - 2 * g["mse_se"], g["mse"] + 2 * g["mse_se"],
                                color=color, alpha=0.12, lw=0, zorder=2)
            ax.annotate(label, (g["eps"].iloc[0], g[metric].iloc[0]), xytext=(-6, 0),
                        textcoords="offset points", ha="right", va="center", fontsize=9, color=INK2)
        for cond, style in BASE_STYLE.items():
            ax.axhline(base.loc[cond, metric], zorder=1, **style)
        if band_frac is not None:
            ref = base.loc[COND_NP, metric]
            ax.axhspan(ref * (1 - band_frac), ref * (1 + band_frac), color=INK, alpha=0.06, lw=0,
                       label=f"within {int(100 * band_frac)}% of cond. 2")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"privacy budget $\varepsilon$ (log)")
        ax.set_title(ARM_TITLE[arm], loc="left")
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    paths = _save(fig, out, stem)
    plt.close(fig)
    return paths


def fig_excess(summary: pd.DataFrame, out: Path) -> list[str]:
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, arm in zip(axes, ARMS):
        base = _baselines(summary, arm)
        for mech in _mechs(summary):
            g = summary[(summary["arm"] == arm) & (summary["mech"] == mech)].sort_values("eps")
            if g.empty:
                continue
            color, marker, label = SERIES[mech]
            pos = g["excess"] > 0
            ax.plot(g["eps"][pos], g["excess"][pos], color=color, marker=marker, ms=6, label=label, zorder=3)
            ax.plot(g["eps"], g["cert"], color=color, ls="--", lw=1.3, zorder=2)
            if arm == "A":
                ax.plot(g["eps"], g["bound"], color=color, ls=":", lw=1.6, zorder=2)
        ax.axhline(base.loc[COND_RANDOM, "excess"], **BASE_STYLE[COND_RANDOM])
        # slope -1 guide
        eps = summary.loc[np.isfinite(summary["eps"]), "eps"]
        x0, x1 = float(eps.quantile(0.55)), float(eps.max())
        y0 = float(summary.loc[(summary["arm"] == arm) & np.isfinite(summary["eps"]), "excess"].replace(0, np.nan).min()) * 3
        y0 = y0 if np.isfinite(y0) and y0 > 0 else 1e-3
        ax.plot([x0, x1], [y0 * x1 / x0, y0], color=MUTED, lw=1.0)
        ax.annotate("slope −1", (x1, y0), xytext=(-4, 4), textcoords="offset points", ha="right",
                    fontsize=8.5, color=MUTED)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"privacy budget $\varepsilon$ (log)")
        ax.set_title(ARM_TITLE[arm] + ("" if arm == "A" else "\n(no theorem applies; certificate only)"), loc="left")
    axes[0].set_ylabel(r"normalized excess cost $(u(\hat\sigma)-\mathrm{OPT})/n_1$ (log)")
    from matplotlib.lines import Line2D

    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color=INK2, ls="--", lw=1.3), Line2D([], [], color=INK2, ls=":", lw=1.6)]
    labels += ["mean pointwise certificate (dashed)", "theorem bound, Arm A only (dotted)"]
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.1), fontsize=9)
    fig.suptitle("Excess matching cost on true costs, with certificates and bounds", x=0.01, ha="left",
                 fontsize=11.5, color=INK)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    paths = _save(fig, out, "fig2_excess_vs_eps")
    plt.close(fig)
    return paths


def fig_bias_scatter(records: pd.DataFrame, out: Path) -> list[str]:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    priv = records[np.isfinite(records["eps"])]
    for mech in [m for m in SERIES if m in set(priv["mech"])]:
        g = priv[priv["mech"] == mech]
        color, marker, label = SERIES[mech]
        ax.scatter(g["u_norm"], g["match_bias"], s=3, alpha=0.18, color=color, marker=marker,
                   lw=0, label=label, rasterized=True)
    base = records[~np.isfinite(records["eps"]) & (records["arm"] == "A")]
    for cond, style in BASE_STYLE.items():
        g = base[base["mech"] == cond]
        if cond == COND_RANDOM:
            ax.scatter(g["u_norm"], g["match_bias"], s=4, alpha=0.3, color=MUTED, lw=0, rasterized=True,
                       label="random matching")
    # binned mean over all runs, pooled
    allr = records[records["arm"].isin(ARMS)]
    bins = np.quantile(allr["u_norm"], np.linspace(0, 1, 26))
    idx = np.clip(np.digitize(allr["u_norm"], bins) - 1, 0, len(bins) - 2)
    centers = [allr["u_norm"][idx == k].mean() for k in range(len(bins) - 1)]
    means = [allr["match_bias"][idx == k].mean() for k in range(len(bins) - 1)]
    meds = [allr["match_bias"][idx == k].median() for k in range(len(bins) - 1)]
    ax.plot(centers, means, color=INK, lw=2.0, label="binned mean (all runs)")
    ax.plot(centers, meds, color=INK, lw=1.4, ls="--", label="binned median (all runs)")
    np_u = float(base[base["mech"] == COND_NP]["u_norm"].iloc[0])
    ax.axvline(np_u, color=INK, lw=0.8, ls=":")
    ax.annotate("OPT/n₁", (np_u, ax.get_ylim()[1]), xytext=(3, -12), textcoords="offset points",
                fontsize=8.5, color=INK2)
    ax.set_yscale("symlog", linthresh=1.0)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xlabel(r"matched cost on true distances $u(\hat\sigma)/n_1$")
    ax.set_ylabel(r"matching-induced bias $\frac{1}{n_1}\sum_i [f_0(X_i)-f_0(X_{\hat\sigma(i)})]$ (symlog)")
    ax.set_title("Matching cost drives bias: every private run, both arms, all ε", loc="left")
    leg = ax.legend(loc="upper left", fontsize=8.5, markerscale=4)
    for h in leg.legend_handles:
        if hasattr(h, "set_alpha"):
            h.set_alpha(1.0)
    fig.tight_layout()
    paths = _save(fig, out, "fig3_bias_vs_cost")
    plt.close(fig)
    return paths


def fig_eps10(head: pd.DataFrame, out: Path) -> list[str]:
    plt = _plt()
    mechs = [m for m in SERIES if m in set(head["mech"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    width = 0.8 / len(mechs)
    for k, mech in enumerate(mechs):
        color, _, label = SERIES[mech]
        g = head[head["mech"] == mech].set_index("arm").reindex(list(ARMS))
        x = np.arange(len(ARMS)) + (k - (len(mechs) - 1) / 2) * width
        val = g["eps10"].to_numpy(dtype=float)
        lo = np.clip(val - g["eps10_lo"].to_numpy(dtype=float), 0, None)
        hi = np.clip(g["eps10_hi"].to_numpy(dtype=float) - val, 0, None)
        ax1.bar(x, np.nan_to_num(val), width * 0.92, color=color, label=label, zorder=3)
        ax1.errorbar(x, val, yerr=[np.nan_to_num(lo), np.nan_to_num(hi)], fmt="none", ecolor=INK2,
                     elinewidth=1, capsize=3, zorder=4)
        for xi, v, st in zip(x, val, g["eps10_status"]):
            txt = f"{v:.3g}" if np.isfinite(v) else "not reached"
            if st != "interpolated":
                txt += f"\n({st})"
            ax1.annotate(txt, (xi, v if np.isfinite(v) else 1), xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=8, color=INK2)
        if mech != MECH_LAMBDA:
            obs = g["observed_ratio_to_Lambda"].to_numpy(dtype=float)
            pred = g["predicted_ratio_to_Lambda"].to_numpy(dtype=float)
            xx = np.arange(len(ARMS)) + (k - 1 - (len(mechs) - 2) / 2) * (0.8 / max(len(mechs) - 1, 1))
            w2 = 0.8 / max(len(mechs) - 1, 1) / 2
            ax2.bar(xx - w2 / 2, np.nan_to_num(obs), w2 * 0.9, color=color, zorder=3,
                    label=f"{label} observed")
            ax2.bar(xx + w2 / 2, pred, w2 * 0.9, facecolor="none", edgecolor=color, hatch="///", lw=1.2,
                    zorder=3, label=f"{label} predicted by bounds")
            for xo, o, p in zip(xx, obs, pred):
                ax2.annotate(f"{o:.2g}" if np.isfinite(o) else "n/a", (xo - w2 / 2, o if np.isfinite(o) else 1),
                             xytext=(0, 2), textcoords="offset points", ha="center", fontsize=8, color=INK2)
                ax2.annotate(f"{p:.2g}", (xo + w2 / 2, p), xytext=(0, 2), textcoords="offset points",
                             ha="center", fontsize=8, color=INK2)
    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(len(ARMS)), ["Arm A (treated private)", "Arm B (controls private)"])
        ax.set_yscale("log")
        ax.grid(axis="x", visible=False)
    ax2.axhline(1.0, color=MUTED, lw=0.8)
    ax1.set_ylabel(r"$\varepsilon_{10}$: smallest $\varepsilon$ with MSE within 10% of cond. 2 (log)")
    ax1.set_title(r"$\varepsilon_{10}$ with 95% paired-bootstrap intervals", loc="left")
    ax2.set_ylabel(r"$\varepsilon_{10}$ / $\varepsilon_{10}(\Lambda_\varepsilon)$ (log)")
    ax2.set_title("Ratio to Λε: observed vs bound-predicted\n(|Q|/d for Φ∞, (2√β_Q+1)²/(2d) for Ξε)", loc="left")
    ax1.legend(loc="upper left", fontsize=8.5)
    ax2.legend(loc="upper left", fontsize=8, ncol=1)
    fig.tight_layout()
    paths = _save(fig, out, "fig4_eps10_bars")
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- tables

def _fmt(x: Any, nd: int = 3) -> str:
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    ax = abs(float(x))
    if ax != 0 and (ax < 1e-3 or ax >= 1e4):
        return f"{x:.{nd - 1}e}"
    return f"{x:.{nd}g}" if ax >= 1 else f"{x:.{nd}f}".rstrip("0").rstrip(".") if ax else "0"


def _md_table(df: pd.DataFrame, cols: list[str], headers: list[str]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def build_table(summary: pd.DataFrame) -> pd.DataFrame:
    t = summary.copy()
    t["mechanism"] = t["mech"].map(MECH_NAME)
    t["bound_kind"] = np.where(~np.isfinite(t["eps"]), "",
                               np.where(t["arm"] == "A", "theorem", "heuristic formula (no theorem)"))
    order = {m: i for i, m in enumerate([COND_SQ, COND_NP, COND_RANDOM, MECH_LAMBDA, MECH_PHI, MECH_XI, MECH_PSI2])}
    t["_o"] = t["mech"].map(order)
    t = t.sort_values(["arm", "_o", "eps"]).drop(columns="_o")
    cols = ["arm", "mechanism", "eps", "n_reps", "bias", "sd", "mse", "mse_se", "median_abs_err",
            "excess", "cert", "bound", "excess_over_bound", "bound_kind", "balance", "match_bias", "u_norm"]
    return t[cols]


def write_report(root: Path, cfg: dict[str, Any], exp, geometry: dict[str, Any], grid: np.ndarray,
                 records: pd.DataFrame, summary: pd.DataFrame, head: pd.DataFrame,
                 checks: list[dict[str, Any]], log=print) -> None:
    fig_dir = root / "figures"
    tab_dir = root / "tables"
    fig_dir.mkdir(exist_ok=True)
    tab_dir.mkdir(exist_ok=True)
    figs = []
    figs += fig_metric_vs_eps(summary, fig_dir, "mse", r"MSE of $\hat\tau$ (log)", "fig1_mse_vs_eps",
                              "Headline: MSE of the matched ATT estimate vs privacy budget",
                              band_frac=float(cfg["within_frac"]))
    figs += fig_excess(summary, fig_dir)
    figs += fig_bias_scatter(records, fig_dir)
    figs += fig_eps10(head, fig_dir)
    figs += fig_metric_vs_eps(summary, fig_dir, "median_abs_err", r"median $|\hat\tau-4|$ (log)",
                              "figS1_median_abs_err_vs_eps",
                              "Median absolute error (robust companion to MSE; surface B is heavy-tailed)")
    figs += fig_metric_vs_eps(summary, fig_dir, "balance", "mean |SMD| over 25 covariates (log)",
                              "figS2_balance_vs_eps", "Covariate balance of the released matching")

    table = build_table(summary)
    table.to_csv(tab_dir / "summary_table.csv", index=False)
    md_cols = ["arm", "mechanism", "eps", "bias", "sd", "mse", "median_abs_err", "excess",
               "excess_over_bound", "balance"]
    md_head = ["arm", "mechanism", "ε", "bias", "SD", "MSE", "median abs err", "excess/n₁",
               "excess ÷ bound", "mean |SMD|"]
    (tab_dir / "summary_table.md").write_text(
        "# Summary table\n\nε = inf rows are the non-private and random baselines. `excess ÷ bound` uses "
        "the theorem bound in Arm A and the same formula (no theorem) in Arm B.\n\n"
        + _md_table(table, md_cols, md_head) + "\n")
    head.to_csv(tab_dir / "headline.csv", index=False)
    write_json(tab_dir / "sanity_checks.json", checks)

    base_a = summary[(summary["arm"] == "A") & ~np.isfinite(summary["eps"])].set_index("mech")
    gA, gB = geometry["arms"]["A"], geometry["arms"]["B"]
    hd = head.copy()
    hd["mechanism"] = hd["mech"].map(MECH_NAME)
    hd["eps10 [95% CI]"] = [f"{_fmt(r.eps10)} [{_fmt(r.eps10_lo)}, {_fmt(r.eps10_hi)}]"
                            + ("" if r.eps10_status == "interpolated" else f" ({r.eps10_status})")
                            for r in hd.itertuples()]
    hd["eps0 [95% CI]"] = [f"{_fmt(r.eps0)} [{_fmt(r.eps0_lo)}, {_fmt(r.eps0_hi)}]"
                           + ("" if r.eps0_status == "interpolated" else f" ({r.eps0_status})")
                           for r in hd.itertuples()]
    head_md = _md_table(hd, ["arm", "mechanism", "eps10 [95% CI]", "d_over_eps10_over_mbar", "eps0 [95% CI]",
                             "d_over_eps0_over_mbar", "observed_ratio_to_Lambda", "predicted_ratio_to_Lambda",
                             "eps10_B_over_A"],
                        ["arm", "mechanism", "ε₁₀ [95% CI]", "(d/ε₁₀)/m̄", "ε₀ [95% CI]", "(d/ε₀)/m̄",
                         "ε₁₀ ÷ ε₁₀(Λ) observed", "predicted by bounds", "ε₁₀(B) ÷ ε₁₀(A)"])
    (tab_dir / "headline.md").write_text("# Headline numbers\n\n" + head_md + "\n")
    checks_md = "\n".join(f"| {c['status']} | {c['check']} | {c['detail']} |" for c in checks)
    n_fail = sum(c["status"] == "FAIL" for c in checks)
    n_flag = sum(c["status"] == "FLAG" for c in checks)

    grid_info = (root / "pilot" / "grid_choice.json").read_text()
    import json

    gc = json.loads(grid_info)
    report = f"""# Private 1:1 matching for ATT estimation on IHDP: results

Implements `causal_exp_plan.md` against the current draft (Definitions 6, 8 and 12, Lemmas 11 and 28, Theorems 12, 16 and 30).
Generated by `python -m private_matching causal-ihdp`. Every number below is computed by the code from `production/records.csv.gz`.

## What you are looking at

* **Estimand**: ATT = 4 under Hill's response surface B, which is redrawn in every one of {cfg['replications']} replications. The estimator is the 1:1 matched mean difference.
* **Data**: IHDP (Hill's `ihdp.RData` from `vdorie/npci`, sha256 `{geometry['data_sha256'][:16]}…`). {' '.join(geometry['data_notes'])}
  This gives n₁ = {geometry['n1']} treated and n₀ = {geometry['n0']} controls, with d = {geometry['d']} covariates.
* **Distance**: unsquared Mahalanobis with Σ = Σ̂₀ (ridge {geometry['ridge']:.2g}). After whitening it is Euclidean in ℝ^{geometry['d']}. OPT/n₁ = m̄ = **{geometry['mbar']:.4f}**.
* **Arm A** (treated private, controls public): k = {gA['k']}, m = {gA['m']}, N_Q = {gA['N_Q']}, β_Q = {gA['beta_Q']:.3f}, ε_y = {gA['eps_y_frac']:.4f}ε, ε_s = {gA['eps_s_frac']:.4f}ε. Mechanisms: {', '.join(gA['mechanisms'])}.
* **Arm B** (controls private, treated public): k = {gB['k']}, m = {gB['m']}, N_Q = {gB['N_Q']}, β_Q = {gB['beta_Q']:.3f}, ε_y = {gB['eps_y_frac']:.4f}ε, ε_s = {gB['eps_s_frac']:.4f}ε. Mechanisms: {', '.join(gB['mechanisms'])}.
  {'Ψ₂ is skipped because m = d in both arms, so Π_Q is the identity and Ψ₂ equals Λ_ε.' if gA['m'] == geometry['d'] and gB['m'] == geometry['d'] else 'Ψ₂ runs in each arm where m < d.'}
* **ε grid**: {len(grid)} log-spaced values from {grid[0]:.4g} to {grid[-1]:.4g}. The start is d/ε = 10·m̄. The end was set from a separate pilot ({cfg['pilot_replications']} replications on a disjoint seed stream, never pooled): {gc['reason']}. The pilot's ε₁₀(Φ∞) was {', '.join(f"arm {a}: {_fmt(v['eps10'])} ({v['status']})" for a, v in gc['pilot_phi_eps10'].items())}.
* **Noise coupling**: in replication r, β and the outcome noise are shared by every condition and arm. Each mechanism's base noise is drawn once at ε = 1 and rescaled by 1/ε, so the curves are paired across ε.

### Predicted normalized excess (from the data)

| | Λ_ε | Φ∞ | Ξ_ε |
|---|---|---|---|
| Arm A (theorems as stated) | 2d/ε = {2 * geometry['d']}/ε | 2\\|Q\\|/ε = {2 * gA['n_public']}/ε | (2√β_Q+1)²/ε = {(2 * math.sqrt(gA['beta_Q']) + 1) ** 2:.1f}/ε |
| Arm B (formula only, no theorem) | {2 * geometry['d']}/ε | {2 * gB['n_public']}/ε | {(2 * math.sqrt(gB['beta_Q']) + 1) ** 2:.1f}/ε |

## Baselines (identical in both arms)

| condition | MSE | median \\|err\\| | bias | excess/n₁ | mean \\|SMD\\| |
|---|---|---|---|---|---|
| 1: non-private, squared Mahalanobis | {_fmt(base_a.loc[COND_SQ, 'mse'])} | {_fmt(base_a.loc[COND_SQ, 'median_abs_err'])} | {_fmt(base_a.loc[COND_SQ, 'bias'])} | {_fmt(base_a.loc[COND_SQ, 'excess'])} | {_fmt(base_a.loc[COND_SQ, 'balance'])} |
| 2: non-private, unsquared (ε = ∞) | {_fmt(base_a.loc[COND_NP, 'mse'])} | {_fmt(base_a.loc[COND_NP, 'median_abs_err'])} | {_fmt(base_a.loc[COND_NP, 'bias'])} | 0 | {_fmt(base_a.loc[COND_NP, 'balance'])} |
| 4: random matching | {_fmt(base_a.loc[COND_RANDOM, 'mse'])} | {_fmt(base_a.loc[COND_RANDOM, 'median_abs_err'])} | {_fmt(base_a.loc[COND_RANDOM, 'bias'])} | {_fmt(base_a.loc[COND_RANDOM, 'excess'])} | {_fmt(base_a.loc[COND_RANDOM, 'balance'])} |

Kallus's (2020) condition-1 anchor is {KALLUS_COND1_MSE}.

## Headline numbers

ε₁₀ is the smallest ε from which MSE stays within {int(100 * cfg['within_frac'])}% of condition 2 across the rest of the grid, log-interpolated between grid points.
ε₀ is the largest ε at which MSE is no better than random matching.
The intervals are 95% paired bootstraps over replications ({cfg['bootstrap']} resamples).

{head_md}

## Sanity checks ({n_fail} FAIL, {n_flag} FLAG)

| status | check | detail |
|---|---|---|
{checks_md}

## Caveats that must accompany these results

* Only the covariates are protected. Outcomes are used without protection.
* Σ = Σ̂₀, the centring point q₁, Π_Q, N_Q, β_Q and the Ξ_ε budget split are computed from public data before any private message. **In Arm B, Σ̂₀ comes from the private controls, so treating it as public is an assumption** (for example, that it was published at population level).
* Arm B quotes no theorem (not Theorem 12, 16 or 30). Its "bound" column uses the Arm A formula with Arm B's |Q|, m and β_Q, and serves only as a scale reference. The pointwise rectangular certificate Σ_(j∈σ*) b_j + Σ_(j∈σ̂) b_j is checked in every run.
* Privacy results use unsquared distances only. Squared Mahalanobis appears only in the non-private condition 1.
* Surface B is heavy-tailed, so read median |error| (fig S1) next to MSE.
* The plan says to remove controls whose mothers are "aged 26 or older", which gives 257 controls. Keeping controls aged **≤ 26** reproduces the plan's n₀ = 279, and that rule is what was used.
* The raw `first` variable is coded 1/2 and was recoded to 0/1. This changes surface B but not the Mahalanobis geometry.

## Files

* `figures/fig1_mse_vs_eps.(png|pdf)`: headline MSE vs ε, one panel per arm, with conditions 1, 2 and 4 as horizontal lines and the 10% band around condition 2 shaded.
* `figures/fig2_excess_vs_eps.(png|pdf)`: normalized excess on log-log axes, with the mean certificate (dashed), the theorem bound (dotted, Arm A only) and a slope −1 guide.
* `figures/fig3_bias_vs_cost.(png|pdf)`: matching-induced bias against u(σ̂)/n₁ for every run.
* `figures/fig4_eps10_bars.(png|pdf)`: ε₁₀ per mechanism and arm, and the observed vs predicted ratio to Λ_ε.
* `figures/figS1_median_abs_err_vs_eps`, `figures/figS2_balance_vs_eps`: supplementary robust-error and balance curves.
* `tables/summary_table.(csv|md)`: bias, SD, MSE, median |error|, excess, certificate, bound, excess ÷ bound and balance for each arm × mechanism × ε.
* `tables/headline.(csv|md)`, `tables/sanity_checks.json`.
* `production/records.csv.gz`: one row per replication × arm × condition × ε. `production/noise_checks.csv`: noise-norm sums.
* `pilot/`: pilot records and summary, plus `grid_choice.json`. `geometry.json` and `config.json` hold provenance.
"""
    (root / "REPORT.md").write_text(report)
    log(f"Wrote {len(figs)} figure files, tables and {root / 'REPORT.md'}")
    log(f"Sanity: {n_fail} FAIL, {n_flag} FLAG")
    for c in checks:
        log(f"  [{c['status']}] {c['check']}: {c['detail']}")
