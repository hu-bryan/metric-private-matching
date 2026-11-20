"""Figures and source CSVs for the opt-in `interchangeability` study (compute only).

Stochastic geometries (more than one geometry replicate): mean over geometries of each geometry's
conditional noise mean, with a Student-t 95% interval ACROSS geometries. Deterministic-up-to-labels
cases (one geometry): the conditional noise-mean t interval from the canonical table.

    python -m private_matching.interchangeability_plots --table TABLE.csv --out DIR
"""
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from private_matching.login_guard import refuse_login_compute

ARMS = [("ambient", "Λε ambient", "#eda100", "^"),
        ("phi_infinity", "Φ∞", "#1baf7a", "s"),
        ("folded", "Ψ₂ exact fold", "#2a78d6", "o"),
        ("anchor", "Ξε anchor", "#eb6834", "D"),
        ("folded:representatives:selected", "Ψ₂ public-selected representatives", "#7b4fc9", "v"),
        ("folded:pca:selected", "Ψ₂ public-selected PCA", "#c94f9b", "P"),
        ("fixed", "fixed (identity) matching", "#555555", "x")]
KEYS = ["fingerprint", "run_phase", "panel", "geometry_family", "n", "d", "jitter_sigma", "type_count", "jitter_tau", "arm_id"]


def aggregate(table, fingerprint=None):
    df = pd.read_csv(table, low_memory=False)
    df = df[(df.study_id == "interchangeability") & (df.status != "certificate_only")].copy()
    if fingerprint is not None:
        # The handoff registry can retain other fingerprints of the same phase; never pool them.
        df = df[df.fingerprint == fingerprint]
    if df.empty:
        raise ValueError(f"no interchangeability rows for fingerprint {fingerprint}")
    for k in KEYS:
        if k not in df:
            df[k] = np.nan
    df["norm_excess"] = df.epsilon*df.mean_excess_cost/df.n**2
    rows = []
    for key, g in df.groupby(KEYS, dropna=False):
        G = g.geometry_replicate.nunique()
        assert G == len(g), "one row per geometry and arm expected"
        rec = dict(zip(KEYS, key), geometries=G, K=int(g.K.max()))
        scale = g.epsilon.iloc[0]/g.n.iloc[0]**2
        for col, values in [("excess", g.mean_excess_cost), ("wrong_fraction", g.get("mean_wrong_cluster_fraction")),
                            ("random_minus_OPT", g.random_minus_OPT)]:
            if values is None or values.isna().all():
                continue
            m = float(values.mean())
            if G > 1:
                half = stats.t.ppf(.975, G-1)*values.std(ddof=1)/np.sqrt(G)
                lo, hi, kind = m-half, m+half, "student_t_95_across_geometries"
            elif col == "excess":
                lo, hi, kind = float(g.ci95_low.iloc[0]), float(g.ci95_high.iloc[0]), "student_t_95_conditional_noise"
            else:
                lo = hi = np.nan
                kind = "student_t_95_conditional_noise"
            rec.update({f"{col}_mean": m, f"{col}_lo": lo, f"{col}_hi": hi})
            if col == "excess":
                rec.update(interval_type=kind, norm_excess_mean=m*scale, norm_excess_lo=lo*scale, norm_excess_hi=hi*scale)
        rec["norm_random_minus_OPT"] = rec.get("random_minus_OPT_mean", np.nan)*scale
        if "OPT_wrong_cluster_fraction" in g:
            rec["OPT_wrong_fraction_max"] = float(g.OPT_wrong_cluster_fraction.max())
        rows.append(rec)
    return pd.DataFrame(rows)


def _label(row):
    if row.geometry_family == "permuted_unique_simplex":
        return "unique-optimum simplex (labels only)"
    if row.jitter_sigma == 0:
        return "two-population, σ=0 original (labels only)"
    return f"two-population, jitter σ={row.jitter_sigma:g} ({row.geometries} geometries)"


def _band(ax, g, x, prefix, arm):
    key, label, colour, marker = arm
    g = g[g.arm_id == key].sort_values(x)
    if g.empty:
        return
    ax.plot(g[x], g[f"{prefix}_mean"], color=colour, marker=marker, ms=4, lw=1.5, label=label)
    # Excess and mismatch are nonnegative; display bands clipped at 0 (CSV keeps raw bounds).
    ax.fill_between(g[x], g[f"{prefix}_lo"].clip(lower=0), g[f"{prefix}_hi"], color=colour, alpha=.15, lw=0)


def _save(fig, out, name, note, data):
    fig.text(.01, .005, textwrap.fill(note, int(fig.get_figwidth()*22)), fontsize=6.5, va="bottom")
    fig.tight_layout(rect=(0, .08, 1, 1))
    fig.savefig(out/f"{name}.png", dpi=160)
    fig.savefig(out/f"{name}.pdf")
    plt.close(fig)
    data.to_csv(out/f"{name}.csv", index=False)


def regret_vs_n(s, out):
    s = s[s.panel.isin(["robust_two_population", "unique_simplex"])].copy()
    s["facet"] = s.apply(_label, axis=1)
    facets = sorted(s.facet.unique())
    fig, axes = plt.subplots(1, len(facets), figsize=(4.2*len(facets), 4.2), sharey=False, squeeze=False)
    for ax, f in zip(axes[0], facets):
        g = s[s.facet == f]
        for arm in ARMS:
            _band(ax, g, "n", "norm_excess", arm)
        base = g.drop_duplicates("n").sort_values("n")
        ax.plot(base.n, base.norm_random_minus_OPT, ":", color="#999999", label="uniform random matching")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(g.n.unique()))
        ax.set_xticklabels([str(int(v)) for v in sorted(g.n.unique())])
        ax.set(title=f, xlabel="n = d", ylabel="ε·(E[u] − OPT)/n²")
        ax.set_ylim(bottom=0)
    axes[0][0].legend(fontsize=7)
    _save(fig, out, "fig1_normalized_regret_vs_n",
          "ε=1. Λε reuses Ψ₂'s samples at n=d (hidden under Ψ₂). Bands (clipped at 0): multi-geometry cases use Student-t 95% intervals across geometries of conditional noise means (K draws each); "
          "labels-only cases (one geometry) use conditional noise t intervals. Finite n only; no asymptotic rate is implied.", s)


def regret_vs_g(s, out):
    s = s[s.panel == "interchangeable_types"]
    taus = sorted(s.jitter_tau.unique())
    fig, axes = plt.subplots(1, len(taus), figsize=(4.2*len(taus), 4.2), sharey=True, squeeze=False)
    for ax, tau in zip(axes[0], taus):
        g = s[s.jitter_tau == tau]
        for arm in ARMS:
            _band(ax, g, "type_count", "excess", arm)
        base = g.drop_duplicates("type_count").sort_values("type_count")
        ax.plot(base.type_count, base.random_minus_OPT_mean, ":", color="#999999", label="uniform random matching")
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(sorted(g.type_count.unique()))
        ax.set_xticklabels([str(int(v)) for v in sorted(g.type_count.unique())])
        G = int(g.geometries.max())
        ax.set(title=f"τ={tau:g} ({'1 geometry, labels only' if G == 1 else f'{G} geometries'})",
               xlabel="number of public types g", ylabel="E[u] − OPT (raw)")
    axes[0][0].legend(fontsize=7)
    n = int(s.n.iloc[0])
    _save(fig, out, "fig2_regret_vs_types",
          f"n=d={n}, ε=1, symlog y-axis. τ>0: t intervals across geometries; τ=0: conditional noise intervals.", s)


def mismatch_vs_n(s, out):
    s = s[(s.panel == "robust_two_population") & s.wrong_fraction_mean.notna()].copy()
    s["facet"] = s.apply(_label, axis=1)
    # Semantics are validated only for facets where OPT respects A-left/B-right in every instance.
    valid = s.groupby("facet").OPT_wrong_fraction_max.transform("max") == 0
    if (~valid).any():
        print(f"group-mismatch: dropped unvalidated facets {sorted(s.facet[~valid].unique())}", flush=True)
    s = s[valid]
    if s.empty:
        return
    facets = sorted(s.facet.unique())
    fig, axes = plt.subplots(1, len(facets), figsize=(4.2*len(facets), 4.2), sharey=True, squeeze=False)
    for ax, f in zip(axes[0], facets):
        g = s[s.facet == f]
        for arm in ARMS[:6]:
            _band(ax, g, "n", "wrong_fraction", arm)
        ax.set_xscale("log", base=2)
        ax.set(title=f, xlabel="n = d", ylabel="fraction violating A-left/B-right", ylim=(0, 1))
    axes[0][0].legend(fontsize=7)
    _save(fig, out, "fig3_group_mismatch_vs_n",
          "Mean over draws of the fraction of public tasks matched against the optimal A-left/B-right grouping "
          "(only facets where OPT has 0 mismatch in every instance; others are dropped as unvalidated). multi-geometry cases: t intervals across geometries; labels-only: point means.", s)


def make_report(table, out, fingerprint=None):
    refuse_login_compute("plot interchangeability study")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    s = aggregate(table, fingerprint)
    if s.fingerprint.nunique() != 1:
        raise ValueError("pass --fingerprint: the table holds several interchangeability fingerprints")
    s.to_csv(out/"interchangeability_summary.csv", index=False)
    regret_vs_n(s, out)
    regret_vs_g(s, out)
    mismatch_vs_n(s, out)
    print(f"interchangeability figures written to {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--table", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fingerprint", default=None)
    a = p.parse_args()
    make_report(a.table, a.out, a.fingerprint)
