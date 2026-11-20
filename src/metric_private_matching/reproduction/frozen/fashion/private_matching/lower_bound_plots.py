"""Section 7 (Theorem 31) separation figures: the four local mechanisms on the Psi_2 lower-bound instances.

Instances: n/2 public tasks at -L e1 + delta e_i and n/2 at +L e1 + delta e_i, with L = n/(2 eps)
and delta = L/n^3; half the drivers at (L/2) e1 + 2L e2 (type A), half at (3/10) L e1 (type B).
Everything scales like 1/eps, so eps * excess is eps-invariant; the study uses eps = 1.

    python -m private_matching.lower_bound_plots --table TABLE.csv --out DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from private_matching.cube_plots import MECHS, YLABEL, _legend_below, _save
from private_matching.login_guard import refuse_login_compute
from private_matching.projection_plots import INK2, PROXY, _style

C0 = (5*np.sqrt(17)-19)/40  # Theorem 31 constant


def source(df):
    return f"Source: psi_lower_bound study, K={int(df.K.max())} noise draws per point; pointwise 95% noise intervals."


def load(table):
    df = pd.read_csv(table, low_memory=False)
    df = df[(df.study_id == "psi_lower_bound") & df.arm_id.isin([m[0] for m in MECHS])].copy()
    df["mechanism"] = df.arm_id
    scale = df.epsilon/df.n**2
    for col in ["mean_excess_cost", "ci95_low", "ci95_high", "random_minus_OPT"]:
        df[f"norm_{col}"] = df[col]*scale
    return df


def _lines(ax, df, x, prefix=""):
    for key, label, colour, marker in MECHS:
        g = df[df.mechanism == key].sort_values(x)
        # At n = d, Lambda uses exactly Psi_2's samples: draw it dashed and on top so both stay visible.
        style = dict(ls="--", zorder=4) if key == "ambient" else dict(ls="-", zorder=3)
        ax.plot(g[x], g[f"{prefix}mean_excess_cost"], color=colour, lw=1.6, marker=marker, ms=4, label=label, **style)
        ax.fill_between(g[x], g[f"{prefix}ci95_low"], g[f"{prefix}ci95_high"], color=colour, alpha=.15, lw=0)


def excess_vs_n(df, out):
    s = df[df.sweep == "n_equals_d"]
    fig, ax = plt.subplots(figsize=(8, 5))
    _lines(ax, s, "n")
    gap = s.drop_duplicates("n").sort_values("n")
    ax.plot(gap.n, gap.random_minus_OPT, ls=":", color=PROXY, lw=1.2, label="Uniformly random matching")
    ax.set(xlabel="n = d", ylabel=YLABEL, title="Section 7 separation instances: excess cost vs n (n = d, ε = 1)")
    _legend_below(ax, ncol=2)
    _save(fig, out, s, "At n = d the fold is the identity, so Λε and Ψ₂ are the same mechanism (identical samples; Λε dashed). "
                       "Theory: Ψ₂'s excess is at least (c₀ − o(1))n²/ε (Theorem 31); Ξε's is O(n^{3/2}√log n/ε) (Theorem 30). "
                       + source(s))


def normalized_vs_n(df, out):
    s = df[df.sweep == "n_equals_d"]
    fig, ax = plt.subplots(figsize=(8, 5))
    _lines(ax, s, "n", prefix="norm_")
    gap = s.drop_duplicates("n").sort_values("n")
    ax.plot(gap.n, gap.norm_random_minus_OPT, ls=":", color=PROXY, lw=1.2, label="Uniformly random matching")
    ax.axhline(C0, color=INK2, ls="-.", lw=1.1, label=f"Theorem 31 limit for Ψ₂: c₀ = (5√17−19)/40 ≈ {C0:.4f}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(s.n.unique()))
    ax.set_xticklabels([str(int(v)) for v in sorted(s.n.unique())], fontsize=7)
    ax.set_ylim(bottom=0)
    ax.set(xlabel="n = d (log scale)", ylabel="ε · (E[u] − OPT) / n²",
           title="Section 7 separation instances: excess scaled by n²/ε")
    _legend_below(ax, ncol=2)
    _save(fig, out, s, "Ψ₂ = Λε at n = d. Dash-dot: Theorem 31's asymptotic constant, ε·excess/n² ≥ c₀ − o(1) for Ψ₂ "
                       "(finite n: (c₀ − 1/n³) on an event of probability 1 − o(1)). Theorem 30 gives O(√(log n / n)) for Ξε "
                       "on this scale. Dotted: random matching. " + source(s))


def excess_vs_d(df, out):
    s = df[df.sweep == "d_zero_padded"]
    n = int(s.n.iloc[0])
    fig, ax = plt.subplots(figsize=(8, 5))
    _lines(ax, s, "d")
    gap = s.drop_duplicates("d").sort_values("d")
    ax.plot(gap.d, gap.random_minus_OPT, ls=":", color=PROXY, lw=1.2, label="Uniformly random matching")
    ax.set(xlabel="Dimension d", ylabel=YLABEL,
           title=f"Section 7 separation instances: excess cost vs d (n = {n}, ε = 1)")
    ax.set_xlim(0, s.d.max()*1.05)
    _legend_below(ax, ncol=2)
    _save(fig, out, s, f"The n = {n} construction embedded in R^d with zero trailing coordinates: the public affine rank stays "
                       f"{n-1}, so Ψ₂ and Ξε (folded dimension {n}) and Φ∞ do not depend on d, while Λε's noise lives in all d "
                       "coordinates. " + source(s))


def wrong_side_vs_n(df, out):
    s = df[df.sweep == "n_equals_d"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, colour, marker in MECHS:
        g = s[s.mechanism == key].sort_values("n")
        ax.plot(g.n, g.mean_wrong_cluster_fraction, color=colour, lw=1.6, marker=marker, ms=4, label=label,
                ls="--" if key == "ambient" else "-", zorder=4 if key == "ambient" else 3)
    ax.axhline(.5, color=PROXY, ls=":", lw=1.2, label="Uniformly random matching (0.5)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(s.n.unique()))
    ax.set_xticklabels([str(int(v)) for v in sorted(s.n.unique())], fontsize=7)
    ax.set_ylim(0, 1.02)
    ax.set(xlabel="n = d (log scale)", ylabel="Fraction of drivers sent to the wrong side",
           title="Section 7 separation instances: how often the matching is reversed")
    _legend_below(ax, ncol=2)
    _save(fig, out, s, "Wrong side = driver type opposite to the optimal grouping (type A left, type B right). "
                       "Theorem 31's proof shows Ψ₂'s noisy costs reverse every assignment with probability → 1 (fraction → 1). "
                       + source(s))


def make_figures(table, out):
    refuse_login_compute("plot Section 7 figures")
    _style()
    df, out = load(table), Path(out)
    excess_vs_n(df, out/"08_section7_excess_vs_n.png")
    normalized_vs_n(df, out/"09_section7_normalized_excess_vs_n.png")
    excess_vs_d(df, out/f"10_section7_excess_vs_d_n{int(df[df.sweep == 'd_zero_padded'].n.iloc[0])}.png")
    wrong_side_vs_n(df, out/"11_section7_wrong_side_fraction_vs_n.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    make_figures(args.table, args.out)


if __name__ == "__main__":
    main()
