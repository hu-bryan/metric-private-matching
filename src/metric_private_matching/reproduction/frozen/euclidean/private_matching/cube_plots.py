"""Uniform-cube figures for the four local mechanisms and their PCA/representative approximations.

Sources (never pooled): the archived anchor follow-up (four paper mechanisms, K=100 per point)
for the epsilon, d and n sweeps, and the cube_approximation study (same frozen masters, new
labeled noise coupled across epsilon) for the approximation sweeps. Each master configuration s
is plotted on its own; intervals are pointwise 95% t intervals over mechanism noise.

    python -m private_matching.cube_plots --archive-summary SUMMARY.csv --approx-table TABLE.csv --out DIR
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from private_matching.login_guard import refuse_login_compute
from private_matching.projection_plots import EPS_RAMP, INK2, PROXY, _style

MECHS = [("ambient", "Λε (ambient location noise)", "#eda100", "^"),
         ("phi_infinity", "Φ∞ (task-distance sup-norm noise)", "#1baf7a", "s"),
         ("folded", "Ψ₂ (folded location noise)", "#2a78d6", "o"),
         ("anchor", "Ξε (folded location + private radius)", "#eb6834", "D")]
SHORT = {"folded": "Ψ₂", "anchor": "Ξε (anchor)"}
FAMILY = {"pca": ("PCA rank", "PCA"), "representatives": ("Representative count", "representative-count")}
YLABEL = "Expected excess cost  E[u] − OPT"


def _legend_below(ax, ncol=3):
    """Legend under the axes so it never covers data or the random-matching reference."""
    ax.legend(fontsize=8, ncol=ncol, loc="upper center", bbox_to_anchor=(.5, -.16), frameon=False)


def _save(fig, path, data, note):
    import textwrap
    fig.text(.01, .01, textwrap.fill(note, int(fig.get_figwidth()*17)), ha="left", va="bottom", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, .06, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    data.to_csv(path.with_suffix(".csv"), index=False)
    print(f"Figure: {path}", flush=True)


def _mechanism_lines(ax, df, x, end_labels=False):
    """Four paper mechanisms with 95% noise bands, direct end labels plus legend."""
    for key, label, colour, marker in MECHS:
        g = df[df.mechanism == key].sort_values(x)
        ax.plot(g[x], g.mean_regret, "-", color=colour, lw=1.6, marker=marker, ms=3, label=label)
        ax.fill_between(g[x], g.ci95_low, g.ci95_high, color=colour, alpha=.15, lw=0)
        if end_labels and len(g):
            ax.annotate(label.split(" ")[0], (g[x].iloc[-1], g.mean_regret.iloc[-1]), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=8, color=INK2)


def epsilon_sweep(arch, s, n, d, out):
    df = arch[(arch.s == s) & (arch.n == n) & (arch.d == d)]
    fig, ax = plt.subplots(figsize=(8, 5))
    _mechanism_lines(ax, df, "epsilon")
    ax.axhline(df.gap.iloc[0], color=PROXY, ls=":", lw=1.2, label="Uniformly random matching")
    ax.set(xlabel="Privacy budget ε", ylabel=YLABEL,
           title=f"Excess cost vs ε\nUniform cube, configuration {s+1}, n={n}, d={d}")
    ax.set_xlim(0, df.epsilon.max()*1.08)
    _legend_below(ax)
    _save(fig, out, df, f"One fixed point set (no averaging over configurations). K=100 noise draws per point; "
                        f"bands are pointwise 95% intervals for the noise mean. Source: archived anchor follow-up.")


def dimension_sweep(arch, s, n, eps, out):
    df = arch[(arch.s == s) & (arch.n == n) & (arch.epsilon == eps)]
    fig, ax = plt.subplots(figsize=(8, 5))
    _mechanism_lines(ax, df, "d")
    gap = df.drop_duplicates("d").sort_values("d")
    ax.plot(gap.d, gap.gap, ls=":", color=PROXY, lw=1.2, label="Uniformly random matching")
    ax.set(xlabel="Dimension d", ylabel=YLABEL,
           title=f"Excess cost vs dimension\nUniform cube, configuration {s+1}, n={n}, ε={eps:g}")
    ax.set_xlim(0, df.d.max()*1.1)
    _legend_below(ax)
    _save(fig, out, df, "Nested slices of one master point set (first d coordinates). K=100 noise draws per point; "
                        "pointwise 95% noise intervals. Source: archived anchor follow-up.")


def size_sweep(arch, s, d, eps, out):
    df = arch[(arch.s == s) & (arch.d == d) & (arch.epsilon == eps)]
    fig, ax = plt.subplots(figsize=(8, 5))
    _mechanism_lines(ax, df, "n")
    gap = df.drop_duplicates("n").sort_values("n")
    ax.plot(gap.n, gap.gap, ls=":", color=PROXY, lw=1.2, label="Uniformly random matching")
    ax.set(xlabel="Number of tasks and drivers n", ylabel=YLABEL,
           title=f"Excess cost vs n\nUniform cube, configuration {s+1}, d={d}, ε={eps:g}")
    ax.set_xlim(0, df.n.max()*1.1)
    _legend_below(ax)
    _save(fig, out, df, "Nested slices of one master point set (first n tasks/drivers). K=100 noise draws per point; "
                        f"pointwise 95% noise intervals. For n ≥ d+1 = {d+1} the folded dimension is d, so Ψ₂ coincides with Λε "
                        "(overlapping curves). Source: archived anchor follow-up.")


def load_approx(table):
    df = pd.read_csv(table, low_memory=False)
    df = df[(df.study_id == "cube_approximation") & (df.status != "certificate_only")
            & df.approximation_family.isin(["pca", "representatives"])
            & ~df.arm_label.astype(str).str.endswith(":selected")].copy()
    df["mechanism"] = df.requested_base_mechanism.fillna(df.base_mechanism)
    df["value"] = df.approximation_rank.fillna(df.representative_count).astype(int)
    df["s"] = df.geometry_replicate.astype(int)
    return df


def approximation_sweep(approx, s, mechanism, family, out):
    df = approx[(approx.s == s) & (approx.approximation_family == family)]
    n, d = int(df.n.iloc[0]), int(df.d.iloc[0])
    axis_label, family_name = FAMILY[family]
    fig, ax = plt.subplots(figsize=(9, 5))
    mech = df[df.mechanism == mechanism]
    values = sorted(mech.value.unique())
    dims = mech.groupby("value").coordinate_noise_dimension.first()
    for colour, eps in zip(EPS_RAMP, sorted(mech.epsilon.unique())):
        g = mech[mech.epsilon == eps].sort_values("value")
        ax.plot(g.value, g.mean_excess_cost, "-o", color=colour, lw=1.6, ms=4, label=f"ε = {eps:g}")
        ax.fill_between(g.value, g.ci95_low, g.ci95_high, color=colour, alpha=.15, lw=0)
    proxy = df[(df.mechanism == "proxy") & (df.epsilon == df.epsilon.min())].sort_values("value")
    ax.plot(proxy.value, proxy.mean_excess_cost, "--", color=PROXY, lw=1.4, marker="o", ms=3,
            label="Noiseless proxy (NONPRIVATE: approximation error only)")
    ax.axhline(df.random_minus_OPT.iloc[0], color=INK2, ls=":", lw=1, label="Uniformly random matching")
    ax.set_xticks(values)
    ax.set_xticklabels([f"{v}\n[{int(dims[v])}]" for v in values], fontsize=7)
    ax.set(xlabel=f"{axis_label} [noise dimension]", ylabel=YLABEL,
           title=f"{SHORT[mechanism]} with {family_name} approximation\nUniform cube, configuration {s+1}, n={n}, d={d}")
    _legend_below(ax)
    endpoint = "rank n−1" if family == "pca" else "count n"
    ties = ("" if family == "pca" else
            " Tasks sharing a representative are indistinguishable to the server, so the noiseless proxy breaks those ties by "
            "input order; at small counts it can be above random matching.")
    _save(fig, out, df[df.mechanism.isin([mechanism, "proxy"])],
          f"Left end = no release (fixed matching); right end ({endpoint}) = exact mechanism. K=100 noise draws, coupled across ε; "
          f"pointwise 95% noise intervals.{ties} Source: cube_approximation study.")


def check_same_instances(arch, approx):
    """The new study must use exactly the archived frozen point sets."""
    rows = []
    for s in sorted(approx.s.unique()):
        n, d = int(approx.n.iloc[0]), int(approx.d.iloc[0])
        new = float(approx[approx.s == s].OPT.iloc[0])
        old = float(arch[(arch.s == s) & (arch.n == n) & (arch.d == d)].OPT.iloc[0])
        rows.append(dict(s=s, n=n, d=d, OPT_new=new, OPT_archive=old, abs_diff=abs(new-old)))
        if abs(new-old) > 1e-9*max(1, abs(old)):
            raise ValueError(f"cube_approximation instance differs from archive for s={s}: {new} vs {old}")
    return pd.DataFrame(rows)


def make_cube_figures(archive_summary, approx_table, out, n=20, d=50, sweep_eps=16.):
    refuse_login_compute("plot cube figures")
    _style()
    out = Path(out)
    arch = pd.read_csv(archive_summary)
    arch = arch[arch.mechanism.isin([m[0] for m in MECHS])]
    approx = load_approx(approx_table)
    out.mkdir(parents=True, exist_ok=True)
    check_same_instances(arch, approx).to_csv(out/"instance_check.csv", index=False)
    for s in sorted(approx.s.unique()):
        base = out if s == 0 else out/"supplement"/f"configuration_{s+1}"
        epsilon_sweep(arch, s, n, d, base/f"01_excess_vs_epsilon_n{n}_d{d}.png")
        dimension_sweep(arch, s, n, sweep_eps, base/f"02_excess_vs_d_n{n}_eps{sweep_eps:g}.png")
        size_sweep(arch, s, d, sweep_eps, base/f"03_excess_vs_n_d{d}_eps{sweep_eps:g}.png")
        for i, (family, mechanism) in enumerate([("pca", "folded"), ("pca", "anchor"),
                                                 ("representatives", "folded"), ("representatives", "anchor")]):
            name = f"{4+i:02d}_{'psi' if mechanism == 'folded' else 'anchor'}_{family}_n{n}_d{d}.png"
            approximation_sweep(approx, s, mechanism, family, base/name)
    for eps in sorted(set(arch.epsilon.unique()) & {1., 8.}):
        extra = out/"supplement"/f"sweeps_eps{eps:g}"
        dimension_sweep(arch, 0, n, eps, extra/f"02_excess_vs_d_n{n}_eps{eps:g}.png")
        size_sweep(arch, 0, d, eps, extra/f"03_excess_vs_n_d{d}_eps{eps:g}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-summary", required=True)
    parser.add_argument("--approx-table", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sweep-epsilon", type=float, default=16.)
    args = parser.parse_args()
    make_cube_figures(args.archive_summary, args.approx_table, args.out, sweep_eps=args.sweep_epsilon)


if __name__ == "__main__":
    main()
