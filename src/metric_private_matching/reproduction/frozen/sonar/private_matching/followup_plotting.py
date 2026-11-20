"""Conditional noise averages for dimension, privacy, and public approximation sweeps."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

from private_matching.config import epsilon_key
from private_matching.followup import MAIN_ARMS, FollowupConfig, load_complete_records, log
from private_matching.proxies import endpoint_grid
from private_matching.results import atomic_write_json

NORMALIZED_YLABEL = "Mean regret / (random cost − OPT)"
EPSILON_ARMS = ("ambient", "phi_infinity", "folded", "anchor", "centered", "fixed", "auto")
EPSILON_INTERVAL_ARMS = {"anchor", "folded"}
LABELS = {
    "ambient": "Ambient", "phi_infinity": r"$\Phi_\infty$", "folded": r"Original $\Psi$",
    "anchor": "Anchor-distance", "centered": "Centered-distance", "fixed": "Fixed (no release)",
    "auto": "Public min: original / anchor / fixed", "anchor_pca_best": "Anchor + public-selected PCA",
    "folded_pca_best": "Original + public-selected PCA", "anchor_representatives_best": "Anchor + public-selected reps",
    "folded_representatives_best": "Original + public-selected reps", "exponential": "Exponential (central; n ≤ 8)",
}
COLORS = {name: plt.get_cmap("tab20")(i) for i, name in enumerate((*MAIN_ARMS, "exponential"))}


def noise_summary(values: list[float]) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    half = float(t.ppf(.975, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
    return mean, half


def _line(ax, xs, rows, name, *, color=None, label=None, interval=True):
    ys, intervals = [], []
    for record, arm in rows:
        mean, half = noise_summary(arm["excess"])
        scale = record["gap"]
        if scale <= 0:
            ys.append(np.nan)
            intervals.append(np.nan)
        else:
            ys.append(mean / scale)
            intervals.append(half / scale)
    color = color if color is not None else COLORS[name]
    ax.plot(xs, ys, label=label or LABELS[name], color=color, linewidth=1.35,
            linestyle="-", marker="." if len(xs) < 60 else None, markersize=3)
    if interval:
        ax.fill_between(xs, np.asarray(ys) - intervals, np.asarray(ys) + intervals, color=color, alpha=.12)
    ax.grid(alpha=.2)
    ax.set_ylim(bottom=0)


def _share_ylim(axes, *, floor=1.0, cap=1.0) -> None:
    ymax = floor
    for ax in axes:
        for line in ax.get_lines():
            y = np.asarray(line.get_ydata(), dtype=float)
            if y.size and np.isfinite(y).any():
                ymax = max(ymax, float(np.nanmax(y)))
    top = min(cap, max(floor, ymax)) if cap is not None else max(floor, ymax)
    for ax in axes:
        ax.set_ylim(0, top)


def _save(fig, directory: Path, name: str, title: str, *, legend_rows=0.12) -> list[str]:
    fig.suptitle(title, fontsize=13)
    handles, labels = [], []
    for ax in fig.axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
    fig.tight_layout(rect=(0, legend_rows, 1, .95))
    paths = []
    for extension in ("png", "pdf"):
        path = directory / f"{name}.{extension}"
        fig.savefig(path, dpi=180)
        paths.append(str(path))
    plt.close(fig)
    log(f"FIGURE {name}.png/.pdf")
    return paths


def plot_followup(cfg: FollowupConfig, directory: Path, fp: str) -> list[str]:
    records = load_complete_records(cfg, directory, fp)
    index = {(r["s"], r["n"], r["d"], r["epsilon"]): r for r in records}
    figures = directory / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    # Flat table includes true regret, error of public proxies, release dimensions,
    # actual privacy splits, branch choices, and conditional noise-only intervals.
    table = []
    for record in records:
        for name, arm in record["arms"].items():
            mean, half = noise_summary(arm["excess"])
            gap = record["gap"]
            row = {k: record[k] for k in ("s", "n", "d", "epsilon", "K", "OPT", "gap")}
            row.update(mechanism=name, mean_regret=mean, ci95_low=mean-half, ci95_high=mean+half,
                       normalized_regret=mean/gap if gap > 0 else None,
                       **{k: v for k, v in arm.items() if k != "excess"})
            table.append(row)
    columns = sorted({key for row in table for key in row})
    with (directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(table)
    paths = []
    n_eps_panels = len(cfg.comparison_settings)
    n_eps_cols = min(2, n_eps_panels)
    n_eps_rows = math.ceil(n_eps_panels / n_eps_cols)
    for s in range(cfg.base.configurations_S):
        for eps in cfg.base.plot_epsilons:
            fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), squeeze=False)
            for col, (settings, xlabel) in enumerate(((cfg.base.n_sweep_settings(), "Number of public tasks n"),
                                                      (cfg.base.d_sweep_settings(), "Ambient dimension d"))):
                for name in (*MAIN_ARMS, "exponential"):
                    selected = [index[(s, n, d, eps)] for n, d in settings if name in index[(s, n, d, eps)]["arms"]]
                    if not selected:
                        continue
                    xs = [r["n"] if col == 0 else r["d"] for r in selected]
                    rows = [(r, r["arms"][name]) for r in selected]
                    _line(axes[0, col], xs, rows, name)
                axes[0, col].set_xlabel(xlabel)
                axes[0, col].set_ylabel(NORMALIZED_YLABEL)
                axes[0, col].set_title(f"d={cfg.base.varying_n_fixed_d}" if col == 0 else f"n={cfg.base.varying_d_fixed_n}")
            paths += _save(fig, figures, f"dimensions_config{s+1}_eps{epsilon_key(eps)}",
                           f"Point configuration {s+1}, ε={eps:g} — means and 95% noise-only intervals")
        fig, axes = plt.subplots(n_eps_rows, n_eps_cols, figsize=(7.5 * n_eps_cols, 4.8 * n_eps_rows), squeeze=False)
        plotted = []
        for i, (n, d) in enumerate(cfg.comparison_settings):
            ax = axes[i // n_eps_cols, i % n_eps_cols]
            selected = [index[(s, n, d, eps)] for eps in cfg.epsilons_for(n, d)]
            xs = [r["epsilon"] for r in selected]
            for name in (*EPSILON_ARMS, "exponential"):
                if name not in selected[0]["arms"]:
                    continue
                rows = [(r, r["arms"][name]) for r in selected]
                _line(ax, xs, rows, name, interval=name in EPSILON_INTERVAL_ARMS)
            ax.set_title(f"n={n}, d={d}")
            ax.set_ylabel(NORMALIZED_YLABEL)
            ax.set_xlabel("Privacy budget ε")
            plotted.append(ax)
        for ax in axes.ravel()[n_eps_panels:]:
            ax.set_visible(False)
        _share_ylim(plotted, floor=1.0, cap=1.0)
        paths += _save(fig, figures, f"epsilon_config{s+1}",
                       f"Point configuration {s+1} — epsilon scaling; means and 95% noise-only intervals")
        for n, d in cfg.comparison_settings:
            for eps in cfg.approximation_epsilons:
                record = index[(s, n, d, eps)]
                fig, axes = plt.subplots(2, 2, figsize=(15, 9), squeeze=False)
                for col, (family, maximum, xlabel) in enumerate((
                    ("pca", record["public_rank"], "Public PCA rank r (0 through affine rank)"),
                    ("representatives", n, "Public representative count s (1 through |Q|)"))):
                    xs = endpoint_grid(maximum, zero=family == "pca", step=cfg.approximation_step)
                    for base, color, label in (("anchor", COLORS["anchor"], "Approximation + anchor correction"),
                                                ("folded", COLORS["folded"], "Approximation + original decoder")):
                        rows = [(record, record["arms"][f"{base}_{family}_{value}"]) for value in xs]
                        _line(axes[0, col], xs, rows, base, color=color, label=label)
                    for name in ("anchor", "folded"):
                        rows = [(record, record["arms"][name])] * len(xs)
                        _line(axes[0, col], xs, rows, name, color=COLORS[name],
                              label=LABELS[name]+" (full references)", interval=False)
                        axes[0, col].lines[-1].set_linestyle(":")
                    proxy_rows = [record["arms"][f"anchor_{family}_{value}"] for value in xs]
                    axes[1, col].plot(xs, [r["D_proxy"]/n for r in proxy_rows], color="black", label="Public proxy error D / n")
                    axes[1, col].set_ylabel("Mean public proxy distance D / n")
                    axes[1, col].grid(alpha=.2)
                    dim_ax = axes[1, col].twinx()
                    dim_ax.plot(xs, [r["report_dimension"] for r in proxy_rows], color="gray", linestyle="--",
                                label="Anchor release dimension")
                    dim_ax.set_ylabel("Numbers released per private point")
                    axes[0, col].set_ylabel(NORMALIZED_YLABEL)
                    axes[0, col].set_ylim(0, 1)
                    for ax in axes[:, col]:
                        ax.set_xlabel(xlabel)
                        ax.set_xlim(xs[0], xs[-1] if xs[-1] != xs[0] else xs[0]+1)
                paths += _save(fig, figures, f"approximation_config{s+1}_n{n}_d{d}_eps{epsilon_key(eps)}",
                               f"Point configuration {s+1}, n={n}, d={d}, ε={eps:g} — public approximation scaling")
    atomic_write_json(directory / "figures.json", {"complete": True, "fingerprint": fp, "paths": paths,
                                                   "png_count": sum(p.endswith('.png') for p in paths)})
    log(f"COMPLETE: {sum(p.endswith('.png') for p in paths)} figures; summary={directory/'summary.csv'}")
    return paths
