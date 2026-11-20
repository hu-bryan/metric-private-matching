"""Nine configuration-and-epsilon figures from aggregated results."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from private_matching.config import (
    ExperimentBundle,
    epsilon_key,
    point_config_dirname,
    point_config_label,
)
from private_matching.results import atomic_write_json

MECHANISM_STYLE = {
    "ambient": {"color": "#0072B2", "label": r"A (ambient Euclidean Laplace)"},
    "phi_infinity": {"color": "#D55E00", "label": r"$\Phi_\infty$ (task-distance sup-norm Laplace)"},
    "folded": {"color": "#009E73", "label": r"$\Psi$ (folded Euclidean Laplace, original server)"},
}

MECHANISM_ORDER = ("ambient", "phi_infinity", "folded")


def _finite(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None:
            continue
        number = float(value)
        if np.isfinite(number):
            out.append(number)
    return out


def _select_rows(
    rows: Sequence[dict[str, Any]],
    *,
    s: int,
    epsilon: float,
    n: int | None = None,
    d: int | None = None,
) -> list[dict[str, Any]]:
    ek = epsilon_key(epsilon)
    selected = []
    for row in rows:
        if int(row["s"]) != int(s):
            continue
        if row["epsilon_key"] != ek and float(row["epsilon"]) != float(epsilon):
            continue
        if n is not None and int(row["n"]) != int(n):
            continue
        if d is not None and int(row["d"]) != int(d):
            continue
        selected.append(row)
    return selected


def _series(rows: Sequence[dict[str, Any]], mechanism: str, x_key: str) -> dict[str, np.ndarray]:
    mech_rows = sorted(
        (row for row in rows if row["mechanism"] == mechanism),
        key=lambda row: int(row[x_key]),
    )
    return {
        "x": np.asarray([int(row[x_key]) for row in mech_rows], dtype=float),
        "y": np.asarray([row["normalized_mean"] if row["normalized_mean"] is not None else np.nan for row in mech_rows], dtype=float),
        "ci_low": np.asarray([row["ci_low"] if row["ci_low"] is not None else np.nan for row in mech_rows], dtype=float),
        "ci_high": np.asarray([row["ci_high"] if row["ci_high"] is not None else np.nan for row in mech_rows], dtype=float),
        "bound": np.asarray([row["bound_normalized"] if row["bound_normalized"] is not None else np.nan for row in mech_rows], dtype=float),
        "gap_zero": [bool(row["gap_zero"]) for row in mech_rows],
        "complete": [bool(row["complete_K"]) for row in mech_rows],
        "K": [int(row["K"]) for row in mech_rows],
        "m_fold": [int(row["m_fold"]) for row in mech_rows],
    }


def _detail_ylim(rows: Sequence[dict[str, Any]]) -> tuple[float, float]:
    empirical = _finite(
        [row["normalized_mean"] for row in rows]
        + [row["ci_low"] for row in rows]
        + [row["ci_high"] for row in rows]
    )
    lo = min([0.0, *empirical]) if empirical else 0.0
    hi = max([1.0, *empirical]) if empirical else 1.0
    span = hi - lo
    pad = 0.08 * span if span > 0 else 0.1
    return lo - pad, hi + pad


def _plot_panel(
    ax: plt.Axes,
    rows: Sequence[dict[str, Any]],
    *,
    x_key: str,
    xlabel: str,
    zoom: bool,
    title: str,
    show_legend: bool,
) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("normalized excess (lower is better)")
    ax.set_xscale("linear")
    ax.set_yscale("linear")
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.axhline(1.0, color="0.45", linestyle=":", linewidth=0.9, label="random baseline = 1")
    handles = []
    labels = []
    for mechanism in MECHANISM_ORDER:
        style = MECHANISM_STYLE[mechanism]
        series = _series(rows, mechanism, x_key)
        if series["x"].size == 0:
            continue
        line_emp = ax.plot(
            series["x"],
            series["y"],
            color=style["color"],
            marker="o",
            markersize=3.5,
            linewidth=1.6,
            label=style["label"],
        )[0]
        finite = np.isfinite(series["ci_low"]) & np.isfinite(series["ci_high"])
        if np.any(finite):
            ax.fill_between(
                series["x"],
                series["ci_low"],
                series["ci_high"],
                color=style["color"],
                alpha=0.18,
                linewidth=0,
                where=finite,
            )
        line_bound = ax.plot(
            series["x"],
            series["bound"],
            color=style["color"],
            linestyle="--",
            linewidth=1.3,
            label=style["label"] + " bound",
        )[0]
        handles.extend([line_emp, line_bound])
        labels.extend([style["label"], style["label"] + " bound $H/g$"])
        if any(series["gap_zero"]):
            ax.scatter(
                series["x"][np.asarray(series["gap_zero"])],
                np.zeros(sum(series["gap_zero"])),
                marker="x",
                color=style["color"],
                zorder=5,
            )
    ax.grid(True, alpha=0.25)
    if zoom:
        ax.set_ylim(*_detail_ylim(rows))
        ax.text(
            0.02,
            0.98,
            "Zoom: empirical range, including 0 and 1.\nOut-of-range bounds are in the full-range panel.",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
        )
    if show_legend:
        ax.legend(handles, labels, loc="best", fontsize=7, framealpha=0.92)


def _partial_banner(rows: Sequence[dict[str, Any]]) -> str | None:
    if not rows:
        return "NO DATA"
    if not all(bool(row["complete_K"]) for row in rows):
        k_min = min(int(row["K"]) for row in rows)
        k_max = max(int(row["K"]) for row in rows)
        return f"PARTIAL / INCOMPLETE  (K range {k_min}–{k_max}; not a final plot)"
    return None


def render_figure(
    bundle: ExperimentBundle,
    aggregate: dict[str, Any],
    *,
    s: int,
    epsilon: float,
    out_dir: Path,
    smoke: bool = False,
) -> list[Path]:
    cfg = bundle.scientific
    rows = aggregate["rows"]
    n_rows = _select_rows(rows, s=s, epsilon=epsilon, d=cfg.varying_n_fixed_d)
    d_rows = _select_rows(rows, s=s, epsilon=epsilon, n=cfg.varying_d_fixed_n)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.2), sharex="col", constrained_layout=True)
    banner = _partial_banner(n_rows + d_rows)
    k_values = [int(row["K"]) for row in n_rows + d_rows] or [cfg.noise_repetitions_K]
    k_note = str(k_values[0]) if len(set(k_values)) == 1 else f"{min(k_values)}–{max(k_values)}"
    title_bits = [
        point_config_label(s),
        rf"$\varepsilon={epsilon_key(epsilon)}$",
        "L=1",
        f"K={k_note}",
        "lower is better",
    ]
    if smoke:
        title_bits.insert(0, "SMOKE TEST")
    if banner:
        title_bits.insert(0, banner)
    fig.suptitle(
        "  |  ".join(title_bits)
        + "\nCurves average only over mechanism noise on this master configuration, not over S=3."
        + "\nSolid: conditional noise-mean estimate with pointwise 95% t interval. Dashed: H_M / g_s on this instance."
        + "\nServer rules: ambient and folded use euclidean_report_distances (original Psi, not debiased).",
        fontsize=10,
    )
    _plot_panel(
        axes[0, 0],
        n_rows,
        x_key="n",
        xlabel=f"n  (d = {cfg.varying_n_fixed_d} fixed)",
        zoom=False,
        title="n-sweep, full-range bounds",
        show_legend=True,
    )
    _plot_panel(
        axes[0, 1],
        d_rows,
        x_key="d",
        xlabel=f"d  (n = {cfg.varying_d_fixed_n} fixed)",
        zoom=False,
        title="d-sweep, full-range bounds",
        show_legend=False,
    )
    _plot_panel(
        axes[1, 0],
        n_rows,
        x_key="n",
        xlabel=f"n  (d = {cfg.varying_n_fixed_d} fixed)",
        zoom=True,
        title="n-sweep, empirical detail (zoom)",
        show_legend=False,
    )
    _plot_panel(
        axes[1, 1],
        d_rows,
        x_key="d",
        xlabel=f"d  (n = {cfg.varying_d_fixed_n} fixed)",
        zoom=True,
        title="d-sweep, empirical detail (zoom)",
        show_legend=False,
    )
    cfg_dir = out_dir / point_config_dirname(s)
    cfg_dir.mkdir(parents=True, exist_ok=True)
    stem = f"epsilon_{epsilon_key(epsilon)}"
    png_path = cfg_dir / f"{stem}.png"
    pdf_path = cfg_dir / f"{stem}.pdf"
    data_path = cfg_dir / f"{stem}.json"
    fig.savefig(png_path, dpi=200)
    fig.savefig(pdf_path)
    plt.close(fig)
    atomic_write_json(
        data_path,
        {
            "s": int(s),
            "label": point_config_label(s),
            "epsilon": float(epsilon),
            "smoke": bool(smoke),
            "partial": banner,
            "n_sweep_fixed_d": int(cfg.varying_n_fixed_d),
            "d_sweep_fixed_n": int(cfg.varying_d_fixed_n),
            "intersection": list(cfg.intersection_setting()),
            "rows_n_sweep": n_rows,
            "rows_d_sweep": d_rows,
        },
    )
    return [png_path, pdf_path, data_path]


def plot_all(
    bundle: ExperimentBundle,
    aggregate: dict[str, Any],
    *,
    out_dir: Path,
    s_filter: int | None = None,
    epsilon_filter: float | None = None,
    smoke: bool = False,
) -> list[Path]:
    cfg = bundle.scientific
    s_values = [s_filter] if s_filter is not None else list(range(cfg.configurations_S))
    epsilons = [epsilon_filter] if epsilon_filter is not None else list(cfg.plot_epsilons)
    paths: list[Path] = []
    for s in s_values:
        for epsilon in epsilons:
            paths.extend(
                render_figure(
                    bundle,
                    aggregate,
                    s=int(s),
                    epsilon=float(epsilon),
                    out_dir=out_dir,
                    smoke=smoke,
                )
            )
    return paths


def expected_main_figure_count(bundle: ExperimentBundle) -> int:
    return int(bundle.scientific.configurations_S) * len(bundle.scientific.plot_epsilons)
