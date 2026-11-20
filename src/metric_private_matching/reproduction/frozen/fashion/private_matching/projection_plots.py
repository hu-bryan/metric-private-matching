"""Figures and geometry-level summary for the projection_epsilon study.

Reads only the exported analysis table. Each geometry contributes its own
K-draw noise mean; intervals here are across the independent geometries
(Student-t 95%), not conditional noise intervals. The noiseless proxy is a
NONPRIVATE DIAGNOSTIC. The pilot-only epsilon=1e9 rows are a numerical limit
check and are summarized but never plotted.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from private_matching.login_guard import refuse_login_compute
from private_matching.results import atomic_write_bytes

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MECH = {"folded": ("Psi (reduced-dimension Laplace)", "#2a78d6"),
        "anchor": ("Anchor (location + private radius)", "#eb6834"),
        "phi_infinity": ("Phi-infinity (task-distance noise)", "#1baf7a"),
        "ambient": ("Ambient (full-d location noise)", "#eda100")}
PROXY = "#8a8984"
EPS_RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # ordinal blue steps 250/400/550/700
PLOT_EPS = [4., 16., 64., 256.]
GEOMETRIES = [("unstructured", None, "Unstructured"), ("low_rank", None, "Exact rank 4"),
              ("near_low_rank", .02, "Near rank 4, residual .02"), ("near_low_rank", .2, "Near rank 4, residual .2"),
              ("clustered", .02, "Four clusters, spread .02"), ("clustered", .2, "Four clusters, spread .2")]
TRUE_STRUCTURE = 4


def _style():
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                         "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
                         "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False})


def geometry_key(row) -> tuple:
    thickness = row["residual_thickness"] if row["geometry_family"] == "near_low_rank" else \
        row["cluster_spread"] if row["geometry_family"] == "clustered" else None
    return row["geometry_family"], (None if thickness is None or pd.isna(thickness) else round(float(thickness), 4))


def load(table: Path) -> pd.DataFrame:
    df = pd.read_csv(table, low_memory=False)
    df = df[(df["study_id"] == "projection_epsilon") & (df["status"] != "certificate_only")].copy()
    df["geometry"] = [GEOMETRIES_LABEL.get(geometry_key(r), "?") for _, r in df.iterrows()]
    df["mechanism"] = df["requested_base_mechanism"].fillna(df["base_mechanism"])
    label = df["arm_label"].astype(str)
    df["operating_point"] = np.where(label.str.endswith(":selected"), "selected",
                                     np.where(df["approximation_family"].isin(["pca", "representatives"]), "candidate", "baseline"))
    value = df["approximation_rank"].fillna(df["representative_count"])
    chosen = df["selected_branch"].astype(str).str.rsplit(":", n=1).str[-1]
    df["candidate_value"] = np.where(df["operating_point"] == "selected", pd.to_numeric(chosen, errors="coerce"), value)
    df["noise_to_gap"] = df["mean_mean_noise_norm"].fillna(0)/(df["random_minus_OPT"]/df["n"])
    return df


GEOMETRIES_LABEL = {(f, th): name for f, th, name in GEOMETRIES}


def _ci(values: np.ndarray) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    mean = float(values.mean()) if len(values) else np.nan
    if len(values) < 2:
        return mean, np.nan, np.nan
    half = float(t.ppf(.975, len(values)-1)*values.std(ddof=1)/np.sqrt(len(values)))
    return mean, mean-half, mean+half


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["d", "geometry", "mechanism", "approximation_family", "operating_point", "group_value", "epsilon"]
    df = df.copy()
    # A public selector is ONE operating point per geometry, whatever value it chose there.
    df["group_value"] = np.where(df["operating_point"] == "selected", np.nan, df["candidate_value"])
    out = []
    for key, g in df.groupby(keys, dropna=False, sort=True):
        mean, lo, hi = _ci(g["normalized_mean_regret"].to_numpy(float))
        record = dict(zip(keys, key))
        record["candidate_value"] = record.pop("group_value")
        record.update(privacy=("NONPRIVATE_DIAGNOSTIC" if key[2] == "proxy" else "local"),
                      n_geometries=int(g["instance_id"].nunique()), K_per_geometry=int(g["K"].max()),
                      coordinate_noise_dimension_median=float(g["coordinate_noise_dimension"].median()),
                      mean_normalized_regret=mean, between_geometry_ci95_low=lo, between_geometry_ci95_high=hi,
                      mean_noise_norm=float(g["mean_mean_noise_norm"].mean()) if g["mean_mean_noise_norm"].notna().any() else None,
                      matching_gap_per_driver=float((g["random_minus_OPT"]/g["n"]).mean()),
                      noise_to_gap_ratio=float(g["noise_to_gap"].mean()),
                      D_proxy_per_point=float(g["D_proxy_per_point"].mean()) if g["D_proxy_per_point"].notna().any() else None,
                      # Regret measured on the proxy's own costs; 0 means the matching is proxy-optimal.
                      mean_normalized_proxy_regret=float((g["mean_proxy_regret"]/g["random_minus_OPT"]).mean())
                      if g["mean_proxy_regret"].notna().any() else None,
                      mean_agreement_with_proxy_optimum=float(g["mean_agreement_with_proxy_optimum"].mean())
                      if "mean_agreement_with_proxy_optimum" in g and g["mean_agreement_with_proxy_optimum"].notna().any() else None,
                      mean_near_minus_far_task_radius=float(g["mean_near_minus_far_task_radius"].mean())
                      if "mean_near_minus_far_task_radius" in g and g["mean_near_minus_far_task_radius"].notna().any() else None,
                      OPT_near_minus_far_task_radius=float(g["OPT_near_minus_far_task_radius"].mean())
                      if "OPT_near_minus_far_task_radius" in g and g["OPT_near_minus_far_task_radius"].notna().any() else None,
                      selected_value_min=float(g["candidate_value"].min()) if key[4] == "selected" else None,
                      selected_value_median=float(g["candidate_value"].median()) if key[4] == "selected" else None,
                      selected_value_max=float(g["candidate_value"].max()) if key[4] == "selected" else None)
        out.append(record)
    return pd.DataFrame(out)


def _panels(title: str, pilot: bool):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.4), sharey=True)
    fig.suptitle(("PILOT (1 geometry, K=5): " if pilot else "")+title, x=.01, ha="left", fontsize=12, fontweight="bold")
    return fig, axes.ravel()


def _empty(ax, frame, label) -> bool:
    """Hide a panel whose geometry is absent (e.g. partial test designs)."""
    if len(frame):
        return False
    ax.set_axis_off()
    ax.text(.5, .5, f"{label}: no data", transform=ax.transAxes, ha="center", color=INK2)
    return True


def _figure_legend(fig, axes):
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper right", ncol=5, fontsize=7.5, bbox_to_anchor=(1, .955))
            return


def _note(fig, text):
    fig.text(.01, .01, text, ha="left", va="bottom", fontsize=7.5, color=INK2)


def rank_sweep(df, summary, d, mechanism, out, pilot):
    name, _ = MECH[mechanism]
    fig, axes = _panels(f"{name}, d={d}: regret across approximation candidates", pilot)
    for ax, (fam, th, label) in zip(axes, GEOMETRIES):
        s = summary[(summary.d == d) & (summary.geometry == label) & (summary.operating_point == "candidate")]
        if _empty(ax, s, label):
            continue
        # Prespecified grid only (exactly the values with a noiseless-proxy row); values that
        # a public selector added for some geometries are in the summary CSV, not here.
        values = sorted(s.loc[s.mechanism == "proxy", "candidate_value"].dropna().unique())
        s = s[s.candidate_value.isin(values)]
        pos = {v: i for i, v in enumerate(values)}
        dims = s[s.mechanism == mechanism].groupby("candidate_value")["coordinate_noise_dimension_median"].first()
        for colour, eps in zip(EPS_RAMP, PLOT_EPS):
            line = s[(s.mechanism == mechanism) & (s.epsilon == eps)].sort_values("candidate_value")
            x = [pos[v] for v in line.candidate_value]
            ax.plot(x, line.mean_normalized_regret, "-o", color=colour, lw=1.6, ms=4, label=f"epsilon = {eps:g}")
            if line.between_geometry_ci95_low.notna().any():
                ax.fill_between(x, line.between_geometry_ci95_low, line.between_geometry_ci95_high, color=colour, alpha=.12, lw=0)
        proxy = s[(s.mechanism == "proxy") & (s.epsilon == PLOT_EPS[0])].sort_values("candidate_value")
        ax.plot([pos[v] for v in proxy.candidate_value], proxy.mean_normalized_regret, "--", color=PROXY, lw=1.4,
                marker="o", ms=3, label="Noiseless proxy (NONPRIVATE)")
        ax.axhline(1, color=INK2, lw=.8, ls=":")
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels([f"{int(v)}\n[{int(dims.get(v, 0))}]" for v in values], fontsize=7)
        ax.set_title(label, loc="left")
        ax.set_xlabel(("PCA rank" if fam != "clustered" else "Representative count")+" [noise dimension]")
    axes[0].set_ylabel("Normalized regret\n(0 = OPT, 1 = random matching)")
    axes[3].set_ylabel("Normalized regret\n(0 = OPT, 1 = random matching)")
    _figure_legend(fig, axes)
    _note(fig, "Mean over independent geometries of each geometry's noise-averaged regret; bands are between-geometry 95% t intervals. "
               "Dotted line = random matching. Candidates equally spaced. Noise is coupled across epsilon.")
    fig.tight_layout(rect=(0, .03, 1, .92))
    path = out/f"fig_rank_sweep_{'psi' if mechanism == 'folded' else mechanism}_d{d}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def epsilon_sweep(df, summary, d, out, pilot):
    fig, axes = _panels(f"d={d}: normalized regret versus epsilon at fixed operating points", pilot)
    for ax, (fam, th, label) in zip(axes, GEOMETRIES):
        s = summary[(summary.d == d) & (summary.geometry == label) & summary.epsilon.isin(PLOT_EPS)]
        if _empty(ax, s, label):
            continue
        cand = s[s.operating_point == "candidate"]
        full = cand.candidate_value.max()
        truth = TRUE_STRUCTURE if TRUE_STRUCTURE in set(cand.candidate_value) else None
        for mech in ["folded", "anchor"]:
            name, colour = MECH[mech]
            short = name.split(" (")[0]
            series = [(cand[(cand.mechanism == mech) & (cand.candidate_value == truth)] if truth is not None else None, "-", f"{short}, rank/count 4"),
                      (s[(s.mechanism == mech) & (s.operating_point == "selected")], "--", f"{short}, public selector"),
                      (cand[(cand.mechanism == mech) & (cand.candidate_value == full)], ":", f"{short}, no approximation (exact fold)")]
            for line, ls, lab in series:
                if line is None or line.empty:
                    continue
                line = line.sort_values("epsilon")
                ax.plot(line.epsilon, line.mean_normalized_regret, ls, marker="o", ms=4, lw=1.6, color=colour, label=lab)
        ambient = s[s.mechanism == "ambient"].sort_values("epsilon")
        ax.plot(ambient.epsilon, ambient.mean_normalized_regret, "-", marker="^", ms=4, lw=1.3, color=MECH["ambient"][1],
                label=f"Ambient (d={d} noise)")
        phi = s[s.mechanism == "phi_infinity"].sort_values("epsilon")
        ax.plot(phi.epsilon, phi.mean_normalized_regret, "-", marker="s", ms=4, lw=1.3, color=MECH["phi_infinity"][1],
                label="Phi-infinity")
        if truth is not None:
            proxy = cand[(cand.mechanism == "proxy") & (cand.candidate_value == truth)]
            if not proxy.empty:
                ax.axhline(proxy.mean_normalized_regret.mean(), color=PROXY, ls="--", lw=1.2,
                           label="Noiseless proxy, rank/count 4 (NONPRIVATE)")
        ax.axhline(1, color=INK2, lw=.8, ls=":")
        ax.set_xscale("log", base=2)
        ax.set_xlim(PLOT_EPS[0]/1.4, PLOT_EPS[-1]*1.4)
        ax.set_xticks(PLOT_EPS)
        ax.set_xticklabels([f"{e:g}" for e in PLOT_EPS])
        ax.set_title(label, loc="left")
        ax.set_xlabel("epsilon")
    axes[0].set_ylabel("Normalized regret")
    axes[3].set_ylabel("Normalized regret")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=4, fontsize=7.5, bbox_to_anchor=(1, .955))
    _note(fig, "Color = mechanism; line style = operating point (solid: true structure 4, dashed: public selector, dotted: exact fold, no approximation). "
               "Means over geometries; see summary CSV for intervals.")
    fig.tight_layout(rect=(0, .03, 1, .9))
    path = out/f"fig_epsilon_sweep_d{d}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def noise_ratio(df, d, out, pilot):
    fig, axes = _panels(f"Psi, d={d}: damage from noise versus noise size relative to the per-driver matching gain", pilot)
    key = ["instance_id", "epsilon", "approximation_family", "candidate_value"]
    proxy = df[(df.mechanism == "proxy") & (df.operating_point == "candidate")][key+["normalized_mean_regret"]]
    for ax, (fam, th, label) in zip(axes, GEOMETRIES):
        rows = df[(df.d == d) & (df.geometry == label) & (df.mechanism == "folded") & (df.operating_point == "candidate")
                  & (df.coordinate_noise_dimension > 0) & df.epsilon.isin(PLOT_EPS)]
        # Noise damage = regret minus the same candidate's noiseless-proxy regret (same geometry).
        rows = rows.merge(proxy, on=key, suffixes=("", "_proxy"))
        if _empty(ax, rows, label):
            continue
        for colour, eps in zip(EPS_RAMP, PLOT_EPS):
            r = rows[rows.epsilon == eps]
            ax.scatter(r.noise_to_gap, r.normalized_mean_regret-r.normalized_mean_regret_proxy, s=10, color=colour,
                       alpha=.6, lw=0, label=f"epsilon = {eps:g}")
        ax.axhline(0, color=INK2, lw=.8, ls=":")
        ax.axvline(1, color=INK2, lw=.8, ls=":")
        ax.set_xscale("log")
        ax.set_title(label, loc="left")
        ax.set_xlabel("Mean noise norm / (random - OPT per driver)")
    axes[0].set_ylabel("Regret minus noiseless-proxy regret\n(normalized)")
    axes[3].set_ylabel("Regret minus noiseless-proxy regret\n(normalized)")
    _figure_legend(fig, axes)
    _note(fig, "One point per geometry and prespecified PCA rank/representative count (no-release excluded). "
               "Right of the vertical line, a driver's noise exceeds the whole per-driver benefit of optimal matching.")
    fig.tight_layout(rect=(0, .03, 1, .92))
    path = out/f"fig_noise_to_gap_d{d}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def make_report(table, out, pilot=False):
    refuse_login_compute("plot projection study")
    _style()
    table, out = Path(table), Path(out)
    df = load(table)
    summary = summarize(df)
    atomic_write_bytes(out/"projection_epsilon_summary.csv", summary.to_csv(index=False).encode())
    paths = []
    for d in sorted(df.d.unique()):
        for mech in ["folded", "anchor"]:
            paths.append(rank_sweep(df, summary, d, mech, out, pilot))
        paths.append(epsilon_sweep(df, summary, d, out, pilot))
        paths.append(noise_ratio(df, d, out, pilot))
    limit = summary[summary.epsilon > 1e6]
    if not limit.empty:
        # epsilon -> infinity check: every noisy candidate must reproduce the noiseless proxy.
        cand = limit[limit.operating_point == "candidate"]
        proxy = cand[cand.mechanism == "proxy"].set_index(["d", "geometry", "candidate_value"]).mean_normalized_regret
        noisy = cand[cand.mechanism.isin(["folded", "anchor"]) & (cand.coordinate_noise_dimension_median > 0)]
        pca = noisy[noisy.approximation_family == "pca"]
        # Representatives share proxy rows (ties), so only proxy-optimality is expected there.
        gaps = [abs(r.mean_normalized_regret-proxy.get((r.d, r.geometry, r.candidate_value), np.nan)) for r in pca.itertuples()]
        report = dict(epsilon=float(limit.epsilon.max()),
                      pca_max_abs_normalized_gap_to_proxy=float(np.nanmax(gaps)) if gaps else None,
                      pca_min_agreement_with_proxy_matching=float(pca.mean_agreement_with_proxy_optimum.min()) if len(pca) else None,
                      all_max_normalized_proxy_regret=float(noisy.mean_normalized_proxy_regret.max()) if len(noisy) else None)
        atomic_write_bytes(out/"epsilon_limit_check.txt", (str(report)+"\n").encode())
        print(f"epsilon-limit check: {report}", flush=True)
    for p in paths:
        print(f"Figure: {p}", flush=True)
    print(f"Summary: {out/'projection_epsilon_summary.csv'}", flush=True)
    return paths
