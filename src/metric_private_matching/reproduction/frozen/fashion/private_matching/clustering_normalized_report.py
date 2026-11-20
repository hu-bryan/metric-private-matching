"""Streaming audited handoff and figures for the epsilon-8 normalization study."""
from __future__ import annotations

import csv
import gzip
import json
import os
from pathlib import Path
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist

from . import clustering_normalized as study
from .clustering_benchmark import sha256
from .clustering_benchmark_report import summarize, interval
from .integral_study import atomic_json
from .login_guard import refuse_login_compute

ORDER = ["graves/ring_noisy", "wut/circles", "fcps/chainlink", "other/iris", "uci/ecoli", "uci/yeast",
         "uci/glass", "uci/wine", "uci/statlog", "uci/wdbc", "uci/ionosphere", "uci/sonar",
         "sipu/worms_64", "mnist/digits10k", "mnist/fashion10k"]
PLOT_ARMS = {1: ["ambient_radial", "ambient_iid", "terminal_fold", "anchor", "corrected_projection", "cost_infinity", "cost_iid", "gumbel"],
            2: ["ambient_radial", "ambient_iid", "span_radial", "coordinate_allocated", "cost_infinity", "cost_iid", "gumbel"]}
LABELS = {"ambient_radial": "Point\nradial", "ambient_iid": "Point\niid", "terminal_fold": "Terminal\nfold",
          "anchor": "Span\nanchor", "corrected_projection": "Corrected\nprojection", "cost_infinity": "Cost\ninfinity",
          "cost_iid": "Cost\niid", "gumbel": "Gumbel", "span_radial": "Public-span\nradial", "coordinate_allocated": "Coordinate\nallocation"}


def independent_normalization_check(parent, dataset):
    """Reconstruct parameters independently of the production normalizer."""
    raw = parent["raw"]
    X = raw.copy()
    if dataset in study.SCALED:
        for j in range(X.shape[1]):
            if j in study.BINARY.get(dataset, {}):
                lo, hi = study.BINARY[dataset][j]
                X[:, j] = (raw[:, j]-lo)/(hi-lo)
            else:
                mean = float(np.mean(raw[:, j]))
                variance = float(np.mean((raw[:, j]-mean)**2))
                eps = np.finfo(float).eps
                constant = variance <= len(raw)*eps*variance+(len(raw)*mean*eps)**2
                denominator = 1. if constant else np.sqrt(variance)
                X[:, j] = (raw[:, j]-mean)/denominator
                assert bool(parent["feature_constant"][j]) == constant
    center = X.mean(axis=0)
    radius = np.sqrt(np.mean(np.sum((X-center)**2, axis=1)))
    np.testing.assert_allclose((X-center)/radius, parent["normalized"], rtol=1e-12, atol=1e-12)
    assert np.isclose(radius, float(parent["pooled_radius"]), rtol=1e-12)


def baseline_data(manifest):
    parents, summaries = [], []
    for source in manifest["archives"]:
        out = Path(source["directory"])/"analysis_handoff"
        assert sha256(Path(source["directory"])/"manifest.json") == source["manifest_sha256"]
        for filename, target in [("parent_summaries.csv.gz", parents), ("summary.csv", summaries)]:
            frame = pd.read_csv(out/filename)
            frame = frame[(frame.panel == source["panel"]) & (frame.epsilon == 8.)].copy()
            frame["source_fingerprint"] = source["fingerprint"]
            target.append(frame)
    assert sha256(study.CORRECTED/"manifest.json") == manifest["corrected_manifest_sha256"]
    for filename, target in [("parent_summaries.csv.gz", parents), ("summary.csv", summaries)]:
        frame = pd.read_csv(study.CORRECTED/"analysis_handoff"/filename)
        frame = frame[frame.epsilon == 8.].copy()
        frame["source_fingerprint"] = json.loads((study.CORRECTED/"manifest.json").read_text())["fingerprint"]
        target.append(frame)
    return pd.concat(parents, ignore_index=True), pd.concat(summaries, ignore_index=True)


def export(directory, output):
    refuse_login_compute("export normalized benchmark results")
    directory, output = Path(directory), Path(output)
    manifest = json.loads((directory/"manifest.json").read_text())
    study.verify_runtime(manifest)
    output.mkdir(parents=True, exist_ok=True)
    (output/"figures").mkdir(exist_ok=True)
    cfg = manifest["design"]; K = cfg["K"]
    registry = json.loads((directory/"input_registry.json").read_text())
    checked_inputs = set(); instances = []; geometry_rows = []; units_meta = []; raw_count = 0
    destination = output/"analysis_table.csv.gz"
    writer = None
    with gzip.open(str(destination)+".tmp", "wt", newline="", compresslevel=3) as stream:
        for u in manifest["units"]:
            assert study.valid_checkpoint(directory, manifest, u), u["instance_id"]
            key = study.parent_key(u)
            if key not in checked_inputs:
                input_path = directory/registry[key]["file"]
                assert sha256(input_path) == registry[key]["sha256"]
                with np.load(input_path) as parent:
                    independent_normalization_check(parent, u["dataset"])
                checked_inputs.add(key)
            stem = directory/"units"/u["instance_id"]
            meta = json.loads(Path(str(stem)+".json").read_text()); units_meta.append(meta)
            with np.load(str(stem)+".geometry.npz") as geometry, np.load(str(stem)+".npz") as saved:
                Q, R = geometry["Q"], geometry["R"]
                assert np.array_equal(np.sort(np.r_[geometry["q_source_ids"], geometry["r_source_ids"]]),
                                      np.unique(np.r_[geometry["q_source_ids"], geometry["r_source_ids"]]))
                pooled = np.r_[Q, R]
                C1 = cdist(R, Q)
                assert np.allclose(pooled.mean(axis=0), 0., atol=2e-12)
                assert np.isclose(np.mean(np.sum(pooled**2, axis=1)), 1., atol=2e-12)
                geometry_rows.append(dict(instance_id=u["instance_id"], dataset=u["dataset"], parent=u["parent"],
                    partition=u["partition"], d=u["d"], changed=u["changed"],
                    q_rank=meta["diagnostics"]["Q"]["covariance_rank"],
                    q_effective_rank=meta["diagnostics"]["Q"]["effective_rank"],
                    q_rho=meta["diagnostics"]["Q"]["effective_rank_over_rank"],
                    public_diameter=meta["public"]["public_diameter"], pooled_diameter=float(pdist(pooled).max()),
                    W_Q=meta["public"]["coordinate_width_certificate"], fold_dimension=meta["public"]["fold_dimension"]))
                for ia, (p, arm) in enumerate(study.ARM_KEYS):
                    C = C1 if p == 1 else C1*C1
                    optimum_pi = saved["optimum_permutations"][p-1]
                    # Independent exact optimum and random expectation reconstruction.
                    from scipy.optimize import linear_sum_assignment
                    rows, cols = linear_sum_assignment(C)
                    optimum = float(C[rows, cols].sum())
                    random = float(C.sum()/64); gap = random-optimum
                    assert np.isclose(optimum, meta["optima"][str(p)], rtol=1e-12, atol=1e-10)
                    assert np.isclose(random, meta["random_expected"][str(p)], rtol=1e-12, atol=1e-10)
                    pi = saved["permutations"][ia]
                    assert np.all(np.sort(pi, axis=1) == np.arange(64))
                    costs = C[np.arange(64), pi].sum(axis=-1)
                    regrets = costs-optimum
                    np.testing.assert_allclose(costs, saved["total_cost"][ia], rtol=1e-12, atol=1e-10)
                    np.testing.assert_allclose(regrets, saved["total_regret"][ia], rtol=1e-12, atol=1e-10)
                    assert np.all(regrets <= saved["realization_bound"][ia]+1e-7)
                    score = regrets/gap
                    cert, reuse = meta["bounds"][f"{p}:{arm}"], meta["reuse"][f"{p}:{arm}"]
                    common = dict(study=study.NAME, fingerprint=manifest["fingerprint"], phase=cfg["phase"],
                        instance_id=u["instance_id"], dataset=u["dataset"], n=64, d=u["d"], parent=u["parent"],
                        partition=u["partition"], objective_p=p, mechanism=arm, epsilon=8.,
                        preprocessing="feature_zscore_binary_then_RMS" if u["changed"] else "historical_RMS_unchanged",
                        normalization_changed=u["changed"], privacy_metric="sum_record_euclidean_normalized_inputs",
                        privacy_output="curator_final_matching" if arm == "gumbel" else "local_report_and_matching",
                        optimum_total_cost=optimum, random_expected_total_cost=random, random_minus_optimum=gap,
                        bound_total=cert["bound_total"], bound_uncapped=cert["bound_uncapped"], bound_cap=cert["bound_cap"],
                        epsilon_y=cert["epsilon_y"], epsilon_s=cert["epsilon_s"],
                        q_rank=meta["diagnostics"]["Q"]["covariance_rank"],
                        q_effective_rank=meta["diagnostics"]["Q"]["effective_rank"],
                        q_rho=meta["diagnostics"]["Q"]["effective_rank_over_rank"],
                        public_diameter=meta["public"]["public_diameter"], W_Q=meta["public"]["coordinate_width_certificate"],
                        fold_dimension=meta["public"]["fold_dimension"], corrected_dimension=meta["corrected_dimension"],
                        imported_execution=meta["imported"], source_fingerprint=reuse.get("fingerprint", manifest["fingerprint"]),
                        source_arm=reuse["source_arm"], reuse_reason=reuse["reason"],
                        seed_namespace=NAMESPACE_FOR_ROW(manifest, meta),
                        noise_stream="historical source" if meta["imported"] else "SHA256(namespace,phase,instance_id,draw,channel); independent channels",
                        historical_input_fingerprint=u["source_fingerprint"], historical_instance_id=u["origin_id"],
                        geometry_file="../units/"+u["instance_id"]+".geometry.npz",
                        permutation_file="../units/"+u["instance_id"]+".npz", permutation_array="permutations", arm_index=ia,
                        parent_input_file="../"+registry[key]["file"],
                        hardware_host=meta["hardware"]["host"], execution_job_id=meta["hardware"]["job_id"])
                    instances.append(dict(common, K=K, score=float(score.mean()),
                        mean_total_regret=float(regrets.mean()), mean_total_cost=float(costs.mean())))
                    for draw in range(K):
                        record = dict(common, draw=draw, score=float(score[draw]), total_cost=float(costs[draw]),
                            total_regret=float(regrets[draw]), row_oscillation=float(saved["row_oscillation"][ia, draw]),
                            realization_bound=float(saved["realization_bound"][ia, draw]))
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(record)); writer.writeheader()
                        writer.writerow(record); raw_count += 1
    os.replace(str(destination)+".tmp", destination)
    df = pd.DataFrame(instances)
    df.to_csv(output/"instance_summaries.csv.gz", index=False)
    keys = ["dataset", "n", "d", "objective_p", "mechanism", "epsilon", "normalization_changed"]
    metrics = ["score", "mean_total_regret", "mean_total_cost", "bound_total", "q_rank", "q_effective_rank", "q_rho", "public_diameter", "W_Q"]
    parent = df.groupby(keys+["parent"])[metrics].mean().reset_index()
    counts = df.groupby(keys+["parent"]).size()
    assert (counts == cfg["partitions"]).all()
    parent["partition_count"] = counts.values
    parent.to_csv(output/"parent_summaries.csv.gz", index=False)
    summary = None
    for metric in metrics:
        part = summarize(parent, keys, metric)
        summary = part if summary is None else summary.merge(part, on=keys, validate="one_to_one")
    summary.to_csv(output/"summary.csv", index=False)
    geometry = pd.DataFrame(geometry_rows)
    geometry.to_csv(output/"geometry_table.csv.gz", index=False)
    baseline_parent, baseline_summary = baseline_data(manifest)
    baseline_parent.to_csv(output/"baseline_parent_summaries.csv.gz", index=False)
    baseline_summary.to_csv(output/"baseline_summary.csv", index=False)
    paired = parent.merge(baseline_parent[["dataset", "parent", "objective_p", "mechanism", "score"]],
                          on=["dataset", "parent", "objective_p", "mechanism"], suffixes=("_normalized", "_original"), validate="one_to_one")
    assert len(paired) == 225*cfg["parents"]
    paired["score_difference"] = paired.score_normalized-paired.score_original
    unchanged_zero = bool(np.allclose(paired.loc[~paired.normalization_changed, "score_difference"], 0., atol=1e-12))
    if cfg["phase"] == "production":
        assert unchanged_zero
    paired.to_csv(output/"paired_parent_differences.csv.gz", index=False)
    differences = summarize(paired, keys, "score_difference")
    differences.to_csv(output/"paired_comparisons.csv", index=False)
    baseline_summary["normalization_changed"] = baseline_summary.dataset.isin(study.SCALED)
    assert len(summary) == len(baseline_summary) == 225
    assert raw_count == len(manifest["units"])*15*K
    # Reconstruct all published setting means and Student-t intervals directly from instances.
    for row in summary.itertuples():
        values = df[(df.dataset == row.dataset) & (df.objective_p == row.objective_p) & (df.mechanism == row.mechanism)]
        independent = values.groupby("parent").score.mean().to_numpy()
        check = interval(independent)
        assert np.isclose(check["mean"], row.score_mean, rtol=1e-12, atol=1e-12)
        if cfg["parents"] > 1:
            assert np.isclose(check["low"], row.score_low) and np.isclose(check["high"], row.score_high)
    checks = dict(status="passed", fingerprint=manifest["fingerprint"], instances=len(manifest["units"]),
        parents_checked=len(checked_inputs), canonical_draw_rows=raw_count, setting_rows=len(summary),
        imported_instances=sum(m["imported"] for m in units_meta),
        newly_simulated_instances=sum(not m["imported"] for m in units_meta),
        actual_new_private_assignment_solves=sum(m["actual_private_solves"] for m in units_meta),
        all_matching_costs_reconstructed=True, all_normalizations_independently_reconstructed=True,
        all_setting_means_and_intervals_checked=True, unchanged_control_differences_zero=unchanged_zero,
        canonical_sha256=sha256(destination), hardware=study.hardware())
    atomic_json(output/"verification.json", checks)
    atomic_json(directory/"COMPLETE.json", checks)
    shutil.copyfile(directory/"manifest.json", output/"manifest.json")
    atomic_json(output/"hardware_provenance.json", dict(new_hardware_constraint=cfg["hardware_constraint"],
        workers=[m["hardware"] for m in units_meta], source_hardware="Imported controls retain the original archive hardware; import host is not original execution host"))
    for path in (directory/"units", directory/"inputs", directory/"source_snapshot"):
        link = output/path.name
        if path.exists() and not link.exists():
            link.symlink_to("../"+path.name)
    figures(summary, baseline_summary, differences, geometry, output)
    report(manifest, summary, baseline_summary, differences, checks, output)
    print(json.dumps(checks), flush=True)


def NAMESPACE_FOR_ROW(manifest, meta):
    return "see source archive manifest" if meta["imported"] else manifest["design"]["seed_namespace"]


def figures(summary, baseline, differences, geometry, output):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "ps.fonttype": 42})
    def save(fig, stem):
        for ext in ("png", "pdf"):
            fig.savefig(output/"figures"/(stem+"."+ext), dpi=180, bbox_inches="tight")
        plt.close(fig)
    meta = summary[(summary.objective_p == 1) & (summary.mechanism == "terminal_fold")].set_index("dataset").loc[ORDER]
    # Overview and paired differences preserve the mechanism order of the original plots.
    for delta in (False, True):
        fig, axes = plt.subplots(1, 2, figsize=(18, 9.5), gridspec_kw={"width_ratios": [8, 7]}, sharey=True)
        fig.subplots_adjust(left=.14, right=.94, bottom=.18, top=.79, wspace=.1)
        for p, ax in zip((1, 2), axes):
            frame = differences if delta else summary
            value = "score_difference_mean" if delta else "score_mean"
            pivot = frame[frame.objective_p == p].pivot(index="dataset", columns="mechanism", values=value).loc[ORDER, PLOT_ARMS[p]]
            values = pivot.to_numpy()
            norm = TwoSlopeNorm(vmin=-max(.05, np.abs(differences.score_difference_mean).max()), vcenter=0,
                                vmax=max(.05, np.abs(differences.score_difference_mean).max())) if delta else Normalize(0, max(1., summary.score_mean.max()))
            im = ax.imshow(values, cmap="RdBu_r" if delta else "YlGnBu_r", norm=norm, aspect="auto")
            ax.set_xticks(range(len(PLOT_ARMS[p])), [LABELS[a] for a in PLOT_ARMS[p]], fontsize=9)
            ax.xaxis.tick_top(); ax.tick_params(length=0, pad=8)
            ax.set_yticks(range(15), [d+(" *" if d in study.SCALED else "") for d in ORDER], fontsize=10)
            ax.set_title("Euclidean distance" if p == 1 else "Squared Euclidean distance", fontsize=14, pad=55)
            for i in range(15):
                for j, arm in enumerate(PLOT_ARMS[p]):
                    value_ij = values[i, j]
                    color = "white" if (not delta and norm(value_ij) < .45) or (delta and abs(norm(value_ij)-.5) > .32) else "#182e32"
                    ax.text(j, i, f"{value_ij:+.3f}" if delta else f"{value_ij:.3f}", ha="center", va="center", color=color, fontsize=9)
                    if delta:
                        row = frame[(frame.dataset == ORDER[i]) & (frame.objective_p == p) & (frame.mechanism == arm)].iloc[0]
                        if pd.notna(row.score_difference_low) and row.score_difference_low <= 0 <= row.score_difference_high and ORDER[i] in study.SCALED:
                            ax.add_patch(Rectangle((j-.45, i-.43), .9, .86, fill=False, edgecolor="#555", linestyle=":", linewidth=1))
            ax.axvline(len(PLOT_ARMS[p])-1.5, color="#aa762a", linewidth=2)
            for spine in ax.spines.values(): spine.set_visible(False)
        fig.colorbar(im, ax=axes, shrink=.65, pad=.02)
        fig.suptitle("Normalization changes: normalized minus original score" if delta else "All 15 benchmarks after normalization · epsilon = 8", fontsize=20, y=.98)
        fig.text(.14, .09, "* Feature-scaled datasets. Other datasets reuse exact historical inputs and draws. Gumbel (last column) protects curator output.", fontsize=10)
        fig.text(.14, .05, "Negative favors normalization; dotted outlines mark pointwise intervals containing zero. Different preprocessing defines different cost/privacy metrics." if delta else
                 "Mean normalized regret: 0 = OPT, 1 = RAND. Noise → partition → parent averaging; n=64, 20 parents, five partitions, K=50.", fontsize=10)
        save(fig, "normalization_paired_differences_epsilon8" if delta else "normalized_overview_epsilon8")
    # Dataset trajectories, matching the earlier connected-dot figure.
    colors = plt.get_cmap("tab20")
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), sharey=True)
    fig.subplots_adjust(left=.065, right=.985, bottom=.32, top=.8, wspace=.12)
    handles = []
    for p, ax in zip((1, 2), axes):
        frame = summary[summary.objective_p == p].pivot(index="dataset", columns="mechanism", values="score_mean")
        for i, ds in enumerate(ORDER):
            line, = ax.plot(range(len(PLOT_ARMS[p])), frame.loc[ds, PLOT_ARMS[p]], color=colors(i),
                            marker=["o", "s", "^", "D", "v"][i % 5], linestyle=["-", "--", "-."][i//5],
                            linewidth=1.5, markersize=4, label=ds)
            if p == 1: handles.append(line)
        ax.set_xticks(range(len(PLOT_ARMS[p])), [LABELS[a] for a in PLOT_ARMS[p]], fontsize=9)
        ax.set(ylim=(0, max(1., summary.score_high.max()+.01)), title="Euclidean distance" if p == 1 else "Squared Euclidean distance")
        ax.grid(axis="y", alpha=.25)
        ax.axvspan(len(PLOT_ARMS[p])-1.5, len(PLOT_ARMS[p])-.5, color="#fbf2df", zorder=0)
    axes[0].set_ylabel("Mean normalized regret · lower is better")
    fig.suptitle("Benchmark hardness after normalization · epsilon = 8", fontsize=20, y=.95)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.52, .1), ncol=5, frameon=False, fontsize=9)
    fig.text(.065, .04, "Colors/markers follow the same dataset in both panels. Seven unchanged controls reuse original draws; no new epsilon sweep.", fontsize=10)
    save(fig, "normalized_trends_epsilon8")
    # Box summaries show between-dataset variation, never confidence intervals.
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), sharey=True)
    fig.subplots_adjust(left=.065, right=.985, bottom=.25, top=.8, wspace=.13)
    for p, ax in zip((1, 2), axes):
        rows = summary[summary.objective_p == p]
        values = [rows[rows.mechanism == arm].set_index("dataset").loc[ORDER].score_mean.to_numpy() for arm in PLOT_ARMS[p]]
        ax.boxplot(values, whis=(0, 100), patch_artist=True, boxprops=dict(facecolor="#a6cbbd"), medianprops=dict(color="#123f35"))
        jitter = np.linspace(-.13, .13, 15)
        for j, y in enumerate(values): ax.scatter(j+1+jitter, y, s=16, color="#37474f", alpha=.65, zorder=4)
        ax.set_xticks(range(1, len(values)+1), [LABELS[a] for a in PLOT_ARMS[p]], fontsize=9)
        ax.set(ylim=(0, max(1., summary.score_high.max()+.01)), title="Euclidean distance" if p == 1 else "Squared Euclidean distance")
        ax.grid(axis="y", alpha=.25)
    axes[0].set_ylabel("Mean normalized regret · lower is better")
    fig.suptitle("Variation across 15 normalized benchmark datasets · epsilon = 8", fontsize=19, y=.95)
    fig.text(.065, .08, "Boxes: quartiles across 15 dataset means; whiskers: minimum/maximum; points: dataset means. These are not confidence intervals.", fontsize=10)
    save(fig, "normalized_boxplot_epsilon8")
    # Geometry before/after: exact rank, effective rank, widths and diameter.
    old = baseline[(baseline.objective_p == 2) & (baseline.mechanism == "coordinate_allocated")].set_index("dataset")
    new = summary[(summary.objective_p == 2) & (summary.mechanism == "coordinate_allocated")].set_index("dataset")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x = np.arange(8)
    datasets = list(study.SCALED)
    for ax, column, title in [(axes[0], "q_rank_mean", "Public affine rank"), (axes[1], "q_effective_rank_mean", "Public covariance effective rank")]:
        ax.bar(x-.18, old.loc[datasets, column], width=.36, label="Original", color="#9aa5ad")
        ax.bar(x+.18, new.loc[datasets, column], width=.36, label="Feature-scaled", color="#287c70")
        ax.set_xticks(x, [d.split("/")[1] for d in datasets], rotation=25)
        ax.set_title(title); ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    fig.suptitle("Public geometry on the same sampled records · epsilon-independent diagnostics", fontsize=16)
    fig.tight_layout()
    save(fig, "normalized_geometry_comparison")


def report(manifest, summary, baseline, differences, checks, output):
    cfg = manifest["design"]
    lines = ["# Feature-normalized clustering benchmarks at epsilon 8", "",
        f'Completed {cfg["phase"]} run `{manifest["fingerprint"]}`. **15 datasets, n=64 public and 64 private points, {cfg["parents"]} parents, {cfg["partitions"]} partitions per parent, K={cfg["K"]}, epsilon=8 only.**', "",
        "The same saved source rows and Q/R assignments as the original experiments are used. Eight tabular datasets receive the agreed feature scaling. Circles, Ring noisy, Chainlink, Worms, Iris, MNIST digits and Fashion-MNIST retain their historical preprocessing and reuse their exact epsilon-8 executions with explicit provenance. Original experiments remain unchanged.", "",
        "## Normalization design", "",
        "For each parent, order its 128 source records by original row ID and fit one shared transform before making the five existing Q/R partitions. Continuous features use pooled means and population standard deviations (ddof=0). Numerical constants use scale 1 and retain their coordinates; the float64 error-bound rule and masks are saved. Binary indicators use the fixed affine source mappings below and receive no variance scaling.", "",
        "| Dataset | Feature scaling before final RMS normalization |", "|---|---|",
        "| Wine | Z-score all 13 features. |", "| WDBC | Z-score all 30 features. |",
        "| Statlog | Z-score varying features; retain the constant pixel-count coordinate, centered to zero. |",
        "| Glass | Z-score all nine features. |",
        "| Ecoli | Z-score five numerical scores; lip 0.48/1 → 0/1; chg 0.5/1 → 0/1. |",
        "| Yeast | Z-score seven numerical scores including pox; erl 0.5/1 → 0/1. |",
        "| Ionosphere | Column 1 stays binary 0/1; column 2 stays zero; z-score columns 3–34. |",
        "| Sonar | Z-score all 60 features. |",
        "| Other seven datasets | Exact original standardized coordinates and mechanism outcomes are retained. |", "",
        "After feature scaling, center the pooled records and divide by their common RMS distance from the center. The result has pooled RMS radius one; coordinates need not lie in [0,1] or [-1,1]. Binary values remain two-valued after this final scalar transform, but generally are no longer 0 and 1. All fitted offsets, scales, constant masks, source rows and normalized parent arrays are saved in `inputs/`.", "",
        "Normalization constructs benchmark inputs outside the private mechanism. Costs and epsilon-metric privacy refer to the resulting normalized space. The same numerical epsilon across preprocessing choices protects different feature-space neighborhoods; performance changes do not establish improvement under an unchanged raw-data metric.", "",
        "## Performance at epsilon 8", "",
        "Scores are instance-wise mean regret divided by exact RAND minus OPT: 0 is OPT, 1 is RAND. Noise is averaged within an instance, then partitions within a parent, then parents. Values above one are retained. Pointwise 95% Student-t intervals use parent means, with no multiple-comparison correction. Euclidean and squared-cost panels are separate; both use Euclidean metric privacy. Gumbel protects the trusted curator's final matching; all other arms also protect local reports.", "",
        "![All normalized benchmark means](figures/normalized_overview_epsilon8.png)", "",
        "![Distribution across normalized datasets](figures/normalized_boxplot_epsilon8.png)", "",
        "![Connected dataset performance trends](figures/normalized_trends_epsilon8.png)", "",
        "## Changes relative to original feature units", "",
        "Differences below pair the same parent collections. Negative means lower normalized regret after feature scaling. The denominators and optimal assignments are recomputed in each geometry; these are comparisons of relative utility in two different metrics. New mechanism channels are independent of the historical draws; pairing is through parent identity, not draw index. Unchanged controls have zero differences by explicit reuse.", "",
        "![Paired score changes](figures/normalization_paired_differences_epsilon8.png)", "",
        "| Dataset | Objective | Lowest observed mean: original → normalized | Original → normalized score |", "|---|---|---|---:|"]
    for ds in ORDER:
        for p in (1, 2):
            a = baseline[(baseline.dataset == ds) & (baseline.objective_p == p)].sort_values("score_mean").iloc[0]
            b = summary[(summary.dataset == ds) & (summary.objective_p == p)].sort_values("score_mean").iloc[0]
            lines.append(f'| {ds} | {"Euclidean" if p == 1 else "Squared"} | {a.mechanism} → {b.mechanism} | {a.score_mean:.3f} → {b.score_mean:.3f} |')
    lines += ["", "The minimum observed mean is descriptive; near ties are not resolved by this table. Use the paired interval table for a specified arm comparison.", "",
        "## Geometry and privacy-distance interpretation", "",
        "![Exact and effective public ranks](figures/normalized_geometry_comparison.png)", "",
        "Positive diagonal scaling preserves exact affine rank mathematically; effective rank, PCA directions, directional widths, diameters and neighbor relations can change. Numerical rank tolerances are retained. Binary remapping is affine and coordinates are not dropped.", "",
        "At epsilon 8, the factor-two protected separation is ln(2)/8 = **0.08664** in the final normalized units. It is a separation between alternative private records, not expected noise magnitude. The geometry table records public and pooled diameters so its relative scale remains visible.", "",
        "## Reproducibility, validation and storage", "",
        f'Validated **{checks["canonical_draw_rows"]:,} canonical draw labels**, **{checks["instances"]:,} instances**, and **{checks["parents_checked"]} parent transforms**. All saved assignments were verified as bijections and their costs, regrets and realization certificates reconstructed. All normalization parameters were independently reconstructed, and setting means/intervals were rebuilt from instance means. There are **{checks["newly_simulated_instances"]} newly simulated** and **{checks["imported_instances"]} imported** instances, with **{checks["actual_new_private_assignment_solves"]:,} new private assignment solves** after equivalent-endpoint reuse.', "",
        "New execution is pinned to AMD Rome, one numerical thread, with source/library/binary and NumPy CPU-feature checks against the manifest. Historical imported controls retain their original runtime provenance. Source snapshots, exact source-row IDs, raw selected parent records, fitted transforms, normalized inputs, PRNG namespaces, complete assignments and numerical outcomes are retained. Pilot and production are separately fingerprinted; pilot draws are excluded from production.", "",
        "Active files and logs are on `/vast`; durable results are mirrored to `/ix1`. The shared archived Python environment is hard-linked on `/ix1` to avoid duplicating its 400 MB payload. `/ihome` contains small source/report files and symlinks. Scientific replay evidence is added after replay from the extracted environment; timings and compressed-container bytes are not deterministic scientific quantities.", "",
        "Rebuild the handoff on a compatible compute node with the frozen source first on PYTHONPATH:", "", "```bash",
        "python -m private_matching analysis-table --input RUN_DIRECTORY --output RUN_DIRECTORY/analysis_handoff", "```", "",
        "[Canonical per-draw table](analysis_table.csv.gz) · [Summary and intervals](summary.csv) · [Parent means](parent_summaries.csv.gz) · [Instance means](instance_summaries.csv.gz) · [Paired differences and intervals](paired_comparisons.csv) · [Geometry](geometry_table.csv.gz) · [Baseline evidence](baseline_summary.csv) · [Manifest](manifest.json) · [Validation](verification.json) · [Hardware provenance](hardware_provenance.json)", ""]
    (output/"report.md").write_text("\n".join(lines))
    dictionary = dict(objective="SUM of original distances^p in each designated normalized geometry", score="mean noise regret / exact RAND-minus-OPT per instance, then partition and parent means",
        epsilon="8 only, per normalized Euclidean distance; both objectives use same privacy metric", binary_maps=cfg["binary_maps"],
        constant_rule=cfg["constant_policy"], arrays="permutations[arm_index,draw,private_record]; q/r row IDs in geometry file; ARM_KEYS order in frozen source",
        normalization="inputs/*.npz: raw selected records in sorted source-ID order, normalized records, feature_offset, feature_scale, feature_constant, pooled_center, pooled_radius",
        randomness="independent channel PRNGs from SHA256 JSON tuple; imported outcomes use historical manifest protocols; no inferred sample pairing",
        intervals="pointwise unadjusted Student-t 95% over parent means; paired differences use same parents", nulls="No finite general squared-cost diameter cap is asserted; bound_cap may be null",
        reuse="source_arm records equivalent endpoints or imported historical labels; imports are existing draws, never new independent replications")
    atomic_json(output/"data_dictionary.json", dictionary)
