"""Parallel audited draw exports, parent inference and paper-style sweep figures."""
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
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.stats import t

from . import clustering_manuscript as study
from .clustering_benchmark import sha256
from .integral_study import atomic_json
from .login_guard import refuse_login_compute

LABELS = {"gumbel": "Gumbel", "original_points": "Original points",
          "projected_representation": "Projected representation", "distance_vectors": "Distance vectors"}
COLORS = dict(zip(study.ARMS, ("#4477AA", "#228833", "#AA3377", "#CCBB44")))
STYLES = dict(zip(study.ARMS, ("-", "-", "--", "-.")))
METRICS = ["mean_total_cost", "mean_total_regret", "optimum", "random_expected", "random_minus_optimum", "expected_bound"]


def audit_master(path, dataset):
    """Independent columnwise reconstruction of the saved feature transform."""
    from .clustering_normalized import BINARY, SCALED
    with np.load(path) as master:
        raw = master["raw"]
        for j in range(raw.shape[1]):
            offset, scale = 0., 1.
            if dataset in SCALED or dataset == "uci/banknote":
                if j in BINARY.get(dataset, {}):
                    lo, hi = BINARY[dataset][j]
                    offset, scale = lo, hi-lo
                else:
                    offset = float(raw[:, j].mean())
                    variance = float(np.mean((raw[:, j]-offset)**2))
                    eps = np.finfo(float).eps
                    constant = variance <= len(raw)*eps*variance+(len(raw)*offset*eps)**2
                    assert bool(master["feature_constant"][j]) == constant
                    scale = 1. if constant else np.sqrt(variance)
            np.testing.assert_allclose([offset, scale],
                [master["feature_offset"][j], master["feature_scale"][j]], rtol=1e-11, atol=1e-12)


def export_group(root, manifest, group, metas):
    root = Path(root); stem = root/"chunks"/group["group_id"]
    cfg, setting = manifest["design"], group["setting"]
    master_record = json.loads((root/"master_registry.json").read_text())[study.master_key(setting, group["parent"])]
    assert sha256(root/master_record["file"]) == master_record["sha256"]
    audit_master(root/master_record["file"], setting["dataset"])
    instance_rows, geometry_rows, raw_count = [], [], 0
    raw_path = Path(str(stem)+".draws.csv.gz")
    with gzip.open(str(raw_path)+".tmp", "wt", newline="", compresslevel=1) as stream:
        fields = ["instance_id", "objective_p", "mechanism", "epsilon", "draw", "total_cost",
                  "total_regret", "row_oscillation", "objective_index", "arm_index", "epsilon_index"]
        writer = csv.writer(stream); writer.writerow(fields)
        for meta in metas:
            instance = meta["instance_id"]
            for filename, checksum in meta["files"].items():
                assert sha256(root/"units"/filename) == checksum
            Q, R = study.load_input(root, meta)
            C1 = cdist(R, Q)
            assert np.allclose(np.r_[Q, R].mean(axis=0), 0., atol=2e-12)
            rms2 = np.mean(np.sum(np.r_[Q, R]**2, axis=1))
            assert np.isclose(rms2, 0 if meta["input"]["zero_radius"] else 1, atol=2e-12)
            n, d = setting["n"], setting["d"]
            geometry_rows.append(dict(instance_id=instance, setting_id=setting["setting_id"],
                panel=setting["panel"], dataset=setting["dataset"], parent=group["parent"], partition=meta["partition"],
                n=n, d=d, ln_n=float(np.log(n)), low_dimension=d < np.log(n),
                public_rank=meta["public"]["public_rank"], folded_dimension=min(meta["public"]["public_rank"]+1, d),
                effective_rank=meta["diagnostics"]["Q"]["effective_rank"],
                public_diameter=meta["public"]["public_diameter"], pooled_diameter=meta["pooled_diameter"],
                sampled_with_replacement=meta["input"]["sampled_with_replacement"],
                q_unique_source_count=meta["input"]["q_unique_source_count"],
                r_unique_source_count=meta["input"]["r_unique_source_count"],
                source_overlap_count=meta["input"]["source_overlap_count"],
                alias_projection_to_original=meta["alias_projection_to_original"],
                geometry_file="../units/"+instance+".geometry.npz",
                outcome_file="../units/"+instance+".npz", metadata_file="../units/"+instance+".json",
                master_file="../"+meta["input"]["master_file"]))
            with np.load(root/"units"/(instance+".npz")) as saved:
                for ip, p in enumerate(setting["objectives"]):
                    C = C1 if p == 1 else C1*C1
                    rows, columns = linear_sum_assignment(C)
                    optimum = float(C[rows, columns].sum())
                    random = float(C.sum()/n)
                    np.testing.assert_allclose([optimum, random],
                        [saved["optimum"][ip], saved["random_expected"][ip]], rtol=1e-11, atol=1e-9)
                    pi = saved["permutations"][ip]
                    assert np.all(np.sort(pi, axis=-1) == np.arange(n))
                    costs = C[np.arange(n), pi].sum(axis=-1)
                    np.testing.assert_allclose(costs, saved["total_cost"][ip], rtol=1e-11, atol=1e-9)
                    np.testing.assert_allclose(costs-optimum, saved["total_regret"][ip], rtol=1e-11, atol=1e-9)
                    assert np.all(saved["total_regret"][ip] <= saved["row_oscillation"][ip]+1e-6)
                    if meta["alias_projection_to_original"]:
                        for field in (*study.FIELDS, "permutations"):
                            np.testing.assert_array_equal(saved[field][ip, 1], saved[field][ip, 2])
                    for ia, arm in enumerate(study.ARMS):
                        for ie, epsilon in enumerate(setting["epsilon"]):
                            common = dict(instance_id=instance, setting_id=setting["setting_id"], panel=setting["panel"],
                                dataset=setting["dataset"], parent=group["parent"], partition=meta["partition"],
                                n=n, d=d, objective_p=p, mechanism=arm, epsilon=epsilon,
                                source_arm="original_points" if arm == "projected_representation" and meta["alias_projection_to_original"] else "")
                            instance_rows.append(dict(common, K=cfg["K"], mean_total_cost=float(costs[ia, ie].mean()),
                                mean_total_regret=float(saved["total_regret"][ip, ia, ie].mean()), optimum=optimum,
                                random_expected=random, random_minus_optimum=random-optimum,
                                expected_bound=float(saved["expected_bound"][ip, ia, ie])))
                            for draw in range(cfg["K"]):
                                writer.writerow([instance, p, arm, epsilon, draw, costs[ia, ie, draw],
                                    saved["total_regret"][ip, ia, ie, draw], saved["row_oscillation"][ip, ia, ie, draw], ip, ia, ie])
                                raw_count += 1
    os.replace(str(raw_path)+".tmp", raw_path)
    actual = 0
    with gzip.open(raw_path, "rb") as stream:
        while block := stream.read(8 << 20):
            actual += block.count(b"\n")
    assert actual-1 == raw_count == len(metas)*len(setting["objectives"])*4*len(setting["epsilon"])*cfg["K"]
    frame = pd.DataFrame(instance_rows)
    frame.to_csv(str(stem)+".instances.csv.gz", index=False)
    keys = ["setting_id", "panel", "dataset", "n", "d", "parent", "objective_p", "mechanism", "epsilon"]
    parent = frame.groupby(keys, sort=True)[METRICS].mean().reset_index()
    assert (frame.groupby(keys).size() == cfg["partitions"]).all()
    parent["partition_count"] = cfg["partitions"]
    parent.to_csv(str(stem)+".parents.csv.gz", index=False)
    pd.DataFrame(geometry_rows).to_csv(str(stem)+".geometry.csv.gz", index=False)
    files = {p.name: sha256(p) for p in root.glob("chunks/"+group["group_id"]+".*.csv.gz")}
    atomic_json(str(stem)+".json", dict(status="passed", fingerprint=manifest["fingerprint"], group=group,
        files=files, draw_rows=raw_count, instance_rows=len(frame), parent_rows=len(parent),
        all_matching_costs_reconstructed=True, normalization_audited=True,
        actual_assignment_solves=sum(m["actual_assignment_solves"] for m in metas), hardware=study.hardware()))


def summarized(parent):
    keys = ["setting_id", "panel", "dataset", "n", "d", "objective_p", "mechanism", "epsilon"]
    aggregate = parent.groupby(keys)[METRICS].agg(["mean", "std", "count"])
    output = aggregate.index.to_frame(index=False)
    for metric in METRICS:
        mean = aggregate[metric]["mean"].to_numpy()
        count = aggregate[metric]["count"].to_numpy()
        se = aggregate[metric]["std"].to_numpy()/np.sqrt(count)
        half = t.ppf(.975, count-1)*se
        for label, values in [("mean", mean), ("low", mean-half), ("high", mean+half), ("se", se), ("parent_count", count)]:
            output[metric+"_"+label] = values
    return output


def export(root, output):
    refuse_login_compute("aggregate manuscript benchmark handoff")
    root, output = Path(root), Path(output)
    manifest = json.loads((root/"manifest.json").read_text())
    study.verify_runtime(manifest)
    output.mkdir(parents=True, exist_ok=True)
    parents, geometries, index = [], [], []
    total_rows = total_solves = instance_rows = 0
    for group in study.groups(manifest["design"]):
        stem = root/"chunks"/group["group_id"]
        meta = json.loads(Path(str(stem)+".json").read_text())
        assert meta["status"] == "passed" and meta["fingerprint"] == manifest["fingerprint"] and meta["group"] == group
        for filename, checksum in meta["files"].items():
            assert sha256(root/"chunks"/filename) == checksum
            index.append(dict(group_id=group["group_id"], file="../chunks/"+filename, sha256=checksum,
                              draw_rows=meta["draw_rows"] if ".draws." in filename else ""))
        parents.append(pd.read_csv(str(stem)+".parents.csv.gz"))
        geometries.append(pd.read_csv(str(stem)+".geometry.csv.gz"))
        total_rows += meta["draw_rows"]; total_solves += meta["actual_assignment_solves"]
        instance_rows += meta["instance_rows"]
    parent, geometry = pd.concat(parents, ignore_index=True), pd.concat(geometries, ignore_index=True)
    summary = summarized(parent)
    parent.to_csv(output/"parent_summaries.csv.gz", index=False)
    geometry.to_csv(output/"geometry_table.csv.gz", index=False)
    summary.to_csv(output/"summary.csv", index=False)
    pd.DataFrame(index).to_csv(output/"table_index.csv", index=False)
    cfg = manifest["design"]
    expected = sum(len(s["objectives"])*4*len(s["epsilon"])*cfg["K"]*cfg["parents"]*cfg["partitions"] for s in cfg["settings"])
    assert total_rows == expected and len(geometry) == len(cfg["settings"])*cfg["parents"]*cfg["partitions"]
    assert not parent.duplicated(["setting_id", "parent", "objective_p", "mechanism", "epsilon"]).any()
    assert (summary.mean_total_regret_parent_count == cfg["parents"]).all()
    verification = dict(status="passed", fingerprint=manifest["fingerprint"], canonical_draw_rows=total_rows,
        actual_assignment_solves=total_solves, distinct_geometries=len(geometry), parent_rows=len(parent),
        setting_rows=len(summary), instance_rows=instance_rows, chunk_groups=len(parents),
        all_parent_chunks_audited=True, all_costs_optima_rand_and_normalizations_reconstructed=True,
        summary_sha256=sha256(output/"summary.csv"), index_sha256=sha256(output/"table_index.csv"), hardware=study.hardware())
    atomic_json(output/"verification.json", verification)
    shutil.copyfile(root/"manifest.json", output/"manifest.json")
    shutil.copyfile(root/"master_registry.json", output/"master_registry.json")
    for name in ("masters", "units", "chunks", "source_snapshot"):
        link = output/name
        if (root/name).exists() and not link.exists():
            link.symlink_to("../"+name, target_is_directory=True)
    atomic_json(output/"data_dictionary.json", dict(
        join="draws.instance_id -> geometry_table.instance_id -> units metadata; setting_id -> summary; table_index locates compressed chunks",
        draws="one row per mechanism/objective/epsilon/draw label, including explicitly reused identity endpoints",
        permutations="units/<instance_id>.npz: permutations[objective_index,arm_index,epsilon_index,draw,private_row]",
        inputs="masters store selected raw records, source IDs, all partition orders and feature transforms; units geometry stores occurrence IDs, feature IDs, pooled center/radius; reconstruct normalized Q/R by frozen load_input",
        quantity="TOTAL matching cost minus instance OPT; no division by n or RAND gap",
        RAND="exact uniform permutation expected cost sum(C)/n",
        privacy="sum-record Euclidean in designated normalized coordinates; Gumbel final matching, other arms local reports",
        randomness="seed = SHA256 JSON tuple(namespace,phase,instance_id,draw,channel), first16bytes little-endian, NumPy default_rng PCG64; epsilon/objectives share unit draws",
        source_arm="projected representation copies original points exactly when min(rank(Q)+1,d)==d",
        sampling_policy=cfg["sampling"], resampling="none; original source IDs and occurrence IDs are distinct and disjoint across sides",
        inference="noise -> partitions -> parents; pointwise unadjusted Student-t95%; no pilot pooling"))
    figures(summary, geometry, output)
    report(manifest, summary, geometry, verification, output)
    atomic_json(root/"COMPLETE.json", verification)
    print(json.dumps(verification), flush=True)


def figures(summary, geometry, output):
    directory = output/"figures"; directory.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
                        "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        for suffix in ("png", "pdf"):
            fig.savefig(directory/(name+"."+suffix), dpi=200, bbox_inches="tight")
        plt.close(fig)
    def curves(ax, frame, xcolumn):
        handles = []
        for arm in study.ARMS:
            values = frame[frame.mechanism == arm].sort_values(xcolumn)
            x, y = values[xcolumn].to_numpy(), values.mean_total_regret_mean.to_numpy()
            line, = ax.plot(x, y, color=COLORS[arm], linestyle=STYLES[arm], linewidth=1.4, label=LABELS[arm])
            handles.append(line)
            ax.fill_between(x, values.mean_total_regret_low.to_numpy(), values.mean_total_regret_high.to_numpy(),
                            color=COLORS[arm], alpha=.12, linewidth=0)
        random = frame[frame.mechanism == "gumbel"].sort_values(xcolumn)
        line, = ax.plot(random[xcolumn], random.random_minus_optimum_mean, color="#555555", linestyle="--", linewidth=1.1, label="RAND − OPT")
        handles.append(line)
        ax.set_ylim(bottom=0); ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
        return handles
    for p in (1, 2):
        fig, axes = plt.subplots(4, 4, figsize=(14, 10), sharex=True)
        fig.subplots_adjust(left=.06, right=.99, bottom=.12, top=.94, wspace=.24, hspace=.37)
        for ax, dataset in zip(axes.flat, study.DATASETS):
            frame = summary[(summary.panel == "epsilon") & (summary.dataset == dataset) & (summary.objective_p == p)]
            handles = curves(ax, frame, "epsilon")
            ax.set_title(dataset); ax.set_xticks([0, 4, 8, 12, 16]); ax.set_xlim(.25, 16)
        fig.supxlabel("ε", y=.065)
        fig.supylabel("E[u] − OPT", x=.008)
        fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(.52, .003), frameon=False)
        save(fig, f"epsilon_regret_p{p}")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    fig.subplots_adjust(left=.07, right=.99, top=.88, bottom=.24, wspace=.28)
    bound_line = None
    for ax, panel, column, title in [(axes[0], "dimension", "d", "MNIST"), (axes[1], "sample_size", "n", "Banknote")]:
        frame = summary[summary.panel == panel]
        plot_frame = frame.copy()
        if panel == "sample_size":
            for field in ("mean_total_regret_mean", "mean_total_regret_low", "mean_total_regret_high", "random_minus_optimum_mean"):
                plot_frame[field] = plot_frame[field]/plot_frame.n
        handles = curves(ax, plot_frame, column)
        handles[-1].set_label("RAND reference")
        threshold = np.log(400) if panel == "dimension" else np.exp(4)
        if panel == "dimension":
            n = int(frame.n.iloc[0]); epsilon = float(frame.epsilon.iloc[0])
            bound_d = np.unique(np.r_[np.linspace(4, 784, 400), np.log(n)])
            bound_line, = ax.plot(bound_d, 2*n*np.minimum(bound_d, np.log(n))/epsilon,
                                  color="black", linestyle="--", linewidth=1.5,
                                  label="Point/Gumbel bound")
            ax.set_ylim(0, 1.05*max(ax.get_ylim()[1], float(np.max(bound_line.get_ydata()))))
            ax.set_xscale("linear"); ax.set_xlim(0, 784); ax.set_xticks([0, 200, 400, 600, 784])
            ax.axvspan(4, threshold, color="#e9f3fb", zorder=0)
            ax.axvspan(threshold, 784, color="#fff1e4", zorder=0)
            ax.set_xlabel("d")
        else:
            bound_n = np.linspace(20, 200, 400)
            ax.plot(bound_n, 2*np.minimum(4, np.log(bound_n))/8,
                    color="black", linestyle="--", linewidth=1.5)
            ax.set_ylim(0, 1.05*max(ax.get_ylim()[1], 1.))
            ax.axvspan(20, threshold, color="#fff1e4", zorder=0)
            ax.axvspan(threshold, 200, color="#e9f3fb", zorder=0)
            ax.set_xlabel("n")
        ax.axvline(threshold, color="#999999", linestyle=":", linewidth=1)
        ax.set_ylabel("E[u] − OPT" if panel == "dimension" else "(E[u] − OPT) / n"); ax.set_title(title)
    fig.legend(handles=handles+[bound_line], loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.52, -.01))
    save(fig, "dimension_and_sample_size_regret")
    fig, axes = plt.subplots(2, 1, figsize=(15, 8.5))
    fig.subplots_adjust(left=.06, right=.995, bottom=.16, top=.93, hspace=.65)
    low_color, high_color = "#dcebf9", "#f6d5d5"
    mixed_color = tuple((a+b)/2 for a, b in zip(matplotlib.colors.to_rgb(low_color),
                                               matplotlib.colors.to_rgb(high_color)))
    for ax, p in zip(axes, (1, 2)):
        frame = summary[(summary.panel == "epsilon") & (summary.objective_p == p) & (summary.epsilon == 8.)]
        x = np.arange(16); width=.19
        for i, dataset in enumerate(study.DATASETS):
            row = frame[frame.dataset == dataset].iloc[0]
            background = low_color if row.d <= np.log(row.n) else (high_color if row.d >= row.n else mixed_color)
            ax.axvspan(i-.5, i+.5, color=background, zorder=0)
        for ia, arm in enumerate(study.ARMS):
            values = frame[frame.mechanism == arm].set_index("dataset").loc[list(study.DATASETS)]
            y = values.mean_total_regret_mean.to_numpy()
            err = np.stack([y-values.mean_total_regret_low.to_numpy(), values.mean_total_regret_high.to_numpy()-y])
            ax.bar(x+(ia-1.5)*width, y, width*.94, color=COLORS[arm], label=LABELS[arm],
                   yerr=err, error_kw=dict(ecolor="#333333", elinewidth=.7, capsize=1))
        values = frame[frame.mechanism == "gumbel"].set_index("dataset").loc[list(study.DATASETS)]
        ax.plot(x, values.random_minus_optimum_mean, color="#555555", linestyle="--", linewidth=1, label="RAND − OPT")
        ax.set_xticks(x, list(study.DATASETS), rotation=32, ha="right", fontsize=8)
        ax.set_ylabel("E[u] − OPT"); ax.set_ylim(bottom=0); ax.set_xlim(-.6, 15.6)
        ax.set_title("Euclidean cost" if p == 1 else "Squared Euclidean cost")
        ax.grid(axis="y", alpha=.18); ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(.52, -.005))
    fig.legend(handles=[Patch(facecolor=low_color, label=r"$d\leq\log n$"),
                        Patch(facecolor=mixed_color, label=r"$\log n<d<n$"),
                        Patch(facecolor=high_color, label=r"$d\geq n$")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.52, 1.005))
    save(fig, "benchmark_bars_epsilon8")


def report(manifest, summary, geometry, verification, output):
    cfg = manifest["design"]
    lines = ["# Four-mechanism normalized clustering benchmark results", "",
        f'Completed production fingerprint `{manifest["fingerprint"]}`. **16 benchmarks, 64 epsilon values (0.25 to 16 by 0.25), four mechanisms, 20 parents, five partitions, K=50.**', "",
        "This experiment implements the current `math_drafts.tex` mechanisms: original-point perturbation, projected representation (the former Terminal fold), infinity-norm distance-vector perturbation, and Gumbel matching. Its projected representation is distinct from the older corrected-projection and squared span-radial mechanisms. Costs are sums over all matched pairs.", "",
        "The main epsilon sweep uses n=64 public and n=64 private records for every benchmark. The MNIST dimension sweep fixes n=400 and epsilon=8, using d=784,759,…,9,4. The Banknote size sweep uses n=200,180,…,20 at d=4 and epsilon=8. Only the epsilon sweep includes squared cost. There are no label-conditioned experiments.", "",
        "**Sampling:** without replacement, with distinct and disjoint public/private source records. The existing 15 benchmarks retain their historical source rows and five partitions; Banknote adds new parent collections. All mechanism noise executions use this study's new namespace, with no historical noise pooling or bootstrap resampling.", "",
        "[UCI Banknote Authentication](https://archive.ics.uci.edu/dataset/267/banknote) contributes 1,372 observations with four continuous wavelet/image-summary features; its class column is excluded. Original image resolution is not its feature dimension. The other 15 benchmarks retain their pinned historical sources. No historical noise draws are pooled into these new results.", "",
        "**Preprocessing.** Retain the specified continuous z-scores, fixed binary mappings and constant coordinates from the normalized benchmark study. Add population z-scores for all four Banknote features. Fit main-parent feature weights once; fit Banknote n-sweep weights on its 400-record master and retain them across n. After taking the actual source and feature subset, center its pooled Q/R records and divide by their RMS radius. Record all transforms. MNIST uses a fixed publicly seeded pixel permutation and nested coordinate prefixes, without per-pixel z-scores. Normalization defines offline inputs; privacy is measured in this normalized space. Epsilon remains fixed in normalized units in n/d sweeps, with no extra n or d scaling.", "",
        "**Privacy and squared costs.** Every arm uses the sum of recordwise Euclidean displacements as its privacy metric. Gumbel protects the curator's matching; the other mechanisms protect their local reports too. Point/projection reports are unchanged when switching to squared decoding. Squared distance vectors use the centered cost row ||x−q_j||²−||x−c_Q||² and infinity-norm sensitivity 2R_Q; squared Gumbel scale is 2D_Q/epsilon. Unsquared distance-vector sensitivity is one and Gumbel scale is 2/epsilon. Projection noise retains the full min(rank(Q)+1,d) dimension. When that is d, its entire realization is copied from original points, with provenance.", "",
        "**Estimation.** Each plotted point averages noise, then five partitions, then 20 parent means. Shading/error bars show pointwise 95% Student-t intervals over parents, without multiple-comparison adjustment. The dashed RAND−OPT reference uses exact RAND=sum(C)/n, not random Monte Carlo assignments. The y-axis is total E[u]−OPT, without division by n or RAND−OPT. Consequently sample-size trends include the change in the number of summed terms.", "",
        "![Euclidean epsilon sweeps](figures/epsilon_regret_p1.png)", "",
        "Euclidean cost on all 16 benchmarks; epsilon runs in linear increments of 0.25. All panels use the same four mechanism labels and an exact random-matching reference.", "",
        "![Squared Euclidean epsilon sweeps](figures/epsilon_regret_p2.png)", "",
        "Squared cost uses its own optimum, RAND reference and correctly calibrated perturbations; its units and numeric scale differ from Euclidean cost.", "",
        "![Dimension and sample-size sweeps](figures/dimension_and_sample_size_regret.png)", "",
        "MNIST: n=400, epsilon=8; the d-axis is linear. Banknote: d=4, epsilon=8. Blue backgrounds mean d<ln(n), peach d>ln(n), with dotted boundaries at d=ln(400)≈5.991 and n=e⁴≈54.60. Coordinate deletion changes image information as well as dimension; the requested d-grid has only one point below ln(400). Banknote n=20,40 lies above the dimensional threshold; n≥60 below. These are ambient-dimension references, distinct from affine or effective rank. The black dashed MNIST curve is the manuscript's regret upper bound 2n min{d,ln(n)}/epsilon for the public rule selecting original points when d≤ln(n), and Gumbel otherwise. OPT is already subtracted in the theorem's regret definition and must not be subtracted again. The bound is 400 at d=4 and plateaus at 599.146 for d≥ln(400); it is not a simultaneous guarantee for all four mechanisms.", "",
        "![Benchmark bars at epsilon eight](figures/benchmark_bars_epsilon8.png)", "",
        "Four bars per dataset, ordered Gumbel, Original points, Projected representation, Distance vectors. Error bars are parent-level 95% intervals; the dashed curve is RAND−OPT. Background blocks are blue for d≤ln(n), rose for d≥n, and an equal blend of those colors for ln(n)<d<n. Here n=64: the first five benchmarks are low dimensional, Ecoli through Sonar are intermediate, and Worms-64 plus both image benchmarks have d≥n. These classifications use ambient dimension, not effective rank. Gumbel uses ordinary mechanism styling. Figures are available as PNG and PDF, with concise annotations intended for an applied theory paper.", "",
        "**Results at epsilon 8.** Lowest observed means are descriptive, especially when endpoints are exact aliases. They do not establish a simultaneous ranking over all settings.", "",
        "| Dataset | n | d | Mean public rank | Mean effective rank | Best Euclidean arm | Regret | Best squared arm | Regret |", "|---|---:|---:|---:|---:|---|---:|---|---:|"]
    for dataset in study.DATASETS:
        frame = summary[(summary.panel == "epsilon") & (summary.dataset == dataset) & (summary.epsilon == 8.)]
        a = frame[frame.objective_p == 1].sort_values("mean_total_regret_mean").iloc[0]
        b = frame[frame.objective_p == 2].sort_values("mean_total_regret_mean").iloc[0]
        g = geometry[(geometry.panel == "epsilon") & (geometry.dataset == dataset)]
        lines.append(f'| {dataset} | {a.n} | {a.d} | {g.public_rank.mean():.2f} | {g.effective_rank.mean():.2f} | {LABELS[a.mechanism]} | {a.mean_total_regret_mean:.3f} | {LABELS[b.mechanism]} | {b.mean_total_regret_mean:.3f} |')
    lines += ["", "**Reproducibility and archive.**",
        f'Audited **{verification["canonical_draw_rows"]:,} draw labels**, **{verification["distinct_geometries"]:,} geometries**, and **{verification["actual_assignment_solves"]:,} new exact assignment solves**. Every saved assignment, true cost, optimum, random reference and endpoint alias was reconstructed within its export owner. Feature transforms were independently reconstructed columnwise. The final aggregation checks all chunk hashes and parent counts.', "",
        "The canonical handoff is partitioned: table_index.csv locates all compressed per-draw and instance tables with SHA-256 checksums. Geometry and saved metadata join instance IDs to raw master records, normalized-input reconstruction, feature/source/occurrence IDs, complete permutations and noise seed protocols. Full noisy cost matrices are regenerated from pinned seeds rather than stored. Pilot draws have separate seeds and are excluded. Source/runtime binaries, CPU dispatch and single-thread settings are pinned; archived-environment replay evidence is supplied with final publication.", "",
        "Active computation and logs use /vast; the verified durable archive uses /ix1, with shared environment archives hard-linked when possible. /ihome contains the small report and links. No byte-identical timing or arbitrary-platform floating-point claim is made.", "",
        "[Summary and intervals](summary.csv) · [Parent summaries](parent_summaries.csv.gz) · [Canonical table index](table_index.csv) · [Geometry](geometry_table.csv.gz) · [Manifest](manifest.json) · [Input registry](master_registry.json) · [Data dictionary](data_dictionary.json) · [Verification](verification.json) · [Frozen source](source_snapshot/)", "",
        "Rebuild scientific tables/figures on a compatible compute node with the archived environment and frozen source using `python -m private_matching analysis-table --input RUN --output RUN/analysis_handoff`. Thread settings, source and CPU dispatch must match the manifest. Individual result regeneration uses the frozen `study --study clustering_manuscript` stages and archived Slurm launcher.", ""]
    (output/"report.md").write_text("\n".join(lines))
