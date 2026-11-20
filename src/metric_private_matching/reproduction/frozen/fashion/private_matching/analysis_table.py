"""One self-contained CSV from new and archived records; never rerun old noise."""
from __future__ import annotations

import csv
import io
from pathlib import Path
import numpy as np
from scipy.stats import t
from private_matching.results import read_json, atomic_write_json, atomic_write_bytes
from private_matching.login_guard import refuse_login_compute

# Required nullable columns are emitted even when only historical data are present.
REQUIRED = '''experiment_version fingerprint study_id sweep_id geometry_family configuration_label
instance_id instance_seed root_seed geometry_replicate geometry_randomized n d public_affine_rank
numerical_rank_tolerance pca_rank_tolerance analytic_affine_rank rank_status preprocessing exact_fold_residual
intrinsic_rank spectral_decay residual_thickness cluster_count cluster_spread private_A_axis private_B_axis
private_A_height private_B_height private_near_multiplier private_far_multiplier L public_perturbation
scale_convention public_scale public_rms_radius public_diameter fixed_instance_sweep scaled_changing_instance
arm_id arm_label base_mechanism requested_base_mechanism decoder approximation_family approximation_rank
representative_count privacy_model epsilon epsilon_y epsilon_r epsilon_other coordinate_noise_dimension
coordinate_noise_norm expected_noise_norm scalar_noise_count scalar_noise_scale report_dimension no_release
noise_stream coupling_group coupling_method radius_budget_fraction ablation D_proxy D_proxy_per_point
B_directional directional_method mechanism_certificate_uncapped universal_cap mechanism_certificate_capped
selection_certificate selected_branch candidate_set selection_origin operating_point OPT expected_random_cost
random_minus_OPT fixed_matching_cost mean_output_cost mean_excess_cost excess_per_point std_excess se_excess K
ci95_low ci95_high normalized_mean_regret normalized_ci95_low normalized_ci95_high excess_q05 excess_q50
excess_q95 excess_over_nL epsilon_excess_over_n2 mean_wrong_cluster_fraction mean_clipped_radius_fraction
mean_oscillation_original mean_oscillation_proxy mean_realization_bound mean_proxy_regret mean_mean_noise_norm
mean_max_reconstruction_error provenance source_arm_id source_row_id reuse_reason status pointwise_checked
pointwise_ok runtime_seconds failure_reason checkpoint_path legacy_missing_fields uncertainty_scope interval_definition
matching_tie_rule numpy_version scipy_version'''.split()


def summarize(row):
    meta, samples = dict(row["metadata"]), row["samples"]
    x = np.asarray(samples["excess"], dtype=float)
    if meta.get("status") == "certificate_only" and not len(x):
        return dict(meta, K=0, uncertainty_scope="not_simulated")
    if not len(x) or not np.all(np.isfinite(x)) or np.any(x < 0):
        raise ValueError("invalid archive excess samples")
    deterministic = (meta.get("no_release") or meta.get("provenance") == "deterministic"
                     or meta.get("base_mechanism") == "proxy")
    if deterministic:
        x = x[:1]
    mean, K = float(x.mean()), len(x)
    sd = 0. if deterministic else float(x.std(ddof=1)) if K > 1 else None
    se = sd/np.sqrt(K) if sd is not None else None
    width = 0. if deterministic else float(t.ppf(.975, K-1)*se) if K > 1 else None
    lo, hi = (mean-width, mean+width) if width is not None else (None, None)
    meta.update(mean_excess_cost=mean, mean_output_cost=meta["OPT"]+mean,
                excess_per_point=mean/meta["n"], std_excess=sd, se_excess=se, K=K,
                ci95_low=lo, ci95_high=hi, uncertainty_scope="conditional_on_fixed_instance_mechanism_noise",
                interval_definition="pointwise Student-t 95% mean CI; untruncated; deterministic point interval",
                excess_q05=float(np.quantile(x, .05)), excess_q50=float(np.quantile(x, .5)),
                excess_q95=float(np.quantile(x, .95)))
    gap = meta.get("random_minus_OPT")
    for name, value in [("normalized_mean_regret", mean), ("normalized_ci95_low", lo), ("normalized_ci95_high", hi)]:
        meta[name] = value/gap if gap is not None and gap > 0 and value is not None else None
    if meta.get("L"):
        meta.update(excess_over_nL=mean/(meta["n"]*meta["L"]),
                    epsilon_excess_over_n2=meta["epsilon"]*mean/meta["n"]**2)
    for key, values in samples.items():
        if key in {"excess", "output_cost"}:
            continue
        finite = [v for v in values if v is not None]
        meta[f"mean_{key}"] = float(np.mean(finite)) if finite else None
    if meta.get("D_proxy") is not None:
        meta["D_proxy_per_point"] = meta["D_proxy"]/meta["n"]
    return meta


def legacy_followup(record, manifest, path):
    n, d, s, eps = (record[k] for k in ["n", "d", "s", "epsilon"])
    seed = manifest.get("scientific", {}).get("base", {}).get("root_seed")
    fixed = record["arms"].get("fixed")
    cap = fixed.get("expected_bound") if fixed else None
    common = dict(experiment_version="anchor-uniform-cube-v1", fingerprint=record["fingerprint"],
                  study_id="frozen_cube_followup", geometry_family="uniform_cube",
                  instance_id=f"cube:seed{seed}:s{s}:n{n}:d{d}", root_seed=seed,
                  instance_seed=None, geometry_replicate=s, configuration_label=f"cube_configuration_{s+1}",
                  n=n, d=d, public_affine_rank=record.get("public_rank"), OPT=record["OPT"],
                  expected_random_cost=record["random_baseline"], random_minus_OPT=record["gap"],
                  fixed_matching_cost=record["OPT"]+fixed["excess"][0] if fixed else None,
                  public_diameter=cap/n if cap is not None else None,
                  scale_convention="Uniform([0,1]^d); frozen master slices", fixed_instance_sweep=True,
                  scaled_changing_instance=False, geometry_randomized=True, epsilon=eps,
                  checkpoint_path=str(path), status=record["status"],
                  legacy_missing_fields="rank tolerance, realized RMS, per-draw reconstruction/oscillation/noise/cluster diagnostics unavailable",
                  numpy_version=manifest.get("numpy"), scipy_version=manifest.get("scipy"))
    sources = {}
    # Baselines first, so aliases point to a readable source when available.
    items = sorted(record["arms"].items(), key=lambda pair: (pair[0] not in {"ambient", "folded", "anchor", "fixed", "phi_infinity"}, pair[0]))
    for name, old in items:
        base, family, value = old["base_mechanism"], old["family"], old["value"]
        no_release = old["no_release"]
        stream = old.get("random_stream")
        source = sources.get(stream)
        # Only the same documented stream AND exactly identical arrays establish reuse.
        reused = source is not None and record["arms"][source]["excess"] == old["excess"]
        sources.setdefault(stream, name)
        dim = 0 if no_release else d if base == "ambient" else n if base == "phi_infinity" else n-1 if base == "centered" else None if base == "exponential" else old["m"]
        er, ey, eo = old["epsilon_r"], old["epsilon_y"], old["epsilon_other"]
        norm = None if no_release or base == "exponential" else "linf" if base == "phi_infinity" else "oscillation_half_range" if base == "centered" else "l2"
        meta = dict(common, arm_id=name, arm_label=name, base_mechanism=base, approximation_family=family,
                    approximation_rank=value if family == "pca" else None,
                    representative_count=value if family == "representatives" else None,
                    privacy_model=old["privacy_model"], epsilon_y=ey, epsilon_r=er, epsilon_other=eo,
                    coordinate_noise_dimension=dim, coordinate_noise_norm=norm,
                    expected_noise_norm=dim/(ey or eo) if dim and (ey or eo) else None,
                    scalar_noise_count=int(er > 0), scalar_noise_scale=1/er if er else None,
                    report_dimension=old["report_dimension"], no_release=no_release,
                    noise_stream=stream, D_proxy=old["D_proxy"],
                    B_directional=old["B_directional"] if base in {"anchor", "folded"} else None,
                    directional_method="nonzero_secant_pair_cardinality_bound" if base == "anchor" else None,
                    mechanism_certificate_uncapped=old["expected_bound"], universal_cap=cap,
                    mechanism_certificate_capped=min(cap, old["expected_bound"]) if cap is not None else None,
                    selection_certificate=old.get("selection_bound"), selected_branch=old.get("selected_branch"),
                    selection_origin="public_certificate" if old.get("selection_bound") is not None else None,
                    candidate_set=("folded,anchor,fixed" if name == "auto" else f"all integer {family} candidates") if old.get("selection_bound") is not None else None,
                    provenance="derived_reused" if reused else "deterministic" if no_release else "simulated",
                    source_arm_id=source if reused else None, runtime_seconds=old.get("elapsed_seconds"),
                    pointwise_checked=old.get("pointwise_checked"), pointwise_ok=old.get("pointwise_ok"),
                    decoder="private_anchor_distance_inner_product" if base == "anchor" else "euclidean_report_distances" if base in {"ambient", "folded"} else base,
                    mean_clipped_radius_fraction=old.get("clipped_radius_fraction") if base == "anchor" and not no_release else None)
        yield dict(metadata=meta, samples={"excess": old["excess"]})


def legacy_baseline(records, manifest, paths):
    first = records[0]
    # Split blocks must be disjoint and contiguous; never silently double-count retries.
    records.sort(key=lambda r: r["k_start"])
    cursor = 0
    for r in records:
        if r["k_start"] != cursor:
            raise ValueError("overlapping or missing legacy noise blocks")
        cursor = r["k_end"]
    n, d, s = first["n"], first["d"], first["s"]
    seed = manifest.get("scientific_identity", {}).get("root_seed")
    for base in first["mechanisms"]:
        for eps in first["epsilons"]:
            from private_matching.config import epsilon_key
            key = epsilon_key(eps)
            samples = [x for r in records for x in r["excess"][base][key]]
            dim = d if base == "ambient" else n if base == "phi_infinity" else first["m_fold"]
            meta = dict(experiment_version="frozen-three-mechanism-v2", fingerprint=first["scientific_fingerprint"],
                        study_id="frozen_cube_baseline", geometry_family="uniform_cube", n=n, d=d,
                        instance_id=f"cube:seed{seed}:s{s}:n{n}:d{d}", root_seed=seed,
                        geometry_replicate=s, configuration_label=f"cube_configuration_{s+1}",
                        public_affine_rank=first["k_span"], numerical_rank_tolerance=first.get("rank_tol"),
                        epsilon=float(eps), OPT=first["O"],
                        expected_random_cost=first["B"], random_minus_OPT=first["g"], arm_id=base,
                        arm_label=base, base_mechanism=base, approximation_family="exact", privacy_model="local",
                        epsilon_y=eps if base != "phi_infinity" else 0., epsilon_other=eps if base == "phi_infinity" else 0.,
                        epsilon_r=0., scalar_noise_count=0, coordinate_noise_dimension=dim,
                        coordinate_noise_norm="linf" if base == "phi_infinity" else "l2", expected_noise_norm=dim/eps,
                        report_dimension=dim, no_release=False, mechanism_certificate_uncapped=2*n*dim/eps,
                        provenance="simulated", status="complete" if cursor == first["K_requested"] else "partial_noise_blocks",
                        fixed_instance_sweep=True, scaled_changing_instance=False, geometry_randomized=True,
                        scale_convention="Uniform([0,1]^d); frozen master slices",
                        checkpoint_path=";".join(str(p) for p in paths),
                        noise_stream=f"PCG64_SeedSequence_v1:{seed}:{s}:{n}:{d}:{base}:{eps}",
                        decoder="euclidean_report_distances" if base != "phi_infinity" else "distance_vector",
                        legacy_missing_fields="RMS/proxy residual/per-draw diagnostics unavailable; public diameter/fixed cost recovered from followup where available")
            yield dict(metadata=meta, samples={"excess": samples})


def export_table(inputs, output):
    refuse_login_compute("aggregate research checkpoints")
    output = Path(output)
    registry = output/"sources.json"
    existing = read_json(registry).get("directories", []) if registry.exists() else []
    directories = sorted(set(existing + [str(Path(p).resolve()) for p in inputs]))
    rows = []
    for directory in directories:
        directory = Path(directory)
        manifest = read_json(directory/"manifest.json")
        blocks, block_paths = {}, {}
        for path in sorted((directory/"units").glob("*.json")):
            record = read_json(path)
            if record.get("status") != "complete":
                continue
            if "rows" in record:
                source_rows = record["rows"]
            elif "arms" in record:
                source_rows = legacy_followup(record, manifest, path)
            elif "k_start" in record:
                key = (record["scientific_fingerprint"], record["s"], record["n"], record["d"])
                blocks.setdefault(key, []).append(record)
                block_paths.setdefault(key, []).append(path)
                continue
            else:
                raise ValueError(f"unsupported checkpoint {path}")
            for row in source_rows:
                row["metadata"].setdefault("checkpoint_path", str(path))
                rows.append(summarize(row))
        for key, records in blocks.items():
            rows.extend(summarize(r) for r in legacy_baseline(records, manifest, block_paths[key]))
    # Recover geometry constants across verified identical frozen instances, never infer diagnostics.
    constants = {}
    for row in rows:
        if row["study_id"] not in {"frozen_cube_baseline", "frozen_cube_followup"}:
            continue
        bucket = constants.setdefault(row["instance_id"], {})
        for field in ["fixed_matching_cost", "public_diameter", "numerical_rank_tolerance"]:
            if row.get(field) is not None:
                bucket[field] = row[field]
    for row in rows:
        for field, value in constants.get(row["instance_id"], {}).items():
            if row.get(field) is None:
                row[field] = value
        if row.get("public_diameter") is not None and row.get("mechanism_certificate_uncapped") is not None:
            row["universal_cap"] = row["n"]*row["public_diameter"]
            row["mechanism_certificate_capped"] = min(row["universal_cap"], row["mechanism_certificate_uncapped"])
    # Old baseline draws were reused by the old followup runner. Mark cross-study reuse
    # only after equality of identity, seed protocol, sample count AND summaries.
    baseline = {(r["instance_id"], r["epsilon"], r["arm_id"]): r for r in rows if r["study_id"] == "frozen_cube_baseline"}
    for row in rows:
        source = baseline.get((row["instance_id"], row["epsilon"], row["arm_id"]))
        if row["study_id"] == "frozen_cube_followup" and source and all(row[k] == source[k] for k in ["K", "mean_excess_cost", "std_excess", "excess_q05", "excess_q50", "excess_q95"]):
            row.update(provenance="derived_reused", source_arm_id=source["arm_id"],
                       source_row_id=f"{source['fingerprint']}:{source['instance_id']}:{source['epsilon']}:{source['arm_id']}",
                       reuse_reason="frozen baseline noise protocol reused by followup; matching summaries")
    columns = REQUIRED + sorted(set().union(*(r.keys() for r in rows))-set(REQUIRED)) if rows else REQUIRED
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_bytes(output/"analysis_table.csv", stream.getvalue().encode())
    atomic_write_json(registry, dict(directories=directories, row_count=len(rows)))
    from private_matching.table_dictionary import dictionary
    atomic_write_bytes(output/"data_dictionary.md", dictionary(columns).encode())
    print(f"Canonical table: {output/'analysis_table.csv'} ({len(rows)} rows)", flush=True)
    return rows
