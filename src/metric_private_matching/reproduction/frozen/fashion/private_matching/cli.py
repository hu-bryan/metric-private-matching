"""CLI: prepare, validate, run, aggregate, plot. There is no --stage all."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from private_matching.aggregate import aggregate_results, completeness_report
from private_matching.config import ConfigError, load_bundle, resolved_table
from private_matching.execution import ResumeConflict, run_specs, specs_for_run
from private_matching.login_guard import refuse_login_compute
from private_matching.manifest import build_manifest, load_manifest, write_manifest
from private_matching.plotting import expected_main_figure_count, plot_all
from private_matching.results import atomic_write_json


REPO_ROOT = Path(__file__).resolve().parents[2]


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        required=True,
        help="YAML configuration file (repeat to merge; later files override).",
    )
    parser.add_argument(
        "--profile",
        choices=("production", "pilot", "local", "test"),
        default=None,
        help="Extra validation profile. production/pilot freeze master dimensions and S=3.",
    )
    parser.add_argument("--results-root", default=None, help="Override execution.results_root.")
    parser.add_argument("--workers", type=int, default=None, help="Override execution.local_workers.")
    parser.add_argument("--serial", action="store_true", help="Force one process.")
    parser.add_argument("--no-resume", action="store_true", help="Recompute even if complete units exist.")
    parser.add_argument("--array-task-id", type=int, default=None)
    parser.add_argument("--array-task-count", type=int, default=None)


def _bundle_from_args(args: argparse.Namespace):
    overrides: dict[str, Any] = {}
    if args.results_root:
        overrides["results_root"] = args.results_root
    if args.workers is not None:
        overrides["local_workers"] = int(args.workers)
    if args.serial:
        overrides["serial"] = True
        overrides["local_workers"] = 1
    if args.no_resume:
        overrides["resume"] = False
    if args.array_task_id is not None:
        overrides["array_task_id"] = int(args.array_task_id)
    if args.array_task_count is not None:
        overrides["array_task_count"] = int(args.array_task_count)
    return load_bundle(args.configs, profile=args.profile, execution_overrides=overrides or None)


def _print_table(rows: Sequence[tuple[str, str]]) -> None:
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"  {name:<{width}}  {value}")


def cmd_validate(args: argparse.Namespace) -> int:
    bundle = _bundle_from_args(args)
    print("Resolved configuration")
    _print_table(resolved_table(bundle))
    print("Workload counts (computed, not hardcoded):")
    for key, value in bundle.scientific.workload_counts().items():
        print(f"  {key}: {value}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    bundle = _bundle_from_args(args)
    root = bundle.execution.results_root
    if root is None or "REPLACE_ME" in str(root):
        raise ConfigError(
            "execution.results_root is unset or still a REPLACE_ME placeholder. "
            "Pass --results-root to a real directory when preparing a cluster manifest."
        )
    results_dir = bundle.results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    n_batches = bundle.execution.array_task_count or args.batches or 1
    manifest = build_manifest(bundle, repo_root=REPO_ROOT, n_batches=int(n_batches))
    path = results_dir / "manifest.json"
    write_manifest(path, manifest)
    atomic_write_json(results_dir / "resolved_config.json", dict(resolved_table(bundle)))
    print(f"Wrote {path}")
    print(f"Scientific fingerprint: {manifest['scientific_fingerprint']}")
    print(f"Work units: {manifest['counts']['work_units']}")
    print(f"Batches: {len(manifest['batches'])}")
    _print_table(resolved_table(bundle))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    bundle = _bundle_from_args(args)
    root = bundle.execution.results_root
    if root is None or "REPLACE_ME" in str(root):
        raise ConfigError(
            "execution.results_root is unset or still a REPLACE_ME placeholder. "
            "Set a real directory (typically under /vast/arahimian/brh314) before running."
        )
    refuse_login_compute("run the matching simulation")
    results_dir = bundle.results_dir()
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.manifest) if args.manifest else results_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = build_manifest(
            bundle,
            repo_root=REPO_ROOT,
            n_batches=bundle.execution.array_task_count or 1,
        )
        write_manifest(manifest_path, manifest)
    specs = specs_for_run(bundle, manifest_path)
    print(f"Running {len(specs)} configuration work units into {results_dir}")
    outcomes = run_specs(
        bundle,
        specs,
        config_paths=args.configs,
        profile=args.profile,
        progress=lambda msg: print(msg, flush=True),
    )
    wrote = sum(1 for item in outcomes if item["status"] == "wrote")
    skipped = sum(1 for item in outcomes if item["status"] == "skipped_complete")
    print(f"Wrote {wrote}, resumed/skipped {skipped}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    bundle = _bundle_from_args(args)
    report = completeness_report(bundle, bundle.results_dir())
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["complete"] else 2


def cmd_aggregate(args: argparse.Namespace) -> int:
    bundle = _bundle_from_args(args)
    results_dir = bundle.results_dir()
    payload = aggregate_results(bundle, results_dir)
    out = results_dir / "aggregate.json"
    atomic_write_json(out, payload)
    completeness = payload["completeness"]
    atomic_write_json(results_dir / "completeness.json", completeness)
    print(f"Wrote {out}")
    print(
        f"Complete work units: {completeness['complete_work_units']}/"
        f"{completeness['expected_work_units']}"
    )
    if not completeness["complete"]:
        print("WARNING: aggregation is incomplete and must not be treated as final.")
        return 2
    return 0


def cmd_plot(args: argparse.Namespace) -> int:
    bundle = _bundle_from_args(args)
    results_dir = bundle.results_dir()
    agg_path = Path(args.aggregate) if args.aggregate else results_dir / "aggregate.json"
    if not agg_path.exists():
        payload = aggregate_results(bundle, results_dir)
        atomic_write_json(agg_path, payload)
    else:
        payload = json.loads(agg_path.read_text(encoding="utf-8"))
    smoke = bool(args.smoke) or "smoke" in bundle.scientific.name
    out_dir = Path(args.out_dir) if args.out_dir else results_dir / "figures"
    paths = plot_all(
        bundle,
        payload,
        out_dir=out_dir,
        s_filter=args.s,
        epsilon_filter=args.epsilon,
        smoke=smoke,
    )
    pngs = [p for p in paths if p.suffix == ".png"]
    expected = expected_main_figure_count(bundle) if args.s is None and args.epsilon is None else len(pngs)
    print(f"Wrote {len(pngs)} PNG figures (expected {expected} for a full plot command)")
    for path in pngs:
        print(f"  {path}")
    if payload.get("completeness") and not payload["completeness"].get("complete", True):
        print("WARNING: figures were generated from incomplete results and are labeled as such.")
        return 2
    return 0


def cmd_counts(args: argparse.Namespace) -> int:
    bundle = _bundle_from_args(args)
    print(json.dumps(bundle.scientific.workload_counts(), indent=2))
    return 0


def cmd_followup(args: argparse.Namespace) -> int:
    refuse_login_compute("run or plot the follow-up study")
    from private_matching.followup import load_followup, fingerprint, result_directory, run_followup, log
    from private_matching.followup_plotting import plot_followup
    from private_matching.results import read_json
    cfg = load_followup(args.config)
    if args.directory is not None:
        if not args.plot_only:
            raise ConfigError("--directory is only valid with --plot-only")
        directory = Path(args.directory).resolve()
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise ConfigError(f"missing follow-up manifest: {manifest_path}")
        fp = str(read_json(manifest_path)["fingerprint"])
    else:
        if args.results_root is None:
            raise ConfigError("pass --results-root, or --plot-only --directory pointing at existing results")
        fp = fingerprint(cfg)
        directory = result_directory(args.results_root, cfg, fp)
    log(f"Resolved follow-up results: {directory}")
    if not args.plot_only:
        run_followup(cfg, directory, fp, args.workers)
    plot_followup(cfg, directory, fp)
    return 0


def cmd_study(args: argparse.Namespace) -> int:
    if args.study in ("fashion_dimension", "fashion_dimension_calibrated"):
        from private_matching.fashion_dimension import run_study
        run_study(args.results_root,args.pilot,args.stage,args.shard_index,args.shard_count,args.directory,
                  calibrated=args.study=="fashion_dimension_calibrated",epsilon=args.epsilon)
        return 0
    if args.study in ("mnist_random_projection", "mnist_random_projection_unnormalized"):
        from private_matching.mnist_random_projection import run_study
        run_study(args.results_root,args.pilot,args.stage,args.shard_index,args.shard_count,args.directory,
                  unnormalized=args.study == "mnist_random_projection_unnormalized")
        return 0
    if args.study == "controlled_sweeps":
        from private_matching.controlled_sweeps import run_study
        run_study(args.results_root,args.pilot,args.stage,args.shard_index,args.shard_count,args.directory)
        return 0
    if args.study == "clustering_manuscript":
        from private_matching.clustering_manuscript import run_study as run_manuscript
        run_manuscript(args.results_root, args.pilot, args.stage, args.shard_index, args.shard_count, args.directory)
        return 0
    if args.study == "clustering_normalized":
        from private_matching.clustering_normalized import run_study as run_normalized
        run_normalized(args.results_root, args.pilot, args.stage, args.shard_index, args.shard_count, args.directory)
        return 0
    if args.study == "corrected_projection_benchmark":
        from private_matching.corrected_projection_benchmark import run_study as run_corrected
        run_corrected(args.results_root,args.pilot,args.stage,args.shard_index,args.shard_count,args.directory)
        return 0
    if args.study == "sonar_rank":
        from private_matching.sonar_rank import run_study as run_sonar
        run_sonar(args.results_root,args.source_data,args.pilot,args.stage,args.shard_index,args.shard_count,args.directory)
        return 0
    if args.study == "clustering_extension":
        from private_matching.clustering_extension import run_study as run_extension
        run_extension(args.results_root, args.source_data, args.pilot, args.stage,
                      args.shard_index, args.shard_count, args.directory)
        return 0
    if args.study == "clustering_benchmark":
        from private_matching.clustering_benchmark import run_study as run_benchmark
        run_benchmark(args.results_root, args.source_data, args.pilot, args.stage,
                      args.shard_index, args.shard_count, args.directory)
        return 0
    if args.study == "integral_adaptive":
        from private_matching.integral_adaptive_study import run_study as run_adaptive
        if args.directory:
            raise SystemExit("Use --results-root with the same phase for adaptive finalization")
        run_adaptive(args.results_root,args.pilot,args.stage,args.shard_index,args.shard_count)
        return 0
    if args.study == "integral_matching_comparison":
        from private_matching.integral_study import run_study as run_integral
        if args.directory:
            if args.stage != "finalize":
                raise SystemExit("--directory requires --stage finalize")
            from private_matching.integral_report import finalize
            finalize(Path(args.directory))
        else:
            run_integral(args.results_root, args.pilot, args.stage, args.shard_index, args.shard_count)
        return 0
    from private_matching.study import finalize_study, run_study
    if args.directory:
        if args.stage != "finalize":
            raise SystemExit("--directory is only valid with --stage finalize")
        finalize_study(args.directory)
        return 0
    run_study(args.results_root, args.pilot, args.study, args.centered, args.central,
              args.stage, args.shard_index, args.shard_count)
    return 0


def cmd_analysis_table(args: argparse.Namespace) -> int:
    fashion = [Path(p) for p in args.input if (Path(p)/"manifest.json").exists()
               and json.loads((Path(p)/"manifest.json").read_text()).get("kind","").startswith(("fashion_dimension_v1","fashion_dimension_epsilon"))]
    if fashion:
        if len(args.input)!=1:raise SystemExit("Fashion export requires one fingerprint")
        from private_matching.fashion_dimension_report import export
        export(fashion[0],Path(args.output))
        return 0
    random_mnist = [Path(p) for p in args.input if (Path(p)/"manifest.json").exists()
                    and json.loads((Path(p)/"manifest.json").read_text()).get("kind") in
                    ("mnist_random_projection_v1", "mnist_random_projection_unnormalized_v1")]
    if random_mnist:
        if len(args.input)!=1:raise SystemExit("Random-projection export requires one fingerprint")
        from private_matching.mnist_random_projection_report import export
        export(random_mnist[0],Path(args.output))
        return 0
    controlled = [Path(p) for p in args.input if (Path(p)/"manifest.json").exists()
                  and json.loads((Path(p)/"manifest.json").read_text()).get("kind") == "controlled_circle_sweeps_v1"]
    if controlled:
        if len(args.input) != 1:
            raise SystemExit("Controlled sweep export requires one phase/fingerprint")
        from private_matching.controlled_sweeps_report import export
        export(controlled[0],Path(args.output))
        return 0
    manuscript = [Path(p) for p in args.input if (Path(p)/"manifest.json").exists()
                  and json.loads((Path(p)/"manifest.json").read_text()).get("kind") == "clustering_manuscript_v1"]
    if manuscript:
        if len(args.input) != 1:
            raise SystemExit("Manuscript benchmark export requires one phase/fingerprint")
        from private_matching.clustering_manuscript_report import export
        export(manuscript[0], Path(args.output))
        return 0
    normalized = [Path(p) for p in args.input if (Path(p)/"manifest.json").exists()
                  and json.loads((Path(p)/"manifest.json").read_text()).get("kind") == "clustering_normalized_v1"]
    if normalized:
        if len(args.input) != 1:
            raise SystemExit("Normalized benchmark export requires one phase/fingerprint")
        from private_matching.clustering_normalized_report import export
        export(normalized[0], Path(args.output))
        return 0
    corrected = [Path(p) for p in args.input if (Path(p)/"manifest.json").exists()
                 and json.loads((Path(p)/"manifest.json").read_text()).get("kind") == "corrected_projection_benchmark_v1"]
    if corrected:
        if len(args.input) != 1:
            raise SystemExit("Corrected projection export requires one phase/fingerprint")
        from private_matching.corrected_projection_benchmark import export
        export(corrected[0],Path(args.output))
        return 0
    sonar=[Path(p) for p in args.input if (Path(p)/"manifest.json").exists()
           and json.loads((Path(p)/"manifest.json").read_text()).get("kind")=="sonar_rank_energy_v1"]
    if sonar:
        if len(args.input)!=1:raise SystemExit("Sonar export requires exactly one phase/fingerprint")
        from private_matching.sonar_rank_report import export
        export(sonar[0],Path(args.output))
        return 0
    extension = [Path(p) for p in args.input if (Path(p)/"manifest.json").exists()
                 and json.loads((Path(p)/"manifest.json").read_text()).get("kind") == "clustering_benchmark_extension_v1"]
    if extension:
        if len(args.input) != 1:
            raise SystemExit("Extension export requires one phase/fingerprint")
        from private_matching.clustering_extension_report import export
        export(extension[0], Path(args.output))
        return 0
    benchmark = [Path(p) for p in args.input if (Path(p)/"manifest.json").exists()
                 and json.loads((Path(p)/"manifest.json").read_text()).get("kind") == "clustering_benchmark_v1"]
    if benchmark:
        if len(args.input) != 1:
            raise SystemExit("Benchmark export requires one phase/fingerprint; pooling is forbidden")
        from private_matching.clustering_benchmark_report import export
        export(benchmark[0], Path(args.output))
        return 0
    adaptive=[Path(p) for p in args.input if (Path(p)/"manifest.json").exists()
              and json.loads((Path(p)/"manifest.json").read_text()).get("kind")=="integral_adaptive_v1"]
    if adaptive:
        if len(args.input)!=1:raise SystemExit("Adaptive export cannot pool phases or fingerprints")
        from private_matching.integral_adaptive_study import export as export_adaptive
        export_adaptive(adaptive[0],Path(args.output))
        return 0
    integral = [Path(p) for p in args.input if (Path(p)/"manifest.json").exists()
                and json.loads((Path(p)/"manifest.json").read_text()).get("kind") == "integral_matching_comparison_v1"]
    if integral:
        if len(args.input) != 1:
            raise SystemExit("Integral matching export requires exactly one phase/fingerprint; pooling is forbidden")
        from private_matching.integral_report import export
        export(integral[0], Path(args.output))
        return 0
    from private_matching.analysis_table import export_table
    export_table(args.input, args.output)
    return 0


def cmd_causal_ihdp(args: argparse.Namespace) -> int:
    refuse_login_compute("run the IHDP causal experiment")
    from private_matching.causal_ihdp import load_config, run_all

    cfg = load_config(args.config)
    if args.workers is not None:
        cfg["workers"] = int(args.workers)
    if args.output_root is not None:
        cfg["output_root"] = args.output_root
    stages = ("pilot", "production", "report") if args.stage == "all" else (args.stage,)
    root = run_all(cfg, stages=stages)
    print(f"Outputs: {root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="private-matching",
        description=(
            "Metric-private matching simulation, including the anchor-distance follow-up. "
            "This CLI replaces the retired AIMD --stage interface."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    study = sub.add_parser("study", help="Compact versioned research suite (compute only; frozen cube runner unchanged).")
    study.add_argument("--results-root", required=True)
    study.add_argument("--pilot", action="store_true")
    study.add_argument("--epsilon",type=float,default=None,help="Requested budget for a Fashion-MNIST dimension study; geometry and unit noise are reused from epsilon 8.")
    study.add_argument("--study", choices=["appendix_scaled", "controlled_epsilon", "controlled_geometry", "structured_geometry", "cube_diagnostics", "central_benchmark",
                                           "projection_epsilon", "cube_approximation", "psi_lower_bound", "interchangeability", "dimension_tradeoff",
                                           "robust_fitted", "continuous_decoder", "integral_matching_comparison", "integral_adaptive", "clustering_benchmark", "clustering_extension", "corrected_projection_benchmark", "clustering_normalized", "sonar_rank", "clustering_manuscript", "controlled_sweeps", "mnist_random_projection", "mnist_random_projection_unnormalized", "fashion_dimension", "fashion_dimension_calibrated"],
                       help="projection_epsilon, cube_approximation, robust_fitted and continuous_decoder are opt-in, never part of the default suite.")
    study.add_argument("--centered", action="store_true", help="Opt-in centered-distance baseline.")
    study.add_argument("--source-data", default=None, help="Pinned source snapshot for the opt-in clustering benchmark.")
    study.add_argument("--central", action="store_true", help="Opt-in central enumeration, only for n<=8.")
    study.add_argument("--stage", choices=["all", "prepare", "shard", "finalize"], default="all",
                       help="all: serial run (original). prepare/shard/finalize: Slurm array workflow.")
    study.add_argument("--shard-index", type=int, default=None, help="Shard number (e.g. SLURM_ARRAY_TASK_ID).")
    study.add_argument("--shard-count", type=int, default=None, help="Total shards; each owns whole instances.")
    study.add_argument("--directory", default=None, help="Finalize this existing fingerprint directory.")
    study.set_defaults(func=cmd_study)

    table = sub.add_parser("analysis-table", help="Import archival and new checkpoints without simulations (compute only).")
    table.add_argument("--input", action="append", required=True)
    table.add_argument("--output", required=True)
    table.set_defaults(func=cmd_analysis_table)

    followup = sub.add_parser("followup", help="Extended mechanisms, epsilon and approximation sweeps (compute only).")
    followup.add_argument("--config", required=True)
    followup.add_argument("--results-root", default=None)
    followup.add_argument("--directory", default=None,
                          help="Existing result directory. Plot-only; uses the stored fingerprint.")
    followup.add_argument("--workers", type=int, default=1)
    followup.add_argument("--plot-only", action="store_true", help="Require complete checkpoints and regenerate figures.")
    followup.set_defaults(func=cmd_followup)

    causal = sub.add_parser("causal-ihdp", help="Private 1:1 matching for ATT on IHDP (causal_exp_plan.md; compute only).")
    causal.add_argument("--config", default=None, help="YAML overrides (default: built-in plan values).")
    causal.add_argument("--stage", choices=["all", "pilot", "production", "report"], default="all")
    causal.add_argument("--workers", type=int, default=None)
    causal.add_argument("--output-root", default=None)
    causal.set_defaults(func=cmd_causal_ihdp)

    validate = sub.add_parser("validate", help="Load and print the resolved configuration.")
    _add_config_args(validate)
    validate.set_defaults(func=cmd_validate)

    prepare = sub.add_parser("prepare", help="Write the workload manifest.")
    _add_config_args(prepare)
    prepare.add_argument("--batches", type=int, default=None, help="Number of array batches to record.")
    prepare.set_defaults(func=cmd_prepare)

    run = sub.add_parser("run", help="Run configuration work units (compute node only).")
    _add_config_args(run)
    run.add_argument("--manifest", default=None)
    run.set_defaults(func=cmd_run)

    status = sub.add_parser("status", help="Completeness report for an existing result directory.")
    _add_config_args(status)
    status.set_defaults(func=cmd_status)

    agg = sub.add_parser("aggregate", help="Per-configuration noise means and intervals.")
    _add_config_args(agg)
    agg.set_defaults(func=cmd_aggregate)

    plot = sub.add_parser("plot", help="Write the nine main figures from saved aggregates.")
    _add_config_args(plot)
    plot.add_argument("--aggregate", default=None)
    plot.add_argument("--out-dir", default=None)
    plot.add_argument("--s", type=int, default=None, help="Optional master configuration index.")
    plot.add_argument("--epsilon", type=float, default=None, help="Optional single epsilon.")
    plot.add_argument("--smoke", action="store_true", help="Label figures as smoke tests.")
    plot.set_defaults(func=cmd_plot)

    counts = sub.add_parser("counts", help="Print computed workload counts.")
    _add_config_args(counts)
    counts.set_defaults(func=cmd_counts)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except (ConfigError, ResumeConflict) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
