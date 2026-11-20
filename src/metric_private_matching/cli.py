"""Independent commands for the paper experiments and CSV-only plotting."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent / "reproduction"


def _run(script, *arguments):
    env = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    env.setdefault("MPLBACKEND", "Agg")
    subprocess.run([sys.executable, str(ROOT / "scripts" / script), *map(str, arguments)],
                   env=env, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Verify packaged frozen code, inputs, and saved results.")
    commands.add_parser("datasets", help="List the sixteen benchmark dataset identifiers.")
    fetch = commands.add_parser("fetch-data", help="Download source data to the external cache; verify SHA256s.")
    fetch.add_argument("--dataset", default="all", help="Dataset identifier, e.g. mnist/fashion10k, or all.")
    check = commands.add_parser("check-inputs", help="Fetch and verify all raw sources and sampled input artifacts.")
    check.add_argument("--output", required=True, type=Path)
    saved = commands.add_parser("saved-results", help="Export paper results as a tidy CSV without simulating mechanisms.")
    saved.add_argument("--experiment", choices=("all", "benchmarks", "dimension-sweep", "size-sweep", "sonar"), default="all")
    saved.add_argument("--output", type=Path, default=Path("outputs/paper"))
    plot = commands.add_parser("plot", help="Render figures using only a summary CSV.")
    plot.add_argument("--csv", required=True, type=Path)
    plot.add_argument("--output", required=True, type=Path)
    plot.add_argument("--view", choices=("curves", "bars"), default="curves")
    original = commands.add_parser("paper-figures", help="Rebuild the eight original paper figures from saved results.")
    original.add_argument("--output", type=Path, default=Path("outputs/paper-figures"))
    for name in ("benchmarks", "dimension-sweep", "size-sweep", "sonar"):
        command = commands.add_parser(name, help=f"Rerun {name}; quick demonstration or full paper configuration.")
        command.add_argument("--mode", choices=("quick", "paper"), default="quick")
        command.add_argument("--workers", default="auto", help="Conservative automatic process count, or a positive integer.")
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--plan", action="store_true", help="Show settings and workload without computing or downloading.")
        command.add_argument("--stage", choices=("all", "prepare", "run", "aggregate"), default="all")
        command.add_argument("--shard-index", type=int, default=0)
        command.add_argument("--shard-count", type=int, default=1)
        if name == "benchmarks":
            command.add_argument("--dataset", default="all", help="One dataset identifier or all sixteen.")
            command.add_argument("--objective", choices=("euclidean", "squared", "both"), default="both")
        if name == "sonar":
            command.add_argument("--regenerate-null", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "verify":
        _run("verify.py")
    elif args.command == "datasets":
        _run("run_study.py", "--list-datasets")
    elif args.command == "fetch-data":
        _run("fetch_data.py", "--dataset", args.dataset)
    elif args.command == "check-inputs":
        _run("fetch_data.py")
        _run("check_inputs.py", "--output", args.output.resolve())
    elif args.command == "plot":
        _run("plot_csv.py", "--csv", args.csv.resolve(), "--output", args.output.resolve(), "--view", args.view)
    elif args.command == "saved-results":
        _run("summarize.py", "--saved", "--experiment", args.experiment, "--output", args.output.resolve())
    elif args.command == "paper-figures":
        _run("fetch_data.py", "--dataset", "uci/sonar")
        _run("rebuild.py", "--output", args.output.resolve())
    else:
        from .execution import workers
        try:
            count = workers(args.workers)
        except ValueError as error:
            parser.error(str(error))
        if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
            parser.error("Require shard-count >= 1 and 0 <= shard-index < shard-count.")
        if args.shard_count > 1 and args.stage == "all":
            parser.error("For shards, prepare once, run each shard, then aggregate once.")
        if not args.plan:
            selection = dict(experiment=args.command, mode=args.mode,
                dataset=getattr(args, 'dataset', None), objective=getattr(args, 'objective', None),
                regenerate_null=getattr(args, 'regenerate_null', False))
            selection_path = args.output.resolve()/'run_selection.json'
            if selection_path.exists():
                if json.loads(selection_path.read_text()) != selection:
                    parser.error('This output directory belongs to another configuration; use a fresh directory.')
            elif args.stage in ('all', 'prepare'):
                selection_path.parent.mkdir(parents=True, exist_ok=True)
                selection_path.write_text(json.dumps(selection, indent=2)+'\n')
        studies = {"dimension-sweep": ["fashion"], "size-sweep": ["euclidean"], "sonar": ["sonar"]}
        if args.command == "benchmarks":
            studies["benchmarks"] = ["euclidean", "squared"] if args.objective == "both" else [args.objective]
        for study in studies[args.command]:
            destination = args.output.resolve() / study
            command = ["--study", study, "--experiment", args.command, "--mode", args.mode,
                       "--output", destination, "--jobs", count, "--stage", args.stage,
                       "--shard-index", args.shard_index, "--shard-count", args.shard_count]
            if args.command == "benchmarks":
                command += ["--dataset", args.dataset]
            if getattr(args, "regenerate_null", False):
                command.append("--regenerate-null")
            if args.plan:
                command.append("--plan")
            _run("run_study.py", *command)
        if not args.plan and args.stage in ("all", "aggregate"):
            _run("summarize.py", "--input", args.output.resolve(), "--experiment", args.command,
                 "--output", args.output.resolve())


if __name__ == "__main__":
    main()
