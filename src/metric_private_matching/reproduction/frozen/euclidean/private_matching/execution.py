"""Serial and local process execution with resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from private_matching.config import ExperimentBundle
from private_matching.login_guard import refuse_login_compute
from private_matching.manifest import (
    all_work_specs,
    load_manifest,
    repetition_blocks,
    select_batch,
    scientific_fingerprint,
)
from private_matching.results import (
    atomic_write_json,
    read_json,
    unit_filename,
    validate_unit_record,
)
from private_matching.simulation import run_work_unit
from private_matching.thread_limits import apply_thread_limits

apply_thread_limits()


class ResumeConflict(RuntimeError):
    """An existing result cannot be reused because scientific identity differs."""


def units_dir(results_dir: Path) -> Path:
    return results_dir / "units"


def unit_path(results_dir: Path, s: int, n: int, d: int, k_start: int, k_end: int) -> Path:
    return units_dir(results_dir) / unit_filename(s, n, d, k_start, k_end)


def _compatible_existing(path: Path, expected_fingerprint: str, server_rules: dict[str, str]) -> bool:
    record = read_json(path)
    validate_unit_record(record, expected_fingerprint=expected_fingerprint)
    if record["ambient_server_rule"] != server_rules["ambient"]:
        raise ResumeConflict(
            f"{path} has ambient_server_rule={record['ambient_server_rule']!r}, "
            f"expected {server_rules['ambient']!r}"
        )
    if record["folded_server_rule"] != server_rules["folded"]:
        raise ResumeConflict(
            f"{path} has folded_server_rule={record['folded_server_rule']!r}, "
            f"expected {server_rules['folded']!r}"
        )
    return True


def run_one_block(
    bundle: ExperimentBundle,
    s: int,
    n: int,
    d: int,
    k_start: int,
    k_end: int,
) -> dict[str, Any]:
    cfg = bundle.scientific
    results_dir = bundle.results_dir()
    path = unit_path(results_dir, s, n, d, k_start, k_end)
    fingerprint = scientific_fingerprint(cfg)
    server_rules = {
        "ambient": cfg.ambient_server_rule,
        "folded": cfg.folded_server_rule,
    }
    if bundle.execution.resume and path.exists():
        _compatible_existing(path, fingerprint, server_rules)
        return {"path": str(path), "status": "skipped_complete"}
    record = run_work_unit(
        cfg,
        s,
        n,
        d,
        k_start=k_start,
        k_end=k_end,
        record_diagnostics=bundle.execution.record_diagnostics,
    )
    atomic_write_json(path, record)
    return {"path": str(path), "status": "wrote", "total_s": record["timings"]["total_s"]}


def _block_jobs(bundle: ExperimentBundle, specs: Sequence[dict[str, Any]]) -> list[tuple[int, int, int, int, int]]:
    blocks = repetition_blocks(bundle.scientific.noise_repetitions_K, bundle.execution.repetition_block_size)
    jobs = []
    for spec in specs:
        for k_start, k_end in blocks:
            jobs.append((int(spec["s"]), int(spec["n"]), int(spec["d"]), int(k_start), int(k_end)))
    return jobs


def _run_job(payload: dict[str, Any]) -> dict[str, Any]:
    return run_one_block(
        payload["bundle"],
        payload["s"],
        payload["n"],
        payload["d"],
        payload["k_start"],
        payload["k_end"],
    )


def run_specs(
    bundle: ExperimentBundle,
    specs: Sequence[dict[str, Any]],
    *,
    config_paths: Sequence[str],
    profile: str | None,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    refuse_login_compute("run matching simulations")
    apply_thread_limits(bundle.execution.numerical_threads_per_worker, overwrite=True)
    jobs = _block_jobs(bundle, specs)
    workers = 1 if bundle.execution.serial else int(bundle.execution.local_workers)
    payloads = [
        {
            "bundle": bundle,
            "s": s,
            "n": n,
            "d": d,
            "k_start": k_start,
            "k_end": k_end,
        }
        for s, n, d, k_start, k_end in jobs
    ]
    if workers == 1 or len(payloads) == 1:
        outcomes = []
        for payload in payloads:
            outcome = run_one_block(
                bundle,
                payload["s"],
                payload["n"],
                payload["d"],
                payload["k_start"],
                payload["k_end"],
            )
            if progress:
                progress(f"{payload['s']} n={payload['n']} d={payload['d']}: {outcome['status']}")
            outcomes.append(outcome)
        return outcomes

    from joblib import Parallel, delayed, parallel_config

    with parallel_config(backend="loky", inner_max_num_threads=int(bundle.execution.numerical_threads_per_worker)):
        outcomes = Parallel(n_jobs=workers)(delayed(_run_job)(payload) for payload in payloads)
    return list(outcomes)


def specs_for_run(bundle: ExperimentBundle, manifest_path: Path | None) -> list[dict[str, Any]]:
    if manifest_path is not None:
        manifest = load_manifest(manifest_path)
        expected = scientific_fingerprint(bundle.scientific)
        if manifest["scientific_fingerprint"] != expected:
            raise ResumeConflict(
                "manifest fingerprint does not match the loaded scientific configuration"
            )
        if bundle.execution.array_task_id is not None:
            return select_batch(manifest, int(bundle.execution.array_task_id))
        specs = []
        for batch in manifest["batches"]:
            specs.extend(batch)
        return specs
    if bundle.execution.array_task_id is not None:
        raise ValueError("array_task_id requires a prepared manifest")
    return all_work_specs(bundle.scientific)
