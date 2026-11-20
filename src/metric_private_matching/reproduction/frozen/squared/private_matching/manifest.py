"""Canonical work IDs, batches, and experiment identity."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from private_matching.config import ExperimentBundle, ScientificConfig, epsilon_key
from private_matching.results import atomic_write_json, read_json
from private_matching.seeds import BIT_GENERATOR_NAME, SEED_SCHEME_VERSION


def canonical_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def scientific_fingerprint(cfg: ScientificConfig) -> str:
    blob = canonical_dumps(cfg.scientific_identity())
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def work_id(cfg: ScientificConfig, s: int, n: int, d: int) -> str:
    return (
        f"{cfg.name}|s={int(s):02d}|n={int(n):04d}|d={int(d):04d}|"
        f"fp={scientific_fingerprint(cfg)[:12]}"
    )


def block_id(cfg: ScientificConfig, s: int, n: int, d: int, k_start: int, k_end: int) -> str:
    return f"{work_id(cfg, s, n, d)}|k={int(k_start):04d}-{int(k_end):04d}"


def all_work_specs(cfg: ScientificConfig) -> list[dict[str, Any]]:
    specs = []
    for s in range(int(cfg.configurations_S)):
        for n, d in cfg.settings():
            specs.append(
                {
                    "work_id": work_id(cfg, s, n, d),
                    "s": int(s),
                    "n": int(n),
                    "d": int(d),
                    "label": f"Point configuration {s + 1}",
                }
            )
    return specs


def estimated_cost(n: int, d: int) -> int:
    """Rough cost: pairwise distances plus Hungarian."""
    return int(n) ** 2 * int(d) + int(n) ** 3


def make_batches(specs: Sequence[dict[str, Any]], n_batches: int) -> list[list[dict[str, Any]]]:
    if n_batches < 1:
        raise ValueError("n_batches must be >= 1")
    ordered = sorted(specs, key=lambda spec: (estimated_cost(spec["n"], spec["d"]), spec["s"]), reverse=True)
    batches: list[list[dict[str, Any]]] = [[] for _ in range(n_batches)]
    loads = [0] * n_batches
    for spec in ordered:
        idx = min(range(n_batches), key=lambda i: (loads[i], i))
        batches[idx].append(spec)
        loads[idx] += estimated_cost(spec["n"], spec["d"])
    return batches


def repetition_blocks(K: int, block_size: int | None) -> list[tuple[int, int]]:
    K = int(K)
    if K < 1:
        raise ValueError("K must be >= 1")
    if block_size is None or int(block_size) >= K:
        return [(0, K)]
    size = int(block_size)
    return [(start, min(start + size, K)) for start in range(0, K, size)]


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for name in ("numpy", "scipy", "pandas", "joblib", "yaml", "matplotlib"):
        try:
            mod = __import__(name if name != "yaml" else "yaml")
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "missing"
    return versions


def source_checksum(root: Path) -> str:
    src = root / "src" / "private_matching"
    hasher = hashlib.sha256()
    if not src.is_dir():
        return hasher.hexdigest()
    for path in sorted(src.rglob("*.py")):
        hasher.update(str(path.relative_to(src)).encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()


def build_manifest(
    bundle: ExperimentBundle,
    *,
    repo_root: Path,
    n_batches: int | None = None,
) -> dict[str, Any]:
    cfg = bundle.scientific
    specs = all_work_specs(cfg)
    n_batches = int(n_batches or max(int(bundle.execution.array_task_count or 1), 1))
    batches = make_batches(specs, n_batches)
    blocks = repetition_blocks(cfg.noise_repetitions_K, bundle.execution.repetition_block_size)
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": cfg.name,
        "scientific_fingerprint": scientific_fingerprint(cfg),
        "scientific_identity": cfg.scientific_identity(),
        "seed_scheme_version": SEED_SCHEME_VERSION,
        "bit_generator": BIT_GENERATOR_NAME,
        "source_api_version": cfg.source_api_version,
        "source_checksum": source_checksum(repo_root),
        "package_versions": package_versions(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "resolved": {
            "epsilons": [epsilon_key(e) for e in cfg.epsilons],
            "n_values": list(cfg.n_values),
            "d_values": list(cfg.d_values),
            "varying_n_fixed_d": cfg.varying_n_fixed_d,
            "varying_d_fixed_n": cfg.varying_d_fixed_n,
            "master_n": cfg.master_n,
            "master_d": cfg.master_d,
            "configurations_S": cfg.configurations_S,
            "noise_repetitions_K": cfg.noise_repetitions_K,
            "ambient_server_rule": cfg.ambient_server_rule,
            "folded_server_rule": cfg.folded_server_rule,
        },
        "counts": cfg.workload_counts(),
        "work_units": specs,
        "repetition_blocks": [{"k_start": a, "k_end": b} for a, b in blocks],
        "batches": [[{"work_id": w["work_id"], "s": w["s"], "n": w["n"], "d": w["d"]} for w in batch] for batch in batches],
        "source_paths": list(bundle.source_paths),
        "execution_note": (
            "Worker count, array task ID, and batch size are execution parameters "
            "and are not part of the scientific fingerprint."
        ),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    atomic_write_json(path, manifest)


def load_manifest(path: Path) -> dict[str, Any]:
    return read_json(path)


def select_batch(manifest: dict[str, Any], array_task_id: int) -> list[dict[str, Any]]:
    batches = manifest["batches"]
    if not (0 <= int(array_task_id) < len(batches)):
        raise IndexError(
            f"array_task_id={array_task_id} is outside 0..{len(batches) - 1}"
        )
    return list(batches[int(array_task_id)])
