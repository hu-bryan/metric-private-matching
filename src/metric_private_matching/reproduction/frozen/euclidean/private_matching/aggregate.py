"""Per-instance noise means, variances, and conditional t intervals."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t

from private_matching.bounds import normalize_bound
from private_matching.config import ExperimentBundle, epsilon_key, point_config_label
from private_matching.manifest import all_work_specs, scientific_fingerprint
from private_matching.results import read_json, validate_unit_record


class CompletenessError(RuntimeError):
    """Manifest and on-disk units do not form a complete, unduplicated result."""


def _t_critical(df: int, confidence: float) -> float:
    tail = 1.0 - (1.0 - float(confidence)) / 2.0
    return float(student_t.ppf(tail, df))


def _summarize_excess(values: np.ndarray, gap: float, confidence: float) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    k = int(values.size)
    mean_x = float(values.mean()) if k else float("nan")
    out: dict[str, Any] = {
        "K": k,
        "mean_excess": mean_x,
        "sum_excess": float(values.sum()) if k else 0.0,
        "sum_squared_excess": float(np.square(values).sum()) if k else 0.0,
        "normalized_mean": None,
        "se_normalized": None,
        "ci_low": None,
        "ci_high": None,
        "sample_variance": None,
        "zero_sample_variance": False,
        "gap_undefined": not (gap > 0.0),
    }
    if gap > 0.0 and k:
        out["normalized_mean"] = mean_x / gap
        if k >= 2:
            var = float(values.var(ddof=1))
            out["sample_variance"] = var
            out["zero_sample_variance"] = bool(var == 0.0)
            se = float(np.sqrt(var / k) / gap)
            out["se_normalized"] = se
            half = _t_critical(k - 1, confidence) * se
            out["ci_low"] = out["normalized_mean"] - half
            out["ci_high"] = out["normalized_mean"] + half
    return out


def collect_unit_files(results_dir: Path) -> list[Path]:
    units = results_dir / "units"
    if not units.is_dir():
        return []
    return sorted(p for p in units.glob("s*_n*_d*_k*.json") if p.is_file())


def completeness_report(bundle: ExperimentBundle, results_dir: Path) -> dict[str, Any]:
    cfg = bundle.scientific
    fingerprint = scientific_fingerprint(cfg)
    expected_specs = all_work_specs(cfg)
    expected_keys = {(spec["s"], spec["n"], spec["d"]) for spec in expected_specs}
    files = collect_unit_files(results_dir)
    found: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    conflicts: list[str] = []
    invalid: list[str] = []
    for path in files:
        try:
            record = read_json(path)
            validate_unit_record(record, expected_fingerprint=fingerprint)
        except Exception as exc:
            invalid.append(f"{path.name}: {exc}")
            continue
        if record["ambient_server_rule"] != cfg.ambient_server_rule:
            conflicts.append(f"{path.name}: ambient server rule mismatch")
            continue
        if record["folded_server_rule"] != cfg.folded_server_rule:
            conflicts.append(f"{path.name}: folded server rule mismatch")
            continue
        key = (int(record["s"]), int(record["n"]), int(record["d"]))
        found[key].append(record)

    missing = []
    incomplete = []
    complete = []
    K = int(cfg.noise_repetitions_K)
    for spec in expected_specs:
        key = (spec["s"], spec["n"], spec["d"])
        records = found.get(key, [])
        covered: dict[int, str] = {}
        duplicate = False
        for rec in records:
            for k in range(int(rec["k_start"]), int(rec["k_end"])):
                if k in covered:
                    duplicate = True
                covered[k] = rec["block_id"]
        have = sorted(covered)
        if not records:
            missing.append({"s": spec["s"], "n": spec["n"], "d": spec["d"]})
            continue
        if duplicate:
            incomplete.append({"s": spec["s"], "n": spec["n"], "d": spec["d"], "reason": "duplicate_repetitions"})
            continue
        if have != list(range(K)):
            incomplete.append(
                {
                    "s": spec["s"],
                    "n": spec["n"],
                    "d": spec["d"],
                    "reason": "missing_repetitions",
                    "have_K": len(have),
                    "need_K": K,
                }
            )
            continue
        complete.append(key)

    extra = [key for key in found if key not in expected_keys]
    complete_flag = not (missing or incomplete or conflicts or invalid or extra)
    by_s = {
        s: {
            "complete": sum(1 for key in complete if key[0] == s),
            "expected": sum(1 for spec in expected_specs if spec["s"] == s),
        }
        for s in range(cfg.configurations_S)
    }
    return {
        "complete": complete_flag,
        "expected_work_units": len(expected_specs),
        "complete_work_units": len(complete),
        "missing": missing,
        "incomplete": incomplete,
        "conflicts": conflicts,
        "invalid": invalid,
        "extra_settings": extra,
        "by_configuration": by_s,
        "scientific_fingerprint": fingerprint,
        "results_dir": str(results_dir),
        "partial_label_required": not complete_flag,
    }


def aggregate_results(bundle: ExperimentBundle, results_dir: Path) -> dict[str, Any]:
    cfg = bundle.scientific
    report = completeness_report(bundle, results_dir)
    fingerprint = scientific_fingerprint(cfg)
    grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = defaultdict(list)
    for path in collect_unit_files(results_dir):
        record = read_json(path)
        try:
            validate_unit_record(record, expected_fingerprint=fingerprint)
        except Exception:
            continue
        key = (int(record["s"]), int(record["n"]), int(record["d"]))
        grouped[key].append(record)

    rows: list[dict[str, Any]] = []
    for spec in all_work_specs(cfg):
        key = (spec["s"], spec["n"], spec["d"])
        records = sorted(grouped.get(key, []), key=lambda rec: int(rec["k_start"]))
        if not records:
            continue
        base = records[0]
        excess_map: dict[str, dict[str, list[float]]] = {
            mechanism: {epsilon_key(eps): [] for eps in cfg.epsilons} for mechanism in cfg.mechanisms
        }
        for rec in records:
            for mechanism in cfg.mechanisms:
                for eps in cfg.epsilons:
                    ek = epsilon_key(eps)
                    excess_map[mechanism][ek].extend(rec["excess"][mechanism][ek])
        gap = float(base["g"])
        n = int(base["n"])
        for mechanism in cfg.mechanisms:
            for eps in cfg.epsilons:
                ek = epsilon_key(eps)
                values = np.asarray(excess_map[mechanism][ek], dtype=np.float64)
                summary = _summarize_excess(values, gap, cfg.plot_confidence_level)
                mean_x = summary["mean_excess"]
                summary["mean_excess_per_driver"] = (
                    None if mean_x is None or not np.isfinite(mean_x) else float(mean_x) / n
                )
                bound = float(base["bounds"][mechanism][ek])
                bound_norm = normalize_bound(bound, gap)
                rows.append(
                    {
                        "s": int(base["s"]),
                        "label": point_config_label(base["s"]),
                        "n": n,
                        "d": int(base["d"]),
                        "mechanism": mechanism,
                        "epsilon": float(eps),
                        "epsilon_key": ek,
                        "K": summary["K"],
                        "K_requested": int(cfg.noise_repetitions_K),
                        "complete_K": bool(summary["K"] == int(cfg.noise_repetitions_K)),
                        "O": float(base["O"]),
                        "B": float(base["B"]),
                        "g": gap,
                        "gap_zero": bool(base["gap_zero"]),
                        "k_span": int(base["k_span"]),
                        "m_fold": int(base["m_fold"]),
                        "unexpected_rank_loss": bool(base["unexpected_rank_loss"]),
                        "bound": bound,
                        "bound_normalized": bound_norm,
                        "mean_excess": summary["mean_excess"],
                        "mean_excess_per_driver": summary["mean_excess_per_driver"],
                        "normalized_mean": summary["normalized_mean"],
                        "se_normalized": summary["se_normalized"],
                        "ci_low": summary["ci_low"],
                        "ci_high": summary["ci_high"],
                        "sample_variance": summary["sample_variance"],
                        "zero_sample_variance": summary["zero_sample_variance"],
                        "sum_excess": summary["sum_excess"],
                        "sum_squared_excess": summary["sum_squared_excess"],
                        "ambient_server_rule": base["ambient_server_rule"],
                        "folded_server_rule": base["folded_server_rule"],
                    }
                )
    return {
        "completeness": report,
        "rows": rows,
        "experiment_name": cfg.name,
        "scientific_fingerprint": fingerprint,
        "interval_method": cfg.plot_interval_method,
        "confidence_level": cfg.plot_confidence_level,
        "note": (
            "Estimates are conditional Monte Carlo means over mechanism noise "
            "on each fixed master configuration. There is no average over s."
        ),
    }
