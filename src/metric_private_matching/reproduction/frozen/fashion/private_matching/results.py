"""Validated records and atomic partition writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from private_matching.config import epsilon_key as format_epsilon

SCHEMA_VERSION = 1


def epsilon_key(epsilon: float) -> str:
    return format_epsilon(epsilon)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    blob = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    atomic_write_bytes(path, blob)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a JSON object")
    return data


def unit_filename(s: int, n: int, d: int, k_start: int, k_end: int) -> str:
    return f"s{int(s):02d}_n{int(n):04d}_d{int(d):04d}_k{int(k_start):04d}_{int(k_end):04d}.json"


def validate_unit_record(record: Mapping[str, Any], *, expected_fingerprint: str | None = None) -> None:
    required = (
        "schema_version",
        "work_id",
        "scientific_fingerprint",
        "s",
        "n",
        "d",
        "k_span",
        "m_fold",
        "O",
        "B",
        "g",
        "K_requested",
        "k_start",
        "k_end",
        "mechanisms",
        "epsilons",
        "excess",
        "ambient_server_rule",
        "folded_server_rule",
        "bounds",
        "status",
    )
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"unit record missing {missing}")
    if int(record["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {record['schema_version']!r}")
    if record.get("status") != "complete":
        raise ValueError(f"unit {record.get('work_id')} is not complete")
    if expected_fingerprint and record["scientific_fingerprint"] != expected_fingerprint:
        raise ValueError(
            "scientific fingerprint mismatch: "
            f"{record['scientific_fingerprint']} != {expected_fingerprint}"
        )
    k_start = int(record["k_start"])
    k_end = int(record["k_end"])
    expected_len = k_end - k_start
    if expected_len < 1:
        raise ValueError("empty repetition block")
    for mechanism in record["mechanisms"]:
        if mechanism not in record["excess"]:
            raise ValueError(f"missing excess for {mechanism}")
        for eps in record["epsilons"]:
            key = epsilon_key(float(eps))
            values = record["excess"][mechanism][key]
            if len(values) != expected_len:
                raise ValueError(
                    f"{mechanism} epsilon={key} has {len(values)} excesses, expected {expected_len}"
                )
