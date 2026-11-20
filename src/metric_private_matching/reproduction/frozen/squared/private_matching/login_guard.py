"""Refuse heavy work on CRC login nodes."""

from __future__ import annotations

import os
import socket


def hostname() -> str:
    return socket.gethostname()


def is_login_node(host: str | None = None) -> bool:
    host = (host or hostname()).lower()
    if os.environ.get("PRIVATE_MATCHING_ALLOW_LOGIN") == "1":
        return False
    return host.startswith("login") or host in {"h2p.crc.pitt.edu", "htc.crc.pitt.edu"}


def refuse_login_compute(action: str) -> None:
    if is_login_node():
        raise RuntimeError(
            f"Refusing to {action} on CRC login node {hostname()!r}. "
            "Run this on an smp compute allocation (srun/sbatch). "
            "This package will not fall back to login-node Python."
        )
