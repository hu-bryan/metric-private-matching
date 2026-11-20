"""Set numerical library thread caps before NumPy is imported."""

from __future__ import annotations

import os

_THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def apply_thread_limits(n_threads: int | None = None, *, overwrite: bool = False) -> int:
    """Pin BLAS/OpenMP thread pools to ``n_threads`` (default 1).

    Call this before importing NumPy in process entry points. Existing
    environment values are preserved unless ``overwrite`` is true, so a
    caller can still raise the cap explicitly.
    """
    if n_threads is None:
        raw = os.environ.get("PRIVATE_MATCHING_NUM_THREADS", "1")
        n_threads = max(int(raw), 1)
    value = str(max(int(n_threads), 1))
    for key in _THREAD_KEYS:
        if overwrite or not os.environ.get(key):
            os.environ[key] = value
    os.environ["PRIVATE_MATCHING_NUM_THREADS"] = value
    return int(value)
