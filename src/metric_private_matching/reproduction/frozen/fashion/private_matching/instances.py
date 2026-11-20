"""Master configuration generation and slicing.

Always draw the frozen MASTER shape ``(master_n, master_d)`` before slicing
``[:n, :d]``. Generating an ``(n, d)`` array from the same seed does not
preserve the required nesting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from private_matching.config import ScientificConfig
from private_matching.seeds import STREAM_MASTER_Q, STREAM_MASTER_R, master_generator


@dataclass(frozen=True)
class Instance:
    s: int
    n: int
    d: int
    Q: np.ndarray
    R: np.ndarray


def numpy_dtype(name: str) -> np.dtype:
    if name != "float64":
        raise ValueError(f"unsupported dtype {name!r}")
    return np.dtype(np.float64)


def generate_master_pair(cfg: ScientificConfig, s: int) -> tuple[np.ndarray, np.ndarray]:
    if not (0 <= int(s) < int(cfg.configurations_S)):
        raise ValueError(f"configuration index s={s} is outside 0..{cfg.configurations_S - 1}")
    dtype = numpy_dtype(cfg.dtype_name)
    rng_q = master_generator(cfg.root_seed, STREAM_MASTER_Q, s)
    rng_r = master_generator(cfg.root_seed, STREAM_MASTER_R, s)
    Q = rng_q.uniform(0.0, float(cfg.cube_side), size=(cfg.master_n, cfg.master_d)).astype(dtype, copy=False)
    R = rng_r.uniform(0.0, float(cfg.cube_side), size=(cfg.master_n, cfg.master_d)).astype(dtype, copy=False)
    return Q, R


def slice_instance(Q_master: np.ndarray, R_master: np.ndarray, s: int, n: int, d: int) -> Instance:
    if Q_master.shape != R_master.shape:
        raise ValueError("Q_master and R_master must have the same shape")
    if n > Q_master.shape[0] or d > Q_master.shape[1]:
        raise ValueError(
            f"requested (n,d)=({n},{d}) exceeds master shape {Q_master.shape}"
        )
    return Instance(
        s=int(s),
        n=int(n),
        d=int(d),
        Q=np.ascontiguousarray(Q_master[:n, :d]),
        R=np.ascontiguousarray(R_master[:n, :d]),
    )


def load_instance(cfg: ScientificConfig, s: int, n: int, d: int) -> Instance:
    Q_master, R_master = generate_master_pair(cfg, s)
    return slice_instance(Q_master, R_master, s, n, d)
