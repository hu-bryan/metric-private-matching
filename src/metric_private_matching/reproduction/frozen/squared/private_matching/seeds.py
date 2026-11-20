"""Versioned, order-independent random streams.

Master Q/R streams depend only on root seed, stream tag, and configuration
index s. Noise streams depend on those plus (n, d, mechanism, epsilon, k).
Python's hash(), timestamps, and process IDs are never used.
"""

from __future__ import annotations

import struct
from typing import Iterable

import numpy as np

SEED_SCHEME_VERSION = "v1-20260914"
SEED_SCHEME_VERSION_INT = 1
BIT_GENERATOR_NAME = "PCG64"

STREAM_MASTER_Q = 1
STREAM_MASTER_R = 2
STREAM_NOISE = 3

MECHANISM_IDS = {
    "ambient": 1,
    "phi_infinity": 2,
    "folded": 3,
}


def float64_to_uint32_pair(value: float) -> tuple[int, int]:
    packed = struct.pack("<d", float(value))
    low, high = struct.unpack("<II", packed)
    return int(low), int(high)


def make_generator(root_seed: int, spawn_key: Iterable[int]) -> np.random.Generator:
    key = tuple(int(x) for x in spawn_key)
    ss = np.random.SeedSequence(int(root_seed), spawn_key=key)
    return np.random.Generator(np.random.PCG64(ss))


def master_generator(root_seed: int, stream: int, s: int) -> np.random.Generator:
    if stream not in (STREAM_MASTER_Q, STREAM_MASTER_R):
        raise ValueError(f"invalid master stream {stream}")
    return make_generator(
        root_seed,
        (SEED_SCHEME_VERSION_INT, stream, int(s), 0, 0, 0, 0, 0, 0, 0),
    )


def noise_generator(
    root_seed: int,
    s: int,
    n: int,
    d: int,
    mechanism: str,
    epsilon: float,
    k: int,
) -> np.random.Generator:
    try:
        mech_id = MECHANISM_IDS[mechanism]
    except KeyError as exc:
        raise ValueError(f"unknown mechanism {mechanism!r}") from exc
    eps_lo, eps_hi = float64_to_uint32_pair(epsilon)
    return make_generator(
        root_seed,
        (
            SEED_SCHEME_VERSION_INT,
            STREAM_NOISE,
            int(s),
            int(n),
            int(d),
            int(mech_id),
            eps_lo,
            eps_hi,
            int(k),
        ),
    )
