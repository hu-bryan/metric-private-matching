"""Private 1:1 matching for ATT estimation on IHDP (``causal_exp_plan.md``).

Covariates only are protected; outcomes are used unprotected. The server always
solves a (treated rows) x (control columns) assignment on noisy unsquared
Mahalanobis costs, and every score is computed on the true costs.

Arm A: treated are private (the draft's theorems hold with n = n1).
Arm B: controls are private (only the rectangular pointwise certificate holds;
Sigma = Sigma_0 is assumed public, e.g. published at population level).
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from private_matching.mechanisms import sample_euclidean_laplace, sample_supnorm_cone_laplace

TRUE_ATT = 4.0
KALLUS_COND1_MSE = 7.56

CONTINUOUS = ["bw", "b.head", "preterm", "birth.o", "nnhealth", "momage"]
BINARY = ["sex", "twin", "b.marr", "mom.lths", "mom.hs", "mom.scoll", "cig", "first", "booze",
          "drugs", "work.dur", "prenatal", "ark", "ein", "har", "mia", "pen", "tex", "was"]
COVARIATES = CONTINUOUS + BINARY

MECH_LAMBDA = "Lambda"
MECH_PHI = "Phi_inf"
MECH_XI = "Xi"
MECH_PSI2 = "Psi2"
COND_SQ = "nonprivate_sq"      # condition 1
COND_NP = "nonprivate"         # condition 2 (epsilon = infinity)
COND_RANDOM = "random"         # condition 4
BASELINES = (COND_SQ, COND_NP, COND_RANDOM)
ARMS = ("A", "B")

# Seed-sequence component codes. Stage codes keep pilot and production disjoint.
STAGE_CODES = {"pilot": 1, "production": 2}
_MECH_CODES = {MECH_LAMBDA: 1, MECH_PHI: 2, MECH_XI: 3, MECH_PSI2: 4}


# --------------------------------------------------------------------------- config

DEFAULT_CONFIG: dict[str, Any] = {
    "name": "causal_ihdp_v1",
    "data_path": "/vast/arahimian/brh314/metric_private_covariate_matching/data/ihdp/ihdp.RData",
    "output_root": "/ix1/arahimian/brh314/metric_private_covariate_matching/causal_ihdp",
    "subset": "kallus",                 # kallus | hill
    "standardize_on": "hill",           # hill (747 units, as in npci) | analysis
    "control_max_age": 26,              # keep controls with nonwhite mothers aged <= this
    "treated_min_age_exclusive": 23,    # keep treated with white mothers aged > this
    "seed": 20260924,
    "replications": 500,
    "pilot_replications": 100,
    "pilot_grid_size": 30,
    "grid_size": 12,
    "grid_start_factor": 10.0,          # d / eps_start = factor * mbar
    "grid_cap_factor": 1.0e4,           # eps_max <= cap / mbar
    "grid_end_margin": 1.25,            # eps_end = margin * pilot eps10(Phi_inf), capped
    "within_frac": 0.10,
    "bootstrap": 1000,
    "workers": 4,
}


def load_config(path: str | Path | None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if path is not None:
        with open(path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        unknown = set(loaded) - set(DEFAULT_CONFIG)
        if unknown:
            raise ValueError(f"unknown causal_ihdp config keys: {sorted(unknown)}")
        cfg.update(loaded)
    if cfg["subset"] not in {"kallus", "hill"}:
        raise ValueError("subset must be 'kallus' or 'hill'")
    if cfg["standardize_on"] not in {"hill", "analysis"}:
        raise ValueError("standardize_on must be 'hill' or 'analysis'")
    return cfg


# --------------------------------------------------------------------------- data

@dataclass
class IHDPData:
    X_t: np.ndarray                 # treated covariates (n1 x 25), continuous standardized
    X_c: np.ndarray                 # control covariates (n0 x 25)
    names: list[str]
    subset: str
    standardize_on: str
    source_sha256: str
    notes: list[str] = field(default_factory=list)

    @property
    def n1(self) -> int:
        return int(self.X_t.shape[0])

    @property
    def n0(self) -> int:
        return int(self.X_c.shape[0])

    @property
    def d(self) -> int:
        return int(self.X_t.shape[1])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_raw_ihdp(path: str | Path) -> pd.DataFrame:
    import pyreadr

    frames = pyreadr.read_r(str(path))
    if "ihdp" not in frames:
        raise ValueError(f"{path} does not contain an 'ihdp' data frame")
    return frames["ihdp"]


def prepare_ihdp(df: pd.DataFrame, cfg: dict[str, Any], sha: str = "") -> IHDPData:
    """Hill's 747-unit sample, then (optionally) Kallus's harder-overlap subset."""
    missing = [c for c in COVARIATES + ["treat", "momwhite"] if c not in df.columns]
    if missing:
        raise ValueError(f"IHDP frame lacks columns {missing}")
    df = df.copy()
    notes: list[str] = []
    # `first` is coded 1/2 in the raw file; the plan leaves binaries as 0/1.
    if set(np.unique(df["first"])) <= {1, 2}:
        df["first"] = df["first"] - 1
        notes.append("`first` recoded from {1,2} to {0,1} (first - 1).")
    hill = df[(df["treat"] != 1) | (df["momwhite"] != 0)].reset_index(drop=True)
    if cfg["subset"] == "kallus":
        t_keep = (hill["treat"] == 1) & (hill["momage"] > cfg["treated_min_age_exclusive"])
        c_keep = (hill["treat"] == 0) & (hill["momwhite"] == 0) & (hill["momage"] <= cfg["control_max_age"])
        analysis = hill[t_keep | c_keep].reset_index(drop=True)
        notes.append(
            f"Kallus subset: treated with white mothers aged > {cfg['treated_min_age_exclusive']}; "
            f"controls with nonwhite mothers aged <= {cfg['control_max_age']}."
        )
    else:
        analysis = hill
        notes.append("Hill subset: all controls, treated with white mothers.")
    base = hill if cfg["standardize_on"] == "hill" else analysis
    X = analysis[COVARIATES].to_numpy(dtype=np.float64)
    for k, name in enumerate(COVARIATES):
        if name in CONTINUOUS:
            mu = float(base[name].mean())
            sd = float(base[name].std(ddof=1))
            X[:, k] = (X[:, k] - mu) / sd
    notes.append(f"Continuous covariates standardized on the {cfg['standardize_on']} sample (sd ddof=1).")
    treat = analysis["treat"].to_numpy() == 1
    return IHDPData(X_t=X[treat], X_c=X[~treat], names=list(COVARIATES), subset=cfg["subset"],
                    standardize_on=cfg["standardize_on"], source_sha256=sha, notes=notes)


def load_ihdp(cfg: dict[str, Any]) -> IHDPData:
    path = Path(cfg["data_path"])
    return prepare_ihdp(read_raw_ihdp(path), cfg, sha=_sha256(path))


# --------------------------------------------------------------------------- geometry

def whitening_matrix(X_c: np.ndarray, rel_ridge: float = 1e-6) -> tuple[np.ndarray, float]:
    """Sigma_0^{-1/2} from the control covariance, ridged if some column is constant."""
    S = np.cov(X_c, rowvar=False, ddof=1)
    ridge = 0.0
    if np.any(np.diag(S) <= 1e-12) or np.linalg.eigvalsh(S).min() <= 1e-10 * np.trace(S) / S.shape[0]:
        ridge = rel_ridge * float(np.trace(S)) / S.shape[0]
        S = S + ridge * np.eye(S.shape[0])
    w, V = np.linalg.eigh(S)
    return (V / np.sqrt(w)) @ V.T, ridge


@dataclass
class PublicRepresentation:
    """Definitions 9-10: centre at q1, fold onto W + R u when k <= d - 2."""

    q1: np.ndarray
    basis: np.ndarray | None   # d x k orthonormal basis of W, None for the identity
    k: int
    m: int

    @property
    def identity(self) -> bool:
        return self.basis is None

    def __call__(self, X: np.ndarray) -> np.ndarray:
        Xc = np.asarray(X, dtype=np.float64) - self.q1
        if self.basis is None:
            return Xc
        coef = Xc @ self.basis
        resid = np.linalg.norm(Xc - coef @ self.basis.T, axis=1, keepdims=True)
        return np.hstack([coef, resid])


def public_representation(P: np.ndarray, rel_tol: float = 1e-9) -> PublicRepresentation:
    P = np.asarray(P, dtype=np.float64)
    d = P.shape[1]
    q1 = P[0].copy()
    D = P[1:] - q1
    if D.shape[0] == 0:
        k = 0
        s = np.zeros(0)
        Vt = np.zeros((0, d))
    else:
        _, s, Vt = np.linalg.svd(D, full_matrices=False)
        k = int(np.sum(s > rel_tol * max(s.max(initial=0.0), 1.0)))
    m = min(k + 1, d)
    basis = None if m == d else Vt[:k].T.copy()
    return PublicRepresentation(q1=q1, basis=basis, k=k, m=m)


def public_directions(P_rep: np.ndarray, decimals: int = 10) -> np.ndarray:
    """V_Q: distinct normalized p_i - p_l (i < l, p_i != p_l)."""
    i, l = np.triu_indices(P_rep.shape[0], k=1)
    diff = P_rep[i] - P_rep[l]
    norm = np.linalg.norm(diff, axis=1)
    keep = norm > 0.0
    V = diff[keep] / norm[keep, None]
    _, first = np.unique(np.round(V, decimals), axis=0, return_index=True)
    return np.ascontiguousarray(V[np.sort(first)])


@dataclass
class ArmGeometry:
    arm: str
    private_group: str
    public_group: str
    pub: np.ndarray            # whitened public points
    priv: np.ndarray           # whitened private points
    rep: PublicRepresentation
    pub_rep: np.ndarray
    priv_rep: np.ndarray
    directions: np.ndarray
    n_directions: int
    beta_q: float
    eps_y_frac: float
    eps_s_frac: float
    mechanisms: tuple[str, ...]

    @property
    def n_pub(self) -> int:
        return int(self.pub.shape[0])

    @property
    def n_priv(self) -> int:
        return int(self.priv.shape[0])

    @property
    def d(self) -> int:
        return int(self.pub.shape[1])

    @property
    def m(self) -> int:
        return self.rep.m

    def summary(self) -> dict[str, Any]:
        return {
            "arm": self.arm, "private": self.private_group, "public": self.public_group,
            "n_private": self.n_priv, "n_public": self.n_pub, "d": self.d, "k": self.rep.k,
            "m": self.m, "pi_q_identity": self.rep.identity, "N_Q": self.n_directions,
            "beta_Q": self.beta_q, "eps_y_frac": self.eps_y_frac, "eps_s_frac": self.eps_s_frac,
            "mechanisms": list(self.mechanisms),
        }

    def theorem_bound(self, mech: str, eps: float) -> float:
        """Normalized bound E[u - OPT]/n1 (Arm A theorems; Arm B heuristic formula)."""
        if mech == MECH_LAMBDA:
            return 2.0 * self.d / eps
        if mech == MECH_PHI:
            return 2.0 * self.n_pub / eps
        if mech == MECH_XI:
            return (2.0 * math.sqrt(self.beta_q) + 1.0) ** 2 / eps
        if mech == MECH_PSI2:
            return 2.0 * self.m / eps
        raise ValueError(mech)


def build_arm(arm: str, Z_t: np.ndarray, Z_c: np.ndarray) -> ArmGeometry:
    """Everything here is a function of the public side only (plus the public Sigma)."""
    if arm == "A":
        pub, priv, pg, qg = Z_c, Z_t, "treated", "controls"
    elif arm == "B":
        pub, priv, pg, qg = Z_t, Z_c, "controls", "treated"
    else:
        raise ValueError(arm)
    rep = public_representation(pub)
    pub_rep = rep(pub)
    V = public_directions(pub_rep)
    n_q = int(V.shape[0])
    beta = min(float(rep.m), math.sqrt(2.0 * rep.m * math.log(2.0 * n_q)))
    root = 2.0 * math.sqrt(beta)
    mechs = [MECH_LAMBDA, MECH_PHI, MECH_XI] + ([MECH_PSI2] if rep.m < pub.shape[1] else [])
    return ArmGeometry(arm=arm, private_group=pg, public_group=qg, pub=pub, priv=priv, rep=rep,
                       pub_rep=pub_rep, priv_rep=rep(priv), directions=V, n_directions=n_q,
                       beta_q=beta, eps_y_frac=root / (root + 1.0), eps_s_frac=1.0 / (root + 1.0),
                       mechanisms=tuple(mechs))


@dataclass
class Experiment:
    data: IHDPData
    W: np.ndarray
    ridge: float
    Z_t: np.ndarray
    Z_c: np.ndarray
    C: np.ndarray               # true unsquared Mahalanobis, treated x controls
    sigma_opt: np.ndarray       # OPT matching (condition 2), control index per treated
    sigma_sq: np.ndarray        # squared-Mahalanobis matching (condition 1)
    opt: float
    arms: dict[str, ArmGeometry]
    sd_pooled: np.ndarray

    @property
    def mbar(self) -> float:
        return self.opt / self.data.n1


def lsa_rows(cost: np.ndarray) -> np.ndarray:
    """Assignment for a (rows <= cols) matrix: the column matched to each row."""
    r, c = linear_sum_assignment(cost)
    out = np.empty(cost.shape[0], dtype=np.int64)
    out[r] = c
    return out


def lsa_cols(cost_t: np.ndarray) -> np.ndarray:
    """Same problem posed with rows = controls (Arm B orientation); returns control per treated."""
    r, c = linear_sum_assignment(cost_t)
    out = np.empty(cost_t.shape[1], dtype=np.int64)
    out[c] = r
    return out


def build_experiment(data: IHDPData) -> Experiment:
    W, ridge = whitening_matrix(data.X_c)
    Z_t, Z_c = data.X_t @ W, data.X_c @ W
    C = cdist(Z_t, Z_c)
    sigma_opt = lsa_rows(C)
    sigma_sq = lsa_rows(C ** 2)
    arms = {a: build_arm(a, Z_t, Z_c) for a in ARMS}
    sd = np.sqrt(0.5 * (data.X_t.var(axis=0, ddof=1) + data.X_c.var(axis=0, ddof=1)))
    return Experiment(data=data, W=W, ridge=ridge, Z_t=Z_t, Z_c=Z_c, C=C, sigma_opt=sigma_opt,
                      sigma_sq=sigma_sq, opt=float(C[np.arange(data.n1), sigma_opt].sum()),
                      arms=arms, sd_pooled=sd)


# --------------------------------------------------------------------------- noise

def _rng(seed: int, stage: str, rep: int, *path: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), STAGE_CODES[stage], int(rep), *path]))


@dataclass
class BaseNoise:
    """Noise drawn once per replication at epsilon = 1; rescaled by 1/eps (Lap_B(e) = Lap_B(1)/e)."""

    Z: np.ndarray | None = None        # location noise (Lambda: d, Xi/Psi2: m) or Phi (n_pub)
    eta: np.ndarray | None = None      # Xi scalar noise
    mq: np.ndarray | None = None       # Xi: M_Q(Z_j) at base scale


def draw_base_noise(geom: ArmGeometry, mech: str, rng: np.random.Generator) -> BaseNoise:
    n = geom.n_priv
    if mech == MECH_LAMBDA:
        Z, _ = sample_euclidean_laplace(n, geom.d, 1.0, rng)
        return BaseNoise(Z=Z)
    if mech == MECH_PHI:
        Z, _ = sample_supnorm_cone_laplace(geom.n_pub, n, 1.0, rng)
        return BaseNoise(Z=np.ascontiguousarray(Z.T))
    if mech == MECH_PSI2:
        Z, _ = sample_euclidean_laplace(n, geom.m, 1.0, rng)
        return BaseNoise(Z=Z)
    if mech == MECH_XI:
        Z, _ = sample_euclidean_laplace(n, geom.m, 1.0, rng)
        eta = rng.laplace(0.0, 1.0, size=n)
        mq = np.abs(Z @ geom.directions.T).max(axis=1)
        return BaseNoise(Z=Z, eta=eta, mq=mq)
    raise ValueError(mech)


@dataclass
class NoisyRelease:
    cost_priv: np.ndarray      # server's cost, private x public
    b: np.ndarray              # per-private-unit bound on |E_ij| over all public i
    lemma_col: np.ndarray      # per-unit column-range bound (Lemma 11/28 form)
    radicand_min: float = 0.0  # Xi only: smallest radicand before the square root


def release(geom: ArmGeometry, mech: str, noise: BaseNoise, eps: float, C_priv: np.ndarray) -> NoisyRelease:
    if mech == MECH_LAMBDA:
        R_hat = geom.priv + noise.Z / eps
        b = np.linalg.norm(noise.Z, axis=1) / eps
        return NoisyRelease(cdist(R_hat, geom.pub), b, 2.0 * b)
    if mech == MECH_PSI2:
        x_hat = geom.priv_rep + noise.Z / eps
        b = np.linalg.norm(noise.Z, axis=1) / eps
        return NoisyRelease(cdist(x_hat, geom.pub_rep), b, 2.0 * b)
    if mech == MECH_PHI:
        Zs = noise.Z / eps
        # Clamping at 0 is post-processing (Proposition 4); |max(0, c+z) - c| <= |z| since c >= 0.
        # Clamping can widen a column's error range beyond range(Z), so the Lemma 11 column
        # term uses the realized range of E rather than max(Z) - min(Z).
        cost = np.maximum(C_priv + Zs, 0.0)
        E = cost - C_priv
        return NoisyRelease(cost, np.abs(Zs).max(axis=1), E.max(axis=1) - E.min(axis=1))
    if mech == MECH_XI:
        eps_y, eps_s = geom.eps_y_frac * eps, geom.eps_s_frac * eps
        Y = geom.priv_rep + noise.Z / eps_y
        D2 = cdist(Y, geom.pub_rep, "sqeuclidean")
        a = np.argmin(D2, axis=1)                      # lowest index among ties
        rows = np.arange(geom.n_priv)
        tau = C_priv[rows, a]                          # ||z_j - p_a||, computed locally by unit j
        tau_t = np.maximum(tau + noise.eta / eps_s, 0.0)
        rad = tau_t[:, None] ** 2 + D2 - D2[rows, a][:, None]
        mq = noise.mq / eps_y
        eta_abs = np.abs(noise.eta) / eps_s
        return NoisyRelease(np.sqrt(np.maximum(rad, 0.0)), 2.0 * mq + eta_abs, 4.0 * mq + eta_abs,
                            radicand_min=float(rad.min()))
    raise ValueError(mech)


# --------------------------------------------------------------------------- outcomes & metrics

@dataclass
class Outcomes:
    y_t: np.ndarray
    y_c: np.ndarray
    f0_t: np.ndarray
    f0_c: np.ndarray
    omega: float


def draw_outcomes(data: IHDPData, rng_beta: np.random.Generator, rng_noise: np.random.Generator) -> Outcomes:
    """Hill's response surface B with CATT = 4 on the treated."""
    beta = rng_beta.choice([0.0, 0.1, 0.2, 0.3, 0.4], size=data.d, p=[0.6, 0.1, 0.1, 0.1, 0.1])
    f0_t = np.exp((data.X_t + 0.5) @ beta)
    f0_c = np.exp((data.X_c + 0.5) @ beta)
    lin_t = data.X_t @ beta
    omega = float(np.mean(lin_t - f0_t) - TRUE_ATT)
    f1_t = lin_t - omega
    e = rng_noise.standard_normal(data.n1 + data.n0)
    return Outcomes(y_t=f1_t + e[:data.n1], y_c=f0_c + e[data.n1:], f0_t=f0_t, f0_c=f0_c, omega=omega)


def match_metrics(exp: Experiment, out: Outcomes, sigma: np.ndarray) -> dict[str, float]:
    n1 = exp.data.n1
    rows = np.arange(n1)
    u = float(exp.C[rows, sigma].sum())
    smd = np.abs(exp.data.X_t.mean(axis=0) - exp.data.X_c[sigma].mean(axis=0)) / exp.sd_pooled
    return {
        "u_norm": u / n1,
        "excess": (u - exp.opt) / n1,
        "err": float(np.mean(out.y_t - out.y_c[sigma]) - TRUE_ATT),
        "match_bias": float(np.mean(out.f0_t - out.f0_c[sigma])),
        "balance": float(np.mean(smd[np.isfinite(smd)])),
    }


# --------------------------------------------------------------------------- one replication

def run_replication(exp: Experiment, seed: int, stage: str, rep: int, grid: np.ndarray) -> dict[str, Any]:
    out = draw_outcomes(exp.data, _rng(seed, stage, rep, 0), _rng(seed, stage, rep, 1))
    n1, n0 = exp.data.n1, exp.data.n0
    rows = np.arange(n1)
    rng_rand = _rng(seed, stage, rep, 2)
    sigma_rand = rng_rand.permutation(n0)[:n1]
    records: list[dict[str, Any]] = []
    noise_stats: list[dict[str, Any]] = []
    for arm, geom in exp.arms.items():
        # Non-private conditions, solved in this arm's orientation (should agree across arms).
        if arm == "A":
            s_np, s_sq = lsa_rows(exp.C), lsa_rows(exp.C ** 2)
        else:
            s_np, s_sq = lsa_cols(exp.C.T), lsa_cols((exp.C ** 2).T)
        for cond, sig in ((COND_SQ, s_sq), (COND_NP, s_np), (COND_RANDOM, sigma_rand)):
            rec = {"rep": rep, "arm": arm, "mech": cond, "eps": math.inf}
            rec.update(match_metrics(exp, out, sig))
            records.append(rec)
        C_priv = exp.C if arm == "A" else exp.C.T
        for mech in geom.mechanisms:
            noise = draw_base_noise(geom, mech, _rng(seed, stage, rep, 10 + ARMS.index(arm), _MECH_CODES[mech]))
            ns = {"rep": rep, "arm": arm, "mech": mech, "count": geom.n_priv}
            if mech == MECH_PHI:
                ns["sum_norm"] = float(np.abs(noise.Z).max(axis=1).sum())
                ns["expected_norm"] = float(geom.n_pub)
            else:
                ns["sum_norm"] = float(np.linalg.norm(noise.Z, axis=1).sum())
                ns["expected_norm"] = float(geom.d if mech == MECH_LAMBDA else geom.m)
            ns["sum_sq_norm"] = float(((np.abs(noise.Z).max(axis=1) if mech == MECH_PHI
                                        else np.linalg.norm(noise.Z, axis=1)) ** 2).sum())
            if mech == MECH_XI:
                ns["sum_abs_eta"] = float(np.abs(noise.eta).sum())
                ns["sum_sq_eta"] = float((noise.eta ** 2).sum())
            noise_stats.append(ns)
            for eps in grid:
                rel = release(geom, mech, noise, float(eps), C_priv)
                C_hat = rel.cost_priv if arm == "A" else rel.cost_priv.T
                sig = lsa_rows(C_hat)
                E = C_hat - exp.C
                transfer = float(E[rows, exp.sigma_opt].sum() - E[rows, sig].sum())
                if arm == "A":
                    cert = float(2.0 * rel.b.sum())
                    lemma = float(rel.lemma_col.sum())
                else:
                    cert = float(rel.b[exp.sigma_opt].sum() + rel.b[sig].sum())
                    lemma = math.nan
                rec = {"rep": rep, "arm": arm, "mech": mech, "eps": float(eps)}
                rec.update(match_metrics(exp, out, sig))
                rec.update({
                    "cert": cert / n1, "cert_lemma": lemma / n1, "transfer": transfer / n1,
                    "bound": geom.theorem_bound(mech, float(eps)),
                    "radicand_min": rel.radicand_min,
                    "cost_scale": float(C_hat.max()),
                })
                records.append(rec)
    return {"records": records, "noise": noise_stats}


# --------------------------------------------------------------------------- driver

_WORKER_EXP: Experiment | None = None


def _init_worker(cfg: dict[str, Any]) -> None:
    global _WORKER_EXP
    _WORKER_EXP = build_experiment(load_ihdp(cfg))


def _work(args: tuple[int, str, list[int], list[float]]) -> list[dict[str, Any]]:
    seed, stage, reps, grid = args
    assert _WORKER_EXP is not None
    g = np.asarray(grid, dtype=np.float64)
    return [run_replication(_WORKER_EXP, seed, stage, r, g) for r in reps]


def run_stage(cfg: dict[str, Any], stage: str, n_reps: int, grid: np.ndarray,
              workers: int, log=print) -> tuple[pd.DataFrame, pd.DataFrame]:
    reps = list(range(n_reps))
    chunk = max(1, math.ceil(n_reps / (workers * 8)))
    batches = [(int(cfg["seed"]), stage, reps[i:i + chunk], [float(x) for x in grid])
               for i in range(0, n_reps, chunk)]
    records: list[dict[str, Any]] = []
    noise: list[dict[str, Any]] = []
    t0 = time.time()
    done = 0
    if workers <= 1:
        _init_worker(cfg)
        results = map(_work, batches)
        executor = None
    else:
        import multiprocessing as mp

        executor = ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                                       initializer=_init_worker, initargs=(cfg,))
        results = executor.map(_work, batches)
    try:
        for batch in results:
            for res in batch:
                records.extend(res["records"])
                noise.extend(res["noise"])
            done += 1
            if done % max(1, len(batches) // 10) == 0 or done == len(batches):
                log(f"  [{stage}] {done}/{len(batches)} batches, {time.time() - t0:.1f}s")
    finally:
        if executor is not None:
            executor.shutdown()
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(noise)


def epsilon_grid(start: float, end: float, size: int) -> np.ndarray:
    return np.geomspace(start, end, size)


# --------------------------------------------------------------------------- aggregation

def summarize(records: pd.DataFrame) -> pd.DataFrame:
    """Per (arm, mechanism, eps) summary; baselines have eps = inf."""
    def agg(g: pd.DataFrame) -> pd.Series:
        err = g["err"].to_numpy()
        n = err.size
        out = {
            "n_reps": n,
            "bias": err.mean(),
            "sd": err.std(ddof=1) if n > 1 else math.nan,
            "mse": np.mean(err ** 2),
            "mse_se": np.std(err ** 2, ddof=1) / math.sqrt(n) if n > 1 else math.nan,
            "median_abs_err": np.median(np.abs(err)),
            "excess": g["excess"].mean(),
            "excess_se": g["excess"].std(ddof=1) / math.sqrt(n) if n > 1 else math.nan,
            "u_norm": g["u_norm"].mean(),
            "balance": g["balance"].mean(),
            "match_bias": g["match_bias"].mean(),
        }
        for col in ("cert", "cert_lemma", "transfer", "bound"):
            out[col] = g[col].mean() if col in g and g[col].notna().any() else math.nan
        out["excess_over_bound"] = out["excess"] / out["bound"] if np.isfinite(out["bound"]) else math.nan
        return pd.Series(out)

    cols = ["arm", "mech", "eps"]
    return records.groupby(cols, sort=True).apply(agg, include_groups=False).reset_index()


def _err_cube(records: pd.DataFrame, arms, mechs, grid) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """err[arm, mech, eps, rep] for private arms and baseline err[arm][cond] -> (rep,)."""
    n_rep = int(records["rep"].max()) + 1
    cube = np.full((len(arms), len(mechs), len(grid), n_rep), np.nan)
    priv = records[np.isfinite(records["eps"])]
    log_grid = np.log(np.asarray(grid, dtype=np.float64))
    for (a, mch, e), g in priv.groupby(["arm", "mech", "eps"]):
        if mch in mechs:
            k = int(np.argmin(np.abs(log_grid - math.log(float(e)))))
            if abs(log_grid[k] - math.log(float(e))) > 1e-9:
                raise ValueError(f"record eps {e} is not on the production grid")
            cube[arms.index(a), mechs.index(mch), k, g["rep"].to_numpy()] = g["err"].to_numpy()
    base: dict[str, np.ndarray] = {}
    for (a, cond), g in records[~np.isfinite(records["eps"])].groupby(["arm", "mech"]):
        arr = np.full(n_rep, np.nan)
        arr[g["rep"].to_numpy()] = g["err"].to_numpy()
        base[f"{a}:{cond}"] = arr
    return cube, base


def crossing_eps10(grid: np.ndarray, mse: np.ndarray, ref: float, frac: float) -> tuple[float, str]:
    """Smallest eps from which MSE stays within frac of ref for all larger grid eps (log interpolation)."""
    ok = np.abs(mse / ref - 1.0) <= frac
    if ok.all():
        return float(grid[0]), "<=grid_min"
    if not ok[-1]:
        return math.nan, ">grid_max"
    j = int(np.max(np.flatnonzero(~ok))) + 1     # first index of the final all-ok run
    lo, hi = j - 1, j
    f_lo = abs(mse[lo] / ref - 1.0) - frac
    f_hi = abs(mse[hi] / ref - 1.0) - frac
    t = f_lo / (f_lo - f_hi) if f_lo != f_hi else 1.0
    return float(math.exp(math.log(grid[lo]) + t * (math.log(grid[hi]) - math.log(grid[lo])))), "interpolated"


def crossing_eps0(grid: np.ndarray, mse: np.ndarray, ref: float) -> tuple[float, str]:
    """Largest eps at which MSE is no better than random matching (MSE >= ref), log-interpolated."""
    bad = mse >= ref
    if not bad.any():
        return math.nan, "<grid_min"
    j = int(np.max(np.flatnonzero(bad)))
    if j == len(grid) - 1:
        return float(grid[-1]), ">=grid_max"
    f_lo, f_hi = mse[j] - ref, mse[j + 1] - ref
    t = f_lo / (f_lo - f_hi) if f_lo != f_hi else 0.0
    return float(math.exp(math.log(grid[j]) + t * (math.log(grid[j + 1]) - math.log(grid[j])))), "interpolated"


def headline(records: pd.DataFrame, exp: Experiment, grid: np.ndarray, frac: float,
             n_boot: int, seed: int) -> pd.DataFrame:
    arms = list(ARMS)
    mechs = [m for m in (MECH_LAMBDA, MECH_PHI, MECH_XI, MECH_PSI2) if m in set(records["mech"])]
    cube, base = _err_cube(records, arms, mechs, grid)
    n_rep = cube.shape[-1]
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99]))
    boot_idx = rng.integers(0, n_rep, size=(n_boot, n_rep))
    rows = []
    d = exp.data.d
    for ai, a in enumerate(arms):
        e_np, e_rand = base[f"{a}:{COND_NP}"], base[f"{a}:{COND_RANDOM}"]
        for mi, mch in enumerate(mechs):
            if np.all(np.isnan(cube[ai, mi])):
                continue
            mse = np.nanmean(cube[ai, mi] ** 2, axis=-1)
            e10, s10 = crossing_eps10(grid, mse, float(np.mean(e_np ** 2)), frac)
            e0, s0 = crossing_eps0(grid, mse, float(np.mean(e_rand ** 2)))
            b10, b0 = [], []
            for idx in boot_idx:
                m_b = np.mean(cube[ai, mi][:, idx] ** 2, axis=-1)
                b10.append(crossing_eps10(grid, m_b, float(np.mean(e_np[idx] ** 2)), frac)[0])
                b0.append(crossing_eps0(grid, m_b, float(np.mean(e_rand[idx] ** 2)))[0])
            b10, b0 = np.asarray(b10), np.asarray(b0)
            geom = exp.arms[a]
            rows.append({
                "arm": a, "mech": mch,
                "eps10": e10, "eps10_status": s10,
                "eps10_lo": np.nanpercentile(b10, 2.5) if np.isfinite(b10).any() else math.nan,
                "eps10_hi": np.nanpercentile(b10, 97.5) if np.isfinite(b10).any() else math.nan,
                "eps10_boot_undefined_frac": float(np.mean(~np.isfinite(b10))),
                "eps0": e0, "eps0_status": s0,
                "eps0_lo": np.nanpercentile(b0, 2.5) if np.isfinite(b0).any() else math.nan,
                "eps0_hi": np.nanpercentile(b0, 97.5) if np.isfinite(b0).any() else math.nan,
                "eps0_boot_undefined_frac": float(np.mean(~np.isfinite(b0))),
                "d_over_eps10_over_mbar": d / e10 / exp.mbar if np.isfinite(e10) else math.nan,
                "d_over_eps0_over_mbar": d / e0 / exp.mbar if np.isfinite(e0) else math.nan,
                "predicted_ratio_to_Lambda": geom.theorem_bound(mch, 1.0) / geom.theorem_bound(MECH_LAMBDA, 1.0),
            })
    df = pd.DataFrame(rows)
    lam = df[df["mech"] == MECH_LAMBDA].set_index("arm")["eps10"]
    df["observed_ratio_to_Lambda"] = [r.eps10 / lam.get(r.arm, math.nan) for r in df.itertuples()]
    a_e10 = df[df["arm"] == "A"].set_index("mech")["eps10"]
    df["eps10_B_over_A"] = [r.eps10 / a_e10.get(r.mech, math.nan) if r.arm == "B" else math.nan
                            for r in df.itertuples()]
    return df


# --------------------------------------------------------------------------- sanity checks

def sanity_checks(records: pd.DataFrame, noise: pd.DataFrame, summary: pd.DataFrame,
                  exp: Experiment, grid: np.ndarray, cond1_alt_mse: float | None) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, passed: bool | None, detail: str) -> None:
        status = "PASS" if passed else ("FLAG" if passed is None else "FAIL")
        checks.append({"check": name, "status": status, "detail": detail})

    base = summary[~np.isfinite(summary["eps"])].set_index(["arm", "mech"])
    mse1 = float(base.loc[("A", COND_SQ), "mse"])
    rel = mse1 / KALLUS_COND1_MSE - 1.0
    alt = "" if cond1_alt_mse is None else f"; with continuous covariates standardized on the other sample (hill <-> analysis): {cond1_alt_mse:.3f}"
    add("Condition 1 MSE vs Kallus 7.56 (+/-15%)", True if abs(rel) <= 0.15 else None,
        f"MSE = {mse1:.3f} ({100 * rel:+.1f}%){alt}. Not tuned.")

    bl = records[~np.isfinite(records["eps"])]
    same = True
    for cond in BASELINES:
        a = bl[(bl["arm"] == "A") & (bl["mech"] == cond)].sort_values("rep")
        b = bl[(bl["arm"] == "B") & (bl["mech"] == cond)].sort_values("rep")
        same &= bool(np.allclose(a[["err", "excess", "balance"]].to_numpy(),
                                 b[["err", "excess", "balance"]].to_numpy(), rtol=0, atol=1e-10))
    add("Conditions 1, 2, 4 identical in both arms", same, "per-replication err/excess/balance compared.")

    priv = records[np.isfinite(records["eps"])].copy()
    tol = 1e-9 * np.maximum(priv["cost_scale"].to_numpy(), 1.0)
    viol_plan = int(np.sum(priv["excess"].to_numpy() > priv["cert"].to_numpy() + tol))
    viol_tr = int(np.sum(priv["excess"].to_numpy() > priv["transfer"].to_numpy() + tol))
    viol_tr2 = int(np.sum(priv["transfer"].to_numpy() > priv["cert"].to_numpy() + tol))
    add("Pointwise certificate never violated", viol_plan == 0 and viol_tr == 0 and viol_tr2 == 0,
        f"{len(priv)} runs; excess > certificate: {viol_plan}; excess > transfer gap: {viol_tr}; "
        f"transfer gap > certificate: {viol_tr2}.")
    a_rows = priv[priv["arm"] == "A"]
    viol_lemma = int(np.sum(a_rows["excess"].to_numpy() > a_rows["cert_lemma"].to_numpy()
                            + 1e-9 * np.maximum(a_rows["cost_scale"].to_numpy(), 1.0)))
    add("Arm A Lemma 11/28 range certificate never violated", viol_lemma == 0, f"violations: {viol_lemma}.")

    xi = priv[priv["mech"] == MECH_XI]
    rmin = float(xi["radicand_min"].min()) if len(xi) else math.nan
    add("Xi radicand never negative", bool(rmin >= -1e-9 * max(float(xi["cost_scale"].max()) ** 2, 1.0)),
        f"min radicand = {rmin:.3e} (float round-off tolerance 1e-9 * scale^2).")

    for (a, mch), g in noise.groupby(["arm", "mech"]):
        cnt = float(g["count"].sum())
        mean = float(g["sum_norm"].sum()) / cnt
        var = float(g["sum_sq_norm"].sum()) / cnt - mean ** 2
        exp_norm = float(g["expected_norm"].iloc[0])
        z = (mean - exp_norm) / math.sqrt(var / cnt)
        label = "||Z||_inf" if mch == MECH_PHI else "||Z||_2"
        detail = f"mean {label} at eps=1: {mean:.4f} vs {exp_norm:.0f} (z = {z:+.2f}, {int(cnt)} draws)"
        ok = abs(z) < 4.0
        if mch == MECH_XI:
            me = float(g["sum_abs_eta"].sum()) / cnt
            ve = float(g["sum_sq_eta"].sum()) / cnt - me ** 2
            ze = (me - 1.0) / math.sqrt(ve / cnt)
            geom = exp.arms[a]
            detail += (f"; mean |eta| at eps=1: {me:.4f} vs 1 (z = {ze:+.2f}). So E||Z|| = m/eps_y = "
                       f"{exp_norm:.0f}/({geom.eps_y_frac:.4f} eps), E|eta| = 1/({geom.eps_s_frac:.4f} eps).")
            ok = ok and abs(ze) < 4.0
        add(f"Noise sampler {mch} arm {a}", ok, detail)

    rows = np.arange(exp.data.n1)
    worst = 0.0
    for arm, geom in exp.arms.items():
        C_priv = exp.C if arm == "A" else exp.C.T
        for mech in geom.mechanisms:
            noise = draw_base_noise(geom, mech, _rng(0, "pilot", 0, 99, _MECH_CODES[mech]))
            rel = release(geom, mech, noise, 1e12, C_priv)
            sig = lsa_rows(rel.cost_priv if arm == "A" else rel.cost_priv.T)
            worst = max(worst, float(exp.C[rows, sig].sum() - exp.opt) / exp.data.n1)
    add("eps -> infinity recovers OPT through every mechanism code path", worst <= 1e-9,
        f"max normalized excess at eps = 1e12 over arms x mechanisms: {worst:.2e}.")

    priv_sum = summary[np.isfinite(summary["eps"])]
    rand_excess = float(base.loc[("A", COND_RANDOM), "excess"])
    for (a, mch), g in priv_sum.groupby(["arm", "mech"]):
        g = g.sort_values("eps")
        ex = g["excess"].to_numpy()
        se = g["excess_se"].to_numpy()
        rises = int(np.sum(np.diff(ex) > 2.0 * np.sqrt(se[1:] ** 2 + se[:-1] ** 2)))
        add(f"Excess falls with eps ({mch}, arm {a})", rises == 0,
            f"excess {ex[0]:.3f} -> {ex[-1]:.4f} across grid; significant rises: {rises}; "
            f"at eps_min {ex[0] / rand_excess:.2f} x random excess ({rand_excess:.3f}); "
            f"at eps_max {ex[-1] / rand_excess:.3%} of random excess.")
        mean_bound_ok = bool(np.all(ex <= g["bound"].to_numpy() + 3 * se)) if a == "A" else None
        if a == "A":
            add(f"Mean excess below theorem bound ({mch}, arm A)", mean_bound_ok,
                f"max excess/bound = {np.max(ex / g['bound'].to_numpy()):.3f}")
    return checks


# --------------------------------------------------------------------------- orchestration

def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    raise TypeError(type(o))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    tmp.replace(path)


def condition1_mse(cfg: dict[str, Any], n_reps: int, stage: str = "production") -> float:
    """Condition 1 MSE with the same outcome draws but a different data preparation."""
    exp = build_experiment(load_ihdp(cfg))
    errs = []
    for rep in range(n_reps):
        out = draw_outcomes(exp.data, _rng(cfg["seed"], stage, rep, 0), _rng(cfg["seed"], stage, rep, 1))
        errs.append(match_metrics(exp, out, exp.sigma_sq)["err"])
    return float(np.mean(np.square(errs)))


def run_all(cfg: dict[str, Any], stages: tuple[str, ...] = ("pilot", "production", "report"),
            log=print) -> Path:
    root = Path(cfg["output_root"]) / cfg["name"]
    root.mkdir(parents=True, exist_ok=True)
    exp = build_experiment(load_ihdp(cfg))
    d, mbar = exp.data.d, exp.mbar
    eps_start = d / (cfg["grid_start_factor"] * mbar)
    eps_cap = cfg["grid_cap_factor"] / mbar
    geometry = {
        "n1": exp.data.n1, "n0": exp.data.n0, "d": d, "ridge": exp.ridge, "OPT": exp.opt,
        "mbar": mbar, "eps_start": eps_start, "eps_cap": eps_cap,
        "arms": {a: g.summary() for a, g in exp.arms.items()},
        "data_notes": exp.data.notes, "data_sha256": exp.data.source_sha256,
        "sq_vs_unsq_same_matching": bool(np.array_equal(exp.sigma_sq, exp.sigma_opt)),
    }
    log(json.dumps(geometry, indent=2, default=_json_default))
    write_json(root / "geometry.json", geometry)
    write_json(root / "config.json", cfg)
    workers = int(cfg["workers"])

    grid_path = root / "pilot" / "grid_choice.json"
    if "pilot" in stages:
        pgrid = epsilon_grid(eps_start, eps_cap, int(cfg["pilot_grid_size"]))
        log(f"Pilot: {cfg['pilot_replications']} reps on {len(pgrid)} eps in [{eps_start:.4g}, {eps_cap:.4g}]")
        prec, pnoise = run_stage(cfg, "pilot", int(cfg["pilot_replications"]), pgrid, workers, log)
        (root / "pilot").mkdir(exist_ok=True)
        prec.to_csv(root / "pilot" / "records.csv.gz", index=False, float_format="%.17g")
        psum = summarize(prec)
        psum.to_csv(root / "pilot" / "summary.csv", index=False, float_format="%.17g")
        base = psum[~np.isfinite(psum["eps"])].set_index(["arm", "mech"])
        phi_e10 = {}
        for a in ARMS:
            g = psum[(psum["arm"] == a) & (psum["mech"] == MECH_PHI)].sort_values("eps")
            phi_e10[a] = crossing_eps10(pgrid, g["mse"].to_numpy(), float(base.loc[(a, COND_NP), "mse"]),
                                        float(cfg["within_frac"]))
        finite = [v[0] for v in phi_e10.values() if np.isfinite(v[0])]
        if len(finite) == len(ARMS):
            eps_end = min(eps_cap, cfg["grid_end_margin"] * max(finite))
            reason = f"{cfg['grid_end_margin']} x max pilot eps10(Phi_inf), capped at {eps_cap:.4g}"
        else:
            eps_end = eps_cap
            reason = "Phi_inf never within 10% of condition 2 in the pilot in some arm; grid capped"
        choice = {"eps_start": eps_start, "eps_end": eps_end, "eps_cap": eps_cap, "reason": reason,
                  "pilot_phi_eps10": {a: {"eps10": v[0], "status": v[1]} for a, v in phi_e10.items()},
                  "pilot_seed_stage": "pilot (disjoint seed stream; never pooled with production)",
                  "grid": epsilon_grid(eps_start, eps_end, int(cfg["grid_size"]))}
        write_json(grid_path, choice)
        log(f"Grid choice: {json.dumps(choice, default=_json_default)}")
    if not grid_path.exists():
        raise RuntimeError("production requires pilot/grid_choice.json; run the pilot stage first")
    grid = np.asarray(json.loads(grid_path.read_text())["grid"], dtype=np.float64)

    prod = root / "production"
    if "production" in stages:
        log(f"Production: {cfg['replications']} reps on {len(grid)} eps")
        rec, noise = run_stage(cfg, "production", int(cfg["replications"]), grid, workers, log)
        prod.mkdir(exist_ok=True)
        rec.to_csv(prod / "records.csv.gz", index=False, float_format="%.17g")
        noise.to_csv(prod / "noise_checks.csv", index=False, float_format="%.17g")

    if "report" in stages:
        from private_matching.causal_ihdp_report import write_report

        rec = pd.read_csv(prod / "records.csv.gz")
        noise = pd.read_csv(prod / "noise_checks.csv")
        summary = summarize(rec)
        head = headline(rec, exp, grid, float(cfg["within_frac"]), int(cfg["bootstrap"]), int(cfg["seed"]))
        alt_cfg = dict(cfg, standardize_on="analysis" if cfg["standardize_on"] == "hill" else "hill")
        alt = condition1_mse(alt_cfg, int(cfg["replications"]))
        checks = sanity_checks(rec, noise, summary, exp, grid, alt)
        write_report(root, cfg, exp, geometry, grid, rec, summary, head, checks, log=log)
    return root
