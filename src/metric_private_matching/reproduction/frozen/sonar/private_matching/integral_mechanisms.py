"""Integral matching mechanisms; true objectives are SUMS, private rows first.

This module is independent of the frozen cube report/decoder implementations.
All calibration quantities are public. Unit draws are deliberately coupled
across epsilon; the run record makes this coupling and endpoint reuse explicit.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist, pdist, squareform

from private_matching.mechanisms import sample_euclidean_laplace

ARMS = {
    1: ("gumbel", "terminal_fold", "anchor", "ambient_radial", "ambient_iid", "cost_infinity", "cost_iid"),
    2: ("gumbel", "span_radial", "coordinate_allocated", "ambient_radial", "ambient_iid", "cost_infinity", "cost_iid"),
}


def solve(cost):
    rows, cols = linear_sum_assignment(cost)
    if len(cols) != len(cost) or not np.array_equal(rows, np.arange(len(cost))):
        raise RuntimeError("incomplete assignment")
    return cols


def supnorm_unit(n, rng):
    """Row-wise cone directions times Gamma(n,1): density exp(-||e||_inf)."""
    cube = rng.uniform(-1., 1., (n, n))
    norms = np.max(np.abs(cube), axis=1)
    if np.any(norms == 0):
        raise FloatingPointError("zero supnorm direction")
    return cube * (rng.gamma(n, size=n) / norms)[:, None]


@dataclass
class Geometry:
    X: np.ndarray
    Y: np.ndarray
    center: np.ndarray
    B: np.ndarray
    singular: np.ndarray
    tolerance: float
    rank: int
    U: np.ndarray
    V: np.ndarray
    fold_X: np.ndarray
    fold_Y: np.ndarray
    C1: np.ndarray
    C2: np.ndarray
    base2: np.ndarray
    DY: np.ndarray
    diameter: float
    radius: float
    widths: np.ndarray
    l1_squared: float
    residual: float

    @classmethod
    def build(cls, X, Y):
        X, Y = np.asarray(X, float), np.asarray(Y, float)
        n, d = Y.shape
        center = Y.mean(axis=0)
        yc, xc = Y-center, X-center
        _, singular, vt = np.linalg.svd(yc, full_matrices=False)
        tolerance = float(singular[0]*max(n, d)*np.finfo(float).eps) if singular.size else 0.
        rank = int(np.sum(singular > tolerance))
        B = np.ascontiguousarray(vt[:rank].T)
        U, V = xc @ B, yc @ B
        residual = float(np.max(np.linalg.norm(yc-V @ B.T, axis=1)))
        if rank < d:
            fold_X = np.column_stack((U, np.linalg.norm(xc-U @ B.T, axis=1)))
            fold_Y = np.column_stack((V, np.zeros(n)))
        else:
            fold_X, fold_Y = U.copy(), V.copy()
        C1 = cdist(X, Y)
        DY = squareform(pdist(Y))
        widths = np.ptp(V, axis=0)
        radii = np.linalg.norm(yc, axis=1)
        l1 = 2*min(float(radii.sum()), float(np.sqrt(n)*singular[0])) if rank else 0.
        return cls(X, Y, center, B, singular, tolerance, rank, U, V, fold_X,
                   fold_Y, C1, C1*C1, -2*U @ V.T, DY, float(DY.max()),
                   float(radii.max()), widths, l1, residual)

    def public_metadata(self):
        n, d = self.Y.shape
        return dict(public_rank=self.rank, analytic_rank=min(n-1,d),
                    rank_tolerance=self.tolerance, projection_residual=self.residual,
                    public_diameter=self.diameter, public_radius=self.radius,
                    coordinate_width_certificate=float(np.sum(self.widths**(2/3))**1.5),
                    squared_linf=2*self.radius, squared_l1=self.l1_squared,
                    fold_dimension=self.fold_X.shape[1])

    def arrays(self):
        return {k: getattr(self,k) for k in ("X","Y","center","B","singular","U","V","widths")}


def bound(geo, arm, p, epsilon, family):
    """Analytic expected TOTAL regret certificate; no fitted/Monte Carlo constants."""
    n, d = geo.Y.shape
    r, D = geo.rank, geo.diameter
    a = lambda t: min(t, 2*np.sqrt(t*np.log(n))) if t else 0.
    cap = n*D if p == 1 else (2*np.sqrt(3)*n*D if family == "U" else np.inf)
    L = (1. if p == 1 else 2*geo.radius)
    L1 = float(n) if p == 1 else geo.l1_squared
    ey, es = epsilon, 0.
    if D == 0:
        value = 0.
    elif arm == "gumbel":
        value = 2*n*np.log(n)/epsilon*(1 if p == 1 else D)
    elif arm == "span_radial":
        value = 2*n*D*a(r)/epsilon
    elif arm == "coordinate_allocated":
        value = 2*n*np.sum(geo.widths**(2/3))**1.5/epsilon
    elif arm == "terminal_fold":
        value = 2*n*min(r+1,d)/epsilon
    elif arm == "anchor":
        ey = epsilon*2*np.sqrt(a(r))/(2*np.sqrt(a(r))+1)
        es = epsilon-ey
        value = n*(2*np.sqrt(a(r))+1)**2/epsilon
    elif arm == "ambient_radial":
        value = (2*n*d if p == 1 else 2*n*D*a(d))/epsilon
    elif arm == "ambient_iid":
        value = (2*np.sqrt(2)*n*d if p == 1 else 2*n*D*np.sqrt(2*r*d))/epsilon
    elif arm == "cost_infinity":
        value = 2*n*n*L/epsilon
    elif arm == "cost_iid":
        value = 2*n*L1*np.sum(1/np.arange(1,n+1))/epsilon
    else:
        raise ValueError(arm)
    return dict(bound_uncapped=float(value), bound_cap=float(cap),
                bound_total=float(min(value,cap)), epsilon_y=float(ey), epsilon_s=float(es),
                L_infinity=float(L), L_one=float(L1), bound_id=f"integral-sum-v1:{p}:{arm}")


def unit_channels(geo, rng_for):
    n, d = geo.Y.shape
    r, k = geo.rank, geo.fold_X.shape[1]
    channels, timings = {}, {}
    def add(name, fn):
        t0 = time.perf_counter()
        channels[name] = fn(rng_for(name))
        timings[name] = time.perf_counter()-t0
    add("ambient_radial", lambda rng: sample_euclidean_laplace(n,d,1.,rng)[0])
    add("ambient_iid", lambda rng: np.sqrt(d)*rng.laplace(size=(n,d)))
    add("gumbel", lambda rng: rng.gumbel(size=(n,n)))
    add("cost_infinity", lambda rng: supnorm_unit(n,rng))
    add("cost_iid", lambda rng: rng.laplace(size=(n,n)))
    if r:
        if r != d:
            add("span_radial", lambda rng: sample_euclidean_laplace(n,r,1.,rng)[0])
        add("coordinate_allocated", lambda rng: rng.laplace(size=(n,r)) /
            (geo.widths**(1/3)/np.sqrt(np.sum(geo.widths**(2/3))))[None,:])
        if k != d:
            add("terminal_fold", lambda rng: sample_euclidean_laplace(n,k,1.,rng)[0])
        add("anchor", lambda rng: sample_euclidean_laplace(n,r,1.,rng)[0])
        add("anchor_scalar", lambda rng: rng.laplace(size=n))
    # Linear squared-score perturbations are reused across epsilon and objectives.
    start = time.perf_counter()
    channels["ambient_radial_dot"] = -2*channels["ambient_radial"] @ (geo.Y-geo.center).T
    channels["ambient_iid_dot"] = -2*channels["ambient_iid"] @ (geo.Y-geo.center).T
    if r:
        if r != d:
            channels["span_radial_dot"] = -2*channels["span_radial"] @ geo.V.T
        channels["coordinate_allocated_dot"] = -2*channels["coordinate_allocated"] @ geo.V.T
    timings["reusable_squared_projections"] = time.perf_counter()-start
    return channels, timings


def evaluate_draw(geo, channels, epsilon, family, optima, precomputed=None):
    """Yield p/arm outcomes. Endpoint aliases copy the same realized permutation."""
    n, d = geo.Y.shape
    r, k = geo.rank, geo.fold_X.shape[1]
    results = {}
    def put(p, arm, noisy, *, source="", reuse="", secant=None):
        start = time.perf_counter()
        true = geo.C1 if p == 1 else geo.C2
        baseline = geo.C1 if p == 1 else geo.base2
        if precomputed and (p, arm) in precomputed:
            item = dict(precomputed[p, arm], runtime_seconds=0., solve_seconds=0.)
            pi = item['permutation']
            item.update(total_cost=float(true[np.arange(n), pi].sum()))
            item['total_regret'] = item['total_cost']-optima[p]
            item.update(bound(geo,arm,p,epsilon,family))
            if item['total_regret'] > item['realization_bound']+1e-8*max(1.,item['realization_bound']):
                raise FloatingPointError('reused isometric realization certificate failed')
            results[p, arm] = item
            return
        if source:
            old = results[p,source]
            item = dict(old, source_arm=source, reuse_reason=reuse, runtime_seconds=0., solve_seconds=0.)
            item.update(bound(geo,arm,p,epsilon,family))
        else:
            if callable(noisy):
                noisy = noisy()
            if not np.all(np.isfinite(noisy)):
                raise FloatingPointError(f"nonfinite costs: {p} {arm}")
            tsolve = time.perf_counter()
            pi = solve(noisy)
            solve_seconds = time.perf_counter()-tsolve
            cost = float(true[np.arange(n),pi].sum())
            regret = cost-optima[p]
            osc = np.ptp(noisy-baseline, axis=1)
            realization = float(osc.sum())
            tol = 1e-8*max(1.,abs(cost),abs(optima[p]),realization)
            if regret < -tol or regret > realization+tol:
                raise FloatingPointError(f"stability failed: {p} {arm} {regret} > {realization}")
            if secant is not None:
                if np.any(osc > secant+1e-8*np.maximum(1.,secant)):
                    raise FloatingPointError("anchor secant row certificate failed")
                realization = float(secant.sum())
            item = dict(permutation=pi, total_cost=cost, total_regret=max(0.,regret),
                        row_oscillation=float(osc.sum()), realization_bound=realization,
                        runtime_seconds=time.perf_counter()-start,
                        solve_seconds=solve_seconds, source_arm="", reuse_reason="",
                        pointwise_ok=True)
            item.update(bound(geo,arm,p,epsilon,family))
        results[p,arm] = item

    if r == 0:
        for p in (1,2):
            for arm in ARMS[p]:
                put(p,arm,np.zeros((n,n)))
        return results
    # Cache expensive report decoders and explicitly reuse equivalent endpoints.
    for arm in ("ambient_radial","ambient_iid"):
        t0 = time.perf_counter()
        noisy1 = cdist(geo.X+channels[arm]/epsilon, geo.Y)
        decode_seconds = time.perf_counter()-t0
        put(1,arm,noisy1)
        results[1,arm]["runtime_seconds"] += decode_seconds
        put(2,arm,lambda: geo.base2+channels[arm+"_dot"]/epsilon)
    put(1,"gumbel",lambda: geo.C1-2*channels["gumbel"]/epsilon)
    put(2,"gumbel",lambda: geo.base2-2*geo.diameter*channels["gumbel"]/epsilon)
    for p in (1,2):
        b = bound(geo,"cost_infinity",p,epsilon,family)
        base = geo.C1 if p == 1 else geo.base2
        put(p,"cost_infinity",lambda: base+b["L_infinity"]*channels["cost_infinity"]/epsilon)
        put(p,"cost_iid",lambda: base+b["L_one"]*channels["cost_iid"]/epsilon)
    if r == d:
        put(2,"span_radial",None,source="ambient_radial",reuse="isotropic_full_span_squared_cost_law")
    else:
        put(2,"span_radial",lambda: geo.base2+channels["span_radial_dot"]/epsilon)
    put(2,"coordinate_allocated",lambda: geo.base2+channels["coordinate_allocated_dot"]/epsilon)
    if k == d:
        put(1,"terminal_fold",None,source="ambient_radial",reuse="isotropic_full_fold_euclidean_cost_law")
    else:
        t0 = time.perf_counter()
        noisy = cdist(geo.fold_X+channels["terminal_fold"]/epsilon,geo.fold_Y)
        seconds = time.perf_counter()-t0
        put(1,"terminal_fold",noisy)
        results[1,"terminal_fold"]["runtime_seconds"] += seconds
    t0 = time.perf_counter()
    b = bound(geo,"anchor",1,epsilon,family)
    Z = channels["anchor"]/b["epsilon_y"]
    z = geo.U+Z
    normv = np.sum(geo.V*geo.V,axis=1)
    scores = normv[None,:]-2*z @ geo.V.T
    anchors = scores.argmin(axis=1)
    # Stable difference of squared distances, retaining row nonnegativity.
    delta = scores-scores[np.arange(n),anchors,None]
    radius = np.linalg.norm(geo.X-geo.Y[anchors],axis=1)
    eta = channels["anchor_scalar"]/b["epsilon_s"]
    T = np.maximum(0.,radius+eta)
    noisy = np.sqrt(T[:,None]**2+np.maximum(delta,0.))
    # The proof only needs secants from the ACTUAL anchor: this sharper
    # realization certificate is <= the note's all-pairs secant maximum.
    projections = Z @ geo.V.T
    dots = np.abs(projections-projections[np.arange(n),anchors,None])
    lengths = geo.DY[anchors]
    ratios = np.divide(dots,lengths,out=np.zeros_like(dots),where=lengths>0)
    secant = 4*ratios.max(axis=1)+np.abs(eta)
    seconds = time.perf_counter()-t0
    put(1,"anchor",noisy,secant=secant)
    results[1,"anchor"]["runtime_seconds"] += seconds
    return results
