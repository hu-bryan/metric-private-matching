"""Randomized minimax regret for interval (box) cost uncertainty; no prior on R.

For a doubly stochastic H (a probability mixture of permutation matchings, sampled via a
Birkhoff decomposition; NOT a fractional physical matching) and elementwise bounds l <= C <= u,

    F(H) = sup_{l<=C<=u} [<C,H> - min_P <C,P>] = <u,H> - min_P <l + (u-l)*H, P>.

The maximizing C is l on the selected P and u elsewhere. F is convex and min_H F(H) is an LP.
Two exact solvers are provided and cross-checked in tests:

* ``solve_compact_lp``: the inner min over the Birkhoff polytope is replaced by its LP dual
  (max sum(alpha)+sum(beta) s.t. alpha_i+beta_j <= l_ij+(u_ij-l_ij)H_ij), giving one LP with
  n^2+2n variables; no cutting planes, no iteration cap.
* ``double_oracle``: finite zero-sum game over output permutations S and adversary permutations P,
  payoff A(S,P) = <C^P,S> - <l,P>, restricted LPs plus Hungarian best responses.

Both return a feasible H with a verifiable upper bound U=F(H) (Hungarian) and lower bound
L = min_S <u-(u-l)*G, S> - <l,G> for a doubly stochastic adversary mixture G (Hungarian).
L <= min F <= U. This is exact optimality for the INTERVAL relaxation over randomized matchings,
not minimax optimality over Euclidean geometry, over channels, or for expected risk.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.optimize import linprog, linear_sum_assignment
from scipy.stats import gamma


def confidence_radius(m: int, epsilon: float, alpha: float, n: int) -> float:
    """Gamma(m, 1/eps) quantile at 1-alpha/n: P(max_j ||Z_j|| <= t) >= 1-alpha by a union bound."""
    if not (0 < alpha < 1 and m >= 1 and epsilon > 0 and n >= 1):
        raise ValueError("invalid confidence-radius arguments")
    return float(gamma.ppf(1-alpha/n, a=m, scale=1/epsilon))


def interval_bounds(d: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    return np.maximum(0., d-t), d+t


def _assign(cost):
    rows, cols = linear_sum_assignment(cost)
    perm = np.empty(len(rows), dtype=np.int64)
    perm[rows] = cols
    return perm


def perm_sum(M, perm):
    return float(M[np.arange(len(perm)), perm].sum())


def worst_case_regret(H, l, u):
    """Exact F(H) and the adversary's maximizing permutation (one Hungarian solve)."""
    M = l+(u-l)*H
    P = _assign(M)
    return float(np.sum(u*H))-perm_sum(M, P), P


def lower_bound(G, l, u):
    """Valid lower bound on min_H F(H) for any doubly stochastic adversary mixture G."""
    Cbar = u-(u-l)*G
    S = _assign(Cbar)
    return perm_sum(Cbar, S)-float(np.sum(l*G)), S


def perm_payoff(S, P, l, u):
    """A(S,P) = <C^P,S> - <l,P>, C^P = l on P and u elsewhere."""
    rows = np.arange(len(S))
    same = S == P
    return float(u[rows, S].sum()-(u-l)[rows[same], S[same]].sum()-l[rows, P].sum())


def ds_residual(X):
    return float(max(np.abs(X.sum(axis=0)-1).max(), np.abs(X.sum(axis=1)-1).max(), max(0., -X.min())))


def ds_repair(X):
    """Exact finite-step projection-free repair to a doubly stochastic matrix; returns (B, residual).

    Clip negatives; scale each row by min(1, 1/rowsum), then each column by min(1, 1/colsum), so all
    sums are <= 1. The deficits r = 1 - B1 and c = 1 - B^T 1 are nonnegative with equal totals delta,
    and B + r c^T / delta has unit row and column sums. Replaces Sinkhorn, which stalled at ~1e-7 on
    sparse nearly reducible LP supports (production units unstructured n25 near r3/r6 eps16).
    """
    B = np.maximum(np.asarray(X, dtype=np.float64), 0.)
    for axis in (1, 0):
        sums = B.sum(axis=axis)
        scale = np.minimum(1., np.divide(1., sums, out=np.ones_like(sums), where=sums > 0))
        B = B*(scale[:, None] if axis == 1 else scale[None, :])
    r = np.maximum(1.-B.sum(axis=1), 0.)
    c = np.maximum(1.-B.sum(axis=0), 0.)
    delta = .5*(r.sum()+c.sum())
    if delta > 0:
        B = B+np.outer(r, c)/delta
    return B, ds_residual(B)


def permutation_matrix(perm):
    n = len(perm)
    X = np.zeros((n, n))
    X[np.arange(n), perm] = 1.
    return X


def _structure(n):
    """Sparse constraint pattern shared by every LP of size n (cached)."""
    if n in _STRUCTURE:
        return _STRUCTURE[n]
    N = n*n
    r = np.arange(N)
    i, j = np.divmod(r, n)
    ub_rows = np.concatenate([r, r, r])
    ub_cols = np.concatenate([r, N+i, N+n+j])
    eq = sparse.csr_matrix((np.ones(2*N), (np.concatenate([i, n+j]), np.concatenate([r, r]))), shape=(2*n, N+2*n))
    _STRUCTURE[n] = (ub_rows, ub_cols, eq)
    return _STRUCTURE[n]


_STRUCTURE: dict = {}


LP_TOLERANCE = 1e-10  # HiGHS primal/dual feasibility (default 1e-7 left ~5e-8 row/column-sum errors)


def solve_compact_lp(l, u, fallback=None, time_limit=120.):
    """min_{H in B_n} F(H) as one LP (variables H, alpha, beta). Returns a certified result dict.

    ``fallback`` (e.g. the direct assignment) is returned, labeled not exact, if HiGHS fails.
    Raw LP residuals are recorded; H and G are then repaired exactly (``ds_repair``) and both
    Hungarian bounds are computed on the repaired matrices.
    """
    started = time.monotonic()
    n = len(l)
    N = n*n
    ub_rows, ub_cols, A_eq = _structure(n)
    width = (u-l).ravel()
    data = np.concatenate([-width, np.ones(N), np.ones(N)])
    A_ub = sparse.csr_matrix((data, (ub_rows, ub_cols)), shape=(N, N+2*n))
    c = np.concatenate([u.ravel(), -np.ones(2*n)])
    bounds = [(0, None)]*N+[(None, None)]*(2*n)
    res = linprog(c, A_ub=A_ub, b_ub=l.ravel(), A_eq=A_eq, b_eq=np.ones(2*n), bounds=bounds, method="highs",
                  options=dict(time_limit=time_limit, primal_feasibility_tolerance=LP_TOLERANCE,
                               dual_feasibility_tolerance=LP_TOLERANCE))
    exact = res.status == 0
    if exact:
        H_raw, G_raw = res.x[:N].reshape(n, n), -res.ineqlin.marginals.reshape(n, n)
        h_raw, g_raw = ds_residual(H_raw), ds_residual(G_raw)
        H, h_res = ds_repair(H_raw)
        G, g_res = ds_repair(G_raw)
        lp_value = float(res.fun)
    else:  # Time limit/failure: fall back to the feasible direct assignment and say so.
        H = permutation_matrix(_assign((l+u)/2) if fallback is None else fallback)
        G, h_res, g_res, h_raw, g_raw, lp_value = H.copy(), 0., 0., None, None, None
    upper, _ = worst_case_regret(H, l, u)
    lower, _ = lower_bound(G, l, u)
    return dict(H=H, G=G, upper=upper, lower=lower, gap=upper-lower, lp_value=lp_value, exact=exact,
                status=int(res.status), message=str(res.message), iterations=int(getattr(res, "nit", 0) or 0),
                H_residual=h_res, G_residual=g_res, H_raw_residual=h_raw, G_raw_residual=g_raw,
                seconds=time.monotonic()-started, solver="compact_lp_highs")


def _restricted_game(M):
    """Mixed equilibrium of the restricted matrix game min_h max_g h^T M g."""
    a, b = M.shape
    # Row player: min v s.t. M^T h <= v, sum h = 1.
    res_h = linprog(np.r_[np.zeros(a), 1.], A_ub=np.c_[M.T, -np.ones(b)], b_ub=np.zeros(b),
                    A_eq=np.r_[np.ones(a), 0.][None], b_eq=[1.], bounds=[(0, None)]*a+[(None, None)], method="highs")
    # Column player: max w s.t. M g >= w, sum g = 1.
    res_g = linprog(np.r_[np.zeros(b), -1.], A_ub=np.c_[-M, np.ones(a)], b_ub=np.zeros(a),
                    A_eq=np.r_[np.ones(b), 0.][None], b_eq=[1.], bounds=[(0, None)]*b+[(None, None)], method="highs")
    if res_h.status != 0 or res_g.status != 0:
        raise RuntimeError(f"restricted game LP failed: {res_h.message}; {res_g.message}")
    h = np.maximum(res_h.x[:a], 0.)
    g = np.maximum(res_g.x[:b], 0.)
    return h/h.sum(), g/g.sum()


def double_oracle(l, u, initial=None, rel_tol=1e-9, abs_tol=1e-9, max_iter=500):
    """Double oracle with Hungarian best responses; U-L certificate every iteration."""
    if max_iter < 1:
        raise ValueError("max_iter must be positive")
    started = time.monotonic()
    n = len(l)
    if initial is None:
        initial = [_assign((l+u)/2), np.arange(n)]
    S_list, P_list = [], []
    for perm in initial:
        perm = np.asarray(perm, dtype=np.int64)
        if not any(np.array_equal(perm, s) for s in S_list):
            S_list.append(perm)
        if not any(np.array_equal(perm, p) for p in P_list):
            P_list.append(perm)
    M = np.array([[perm_payoff(S, P, l, u) for P in P_list] for S in S_list])
    history = []
    converged = False
    for iteration in range(1, max_iter+1):
        h, g = _restricted_game(M)
        H = np.tensordot(h, np.array([permutation_matrix(S) for S in S_list]), axes=1)
        G = np.tensordot(g, np.array([permutation_matrix(P) for P in P_list]), axes=1)
        upper, P_new = worst_case_regret(H, l, u)
        lower, S_new = lower_bound(G, l, u)
        history.append((upper, lower))
        if upper-lower <= max(abs_tol, rel_tol*max(1., abs(upper))):
            converged = True
            break
        if iteration == max_iter:
            # Keep action lists aligned with the last solved mixture on a capped run.
            break
        added = False
        if not any(np.array_equal(S_new, s) for s in S_list):
            S_list.append(S_new)
            M = np.r_[M, [[perm_payoff(S_new, P, l, u) for P in P_list]]]
            added = True
        if not any(np.array_equal(P_new, p) for p in P_list):
            P_list.append(P_new)
            M = np.c_[M, [perm_payoff(S, P_new, l, u) for S in S_list]]
            added = True
        if not added:  # Both best responses already present: restricted equilibrium is global up to LP tolerance.
            converged = upper-lower <= 1e-7*max(1., abs(upper))
            break
    keep = h > 0
    return dict(H=H, G=G, upper=upper, lower=lower, gap=upper-lower, exact=converged, iterations=iteration,
                weights=h[keep], perms=np.array(S_list)[keep], adversary_weights=g, adversary_perms=np.array(P_list),
                history=history, seconds=time.monotonic()-started, solver="double_oracle",
                H_residual=ds_residual(H), G_residual=ds_residual(G))


def birkhoff_decomposition(H, tol=1e-10, max_terms=None):
    """Explicit mixture sum_s w_s S_s of permutations with sum w = 1 - leftover (recorded)."""
    R = np.array(H, dtype=np.float64)
    n = len(R)
    max_terms = max_terms or n*n+n
    weights, perms = [], []
    mass = 1.
    while mass > tol and len(weights) < max_terms:
        support = R > tol
        # Maximize retained mass inside the support; off-support entries are prohibitively costly.
        perm = _assign(np.where(support, -R, n+1.))
        entries = R[np.arange(n), perm]
        if np.any(entries <= tol):
            break
        w = float(entries.min())
        R[np.arange(n), perm] -= w
        weights.append(w)
        perms.append(perm)
        mass -= w
    return np.array(weights), np.array(perms, dtype=np.int64).reshape(len(perms), n), float(max(mass, 0.))
