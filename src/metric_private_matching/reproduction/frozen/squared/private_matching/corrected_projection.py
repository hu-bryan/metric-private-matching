"""The v1 manuscript's corrected projected representation, without span substitution.

The public map is identity when min(rank(Q)+1,d)==d. Otherwise it retains
public-span coordinates and the private perpendicular norm. Noise is sampled
in that FULL representation dimension. The budget split counts every nonzero
unordered public pair, as specified in the manuscript's experimental appendix.
"""
from dataclasses import dataclass

import numpy as np
from .anchor import AnchorGeometry, AnchorMessage, anchor_server, nearest_anchor, optimized_split


@dataclass(frozen=True)
class CorrectedProjection:
    Q: np.ndarray
    center: np.ndarray
    basis: np.ndarray
    rank: int
    tolerance: float
    residual: float
    identity: bool
    anchor: AnchorGeometry

    @classmethod
    def build(cls, Q):
        Q = np.asarray(Q, dtype=float)
        if Q.ndim != 2 or min(Q.shape) < 1 or not np.isfinite(Q).all():
            raise ValueError("finite nonempty public points required")
        center = Q.mean(axis=0)
        qc = Q-center
        _, singular, vt = np.linalg.svd(qc, full_matrices=False)
        tolerance = float(singular[0]*max(Q.shape)*np.finfo(float).eps)
        rank = int(np.sum(singular > tolerance))
        basis = vt[:rank].T.copy()
        residual = float(np.linalg.norm(qc-(qc@basis)@basis.T, axis=1).max())
        identity = min(rank+1, Q.shape[1]) == Q.shape[1]
        p = Q.copy() if identity else np.column_stack((qc@basis, np.zeros(len(Q))))
        return cls(Q.copy(), center, basis, rank, tolerance, residual, identity,
                   AnchorGeometry.from_references(p))

    @property
    def dimension(self):
        return self.anchor.p.shape[1]

    def transform(self, R):
        R = np.asarray(R, dtype=float)
        if self.identity:
            return R.copy()
        rc = R-self.center
        along = rc@self.basis
        return np.column_stack((along, np.linalg.norm(rc-along@self.basis.T, axis=1)))

    def split(self, epsilon):
        return optimized_split(self.anchor.B, epsilon)

    def decode_draw(self, R, unit_noise, unit_scalar, epsilon):
        """Decode one actual full-dimensional noise realization; return audit data.

        Unit noise is Euclidean Laplace of rate one in self.dimension, never
        a fresh draw in the public span. Costs are private-record rows first.
        """
        return self.decode_folded(self.transform(R),unit_noise,unit_scalar,epsilon)

    def decode_folded(self, u, unit_noise, unit_scalar, epsilon):
        """The same release/decoder on an already computed exact public fold."""
        if np.shape(unit_noise) != u.shape:
            raise ValueError("noise must use the full folded representation dimension")
        if self.anchor.B == 0:
            return np.zeros((len(u),len(self.Q))), np.zeros(len(u)), np.zeros(len(u),dtype=int), 0., 0.
        ey, es = self.split(epsilon)
        W, eta = unit_noise/ey, unit_scalar/es
        Y = u+W
        a = nearest_anchor(self.anchor.p, Y)
        radius = np.linalg.norm(u-self.anchor.p[a], axis=1)
        message = AnchorMessage(Y, np.maximum(0., radius+eta))
        decoded = anchor_server(message, self.anchor.p).T
        certificate = 4*self.anchor.max_projection(W)+np.abs(eta)
        return decoded, certificate, a, ey, es
