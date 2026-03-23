"""
crw.py – Classical continuous-time random walk (CRW) simulator.

Theory
------
The classical random walk is governed by the master equation:

    dp/dt = −L p

where L is the graph Laplacian  L = D − A  (same matrix as the CTQW
Hamiltonian when γ = 1) and p(t) is the probability distribution over
nodes.

The solution is:

    p(t) = exp(−L t) p(0)

which can again be evaluated efficiently via diagonalisation of L.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.linalg as la


class CRW:
    """Classical continuous-time random walk on a graph.

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph (dense or sparse, shape N×N).
    """

    def __init__(self, adjacency: np.ndarray | sp.spmatrix) -> None:
        if sp.issparse(adjacency):
            adj = adjacency.toarray()
        else:
            adj = np.asarray(adjacency, dtype=float)

        self._n = adj.shape[0]
        degree = adj.sum(axis=1)
        D = np.diag(degree)
        self._L = D - adj  # Graph Laplacian

        # Pre-diagonalise (L is real-symmetric positive semi-definite)
        self._eigenvalues, self._eigenvectors = la.eigh(self._L)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._n

    @property
    def laplacian(self) -> np.ndarray:
        """The graph Laplacian matrix L."""
        return self._L.copy()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        start_node: int,
        times: np.ndarray,
    ) -> np.ndarray:
        """Run a classical random-walk time-series from *start_node*.

        Parameters
        ----------
        start_node:
            Index of the starting node (0-indexed).
        times:
            1-D array of time values at which probabilities are recorded.

        Returns
        -------
        prob_matrix : float ndarray, shape (len(times), N)
            prob_matrix[k, j] = P(node j at time times[k]).
        """
        if not (0 <= start_node < self._n):
            raise ValueError(
                f"start_node {start_node} is out of range [0, {self._n})."
            )

        times = np.asarray(times, dtype=float)
        prob_matrix = np.zeros((len(times), self._n), dtype=float)

        p0 = np.zeros(self._n, dtype=float)
        p0[start_node] = 1.0

        # Project initial distribution onto eigenvectors
        # L = V Λ Vᵀ  →  exp(−Lt) = V exp(−Λt) Vᵀ
        coeffs0 = self._eigenvectors.T @ p0  # shape (N,)

        for k, t in enumerate(times):
            decay = np.exp(-self._eigenvalues * t)
            p_t = self._eigenvectors @ (decay * coeffs0)
            # Numerical noise may yield tiny negatives – clip and renormalise
            p_t = np.clip(p_t, 0.0, None)
            total = p_t.sum()
            if total > 0:
                p_t /= total
            prob_matrix[k] = p_t

        return prob_matrix

    def variance(
        self,
        start_node: int,
        positions: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Compute variance σ²(t) of the classical probability distribution.

        Parameters
        ----------
        start_node:
            Starting node index.
        positions:
            (N, 2) array of node coordinates.
        times:
            1-D array of time values.

        Returns
        -------
        var : float ndarray, shape (len(times),)
        """
        positions = np.asarray(positions, dtype=float)
        prob_matrix = self.simulate(start_node, times)
        mean_r = prob_matrix @ positions
        r_sq = (positions ** 2).sum(axis=1)
        mean_r_sq = prob_matrix @ r_sq
        var = mean_r_sq - (mean_r ** 2).sum(axis=1)
        return var
