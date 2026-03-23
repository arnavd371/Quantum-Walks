"""
ctqw.py – Continuous-time quantum walk (CTQW) simulator.

Theory
------
Given a graph with adjacency matrix A, the CTQW Hamiltonian is taken to be
the graph Laplacian:

    H = γ · (D − A)

where D is the diagonal degree matrix and γ is a coupling constant (default 1).

The time-evolution operator is:

    U(t) = exp(−i H t)

Starting from an initial state |ψ₀⟩ (a unit vector), the state at time t is:

    |ψ(t)⟩ = U(t)|ψ₀⟩

and the probability of being at node j is:

    P(j, t) = |⟨j|ψ(t)⟩|²

Implementation notes
--------------------
For small/medium graphs the Hamiltonian is diagonalised once:

    H = V Λ Vᵀ  (real symmetric → real eigenvalues)

so that  U(t) = V · diag(exp(−i λₖ t)) · Vᵀ

This makes repeated time-steps O(N²) after an O(N³) one-time cost.
For large sparse graphs a Krylov / Padé approximation can be used by setting
`use_expm=True`.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la


class CTQW:
    """Continuous-time quantum walk on a graph.

    Parameters
    ----------
    adjacency:
        Adjacency matrix of the graph (dense or sparse, shape N×N).
    gamma:
        Coupling constant for the Hamiltonian (default 1.0).
    use_laplacian:
        If True (default) use the graph Laplacian H = γ(D−A); otherwise use
        H = γ A directly.
    """

    def __init__(
        self,
        adjacency: np.ndarray | sp.spmatrix,
        gamma: float = 1.0,
        use_laplacian: bool = True,
    ) -> None:
        if sp.issparse(adjacency):
            self._adj = adjacency.toarray()
        else:
            self._adj = np.asarray(adjacency, dtype=float)

        self._n = self._adj.shape[0]
        self._gamma = gamma

        if use_laplacian:
            degree = self._adj.sum(axis=1)
            D = np.diag(degree)
            self._H = gamma * (D - self._adj)
        else:
            self._H = gamma * self._adj

        # Pre-diagonalise (H is real-symmetric)
        self._eigenvalues, self._eigenvectors = la.eigh(self._H)
        self._state: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the graph."""
        return self._n

    @property
    def hamiltonian(self) -> np.ndarray:
        """The Hamiltonian matrix H."""
        return self._H.copy()

    @property
    def state(self) -> np.ndarray | None:
        """Current quantum state vector |ψ(t)⟩ (complex, length N)."""
        return self._state

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def set_initial_state(self, start_node: int) -> None:
        """Initialise to the computational basis state |start_node⟩.

        Parameters
        ----------
        start_node:
            Index of the starting node (0-indexed).
        """
        if not (0 <= start_node < self._n):
            raise ValueError(
                f"start_node {start_node} is out of range [0, {self._n})."
            )
        psi = np.zeros(self._n, dtype=complex)
        psi[start_node] = 1.0
        self._state = psi

    def set_initial_state_superposition(self, nodes: list[int]) -> None:
        """Initialise to a uniform superposition over a set of nodes.

        Parameters
        ----------
        nodes:
            List of node indices (0-indexed).
        """
        if not nodes:
            raise ValueError("nodes must be non-empty.")
        psi = np.zeros(self._n, dtype=complex)
        for node in nodes:
            if not (0 <= node < self._n):
                raise ValueError(
                    f"Node index {node} is out of range [0, {self._n})."
                )
            psi[node] = 1.0
        psi /= np.linalg.norm(psi)
        self._state = psi

    def set_custom_initial_state(self, psi: np.ndarray) -> None:
        """Set an arbitrary normalised initial state.

        Parameters
        ----------
        psi:
            Complex array of length N.  Will be normalised automatically.
        """
        psi = np.asarray(psi, dtype=complex)
        if psi.shape != (self._n,):
            raise ValueError(
                f"State vector must have length {self._n}, got {psi.shape}."
            )
        norm = np.linalg.norm(psi)
        if norm == 0.0:
            raise ValueError("State vector must not be the zero vector.")
        self._state = psi / norm

    # ------------------------------------------------------------------
    # Time evolution
    # ------------------------------------------------------------------

    def evolve(self, t: float) -> np.ndarray:
        """Evolve the walk from *t = 0* for time *t* using the stored initial state.

        The result is cached as *self.state*.

        Parameters
        ----------
        t:
            Evolution time (real, ≥ 0).

        Returns
        -------
        state : complex ndarray, shape (N,)
            The quantum state |ψ(t)⟩.
        """
        if self._state is None:
            raise RuntimeError(
                "Initial state not set.  Call set_initial_state() first."
            )
        # U(t)|ψ₀⟩ = V · diag(exp(−i λ t)) · Vᵀ · |ψ₀⟩
        phases = np.exp(-1j * self._eigenvalues * t)
        coeffs = self._eigenvectors.T.conj() @ self._state   # project
        self._state = self._eigenvectors @ (phases * coeffs)  # evolve
        return self._state

    def probabilities(self) -> np.ndarray:
        """Return the node probability distribution P(j) = |⟨j|ψ⟩|² for the current state.

        Returns
        -------
        probs : float ndarray, shape (N,)
        """
        if self._state is None:
            raise RuntimeError(
                "No state available.  Call set_initial_state() and evolve()."
            )
        probs = np.abs(self._state) ** 2
        # Renormalise to guard against floating-point drift
        probs /= probs.sum()
        return probs

    def simulate(
        self,
        start_node: int,
        times: np.ndarray,
    ) -> np.ndarray:
        """Run a full time-series simulation from *start_node*.

        Parameters
        ----------
        start_node:
            Index of the starting node.
        times:
            1-D array of time values at which probabilities are recorded.

        Returns
        -------
        prob_matrix : float ndarray, shape (len(times), N)
            prob_matrix[k, j] = P(node j at time times[k]).
        """
        times = np.asarray(times, dtype=float)
        prob_matrix = np.zeros((len(times), self._n), dtype=float)

        # Pre-compute overlap of initial state with eigenvectors once
        psi0 = np.zeros(self._n, dtype=complex)
        psi0[start_node] = 1.0
        coeffs0 = self._eigenvectors.T.conj() @ psi0  # shape (N,)

        for k, t in enumerate(times):
            phases = np.exp(-1j * self._eigenvalues * t)
            psi_t = self._eigenvectors @ (phases * coeffs0)
            probs = np.abs(psi_t) ** 2
            probs /= probs.sum()
            prob_matrix[k] = probs

        # Store final state
        self._state = self._eigenvectors @ (
            np.exp(-1j * self._eigenvalues * times[-1]) * coeffs0
        )
        return prob_matrix

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def mean_position(
        self,
        start_node: int,
        positions: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Compute the mean displacement ⟨r(t)⟩ over time.

        Parameters
        ----------
        start_node:
            Starting node index.
        positions:
            (N, d) array of node coordinates.
        times:
            1-D array of time values.

        Returns
        -------
        mean_r : float ndarray, shape (len(times), d)
        """
        positions = np.asarray(positions, dtype=float)
        prob_matrix = self.simulate(start_node, times)
        return prob_matrix @ positions

    def variance(
        self,
        start_node: int,
        positions: np.ndarray,
        times: np.ndarray,
    ) -> np.ndarray:
        """Compute variance σ²(t) = ⟨r²⟩ − ⟨r⟩² of the probability distribution.

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
        prob_matrix = self.simulate(start_node, times)  # (T, N)
        mean_r = prob_matrix @ positions                 # (T, d)
        r_sq = (positions ** 2).sum(axis=1)              # (N,)
        mean_r_sq = prob_matrix @ r_sq                   # (T,)
        var = mean_r_sq - (mean_r ** 2).sum(axis=1)      # (T,)
        return var
