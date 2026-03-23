"""
graph.py – Geometric and structural graph generators.

Each function returns a pair (adjacency_matrix, node_positions) where
adjacency_matrix is a scipy CSR sparse matrix and node_positions is an
(N, 2) NumPy array of 2-D coordinates (or None when positions are not
meaningful).
"""

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# Random Geometric Graph
# ---------------------------------------------------------------------------

def random_geometric_graph(
    n: int,
    radius: float,
    seed: int | None = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build a random geometric graph in the unit square.

    Nodes are placed uniformly at random.  Two nodes are connected when
    their Euclidean distance is at most *radius*.

    Parameters
    ----------
    n:
        Number of nodes.
    radius:
        Connection radius in [0, 1].
    seed:
        Random seed for reproducibility.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix  shape (n, n)
    positions : numpy.ndarray            shape (n, 2)
    """
    rng = np.random.default_rng(seed)
    positions = rng.uniform(0.0, 1.0, size=(n, 2))

    # Vectorised pairwise distance matrix
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (n,n,2)
    dist = np.sqrt((diff ** 2).sum(axis=-1))                          # (n,n)

    adj_dense = ((dist <= radius) & (dist > 0.0)).astype(float)
    adjacency = sp.csr_matrix(adj_dense)
    return adjacency, positions


# ---------------------------------------------------------------------------
# Grid (Lattice) Graph
# ---------------------------------------------------------------------------

def grid_graph(rows: int, cols: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build a 2-D rectangular lattice graph.

    Parameters
    ----------
    rows, cols:
        Dimensions of the grid.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix  shape (rows*cols, rows*cols)
    positions : numpy.ndarray            shape (rows*cols, 2)
    """
    n = rows * cols
    row_ids: list[int] = []
    col_ids: list[int] = []

    def node(r: int, c: int) -> int:
        return r * cols + c

    for r in range(rows):
        for c in range(cols):
            u = node(r, c)
            if c + 1 < cols:
                v = node(r, c + 1)
                row_ids += [u, v]
                col_ids += [v, u]
            if r + 1 < rows:
                v = node(r + 1, c)
                row_ids += [u, v]
                col_ids += [v, u]

    data = np.ones(len(row_ids), dtype=float)
    adjacency = sp.csr_matrix((data, (row_ids, col_ids)), shape=(n, n))

    # Node positions on integer lattice
    positions = np.array(
        [[c, r] for r in range(rows) for c in range(cols)],
        dtype=float,
    )
    return adjacency, positions


# ---------------------------------------------------------------------------
# Complete Graph
# ---------------------------------------------------------------------------

def complete_graph(n: int) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build a complete graph K_n.

    Parameters
    ----------
    n:
        Number of nodes.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix  shape (n, n)
    positions : numpy.ndarray            shape (n, 2)  (on a unit circle)
    """
    adj_dense = np.ones((n, n), dtype=float) - np.eye(n)
    adjacency = sp.csr_matrix(adj_dense)

    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    return adjacency, positions


# ---------------------------------------------------------------------------
# Small-World Graph (Watts–Strogatz)
# ---------------------------------------------------------------------------

def small_world_graph(
    n: int,
    k: int,
    beta: float,
    seed: int | None = None,
) -> tuple[sp.csr_matrix, np.ndarray]:
    """Build a Watts–Strogatz small-world graph.

    Parameters
    ----------
    n:
        Number of nodes arranged on a ring.
    k:
        Each node is connected to *k* nearest neighbours (must be even).
    beta:
        Rewiring probability in [0, 1].
    seed:
        Random seed for reproducibility.

    Returns
    -------
    adjacency : scipy.sparse.csr_matrix  shape (n, n)
    positions : numpy.ndarray            shape (n, 2)  (on a unit circle)
    """
    if k % 2 != 0:
        raise ValueError("k must be even for the Watts–Strogatz model.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1].")

    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=float)

    # Build ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            neighbour = (i + j) % n
            adj[i, neighbour] = 1.0
            adj[neighbour, i] = 1.0

    # Rewiring
    for i in range(n):
        for j in range(1, k // 2 + 1):
            if rng.random() < beta:
                old_neighbour = (i + j) % n
                # Choose a new neighbour that is not i and not already connected
                candidates = [
                    v for v in range(n) if v != i and adj[i, v] == 0.0
                ]
                if not candidates:
                    continue
                new_neighbour = rng.choice(candidates)
                adj[i, old_neighbour] = 0.0
                adj[old_neighbour, i] = 0.0
                adj[i, new_neighbour] = 1.0
                adj[new_neighbour, i] = 1.0

    adjacency = sp.csr_matrix(adj)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    positions = np.column_stack([np.cos(angles), np.sin(angles)])
    return adjacency, positions
