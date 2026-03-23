"""
tests/test_quantum_walk.py – Unit and integration tests for the quantum_walk package.

Run with:
    pytest tests/test_quantum_walk.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from quantum_walk.graph import (
    random_geometric_graph,
    grid_graph,
    complete_graph,
    small_world_graph,
)
from quantum_walk.ctqw import CTQW
from quantum_walk.crw import CRW


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def ring_adjacency():
    """6-node ring graph."""
    n = 6
    rows, cols, data = [], [], []
    for i in range(n):
        j = (i + 1) % n
        rows += [i, j]
        cols += [j, i]
        data += [1.0, 1.0]
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


@pytest.fixture
def complete_5():
    adj, pos = complete_graph(5)
    return adj, pos


# ===========================================================================
# graph.py
# ===========================================================================

class TestRandomGeometricGraph:
    def test_shape(self):
        adj, pos = random_geometric_graph(20, radius=0.4, seed=0)
        assert adj.shape == (20, 20)
        assert pos.shape == (20, 2)

    def test_symmetry(self):
        adj, _ = random_geometric_graph(15, radius=0.45, seed=1)
        diff = (adj - adj.T).toarray()
        np.testing.assert_allclose(diff, 0.0, atol=1e-12)

    def test_no_self_loops(self):
        adj, _ = random_geometric_graph(10, radius=0.5, seed=2)
        np.testing.assert_allclose(adj.diagonal(), 0.0)

    def test_positions_in_unit_square(self):
        _, pos = random_geometric_graph(50, radius=0.3, seed=3)
        assert pos.min() >= 0.0
        assert pos.max() <= 1.0

    def test_reproducibility(self):
        adj1, _ = random_geometric_graph(20, radius=0.4, seed=42)
        adj2, _ = random_geometric_graph(20, radius=0.4, seed=42)
        np.testing.assert_array_equal(adj1.toarray(), adj2.toarray())


class TestGridGraph:
    def test_shape(self):
        adj, pos = grid_graph(4, 5)
        assert adj.shape == (20, 20)
        assert pos.shape == (20, 2)

    def test_symmetry(self):
        adj, _ = grid_graph(3, 3)
        diff = (adj - adj.T).toarray()
        np.testing.assert_allclose(diff, 0.0, atol=1e-12)

    def test_degree_bounds(self):
        adj, _ = grid_graph(5, 5)
        degrees = np.array(adj.sum(axis=1)).flatten()
        # Corner nodes: degree 2; edge nodes: 3; interior: 4
        assert degrees.min() == 2
        assert degrees.max() == 4

    def test_single_node(self):
        adj, pos = grid_graph(1, 1)
        assert adj.shape == (1, 1)
        assert adj.nnz == 0


class TestCompleteGraph:
    def test_shape(self):
        adj, pos = complete_graph(7)
        assert adj.shape == (7, 7)
        assert pos.shape == (7, 2)

    def test_all_ones_off_diagonal(self):
        adj, _ = complete_graph(5)
        dense = adj.toarray()
        expected = np.ones((5, 5)) - np.eye(5)
        np.testing.assert_allclose(dense, expected)

    def test_positions_on_circle(self):
        _, pos = complete_graph(8)
        radii = np.sqrt((pos ** 2).sum(axis=1))
        np.testing.assert_allclose(radii, 1.0, atol=1e-12)


class TestSmallWorldGraph:
    def test_shape(self):
        adj, pos = small_world_graph(10, 4, 0.2, seed=0)
        assert adj.shape == (10, 10)
        assert pos.shape == (10, 2)

    def test_symmetry(self):
        adj, _ = small_world_graph(12, 4, 0.3, seed=1)
        diff = (adj - adj.T).toarray()
        np.testing.assert_allclose(diff, 0.0, atol=1e-12)

    def test_no_self_loops(self):
        adj, _ = small_world_graph(10, 2, 0.5, seed=2)
        np.testing.assert_allclose(adj.diagonal(), 0.0)

    def test_invalid_k_odd(self):
        with pytest.raises(ValueError, match="even"):
            small_world_graph(10, 3, 0.1)

    def test_invalid_beta(self):
        with pytest.raises(ValueError, match="beta"):
            small_world_graph(10, 4, -0.1)


# ===========================================================================
# ctqw.py
# ===========================================================================

class TestCTQWInitialisation:
    def test_set_initial_state(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        ctqw.set_initial_state(0)
        assert ctqw.state is not None
        np.testing.assert_allclose(np.abs(ctqw.state[0]), 1.0, atol=1e-12)

    def test_invalid_start_node(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        with pytest.raises(ValueError):
            ctqw.set_initial_state(100)

    def test_superposition(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        ctqw.set_initial_state_superposition([0, 1, 2])
        norm = np.linalg.norm(ctqw.state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-12)

    def test_custom_state_normalisation(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        psi = np.array([2.0, 0, 0, 0, 0, 0], dtype=complex)
        ctqw.set_custom_initial_state(psi)
        norm = np.linalg.norm(ctqw.state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-12)

    def test_zero_state_raises(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        with pytest.raises(ValueError):
            ctqw.set_custom_initial_state(np.zeros(6, dtype=complex))


class TestCTQWEvolution:
    def test_probability_sum_to_one(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        ctqw.set_initial_state(0)
        ctqw.evolve(1.5)
        probs = ctqw.probabilities()
        np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-10)

    def test_probability_non_negative(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        ctqw.set_initial_state(0)
        ctqw.evolve(3.0)
        assert np.all(ctqw.probabilities() >= 0.0)

    def test_no_state_evolve_raises(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        with pytest.raises(RuntimeError):
            ctqw.evolve(1.0)

    def test_no_state_probs_raises(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        with pytest.raises(RuntimeError):
            ctqw.probabilities()

    def test_t0_recovers_initial(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        ctqw.set_initial_state(2)
        ctqw.evolve(0.0)
        probs = ctqw.probabilities()
        np.testing.assert_allclose(probs[2], 1.0, atol=1e-10)
        np.testing.assert_allclose(probs[:2].sum() + probs[3:].sum(), 0.0, atol=1e-10)


class TestCTQWSimulate:
    def test_output_shape(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        times = np.linspace(0, 5, 50)
        result = ctqw.simulate(0, times)
        assert result.shape == (50, 6)

    def test_each_row_sums_to_one(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        times = np.linspace(0, 10, 100)
        result = ctqw.simulate(0, times)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_initial_condition(self, ring_adjacency):
        ctqw = CTQW(ring_adjacency)
        times = np.array([0.0, 1.0, 2.0])
        result = ctqw.simulate(3, times)
        np.testing.assert_allclose(result[0, 3], 1.0, atol=1e-10)

    def test_complete_graph_symmetry(self, complete_5):
        adj, _ = complete_5
        ctqw = CTQW(adj)
        times = np.linspace(0, 5, 10)
        result = ctqw.simulate(0, times)
        # All non-start nodes should have equal probability by symmetry
        for k in range(len(times)):
            off = result[k, 1:]
            np.testing.assert_allclose(off, off[0], atol=1e-10)

    def test_dense_vs_sparse_adjacency(self):
        """CTQW should accept both dense and sparse adjacency matrices."""
        adj, _ = grid_graph(3, 3)
        dense_adj = adj.toarray()
        ctqw_sparse = CTQW(adj)
        ctqw_dense = CTQW(dense_adj)
        times = np.array([0.0, 1.0, 2.5])
        r_sparse = ctqw_sparse.simulate(0, times)
        r_dense = ctqw_dense.simulate(0, times)
        np.testing.assert_allclose(r_sparse, r_dense, atol=1e-10)


class TestCTQWVariance:
    def test_variance_non_negative(self):
        adj, pos = grid_graph(4, 4)
        ctqw = CTQW(adj)
        times = np.linspace(0.1, 5.0, 30)
        var = ctqw.variance(0, pos, times)
        assert np.all(var >= -1e-8)  # allow tiny numerical negatives

    def test_variance_grows(self):
        """Variance for a CTQW on a grid should generally grow (quadratic spreading)."""
        adj, pos = grid_graph(6, 6)
        ctqw = CTQW(adj)
        times = np.linspace(0.0, 3.0, 60)
        var = ctqw.variance(0, pos, times)
        # Average of second half should exceed average of first half
        assert var[30:].mean() > var[:30].mean()


# ===========================================================================
# crw.py
# ===========================================================================

class TestCRWSimulate:
    def test_output_shape(self, ring_adjacency):
        crw = CRW(ring_adjacency)
        times = np.linspace(0, 5, 50)
        result = crw.simulate(0, times)
        assert result.shape == (50, 6)

    def test_each_row_sums_to_one(self, ring_adjacency):
        crw = CRW(ring_adjacency)
        times = np.linspace(0, 10, 100)
        result = crw.simulate(0, times)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_probabilities_non_negative(self, ring_adjacency):
        crw = CRW(ring_adjacency)
        times = np.linspace(0, 5, 50)
        result = crw.simulate(0, times)
        assert np.all(result >= -1e-10)

    def test_initial_condition(self, ring_adjacency):
        crw = CRW(ring_adjacency)
        times = np.array([0.0, 1.0])
        result = crw.simulate(2, times)
        np.testing.assert_allclose(result[0, 2], 1.0, atol=1e-10)

    def test_stationary_distribution(self):
        """At large t, CRW should converge to uniform on regular graphs."""
        adj, _ = complete_graph(5)
        crw = CRW(adj)
        result = crw.simulate(0, np.array([1000.0]))
        np.testing.assert_allclose(result[0], np.full(5, 0.2), atol=1e-6)

    def test_invalid_start_node(self, ring_adjacency):
        crw = CRW(ring_adjacency)
        with pytest.raises(ValueError):
            crw.simulate(99, np.array([1.0]))


class TestCRWVariance:
    def test_variance_non_negative(self):
        adj, pos = grid_graph(4, 4)
        crw = CRW(adj)
        times = np.linspace(0.1, 5.0, 30)
        var = crw.variance(0, pos, times)
        assert np.all(var >= -1e-8)


# ===========================================================================
# Integration: quantum faster than classical
# ===========================================================================

class TestQuantumAdvantage:
    def test_quantum_spreads_faster(self):
        """On a 1-D ring, quantum walk variance should exceed classical variance
        at intermediate times (quadratic vs linear spreading)."""
        adj, pos = grid_graph(1, 20)  # 1×20 line graph
        times = np.linspace(0.1, 4.0, 80)
        ctqw = CTQW(adj)
        crw = CRW(adj)
        q_var = ctqw.variance(0, pos, times)
        c_var = crw.variance(0, pos, times)
        # Quantum variance should exceed classical on average
        assert q_var.mean() > c_var.mean()

    def test_quantum_interference(self):
        """Quantum walk on a complete graph creates interference patterns
        (non-uniform distribution) unlike the classical walk at early times."""
        adj, _ = complete_graph(8)
        ctqw = CTQW(adj)
        q_result = ctqw.simulate(0, np.array([1.0]))
        # Interference: probabilities should not all be equal at t=1
        probs = q_result[0]
        # The starting node should have higher or lower probability than others
        # (interference effect)
        assert not np.allclose(probs, probs.mean(), atol=1e-3)
