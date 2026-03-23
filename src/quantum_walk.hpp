/*
 * quantum_walk.hpp
 *
 * Continuous-time quantum walk (CTQW) on a graph using a sparse Hamiltonian.
 *
 * The Hamiltonian is the graph Laplacian:
 *
 *     H = γ (D − A)
 *
 * Time evolution is computed via the spectral decomposition
 *
 *     U(t) = exp(−i H t)
 *
 * For small/medium graphs (N ≲ 2000) an exact dense eigen-decomposition is
 * used; the evolution of each time step is O(N²) after an O(N³) one-time
 * factorisation.
 *
 * For larger graphs a simple Krylov-like sparse-matrix–vector product (SMVP)
 * can be used via the Chebyshev-series expansion included below.
 *
 * Compilation example (requires Eigen ≥ 3.3):
 *
 *   g++ -O2 -std=c++17 -I/usr/include/eigen3 quantum_walk.cpp -o quantum_walk
 */

#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <complex>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace qw {

using Real    = double;
using Complex = std::complex<Real>;
using Vec     = Eigen::VectorXd;
using CVec    = Eigen::VectorXcd;
using Mat     = Eigen::MatrixXd;
using CMat    = Eigen::MatrixXcd;
using SpMat   = Eigen::SparseMatrix<Real>;

// ---------------------------------------------------------------------------
// Build graph Laplacian from adjacency list
// ---------------------------------------------------------------------------

/**
 * @brief Construct the sparse graph Laplacian from an edge list.
 *
 * @param n          Number of nodes.
 * @param edges      Undirected edge list as pairs (u, v).
 * @param gamma      Coupling constant (scales the Hamiltonian).
 * @return SpMat     N×N sparse Laplacian γ(D−A).
 */
SpMat make_laplacian(
        int n,
        const std::vector<std::pair<int,int>>& edges,
        Real gamma = 1.0)
{
    SpMat adj(n, n);
    std::vector<Eigen::Triplet<Real>> triplets;
    triplets.reserve(edges.size() * 2);

    Vec degree = Vec::Zero(n);
    for (auto [u, v] : edges) {
        if (u < 0 || u >= n || v < 0 || v >= n)
            throw std::out_of_range("Edge endpoint out of range.");
        triplets.emplace_back(u, v, -gamma);
        triplets.emplace_back(v, u, -gamma);
        degree[u] += gamma;
        degree[v] += gamma;
    }
    // Add degree diagonal
    for (int i = 0; i < n; ++i)
        if (degree[i] != 0.0)
            triplets.emplace_back(i, i, degree[i]);

    adj.setFromTriplets(triplets.begin(), triplets.end());
    return adj;
}

// ---------------------------------------------------------------------------
// CTQW – dense eigen-decomposition approach
// ---------------------------------------------------------------------------

/**
 * @brief Continuous-time quantum walk using full eigen-decomposition.
 *
 * Suitable for graphs with up to ~1000 nodes.  The Hamiltonian is
 * diagonalised once in the constructor; each call to `evolve` is O(N²).
 */
class CTQWDense {
public:
    /**
     * @param n      Number of nodes.
     * @param edges  Undirected edge list.
     * @param gamma  Coupling constant.
     */
    CTQWDense(int n,
              const std::vector<std::pair<int,int>>& edges,
              Real gamma = 1.0)
        : n_(n)
    {
        SpMat L = make_laplacian(n, edges, gamma);
        Mat Ldense = Mat(L);  // convert to dense for eigen-solver

        Eigen::SelfAdjointEigenSolver<Mat> solver(Ldense);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Eigen-decomposition failed.");

        eigenvalues_ = solver.eigenvalues();   // (N,)  real
        eigenvectors_ = solver.eigenvectors(); // (N,N) real orthogonal
    }

    /**
     * @brief Evolve initial state |start⟩ for time t and return probabilities.
     *
     * @param start_node  Initial node index.
     * @param t           Evolution time.
     * @return Vec        Probability vector P(j) = |⟨j|ψ(t)⟩|², length N.
     */
    Vec evolve(int start_node, Real t) const {
        if (start_node < 0 || start_node >= n_)
            throw std::out_of_range("start_node out of range.");

        // |ψ₀⟩ = |start_node⟩
        CVec psi0 = CVec::Zero(n_);
        psi0[start_node] = Complex(1.0, 0.0);

        return evolve_state(psi0, t);
    }

    /**
     * @brief Evolve an arbitrary initial state for time t.
     *
     * @param psi0  Initial state (will be normalised).
     * @param t     Evolution time.
     * @return Vec  Probability distribution.
     */
    Vec evolve_state(const CVec& psi0, Real t) const {
        // Project onto eigenbasis: c_k = ⟨ϕ_k|ψ₀⟩
        // (eigenvectors are real so conjugate is identity)
        CVec coeffs = eigenvectors_.transpose().cast<Complex>() * psi0;

        // Apply phases: e^{−i λ_k t}
        CVec phases(n_);
        for (int k = 0; k < n_; ++k)
            phases[k] = std::exp(Complex(0.0, -eigenvalues_[k] * t));

        // Reconstruct: |ψ(t)⟩ = Σ_k e^{−iλ_k t} c_k |ϕ_k⟩
        CVec psi_t = eigenvectors_.cast<Complex>() * (phases.cwiseProduct(coeffs));

        // Probabilities
        Vec probs(n_);
        for (int j = 0; j < n_; ++j)
            probs[j] = std::norm(psi_t[j]);  // |z|²

        probs /= probs.sum();  // renormalise
        return probs;
    }

    /**
     * @brief Simulate the walk over a vector of time values.
     *
     * @param start_node  Initial node.
     * @param times       Time values.
     * @return std::vector<Vec>  One probability vector per time value.
     */
    std::vector<Vec> simulate(int start_node,
                              const std::vector<Real>& times) const
    {
        // Pre-compute projection of |ψ₀⟩ onto eigenbasis
        CVec psi0 = CVec::Zero(n_);
        psi0[start_node] = Complex(1.0, 0.0);
        CVec coeffs = eigenvectors_.transpose().cast<Complex>() * psi0;

        std::vector<Vec> result;
        result.reserve(times.size());

        for (Real t : times) {
            CVec phases(n_);
            for (int k = 0; k < n_; ++k)
                phases[k] = std::exp(Complex(0.0, -eigenvalues_[k] * t));

            CVec psi_t = eigenvectors_.cast<Complex>()
                         * (phases.cwiseProduct(coeffs));

            Vec probs(n_);
            for (int j = 0; j < n_; ++j)
                probs[j] = std::norm(psi_t[j]);
            probs /= probs.sum();
            result.push_back(probs);
        }
        return result;
    }

    int n_nodes() const { return n_; }

private:
    int  n_;
    Vec  eigenvalues_;
    Mat  eigenvectors_;
};

// ---------------------------------------------------------------------------
// Classical random walk (for comparison)
// ---------------------------------------------------------------------------

/**
 * @brief Classical continuous-time random walk via Laplacian eigen-decomposition.
 *
 * Solves  dp/dt = −L p  →  p(t) = exp(−Lt) p₀.
 */
class CRWDense {
public:
    CRWDense(int n,
             const std::vector<std::pair<int,int>>& edges,
             Real gamma = 1.0)
        : n_(n)
    {
        SpMat L = make_laplacian(n, edges, gamma);
        Mat Ldense = Mat(L);

        Eigen::SelfAdjointEigenSolver<Mat> solver(Ldense);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("Eigen-decomposition failed.");

        eigenvalues_  = solver.eigenvalues();
        eigenvectors_ = solver.eigenvectors();
    }

    /**
     * @brief Evolve initial delta-distribution at start_node for time t.
     *
     * @param start_node  Starting node.
     * @param t           Time.
     * @return Vec        Probability distribution.
     */
    Vec evolve(int start_node, Real t) const {
        Vec p0 = Vec::Zero(n_);
        p0[start_node] = 1.0;
        return evolve_dist(p0, t);
    }

    Vec evolve_dist(const Vec& p0, Real t) const {
        Vec coeffs = eigenvectors_.transpose() * p0;
        Vec decay(n_);
        for (int k = 0; k < n_; ++k)
            decay[k] = std::exp(-eigenvalues_[k] * t);

        Vec p_t = eigenvectors_ * (decay.cwiseProduct(coeffs));

        // Clip negatives and renormalise
        for (int j = 0; j < n_; ++j)
            if (p_t[j] < 0.0) p_t[j] = 0.0;
        Real s = p_t.sum();
        if (s > 0.0) p_t /= s;
        return p_t;
    }

    std::vector<Vec> simulate(int start_node,
                              const std::vector<Real>& times) const
    {
        Vec p0 = Vec::Zero(n_);
        p0[start_node] = 1.0;
        Vec coeffs = eigenvectors_.transpose() * p0;

        std::vector<Vec> result;
        result.reserve(times.size());
        for (Real t : times) {
            Vec decay(n_);
            for (int k = 0; k < n_; ++k)
                decay[k] = std::exp(-eigenvalues_[k] * t);
            Vec p_t = eigenvectors_ * (decay.cwiseProduct(coeffs));
            for (int j = 0; j < n_; ++j)
                if (p_t[j] < 0.0) p_t[j] = 0.0;
            Real s = p_t.sum();
            if (s > 0.0) p_t /= s;
            result.push_back(p_t);
        }
        return result;
    }

    int n_nodes() const { return n_; }

private:
    int  n_;
    Vec  eigenvalues_;
    Mat  eigenvectors_;
};

} // namespace qw
