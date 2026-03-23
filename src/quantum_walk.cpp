/*
 * quantum_walk.cpp
 *
 * Demonstration driver for the CTQW / CRW implementations in quantum_walk.hpp.
 *
 * Builds a small random geometric-style graph (a ring lattice) and prints
 * probability distributions at several time steps for both quantum and
 * classical walks.
 *
 * Compilation (requires Eigen ≥ 3.3):
 *
 *   g++ -O2 -std=c++17 -I/usr/include/eigen3 quantum_walk.cpp -o quantum_walk
 *
 * Run:
 *   ./quantum_walk
 */

#include "quantum_walk.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <utility>

// ---------------------------------------------------------------------------
// Helper: build a 1-D ring lattice with k nearest neighbours
// ---------------------------------------------------------------------------
static std::vector<std::pair<int,int>>
ring_lattice_edges(int n, int k_half)
{
    std::vector<std::pair<int,int>> edges;
    for (int i = 0; i < n; ++i)
        for (int d = 1; d <= k_half; ++d)
            edges.emplace_back(i, (i + d) % n);
    return edges;
}

// ---------------------------------------------------------------------------
// Helper: print a probability vector
// ---------------------------------------------------------------------------
static void print_probs(const qw::Vec& probs, int n_show = 8)
{
    int n = static_cast<int>(probs.size());
    std::cout << std::fixed << std::setprecision(4);
    for (int j = 0; j < std::min(n, n_show); ++j)
        std::cout << "  P(" << j << ")=" << probs[j];
    if (n > n_show)
        std::cout << "  ...";
    std::cout << "\n";
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main()
{
    constexpr int  N      = 16;   // number of nodes
    constexpr int  K_HALF = 2;    // each node connected to 2 neighbours each side
    constexpr int  START  = 0;    // initial node

    auto edges = ring_lattice_edges(N, K_HALF);

    qw::CTQWDense ctqw(N, edges, /*gamma=*/1.0);
    qw::CRWDense  crw (N, edges, /*gamma=*/1.0);

    std::vector<double> times = {0.0, 0.5, 1.0, 2.0, 4.0, 8.0};

    std::cout << "=== Continuous-Time Quantum Walk (N=" << N
              << ", start=" << START << ") ===\n";
    for (double t : times) {
        std::cout << "t=" << t << ": ";
        print_probs(ctqw.evolve(START, t));
    }

    std::cout << "\n=== Classical Random Walk (N=" << N
              << ", start=" << START << ") ===\n";
    for (double t : times) {
        std::cout << "t=" << t << ": ";
        print_probs(crw.evolve(START, t));
    }

    // --- Batch simulation ---
    auto q_all = ctqw.simulate(START, times);
    auto c_all = crw.simulate(START, times);

    std::cout << "\n=== Batch simulation: probability at node " << N/2
              << " over time ===\n";
    std::cout << std::setw(8) << "time"
              << std::setw(14) << "quantum"
              << std::setw(14) << "classical\n";
    for (std::size_t k = 0; k < times.size(); ++k) {
        std::cout << std::setw(8) << times[k]
                  << std::setw(14) << q_all[k][N/2]
                  << std::setw(14) << c_all[k][N/2] << "\n";
    }

    return 0;
}
