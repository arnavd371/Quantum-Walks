# Quantum-Walks

Simulating **continuous-time quantum walks (CTQW)** on dynamic geometric graphs
and comparing them with classical random walks (CRW) to study computational
advantages in traversing highly connected networks.

---

## Overview

| Feature | Detail |
|---|---|
| Language | Python 3.12+ / C++17 |
| Graph types | Random geometric, 2-D grid, complete (K_n), Watts–Strogatz small-world |
| Walk model | Continuous-time (Hamiltonian = graph Laplacian, H = γ(D−A)) |
| Optimisation | Spectral decomposition – O(N²) per time step after O(N³) one-time factorisation |
| Outputs | Node probabilities P(j,t), variance σ²(t), quantum vs classical comparison plots |

---

## Repository structure

```
.
├── quantum_walk/           # Python package
│   ├── __init__.py
│   ├── graph.py            # Graph generators (RGG, grid, complete, small-world)
│   ├── ctqw.py             # Continuous-time quantum walk simulator
│   ├── crw.py              # Classical random walk simulator
│   └── visualization.py   # Matplotlib plotting utilities
├── src/                    # C++ implementation
│   ├── quantum_walk.hpp    # Header-only CTQW/CRW kernels (Eigen)
│   ├── quantum_walk.cpp    # Demo driver
│   └── CMakeLists.txt      # CMake build file
├── tests/
│   └── test_quantum_walk.py
├── main.py                 # Python simulation driver (CLI)
└── requirements.txt
```

---

## Quick start

### Python

```bash
pip install -r requirements.txt

# Run all graph demos and save plots
python main.py --save-plots plots/

# Single graph type
python main.py --graph rgg --nodes 50 --t-max 15 --save-plots plots/
```

Available `--graph` values: `rgg`, `grid`, `complete`, `small_world`, `all` (default).

### Tests

```bash
pip install pytest
pytest tests/ -v
```

### C++ (requires Eigen >= 3.3)

```bash
cd src
cmake -B build && cmake --build build
./build/quantum_walk
```

---

## Physics background

The **continuous-time quantum walk** replaces the stochastic transition matrix
of a classical random walk with unitary time evolution under a Hamiltonian:

```
|ψ(t)⟩ = exp(−i H t) |ψ(0)⟩
```

where the Hamiltonian H is the graph Laplacian `H = γ(D − A)`.  The
probability of finding the walker at node *j* at time *t* is:

```
P(j, t) = |⟨j|ψ(t)⟩|²
```

Key differences from classical walks:
- **Quadratic speedup** in variance growth: σ²(t) ~ t² (quantum) vs t (classical).
- **Interference effects** cause non-uniform spreading and localisation phenomena.
- On highly connected networks the walker can exploit constructive interference to
  reach distant nodes faster than a classical random walker.
