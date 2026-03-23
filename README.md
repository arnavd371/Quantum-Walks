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

## Minimal UI (GitHub Pages friendly)

A lightweight, static interface lives in `docs/` so it can be published directly to GitHub Pages.

- Preview locally: `python -m http.server 8000 -d docs` then open http://localhost:8000/docs/
- Regenerate the deterministic presets used by the UI (updates `docs/assets/sample-data.json`):

  ```bash
  python - <<'PY'
  import json, numpy as np
  from quantum_walk import graph, ctqw

  def make_entry(adjacency, label, parameters, description, times):
      walk = ctqw.CTQW(adjacency)
      walk.set_initial_state(0)
      probabilities = [
          [round(float(p), 6) for p in row] for row in walk.simulate(0, times)
      ]
      return {
          "label": label,
          "nodes": adjacency.shape[0],
          "times": [round(float(t), 2) for t in times],
          "probabilities": probabilities,
          "parameters": parameters,
          "description": description,
      }

  times = np.linspace(0, 3, 5)
  data = {
      "grid": make_entry(
          graph.grid_graph(3, 3)[0],
          "3x3 Grid",
          {"rows": 3, "cols": 3},
          "Uniform lattice illustrating symmetric spread from a corner node.",
          times,
      ),
      "rgg": make_entry(
          graph.random_geometric_graph(8, 0.55, seed=7)[0],
          "Random Geometric (n=8, r=0.55)",
          {"n": 8, "radius": 0.55, "seed": 7},
          "Spatially random connections with radius-based edges.",
          times,
      ),
      "complete": make_entry(
          graph.complete_graph(6)[0],
          "Complete Graph (K6)",
          {"n": 6},
          "Fully connected graph reaching fast interference patterns.",
          times,
      ),
      "small_world": make_entry(
          graph.small_world_graph(10, 4, 0.2, seed=3)[0],
          "Small-world (n=10, k=4, β=0.2)",
          {"n": 10, "k": 4, "beta": 0.2, "seed": 3},
          "Watts–Strogatz ring with shortcuts creating mixed locality.",
          times,
      ),
  }

  with open("docs/assets/sample-data.json", "w", encoding="utf-8") as f:
      json.dump(data, f, indent=2)
  PY
  ```
- Enable Pages in **Settings → Pages**, choose **Deploy from a branch**, then select the branch that contains `docs/` with the `/docs` folder as the source.

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
