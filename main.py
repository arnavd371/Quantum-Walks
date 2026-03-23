#!/usr/bin/env python3
"""
main.py – Quantum Walk simulation driver.

Demonstrates continuous-time quantum walks (CTQW) on several geometric
graph types and compares them with classical random walks (CRW).

Usage
-----
    python main.py                          # run all demos
    python main.py --graph rgg --nodes 30   # random geometric graph, 30 nodes
    python main.py --graph grid             # 5×5 grid graph
    python main.py --graph complete         # complete graph K_10
    python main.py --graph small_world      # Watts–Strogatz small-world
    python main.py --save-plots plots/      # save figures to directory

Saved outputs
-------------
When --save-plots is provided a set of PNG figures is written:
    <dir>/probs_<graph>.png       – probability distribution at t=T_max/2
    <dir>/comparison_<graph>.png  – quantum vs classical at the far node
    <dir>/time_evolution_<graph>.png
    <dir>/variance_<graph>.png
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from quantum_walk import (
    random_geometric_graph,
    grid_graph,
    complete_graph,
    small_world_graph,
    CTQW,
    CRW,
)
from quantum_walk.visualization import (
    plot_probability_distribution,
    plot_walk_comparison,
    plot_time_evolution,
    plot_variance_comparison,
)


# ---------------------------------------------------------------------------
# Individual graph demos
# ---------------------------------------------------------------------------

def run_demo(
    name: str,
    adjacency,
    positions,
    n_times: int = 200,
    t_max: float = 10.0,
    start_node: int = 0,
    save_dir: str | None = None,
) -> None:
    """Run a CTQW / CRW demo on a pre-built graph and optionally save plots."""
    n = adjacency.shape[0]
    far_node = n // 2

    times = np.linspace(0.0, t_max, n_times)

    print(f"\n{'='*60}")
    print(f"Graph: {name}  (N={n}  start={start_node}  far_node={far_node})")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    ctqw = CTQW(adjacency, gamma=1.0, use_laplacian=True)
    q_probs = ctqw.simulate(start_node, times)
    print(f"  CTQW simulation: {time.perf_counter()-t0:.3f}s")

    t0 = time.perf_counter()
    crw = CRW(adjacency)
    c_probs = crw.simulate(start_node, times)
    print(f"  CRW  simulation: {time.perf_counter()-t0:.3f}s")

    # Variance
    if positions is not None:
        q_var = ctqw.variance(start_node, positions, times)
        c_var = crw.variance(start_node, positions, times)
    else:
        q_var = c_var = None

    # Print summary
    mid = n_times // 2
    print(f"  At t={times[mid]:.2f}: P_quantum(node {far_node})  = {q_probs[mid, far_node]:.4f}")
    print(f"  At t={times[mid]:.2f}: P_classical(node {far_node}) = {c_probs[mid, far_node]:.4f}")

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        slug = name.lower().replace(" ", "_")

        plot_probability_distribution(
            q_probs[mid],
            positions=positions,
            title=f"{name}: quantum probabilities at t={times[mid]:.1f}",
            save_path=os.path.join(save_dir, f"probs_{slug}.png"),
        )
        plot_walk_comparison(
            times, q_probs, c_probs, far_node,
            title=f"{name}: quantum vs classical at node {far_node}",
            save_path=os.path.join(save_dir, f"comparison_{slug}.png"),
        )
        plot_time_evolution(
            times, q_probs,
            title=f"{name}: quantum walk time evolution",
            save_path=os.path.join(save_dir, f"time_evolution_{slug}.png"),
        )
        if q_var is not None:
            plot_variance_comparison(
                times, q_var, c_var,
                title=f"{name}: variance comparison",
                save_path=os.path.join(save_dir, f"variance_{slug}.png"),
            )
        print(f"  Figures saved to {save_dir}/")


# ---------------------------------------------------------------------------
# Graph factory
# ---------------------------------------------------------------------------

def _rgg(n: int = 30, seed: int = 42):
    return random_geometric_graph(n, radius=0.35, seed=seed), "Random Geometric Graph"


def _grid(rows: int = 5, cols: int = 5):
    return grid_graph(rows, cols), f"Grid Graph ({rows}×{cols})"


def _complete(n: int = 10):
    return complete_graph(n), f"Complete Graph K_{n}"


def _small_world(n: int = 20, k: int = 4, beta: float = 0.3, seed: int = 42):
    return small_world_graph(n, k, beta, seed=seed), "Small-World Graph (WS)"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quantum Walk simulation driver")
    parser.add_argument(
        "--graph",
        choices=["rgg", "grid", "complete", "small_world", "all"],
        default="all",
        help="Graph type to simulate (default: all).",
    )
    parser.add_argument("--nodes", type=int, default=None,
                        help="Number of nodes (overrides default for chosen graph).")
    parser.add_argument("--t-max", type=float, default=10.0,
                        help="Maximum simulation time (default: 10).")
    parser.add_argument("--n-times", type=int, default=200,
                        help="Number of time steps (default: 200).")
    parser.add_argument("--save-plots", metavar="DIR", default=None,
                        help="Directory to save PNG figures.")
    return parser.parse_args()


def build_graphs(graph_type: str, n_override: int | None):
    """Return a list of (adjacency, positions, name) tuples."""
    graphs = []

    if graph_type in ("rgg", "all"):
        n = n_override or 30
        (adj, pos), name = _rgg(n=n)
        graphs.append((adj, pos, name))

    if graph_type in ("grid", "all"):
        side = int(np.sqrt(n_override)) if n_override else 5
        (adj, pos), name = _grid(side, side)
        graphs.append((adj, pos, name))

    if graph_type in ("complete", "all"):
        n = n_override or 10
        (adj, pos), name = _complete(n=n)
        graphs.append((adj, pos, name))

    if graph_type in ("small_world", "all"):
        n = n_override or 20
        (adj, pos), name = _small_world(n=n)
        graphs.append((adj, pos, name))

    return graphs


def main() -> None:
    args = parse_args()

    graphs = build_graphs(args.graph, args.nodes)

    for adj, pos, name in graphs:
        run_demo(
            name=name,
            adjacency=adj,
            positions=pos,
            n_times=args.n_times,
            t_max=args.t_max,
            start_node=0,
            save_dir=args.save_plots,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
