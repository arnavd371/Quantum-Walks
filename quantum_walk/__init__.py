"""
quantum_walk – Continuous-time quantum walk simulation on geometric graphs.

Modules
-------
graph          Geometric and structural graph generators.
ctqw           Continuous-time quantum walk (CTQW) simulator.
crw            Classical random walk (CRW) simulator for comparison.
visualization  Probability-distribution and comparison plots.
"""

from .graph import (
    random_geometric_graph,
    grid_graph,
    complete_graph,
    small_world_graph,
)
from .ctqw import CTQW
from .crw import CRW
from .visualization import (
    plot_probability_distribution,
    plot_walk_comparison,
    plot_time_evolution,
)

__all__ = [
    "random_geometric_graph",
    "grid_graph",
    "complete_graph",
    "small_world_graph",
    "CTQW",
    "CRW",
    "plot_probability_distribution",
    "plot_walk_comparison",
    "plot_time_evolution",
]
