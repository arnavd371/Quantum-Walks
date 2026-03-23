"""
visualization.py – Probability-distribution and comparison plots.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend – safe for scripts
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_probability_distribution(
    probs: np.ndarray,
    positions: np.ndarray | None = None,
    title: str = "Quantum Walk Probability Distribution",
    ax: plt.Axes | None = None,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot the node probability distribution P(j) as a bar chart (or a 2-D
    scatter when *positions* are provided).

    Parameters
    ----------
    probs:
        1-D array of length N with node probabilities.
    positions:
        Optional (N, 2) array of node coordinates.  When given, a scatter plot
        is drawn with marker size proportional to probability.
    title:
        Plot title.
    ax:
        Matplotlib axes to draw on.  Created automatically when omitted.
    save_path:
        If provided, the figure is saved to this path instead of being shown.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    n = len(probs)

    if positions is not None:
        positions = np.asarray(positions, dtype=float)
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=probs * 3000,
            c=probs,
            cmap="viridis",
            alpha=0.8,
            edgecolors="k",
            linewidths=0.4,
        )
        plt.colorbar(scatter, ax=ax, label="Probability")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    else:
        colors = cm.viridis(probs / (probs.max() + 1e-12))
        ax.bar(range(n), probs, color=colors, edgecolor="k", linewidth=0.4)
        ax.set_xlabel("Node index")
        ax.set_ylabel("Probability")
        ax.set_xlim(-0.5, n - 0.5)

    ax.set_title(title)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    return fig


def plot_walk_comparison(
    times: np.ndarray,
    quantum_probs: np.ndarray,
    classical_probs: np.ndarray,
    node_index: int,
    title: str = "Quantum vs Classical Walk",
    save_path: str | None = None,
) -> plt.Figure:
    """Compare quantum-walk and classical-walk probability at a single node over time.

    Parameters
    ----------
    times:
        1-D array of time values.
    quantum_probs:
        (T, N) array of quantum probabilities.
    classical_probs:
        (T, N) array of classical probabilities.
    node_index:
        Node to compare at.
    title:
        Plot title.
    save_path:
        If given, save figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, quantum_probs[:, node_index],
            label="Quantum walk", color="royalblue", linewidth=2)
    ax.plot(times, classical_probs[:, node_index],
            label="Classical walk", color="tomato", linewidth=2, linestyle="--")
    ax.set_xlabel("Time t")
    ax.set_ylabel(f"P(node {node_index}, t)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    return fig


def plot_time_evolution(
    times: np.ndarray,
    prob_matrix: np.ndarray,
    title: str = "Probability Time Evolution",
    max_nodes: int = 10,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot the probability of each node over time (first *max_nodes* nodes).

    Parameters
    ----------
    times:
        1-D array of time values.
    prob_matrix:
        (T, N) probability matrix.
    title:
        Plot title.
    max_nodes:
        Maximum number of nodes to show (for readability).
    save_path:
        If given, save figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_show = min(max_nodes, prob_matrix.shape[1])
    fig, ax = plt.subplots(figsize=(10, 5))
    palette = cm.tab20(np.linspace(0, 1, n_show))

    for j in range(n_show):
        ax.plot(times, prob_matrix[:, j], color=palette[j],
                linewidth=1.5, label=f"Node {j}")

    ax.set_xlabel("Time t")
    ax.set_ylabel("Probability P(j, t)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    return fig


def plot_variance_comparison(
    times: np.ndarray,
    quantum_var: np.ndarray,
    classical_var: np.ndarray,
    title: str = "Variance: Quantum vs Classical Walk",
    save_path: str | None = None,
) -> plt.Figure:
    """Plot variance σ²(t) for quantum and classical walks.

    Parameters
    ----------
    times:
        1-D array of time values.
    quantum_var:
        Variance array from CTQW (shape T).
    classical_var:
        Variance array from CRW (shape T).
    title:
        Plot title.
    save_path:
        If given, save figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(times, quantum_var, label="Quantum walk", color="royalblue", linewidth=2)
    ax.plot(times, classical_var, label="Classical walk",
            color="tomato", linewidth=2, linestyle="--")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Variance σ²(t)")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    return fig
