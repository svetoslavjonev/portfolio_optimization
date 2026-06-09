"""
Plotting helpers for the study.

Each function builds and returns a Matplotlib figure so the notebook stays
the single place where visuals are displayed. Colours come from the
coolwarm map for visual consistency with the original thesis figures.
"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_correlation_heatmap(
    returns: pd.DataFrame,
    *,
    figsize=(8, 6),
) -> plt.Figure:
    """Annotated correlation heatmap of daily returns."""
    corr = returns.corr()
    plt.rc("font", family="serif")
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    fig.tight_layout()
    return fig


def _autopct_above(threshold: float = 2.0):
    """Return an autopct formatter that hides slices below threshold %."""
    def _fmt(pct: float) -> str:
        return f"{pct:.1f}%" if pct >= threshold else ""
    return _fmt


def plot_weight_donuts(
    average_weights: pd.DataFrame,
    port_types: List[str],
    *,
    ncols: int = 3,
    figsize=(12, 8),
) -> plt.Figure:
    """
    Grid of donut charts showing average weights per portfolio.

    A shared colour map keeps each asset the same colour across every subplot,
    and a single legend row labels them.
    """
    assets = average_weights.index
    cmap = plt.colormaps["coolwarm"].resampled(len(assets))
    colors = [cmap(i) for i in range(len(assets))]

    nrows = int(np.ceil(len(port_types) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = np.atleast_1d(axes).flatten()

    for i, port_type in enumerate(port_types):
        ax = axes[i]
        ax.pie(
            average_weights[port_type],
            labels=None,
            colors=colors,
            autopct=_autopct_above(2.0),
            pctdistance=1.18,
            startangle=140,
            textprops={"fontsize": 12},
        )
        ax.add_artist(plt.Circle((0, 0), 0.65, fc="white"))
        ax.set_title(port_type, fontsize=14)

    for j in range(len(port_types), len(axes)):
        axes[j].axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=colors[i], label=assets[i], markersize=8)
        for i in range(len(assets))
    ]
    legend = fig.legend(
        handles=handles, loc="lower center",
        bbox_to_anchor=(0.5, -0.05), ncol=len(assets),
    )
    legend.get_frame().set_edgecolor("white")
    fig.tight_layout()
    return fig
