"""Multi-asset portfolio optimisation across economic regimes.

A small, well-factored toolkit supporting a rolling-window backtest of six
allocation models (Equal Weight, Tangency, Minimum Variance, Most-Diversified,
Risk Parity and Hierarchical Risk Parity) on daily index data, with descriptive
statistics by sub-period and by economic regime.

The companion notebook (Portfolio_Optimization.ipynb) is the narrative
front end; this package holds the reusable, documented logic.
"""

from __future__ import annotations

from .backtest import BacktestResults, run_backtest
from .config import DEFAULT_CONFIG, BacktestConfig
from .data import load_index_data, split_assets_risk_free
from .optimizers import PORT_TYPES, optimize_portfolio
from .periods import (
    DEFAULT_REGIMES,
    clip_regimes_to_data,
    generate_subperiods,
)
from .plots import plot_correlation_heatmap, plot_weight_donuts
from .stats import (
    annual_return_vol,
    portfolio_summary_table,
    regime_descriptive_table,
    regime_performance_table,
    subperiod_descriptive_table,
    summary_stats,
)
from .synthetic import generate_synthetic_indices

__version__ = "1.0.0"

__all__ = [
    "BacktestConfig",
    "DEFAULT_CONFIG",
    "load_index_data",
    "split_assets_risk_free",
    "generate_synthetic_indices",
    "generate_subperiods",
    "clip_regimes_to_data",
    "DEFAULT_REGIMES",
    "annual_return_vol",
    "summary_stats",
    "subperiod_descriptive_table",
    "regime_descriptive_table",
    "portfolio_summary_table",
    "regime_performance_table",
    "optimize_portfolio",
    "PORT_TYPES",
    "run_backtest",
    "BacktestResults",
    "plot_correlation_heatmap",
    "plot_weight_donuts",
    "__version__",
]
