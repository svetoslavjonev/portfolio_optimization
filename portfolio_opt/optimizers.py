"""
Single-window portfolio optimisers built on Riskfolio-Lib.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pandas as pd
import riskfolio as rp

logger = logging.getLogger(__name__)

#: Canonical ordering of the portfolio models.
PORT_TYPES: List[str] = ["EW", "TANGENCY", "MIN RISK", "MDP", "RP", "HRP"]


def _weights_series(raw: pd.DataFrame, columns: pd.Index) -> pd.Series:
    """
    Coerce a Riskfolio weights frame into a Series aligned to columns.

    Aligning by asset name (rather than by position) and reindexing to the full
    universe makes the result robust to any internal reordering and to assets
    that an optimiser may drop.
    """
    series = raw.iloc[:, 0]
    return series.reindex(columns).fillna(0.0)


def optimize_portfolio(
    port_type: str,
    window_returns: pd.DataFrame,
    risk_free_rate: float,
    *,
    hrp_max_k: int = 10,
) -> pd.Series:
    """
    Optimise one portfolio on a single estimation window.

    Parameters
    ----------
    port_type:
        One of :data:PORT_TYPES.
    window_returns:
        Daily simple returns over the estimation window (rows = days,
        columns = assets).
    risk_free_rate:
        Daily risk-free rate used by the Sharpe and risk-parity models. The
        MDP uses rf = 0 regardless (see Notes).
    hrp_max_k:
        Maximum number of clusters for HRP; capped at the asset count.

    Returns
    -------
    pandas.Series
        Weights indexed by asset, summing to ~1. On optimiser failure the
        function logs a warning and falls back to equal weights (1/N).

    Notes
    -----
    MDP. The most-diversified portfolio maximises the diversification
    ratio (w'σ) / sqrt(w'Σw). Riskfolio implements this as a Sharpe problem
    with the mean vector replaced by the vector of asset volatilities. Тhe
    diversification ratio corresponds to that Sharpe form with rf = 0, so a
    zero risk-free rate is used here (the textbook definition).
    """
    columns = window_returns.columns
    n_assets = window_returns.shape[1]
    equal_weights = pd.Series(1.0 / n_assets, index=columns)

    if port_type == "EW":
        return equal_weights

    try:
        if port_type in {"TANGENCY", "MIN RISK", "MDP", "RP"}:
            port = rp.Portfolio(returns=window_returns)
            port.assets_stats(method_mu="hist", method_cov="hist")

            if port_type == "TANGENCY":
                raw = port.optimization(
                    model="Classic", rm="MV", obj="Sharpe",
                    rf=risk_free_rate, hist=True,
                )
            elif port_type == "MIN RISK":
                raw = port.optimization(
                    model="Classic", rm="MV", obj="MinRisk",
                    rf=risk_free_rate, hist=True,
                )
            elif port_type == "MDP":
                # Maximum diversification ratio: mu <- asset volatilities, rf = 0.
                port.mu = np.sqrt(np.diag(port.cov)).reshape(1, -1)
                raw = port.optimization(
                    model="Classic", rm="MV", obj="Sharpe", rf=0.0, hist=True,
                )
            else:  # RP
                raw = port.rp_optimization(
                    model="Classic", rm="MV", rf=risk_free_rate, b=None, hist=True,
                )

        elif port_type == "HRP":
            hc = rp.HCPortfolio(returns=window_returns)
            raw = hc.optimization(
                model="HRP", codependence="pearson", rm="MV", rf=risk_free_rate,
                linkage="single", max_k=min(hrp_max_k, n_assets), leaf_order=True,
            )
        else:
            raise ValueError(f"Unknown portfolio type: {port_type!r}")

        if raw is None or raw.empty:
            raise RuntimeError("optimiser returned no solution")

        return _weights_series(raw, columns)

    except Exception as exc:
        logger.warning(
            "%s optimisation failed (%s); falling back to equal weights (1/N).",
            port_type,
            exc,
        )
        return equal_weights
