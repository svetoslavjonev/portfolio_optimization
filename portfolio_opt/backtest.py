"""
Rolling-window backtest engine.

Each year is treated as an out-of-sample test period: the portfolios are fit on
the preceding lookback_years of daily returns, then held fixed for the test
year. Daily portfolio and excess returns are recorded, along with the weights
chosen each year.

The first test year and the last test year are derived from the data span
(first_year + lookback to the last full year present), so the engine is not
tied to the original 1999-2023 sample. For 1999-2023 data with a 5-year
lookback this reproduces the original 2004 ... 2023 loop exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from .config import DEFAULT_CONFIG, BacktestConfig
from .optimizers import PORT_TYPES, optimize_portfolio

logger = logging.getLogger(__name__)


@dataclass
class BacktestResults:
    """
    Container for backtest outputs.

    Attributes
    ----------
    combined_results:
        Daily {pt}_return and {pt}_excess_return columns per portfolio.
    weights_over_time:
        Per-portfolio frame of chosen weights indexed by test year.
    average_weights:
        Mean weight per asset across years, in %, one column per portfolio.
    port_types:
        The portfolio keys that were run.
    """

    combined_results: pd.DataFrame
    weights_over_time: Dict[str, pd.DataFrame]
    average_weights: pd.DataFrame
    port_types: List[str]


def run_backtest(
    asset_prices: pd.DataFrame,
    risk_free_prices: Optional[pd.Series],
    *,
    config: BacktestConfig = DEFAULT_CONFIG,
    port_types: List[str] = PORT_TYPES,
) -> BacktestResults:
    """
    Run the rolling-window annual-rebalancing backtest.

    Parameters
    ----------
    asset_prices:
        Daily index levels for the investable universe.
    risk_free_prices:
        Daily index levels for the risk-free instrument, or None to treat
        the risk-free rate as zero throughout.
    config:
        Trading-day count and lookback length.
    port_types:
        Which portfolios to run.

    Returns
    -------
    BacktestResults
    """
    lookback = config.lookback_years
    first_year = int(asset_prices.index.min().year)
    last_year = int(asset_prices.index.max().year)
    start_year = first_year + lookback

    if start_year > last_year:
        raise ValueError(
            f"Need at least {lookback + 1} calendar years of data; got "
            f"{first_year}-{last_year}."
        )

    backtest_returns = pd.DataFrame(index=asset_prices.index)
    excess_returns = pd.DataFrame(index=asset_prices.index)
    weights_over_time = {
        pt: pd.DataFrame(columns=asset_prices.columns) for pt in port_types
    }

    zero_rf = pd.Series(0.0, index=asset_prices.index) if risk_free_prices is None \
        else None

    for year in range(start_year, last_year + 1):
        train_window = asset_prices.loc[
            f"{year - lookback}-01-01":f"{year - 1}-12-31"
        ].pct_change().dropna()
        test_returns = asset_prices.loc[
            f"{year}-01-01":f"{year}-12-31"
        ].pct_change().dropna()

        if train_window.empty or test_returns.empty:
            logger.warning("Skipping %d: insufficient train/test data.", year)
            continue

        if risk_free_prices is None:
            rf_rate = 0.0
            rf_current = zero_rf.loc[test_returns.index]
        else:
            rf_window = risk_free_prices.loc[
                f"{year - lookback}-01-01":f"{year - 1}-12-31"
            ].pct_change().dropna()
            rf_current = risk_free_prices.loc[
                f"{year}-01-01":f"{year}-12-31"
            ].pct_change().dropna()
            rf_rate = float(rf_window.mean())

        for port_type in port_types:
            weights = optimize_portfolio(
                port_type, train_window, rf_rate, hrp_max_k=config.hrp_max_k
            )
            weights_over_time[port_type].loc[year] = weights

            daily_returns = test_returns @ weights
            adjusted = test_returns.sub(rf_current, axis=0)
            daily_excess = adjusted @ weights

            backtest_returns.loc[test_returns.index, f"{port_type}_return"] = daily_returns
            excess_returns.loc[
                test_returns.index, f"{port_type}_excess_return"
            ] = daily_excess

    combined = pd.concat([backtest_returns, excess_returns], axis=1).dropna(how="all")
    average_weights = pd.DataFrame(
        {pt: (w.mean() * 100).round(2) for pt, w in weights_over_time.items()}
    )

    logger.info("Backtest complete: %d test years.", combined.index.year.nunique())
    return BacktestResults(
        combined_results=combined,
        weights_over_time=weights_over_time,
        average_weights=average_weights,
        port_types=list(port_types),
    )
