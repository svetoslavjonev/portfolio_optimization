"""
Return statistics and the descriptive / performance tables.

The estimators here are kept identical to the defended thesis. In particular:

* Annualisation uses arithmetic scaling: mean x D for returns and
  std x sqrt(D) for volatility, with D the trading-day count.
* The reported Sharpe Ratio is a geometric Sharpe: the annualised
  geometric excess return divided by the annualised (arithmetic) volatility.
* Per-period returns are computed within each slice (pct_change then drop
  the leading NaN), so no observation outside a period contributes to it. This
  drops the first day of each slice by construction — a standard, intentional
  edge effect.

"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, BacktestConfig
from .periods import Interval, pool_slices


def annual_return_vol(
    prices: pd.DataFrame,
    *,
    trading_days: int = DEFAULT_CONFIG.trading_days,
) -> pd.DataFrame:
    """
    Annualised mean return and volatility (in %) for each column of prices.

    Parameters
    ----------
    prices:
        Index-level frame, daily simple returns are taken internally.
    trading_days:
        Annualisation factor.

    Returns
    -------
    pandas.DataFrame
        Columns ["Return", "Volatility"] indexed by asset, rounded to 2 dp.
    """
    returns = prices.pct_change().dropna()
    mean_ret = returns.mean() * trading_days * 100
    vol = returns.std() * np.sqrt(trading_days) * 100
    return pd.DataFrame({"Return": mean_ret.round(2), "Volatility": vol.round(2)})


def summary_stats(
    portfolio_return: pd.Series,
    excess_return: pd.Series,
    *,
    trading_days: int = DEFAULT_CONFIG.trading_days,
) -> Dict[str, float]:
    """
    Performance summary for one stream of daily portfolio returns.

    Parameters
    ----------
    portfolio_return:
        Daily total returns of the portfolio.
    excess_return:
        Daily returns in excess of the risk-free rate.
    trading_days:
        Annualisation factor.

    Returns
    -------
    dict
        Annualised mean/geometric/excess returns, volatility, geometric Sharpe
        ratio, maximum drawdown and cumulative return — all in % except the
        Sharpe ratio, rounded to two dp.
    """
    daily_mean = portfolio_return.mean()
    daily_std = portfolio_return.std()

    annual_mean = daily_mean * trading_days
    annual_std = daily_std * np.sqrt(trading_days)

    cumulative_returns = (1 + portfolio_return).cumprod()
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    cum_ret = cumulative_returns.iloc[-1] - 1

    cumulative_excess = (1 + excess_return).cumprod()
    cum_ex_ret = cumulative_excess.iloc[-1] - 1

    years = len(portfolio_return) / trading_days
    geo_ret = (1 + cum_ret) ** (1 / years) - 1 if years > 0 else np.nan
    geo_excess = (1 + cum_ex_ret) ** (1 / years) - 1 if years > 0 else np.nan

    sharpe = geo_excess / annual_std if annual_std != 0 else np.nan

    return {
        "Annual Mean Return (%)": round(annual_mean * 100, 2),
        "Annual Geometric Return (%)": round(geo_ret * 100, 2),
        "Annual Std Dev (%)": round(annual_std * 100, 2),
        "Annual Excess Return (%)": round(geo_excess * 100, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Maximum Drawdown (%)": round(max_drawdown * 100, 2),
        "Cumulative Return (%)": round(cum_ret * 100, 2),
    }


def subperiod_descriptive_table(
    prices: pd.DataFrame,
    subperiods: Dict[str, Interval],
    *,
    trading_days: int = DEFAULT_CONFIG.trading_days,
) -> pd.DataFrame:
    """
    Annualised return/vol per asset for each descriptive sub-period.

    Returns a frame indexed by sub-period with a two-level column index
    (metric, asset) — the layout shown in the notebook.
    """
    per_period = {
        label: annual_return_vol(prices.loc[start:end], trading_days=trading_days)
        for label, (start, end) in subperiods.items()
    }
    combined = pd.concat(per_period, axis=1)
    return combined.T.unstack(level=1)


def regime_descriptive_table(
    prices: pd.DataFrame,
    regimes: Dict[str, List[Interval]],
    *,
    trading_days: int = DEFAULT_CONFIG.trading_days,
) -> pd.DataFrame:
    """Annualised return/vol per asset, pooling daily returns within each regime.

    Returns are computed within each interval and pooled, so the statistics
    describe behaviour *conditional* on the regime.
    """
    per_regime: Dict[str, pd.DataFrame] = {}
    for regime, intervals in regimes.items():
        returns_slices = [
            prices.loc[start:end].pct_change().dropna() for start, end in intervals
        ]
        pooled = pd.concat(returns_slices) if returns_slices else prices.iloc[0:0]
        mean_ret = pooled.mean() * trading_days * 100
        vol = pooled.std() * np.sqrt(trading_days) * 100
        per_regime[regime] = pd.DataFrame(
            {"Return": mean_ret.round(2), "Volatility": vol.round(2)}
        )
    combined = pd.concat(per_regime, axis=1)
    return combined.T.unstack(level=1)


def portfolio_summary_table(
    combined_results: pd.DataFrame,
    port_types: List[str],
    *,
    trading_days: int = DEFAULT_CONFIG.trading_days,
) -> pd.DataFrame:
    """Full-sample performance summary, one row per portfolio."""
    stats = {
        pt: summary_stats(
            combined_results[f"{pt}_return"],
            combined_results[f"{pt}_excess_return"],
            trading_days=trading_days,
        )
        for pt in port_types
    }
    return pd.DataFrame(stats).T


def regime_performance_table(
    combined_results: pd.DataFrame,
    regimes: Dict[str, List[Interval]],
    port_types: List[str],
    *,
    trading_days: int = DEFAULT_CONFIG.trading_days,
) -> pd.DataFrame:
    """
    Per-regime performance summary with a (regime, portfolio) index.

    Daily backtest and excess returns are pooled within each regime before the
    summary statistics are computed. Note that pooling non-contiguous intervals
    chains returns across calendar gaps, means and volatilities are unaffected,
    but the regime-level maximum drawdown can show artificial moves at interval
    seams and should be read with that caveat in mind.
    """
    per_regime: Dict[str, pd.DataFrame] = {}
    for regime, intervals in regimes.items():
        perfs = {}
        for pt in port_types:
            pooled_ret = pool_slices(combined_results[f"{pt}_return"], intervals)
            pooled_ex = pool_slices(combined_results[f"{pt}_excess_return"], intervals)
            if pooled_ret.empty:
                continue
            perfs[pt] = summary_stats(pooled_ret, pooled_ex, trading_days=trading_days)
        if perfs:
            per_regime[regime] = pd.DataFrame(perfs).T
    return pd.concat(per_regime, axis=0)
