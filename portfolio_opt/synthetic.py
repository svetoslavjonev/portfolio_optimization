"""
Synthetic daily index data for reproducible runs.

This module generates a drop-in replacement with the same schema (seven asset-class
total-return indices plus a low-volatility T-Bills risk-free series) and
plausible cross-asset correlations, so the entire notebook runs end-to-end for
anyone who clones the repository.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

#: Column layout matches the private dataset: 7 assets, then the risk-free.
ASSET_COLUMNS: List[str] = [
    "DM Equity",
    "EM Equity",
    "IG Bonds",
    "HY Bonds",
    "Commodities",
    "Private Equity",
    "Real Estate",
]
RISK_FREE_COLUMN: str = "T-Bills"
COLUMNS: List[str] = ASSET_COLUMNS + [RISK_FREE_COLUMN]

# Annualised (drift, volatility) assumptions per series, ordered as COLUMNS.
_ANNUAL_MU = np.array([0.06, 0.07, 0.03, 0.05, 0.03, 0.09, 0.06, 0.020])
_ANNUAL_VOL = np.array([0.16, 0.20, 0.05, 0.08, 0.18, 0.22, 0.19, 0.004])

# Stylised correlation matrix (symmetric, positive-definite). Equities and
# private/real assets co-move, bonds are a partial diversifier, the risk-free
# instrument is essentially uncorrelated with everything.
_CORR = np.array(
    [
        # DM    EM   IG    HY   Comm  PE    RE   TB
        [1.00, 0.78, 0.15, 0.55, 0.35, 0.80, 0.70, 0.00],  # DM Equity
        [0.78, 1.00, 0.10, 0.55, 0.45, 0.72, 0.60, 0.00],  # EM Equity
        [0.15, 0.10, 1.00, 0.45, 0.05, 0.10, 0.25, 0.05],  # IG Bonds
        [0.55, 0.55, 0.45, 1.00, 0.35, 0.55, 0.55, 0.00],  # HY Bonds
        [0.35, 0.45, 0.05, 0.35, 1.00, 0.35, 0.30, 0.00],  # Commodities
        [0.80, 0.72, 0.10, 0.55, 0.35, 1.00, 0.68, 0.00],  # Private Equity
        [0.70, 0.60, 0.25, 0.55, 0.30, 0.68, 1.00, 0.00],  # Real Estate
        [0.00, 0.00, 0.05, 0.00, 0.00, 0.00, 0.00, 1.00],  # T-Bills
    ]
)


def generate_synthetic_indices(
    start: str = "1999-01-01",
    end: str = "2023-12-31",
    *,
    seed: int = 42,
    freq: str = "B",
    trading_days: int = 252,
) -> pd.DataFrame:
    """
    Generate a synthetic daily index-level dataset.

    Parameters
    ----------
    start, end:
        Inclusive date range. The default span (1999-2023) exercises every
        descriptive sub-period and economic regime defined in :mod:periods.
    seed:
        Seed for the random number generator; identical seeds give identical
        data.
    freq:
        Pandas offset alias for the date index ("B" = business days).
    trading_days:
        Used to convert annualised assumptions to a per-period scale.

    Returns
    -------
    pandas.DataFrame
        Index levels (base 100) indexed by a Dates DatetimeIndex, with
        columns matching the private dataset's schema.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end) if freq == "B" else pd.date_range(
        start=start, end=end, freq=freq
    )
    n_obs = len(dates)
    n_series = len(COLUMNS)

    mu_daily = _ANNUAL_MU / trading_days
    vol_daily = _ANNUAL_VOL / np.sqrt(trading_days)

    # Build the daily covariance from the correlation and per-series vols, then
    # draw correlated simple returns.
    cov_daily = _CORR * np.outer(vol_daily, vol_daily)
    chol = np.linalg.cholesky(_nearest_pd(cov_daily))
    shocks = rng.standard_normal(size=(n_obs, n_series)) @ chol.T
    returns = mu_daily + shocks

    # Compound to price levels (base 100).
    levels = 100.0 * np.cumprod(1.0 + returns, axis=0)

    frame = pd.DataFrame(levels, index=dates, columns=COLUMNS)
    frame.index.name = "Dates"
    return frame


def _nearest_pd(matrix: np.ndarray) -> np.ndarray:
    """
    Return a positive-definite copy of matrix via tiny diagonal jitter.

    The stylised covariance is already PD, but this guards generated covariances
    against floating-point round-off before the Cholesky factorisation.
    """
    eigvals = np.linalg.eigvalsh(matrix)
    if eigvals.min() > 0:
        return matrix
    jitter = (-eigvals.min() + 1e-12) * np.eye(matrix.shape[0])
    return matrix + jitter
