"""
Configuration objects for the portfolio-optimization study.

The defaults reproduce the original master's-thesis setup exactly:

* a daily data/Indices.xlsx workbook with a Dates column formatted
  dd.mm.yyyy;
* the risk-free instrument stored in the last column (risk_free_col=None);
* a 252-day trading year, a 5-year rolling estimation window, and 5-year
  descriptive sub-periods.

Override any field to run the pipeline on an arbitrary daily index dataset.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class BacktestConfig:
    """
    Immutable bundle of parameters for the full study.

    Parameters
    ----------
    trading_days:
        Number of trading days used to annualise daily statistics.
    lookback_years:
        Length, in calendar years, of the rolling estimation window used to
        fit each portfolio before it is held out-of-sample for one year.
    subperiod_length_years:
        Length, in calendar years, of the descriptive sub-periods. Sub-periods
        are tiled across the *actual* span of the data, so they are not tied to
        any particular start year.
    date_column:
        Name of the date column in the source file.
    date_format:
        strftime/strptime pattern used to parse date_column.
    risk_free_col:
        Name of the risk-free column. None falls back to the last column,
        preserving the original positional behaviour.
    hrp_max_k:
        Upper bound on the number of clusters for Hierarchical Risk Parity.
        Capped at n_assets at run time so the same config works for
        universes of any size.
    """

    trading_days: int = 252
    lookback_years: int = 5
    subperiod_length_years: int = 5
    date_column: str = "Dates"
    date_format: str = "%d.%m.%Y"
    risk_free_col: Optional[str] = None
    hrp_max_k: int = 10

    def __post_init__(self) -> None:
        if self.trading_days <= 0:
            raise ValueError("trading_days must be positive.")
        if self.lookback_years <= 0:
            raise ValueError("lookback_years must be positive.")
        if self.subperiod_length_years <= 0:
            raise ValueError("subperiod_length_years must be positive.")


#: The default configuration used throughout the notebook.
DEFAULT_CONFIG = BacktestConfig()
