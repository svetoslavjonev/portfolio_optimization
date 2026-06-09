"""
Two types of periods are used in the study:
* Sub-periods — contiguous, equal-length calendar blocks used for
  descriptive statistics. These are derived from the data, so they adapt to
  whatever span the dataset covers rather than being hard-coded.
* Regimes — a fixed, pre-defined calendar of economic states (expansion,
  recession, ...). The calendar is exogenous, but when the supplied data only
  covers part of it, intervals are clipped to the data's span and empty
  regimes are dropped, so a partial dataset never breaks the pipeline.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pandas as pd

from .config import DEFAULT_CONFIG, BacktestConfig

logger = logging.getLogger(__name__)

Interval = Tuple[pd.Timestamp, pd.Timestamp]

#: Pre-defined economic-regime calendar (exogenous to any particular dataset).
DEFAULT_REGIMES: Dict[str, List[Tuple[str, str]]] = {
    "Expansion": [
        ("1999-01-01", "2000-12-31"),
        ("2003-07-01", "2007-12-31"),
        ("2017-01-01", "2018-06-30"),
        ("2021-01-01", "2021-12-31"),
    ],
    "Recession": [
        ("2001-01-01", "2001-11-30"),
        ("2008-09-01", "2009-06-30"),
        ("2020-03-01", "2020-12-31"),
    ],
    "Recovery": [
        ("2001-12-01", "2002-06-30"),
        ("2003-01-01", "2003-06-30"),
        ("2009-07-01", "2010-12-31"),
        ("2023-01-01", "2023-06-30"),
    ],
    "Stagflation": [
        ("2002-07-01", "2002-12-31"),
        ("2008-01-01", "2008-08-31"),
        ("2022-01-01", "2022-12-31"),
    ],
    "Moderate Growth": [
        ("2011-01-01", "2016-12-31"),
        ("2018-07-01", "2020-02-29"),
        ("2023-07-01", "2023-12-31"),
    ],
}


def generate_subperiods(
    data: pd.DataFrame,
    *,
    config: BacktestConfig = DEFAULT_CONFIG,
    include_overall: bool = True,
) -> Dict[str, Interval]:
    """
    Tile equal-length calendar blocks across the data's span.

    Blocks are anchored at the first calendar year present and are
    config.subperiod_length_years years long; the final block is clipped to
    the last available date. For 1999-2023 data with a 5-year length this
    reproduces the original 1999-2003 ... 2019-2023 blocks exactly.

    Parameters
    ----------
    data:
        Datetime-indexed frame; only the index span is used.
    include_overall:
        If True, append an "Overall" entry spanning the full range.

    Returns
    -------
    dict
        Ordered mapping label -> (start_timestamp, end_timestamp).
    """
    first_year = int(data.index.min().year)
    last_year = int(data.index.max().year)
    last_date = data.index.max()
    length = config.subperiod_length_years

    periods: Dict[str, Interval] = {}
    for year in range(first_year, last_year + 1, length):
        block_start = pd.Timestamp(f"{year}-01-01")
        block_end = min(pd.Timestamp(f"{year + length - 1}-12-31"), last_date)
        label = f"{year}-{block_end.year}"
        periods[label] = (block_start, block_end)

    if include_overall:
        periods["Overall"] = (data.index.min(), data.index.max())

    logger.info("Generated %d descriptive sub-periods.", len(periods))
    return periods


def clip_regimes_to_data(
    regimes: Dict[str, List[Tuple[str, str]]],
    data: pd.DataFrame,
) -> Dict[str, List[Interval]]:
    """
    Intersect a regime calendar with the data's available span.

    Each (start, end) interval is clipped to [data_min, data_max];
    intervals with no overlap are dropped, and regimes left with no intervals
    are removed entirely (with a log message). This is what lets a dataset
    covering only part of the regime calendar run without error.

    Returns
    -------
    dict
        Mapping regime -> list of clipped (start, end) timestamp intervals,
        containing only regimes with at least one overlapping interval.
    """
    data_min, data_max = data.index.min(), data.index.max()
    clipped: Dict[str, List[Interval]] = {}

    for regime, intervals in regimes.items():
        kept: List[Interval] = []
        for raw_start, raw_end in intervals:
            start = max(pd.Timestamp(raw_start), data_min)
            end = min(pd.Timestamp(raw_end), data_max)
            if start <= end and not data.loc[start:end].empty:
                kept.append((start, end))
        if kept:
            clipped[regime] = kept
        else:
            logger.warning("Regime %r has no data in range; dropping it.", regime)

    if not clipped:
        logger.warning("No regimes overlap the data span; regime tables will be empty.")
    return clipped


def pool_slices(obj, intervals: List[Interval]):
    """
    Concatenate the rows of obj falling in each of intervals.

    Works for both Series and DataFrames. Returns an empty object of the same
    type when intervals is empty, so callers never hit a bare concat
    of an empty list.
    """
    slices = [obj.loc[start:end] for start, end in intervals]
    slices = [s for s in slices if not s.empty]
    if not slices:
        return obj.iloc[0:0]
    return pd.concat(slices)
