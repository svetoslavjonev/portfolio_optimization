"""
Loading and validation of daily index data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from .config import DEFAULT_CONFIG, BacktestConfig

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

#: Default location of the (private) real dataset, relative to the repo root.
DEFAULT_DATA_PATH = "data/Indices.xlsx"


def load_index_data(
    path: PathLike = DEFAULT_DATA_PATH,
    *,
    config: BacktestConfig = DEFAULT_CONFIG,
    date_column: Optional[str] = None,
    date_format: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a daily index-level dataset into a clean, sorted frame.

    With no arguments this reproduces the original thesis behaviour exactly:
    it reads data/Indices.xlsx, parses the Dates column as
    dd.mm.yyyy and uses it as the index. .csv/.tsv files are also
    supported so the pipeline can ingest arbitrary daily index data.

    Parameters
    ----------
    path:
        Path to a .xlsx/.xls workbook or a .csv/.tsv file.
    config:
        Supplies the default date_column and date_format.
    date_column, date_format:
        Optional overrides for the config values.

    Returns
    -------
    pandas.DataFrame
        Float-valued index levels indexed by a sorted DatetimeIndex.

    Raises
    ------
    FileNotFoundError
        If path does not exist.
    ValueError
        If the date column is missing, dates are duplicated, or any non-date
        column is non-numeric.
    """
    path = Path(path)
    date_column = date_column or config.date_column
    date_format = date_format or config.date_format

    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        raw = pd.read_excel(path)
    elif suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else ","
        raw = pd.read_csv(path, sep=sep)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}")

    if date_column not in raw.columns:
        raise ValueError(
            f"Date column {date_column!r} not found. Available columns: "
            f"{list(raw.columns)}"
        )

    # Parse dates without the deprecated date_parser argument.
    raw[date_column] = pd.to_datetime(raw[date_column], format=date_format)
    data = raw.set_index(date_column).sort_index()
    data.index.name = date_column

    if data.index.has_duplicates:
        dupes = data.index[data.index.duplicated()].unique()
        raise ValueError(f"Duplicate dates found in {path.name}: {list(dupes)[:5]} ...")

    non_numeric = data.columns[~data.apply(pd.api.types.is_numeric_dtype)]
    if len(non_numeric):
        raise ValueError(f"Non-numeric columns are not allowed: {list(non_numeric)}")

    logger.info(
        "Loaded %d rows x %d columns from %s (%s to %s).",
        len(data),
        data.shape[1],
        path.name,
        data.index.min().date(),
        data.index.max().date(),
    )
    return data


def split_assets_risk_free(
    data: pd.DataFrame,
    *,
    config: BacktestConfig = DEFAULT_CONFIG,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Separate the investable universe from the risk-free instrument.

    Mirrors the original data.iloc[:, :-1] / data.iloc[:, -1] split when
    config.risk_free_col is None. If a column name is supplied instead,
    that column is used as the risk-free series and removed from the universe.
    Datasets with no risk-free column are supported: pass a name that is not
    present is an error, but leaving the default and having a genuine
    risk-free-free universe simply means the caller can treat rf as zero.

    Returns
    -------
    (assets, risk_free)
        assets is the asset-level frame; risk_free is the risk-free
        level series (or None if the universe has only one column and no
        explicit risk-free column was requested).
    """
    if config.risk_free_col is None:
        if data.shape[1] < 2:
            logger.warning(
                "Only one column present and no risk-free column named; "
                "treating the dataset as risk-free-free (rf = 0)."
            )
            return data.copy(), None
        assets = data.iloc[:, :-1].copy()
        risk_free = data.iloc[:, -1].copy()
        logger.info("Risk-free instrument taken as last column: %r.", risk_free.name)
        return assets, risk_free

    if config.risk_free_col not in data.columns:
        raise ValueError(
            f"risk_free_col {config.risk_free_col!r} not in columns {list(data.columns)}"
        )
    risk_free = data[config.risk_free_col].copy()
    assets = data.drop(columns=[config.risk_free_col]).copy()
    return assets, risk_free
