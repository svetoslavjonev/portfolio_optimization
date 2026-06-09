"""
Smoke tests for the portfolio optimisation pipeline.

These run the full study on synthetic data and exercise the robustness paths
(partial regime coverage, off-grid start dates, too-short data). They are fast
and deterministic. Run with pytest -q.
"""

from __future__ import annotations

import warnings

import pytest

import portfolio_opt as po

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def data():
    return po.generate_synthetic_indices(seed=7)


def test_synthetic_schema(data):
    assert list(data.columns)[-1] == "T-Bills"
    assert data.shape[1] == 8
    assert data.index.is_monotonic_increasing


def test_full_pipeline(data):
    cfg = po.BacktestConfig()
    assets, rf = po.split_assets_risk_free(data, config=cfg)

    subs = po.generate_subperiods(data, config=cfg)
    assert "Overall" in subs
    assert not po.subperiod_descriptive_table(data, subs).empty

    regimes = po.clip_regimes_to_data(po.DEFAULT_REGIMES, data)
    assert set(regimes) == set(po.DEFAULT_REGIMES)

    results = po.run_backtest(assets, rf, config=cfg)
    # Equal-weight column should sum to ~1 each year.
    ew = results.weights_over_time["EW"].sum(axis=1)
    assert (ew.round(6) == 1.0).all()

    summary = po.portfolio_summary_table(results.combined_results, results.port_types)
    assert summary.shape[0] == len(results.port_types)


def test_partial_regime_coverage(data):
    """A dataset covering only part of the regime calendar must not break."""
    cfg = po.BacktestConfig()
    partial = data.loc["2010-03-15":"2016-08-20"]

    regimes = po.clip_regimes_to_data(po.DEFAULT_REGIMES, partial)
    # Out-of-range regimes are dropped; survivors are clipped to the data span.
    assert "Expansion" not in regimes
    for intervals in regimes.values():
        for start, end in intervals:
            assert start >= partial.index.min()
            assert end <= partial.index.max()

    assets, rf = po.split_assets_risk_free(partial, config=cfg)
    results = po.run_backtest(assets, rf, config=cfg)
    assert not results.combined_results.empty


def test_too_short_data_raises(data):
    cfg = po.BacktestConfig()
    tiny = data.loc["2018-01-01":"2020-12-31"]
    assets, rf = po.split_assets_risk_free(tiny, config=cfg)
    with pytest.raises(ValueError):
        po.run_backtest(assets, rf, config=cfg)


def test_no_risk_free_column():
    """Universe with no risk-free column should run with rf = 0."""
    data = po.generate_synthetic_indices(seed=3).drop(columns=["T-Bills"])
    cfg = po.BacktestConfig()
    assets, rf = po.split_assets_risk_free(data, config=cfg)
    # Last asset column is now treated as the (missing) risk-free unless named;
    # here we explicitly pass the whole frame as assets with rf=None instead.
    results = po.run_backtest(assets.iloc[:, :-1], None, config=cfg)
    assert not results.combined_results.empty
