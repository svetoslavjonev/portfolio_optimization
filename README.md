# Multi-Asset Portfolio Optimisation Across Economic Regimes

This repository accompanies my master's thesis in applied mathematics.
It presents a reproducible study comparing six portfolio-construction methods on daily
multi-asset index data, with a rolling out-of-sample backtest and performance
attribution conditional on the prevailing economic regime.

## Models

| Key | Model | Description |
|-----|-------|-------------|
| `EW` | Equal Weight | Naive 1/N benchmark. |
| `TANGENCY` | Maximum Sharpe | Mean-variance tangency portfolio. |
| `MIN RISK` | Minimum Variance | Global minimum-variance portfolio. |
| `MDP` | Most Diversified | Maximises the diversification ratio (`rf = 0`). |
| `RP` | Risk Parity | Equal risk contribution under variance. |
| `HRP` | Hierarchical Risk Parity | Cluster-based allocation. |

All convex optimisations are solved with [Riskfolio-Lib](https://riskfolio-lib.readthedocs.io).

## What the study produces

- Annualised return/volatility per asset, by **descriptive sub-period** and by
  **economic regime**.
- A full-sample correlation heatmap.
- A rolling 5-year-lookback, annually-rebalanced **out-of-sample backtest** of
  all six portfolios.
- Full-sample and per-regime performance summaries (geometric Sharpe, maximum
  drawdown, cumulative return, ...).
- Average allocations per model, shown as donut charts.

## Repository layout

```
portfolio-optimization/
├── Portfolio_Optimization.ipynb   # the analysis (tables + charts)
├── portfolio_opt/                 # reusable, documented package
│   ├── config.py                  # BacktestConfig: all tunable parameters
│   ├── data.py                    # load + validate index data, asset/RF split
│   ├── synthetic.py               # deterministic synthetic dataset generator
│   ├── periods.py                 # sub-period generation + regime clipping
│   ├── stats.py                   # return stats + descriptive/perf tables
│   ├── optimizers.py              # the six allocation models
│   ├── backtest.py                # rolling-window backtest engine
│   └── plots.py                   # correlation heatmap + weight donuts
├── tests/test_smoke.py            # fast end-to-end + robustness tests
├── data/
│   └── sample_indices.csv         # input-schema example
├── requirements.txt
├── pyproject.toml
└── LICENSE
```

## Installation

```bash
git clone <your-repo-url>
cd portfolio-optimization
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
```

## Quickstart

Open and run the notebook top to bottom:

```bash
jupyter lab Portfolio_Optimization.ipynb
```

With no data file present, it generates a synthetic dataset automatically and
runs the complete study. To use the real data, place the workbook at
`data/Indices.xlsx` and re-run, the notebook picks it up
with no code changes.

The same pipeline from a script:

```python
import portfolio_opt as po

cfg = po.BacktestConfig()
data = po.generate_synthetic_indices(seed=42)        # or po.load_index_data("data/Indices.xlsx")
assets, rf = po.split_assets_risk_free(data, config=cfg)

results = po.run_backtest(assets, rf, config=cfg)
print(po.portfolio_summary_table(results.combined_results, results.port_types))
```

## Using your own data

Any **daily index-level** dataset works. Expected shape: a date column plus one
numeric column per series. See `data/sample_indices.csv` for the format.

- **Date parsing** — set `date_column` / `date_format` on `BacktestConfig`
  (defaults: `Dates`, `%d.%m.%Y`).
- **Risk-free instrument**: by default the **last column** is used as the
  risk-free series (matching the original setup). Name it explicitly with
  `BacktestConfig(risk_free_col="...")`, or pass `risk_free_prices=None` to
  `run_backtest` to treat the risk-free rate as zero.
- **Sub-periods** are tiled from the data's own date range, so they are not
  tied to any start year.
- **Regimes** use a fixed, exogenous calendar (`portfolio_opt.periods.DEFAULT_REGIMES`).
  If your data covers only part of it, intervals are clipped to the available
  span and out-of-range regimes are dropped automatically.

## Methodology notes and assumptions

- Annualisation is arithmetic (`mean x 252`, `std x sqrt(252)`).
- The reported **Sharpe ratio** is geometric: annualised geometric excess
  return over annualised volatility.
- **MDP** is solved as a Sharpe problem with the mean vector replaced by asset
  volatilities and `rf = 0`, the textbook diversification ratio.
- If an optimiser fails to solve on a window, the engine logs a warning and
  falls back to **equal weights (1/N)** rather than allocating nothing.
- Per-period returns drop the first observation of each slice by construction.
- Regime-level statistics pool returns across non-contiguous intervals. Means
  and volatilities are unaffected, but the regime **maximum drawdown** can show
  artificial moves at interval seams and should be read with that caveat.

## Testing

```bash
pip install pytest
pytest -q
```

The suite runs the full pipeline on synthetic data and verifies the
partial-regime, off-grid-start and too-short-data paths.

## References

1. Markowitz, H. (1952). *Portfolio Selection.* Journal of Finance.
2. Choueifaty, Y. & Coignard, Y. (2008). *Toward Maximum Diversification.*
3. Maillard, S., Roncalli, T. & Teiletche, J. (2010). *The Properties of
   Equally Weighted Risk Contribution Portfolios.*
4. López de Prado, M. (2016). *Building Diversified Portfolios that Outperform
   Out of Sample.*
5. DeMiguel, V., Garlappi, L. & Uppal, R. (2009). *Optimal Versus Naive
   Diversification.*

## License

MIT — see [LICENSE](LICENSE). Update the copyright holder before publishing.
