# signal-validation

Validation harness for systematic trading signals. The goal is to test whether a
signal's edge is real or an artefact of overfitting, look-ahead, or ignored
transaction costs. The output is a per-signal verdict (keep / marginal / noise),
not a polished equity curve.

## What it does

1. Builds a few transparent candidate signals on liquid public data (equities via
   `yfinance`; a crypto loader can reuse the same interface).
2. Backtests each one without look-ahead and net of transaction costs.
3. Scores the result with the statistics that matter:
   - Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014), which corrects for the
     number of variants tried, non-normal returns, and sample length.
   - Probability of Backtest Overfitting via CSCV (Bailey et al., 2017).
   - Signal decay half-life from the lag-1 autocorrelation.

Signals are kept deliberately simple. A simple signal that survives the deflated
Sharpe and PBO checks is easier to trust than a complex one that does not.

## Candidate signals

- `ts_momentum`: time-series momentum on the trailing return.
- `mean_reversion`: z-score reversion to a rolling mean.
- `vol_scaled_momentum`: momentum scaled by inverse realised volatility.

## Layout

```
src/signalval/
  data.py       price loaders and returns
  signals.py    candidate signals, each a position series in [-1, 1]
  backtest.py   look-ahead-safe backtest with costs and summary metrics
  stats.py      Sharpe, PSR, deflated Sharpe, expected-max-Sharpe, PBO, decay
tests/
  test_no_lookahead.py
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
python scripts/run_study.py
```

## Results

`results.json` holds the latest run. On SPY (2010-2024), across 37 signal variants,
the best variant reached a net Sharpe of 0.50 but failed the Deflated Sharpe
(p = 0.77) and PBO (0.61) checks, so no edge was claimed.

## Reporting rule

A Sharpe figure is reported as an edge only if it is out-of-sample, net of costs,
and clears the Deflated Sharpe and PBO checks. Otherwise the honest summary is that
the pipeline ran and no signal cleared the bar.
