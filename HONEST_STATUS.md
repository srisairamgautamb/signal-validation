# Honest status

## Rules

1. No look-ahead. Positions act on lagged information; `tests/test_no_lookahead.py`
   must pass before any result is reported.
2. Net of costs. Headline numbers are after transaction costs. Gross is shown only
   alongside net.
3. Out-of-sample only. No in-sample Sharpe is quoted as a result.
4. Trial count is logged and fed into the Deflated Sharpe Ratio. Reporting the best
   of many variants at its raw Sharpe is the failure mode this repo exists to catch.
5. A number reaches the CV only if Deflated Sharpe p-value > 0.95 and PBO < 0.5.
   Otherwise the CV line is "built a signal-validation pipeline", with no Sharpe figure.

## Progress

- [ ] data loaders run
- [ ] signals implemented
- [ ] look-ahead tests green
- [ ] deflated Sharpe and PBO computed across the variant sweep
- [ ] results notebook with the verdict
- [ ] pushed and pinned on GitHub

## CV line (fill after the run)

> Built an out-of-sample, cost-aware signal-validation pipeline and screened N
> candidate signals with the Deflated Sharpe Ratio and the Probability of Backtest
> Overfitting (CSCV). [result, stated honestly]
