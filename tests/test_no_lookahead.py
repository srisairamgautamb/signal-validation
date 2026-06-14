"""Guards against future information leaking into the backtest."""

import numpy as np
import pandas as pd

from signalval.backtest import run


def _prices(n=300, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=idx, name="close")


def test_perfect_foresight_signal_is_not_rewarded():
    """A signal equal to the same-day return sign must not earn an oracle Sharpe
    once the backtester applies its one-step lag."""
    close = _prices()
    cheating = np.sign(close.pct_change()).fillna(0.0)
    res = run(close, cheating, cost_bps=0.0)
    assert res["sharpe_net"] < 5.0


def test_constant_position_has_low_turnover():
    close = _prices()
    pos = pd.Series(1.0, index=close.index)
    res = run(close, pos, cost_bps=10.0)
    assert res["ann_turnover"] < 1.0


def test_position_earns_next_period_return():
    close = _prices(n=10)
    pos = pd.Series(0.0, index=close.index)
    pos.iloc[3] = 1.0
    res = run(close, pos, cost_bps=0.0)
    ret = close.pct_change()
    d3, d4 = close.index[3], close.index[4]
    assert abs(res["gross_returns"].loc[d3]) < 1e-12
    assert abs(res["gross_returns"].loc[d4] - ret.loc[d4]) < 1e-12
