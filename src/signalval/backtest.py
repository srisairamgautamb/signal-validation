"""A position decided at t earns the t -> t+1 return: the position series is shifted
forward one period before being applied to returns. Costs are charged on traded notional."""

from __future__ import annotations

import pandas as pd

from . import stats as st

PERIODS_PER_YEAR = 252


def run(close: pd.Series, position: pd.Series, cost_bps: float = 1.0) -> dict:
    ret = close.pct_change()
    pos = position.shift(1).fillna(0.0)
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (cost_bps / 1e4)
    gross = pos * ret
    net = (gross - cost).dropna()
    return {
        "net_returns": net,
        "gross_returns": gross.dropna(),
        "turnover": turnover,
        "sharpe_net": st.annualised_sharpe(net, PERIODS_PER_YEAR),
        "sharpe_gross": st.annualised_sharpe(gross.dropna(), PERIODS_PER_YEAR),
        "max_drawdown": st.max_drawdown((1 + net).cumprod()),
        "ann_turnover": float(turnover.mean() * PERIODS_PER_YEAR),
        "n_obs": int(len(net)),
    }
