"""Vectorised backtest with transaction costs. A position decided at t earns the
t -> t+1 return: the position is shifted forward one step before being applied to
returns, which prevents look-ahead. Costs are charged on traded notional |dpos|."""

from __future__ import annotations

import pandas as pd

from . import signals as sig
from . import stats as st

PERIODS_PER_YEAR = 252


def run(close: pd.Series, position: pd.Series, cost_bps: float = 1.0) -> dict:
    """Backtest a target-position series. cost_bps is per-side cost on traded notional."""
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


def demo() -> None:
    from .data import load_equity

    close = load_equity("SPY", "2010-01-01", "2024-12-31")["close"]
    print(f"{'signal':<22}{'SR_net':>9}{'SR_gross':>10}{'maxDD':>9}{'turn/yr':>9}")
    for name, fn in sig.REGISTRY.items():
        r = run(close, fn(close), cost_bps=1.0)
        print(f"{name:<22}{r['sharpe_net']:>9.2f}{r['sharpe_gross']:>10.2f}"
              f"{r['max_drawdown']:>9.2%}{r['ann_turnover']:>9.1f}")
    print("Significance requires deflated_sharpe and pbo_cscv over the full variant sweep.")


if __name__ == "__main__":
    demo()
