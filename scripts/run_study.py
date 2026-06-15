from __future__ import annotations

import itertools
import json
from pathlib import Path

import pandas as pd

from signalval import signals as sig
from signalval.backtest import run
from signalval.data import load_equity
from signalval.stats import decay_half_life, deflated_sharpe, pbo_cscv

TICKER = "SPY"
START = "2010-01-01"
END = "2024-12-31"
COST_BPS = 1.0


def variants():
    v = []
    for lb in (20, 40, 60, 90, 120, 180, 250):
        v.append((f"ts_mom_{lb}", lambda c, lb=lb: sig.ts_momentum(c, lb)))
    for lb, zc in itertools.product((5, 10, 15, 20, 30, 40), (1.5, 2.0, 2.5)):
        v.append((f"meanrev_{lb}_{zc}", lambda c, lb=lb, zc=zc: sig.mean_reversion(c, lb, zc)))
    for lb, vw in itertools.product((40, 60, 90, 120), (10, 20, 30)):
        v.append((f"volmom_{lb}_{vw}", lambda c, lb=lb, vw=vw: sig.vol_scaled_momentum(c, lb, vw)))
    return v


def main():
    close = load_equity(TICKER, START, END)["close"]
    specs = variants()
    rets, positions, rows = {}, {}, []
    for name, fn in specs:
        pos = fn(close)
        r = run(close, pos, cost_bps=COST_BPS)
        rets[name] = r["net_returns"]
        positions[name] = pos
        rows.append({"variant": name, "sharpe_net": r["sharpe_net"],
                     "max_drawdown": r["max_drawdown"], "turnover_yr": r["ann_turnover"]})

    M = pd.DataFrame(rets).dropna()
    trial_sharpes = [M[c].mean() / M[c].std(ddof=1) for c in M.columns]
    best = max(rows, key=lambda x: x["sharpe_net"])
    dsr = deflated_sharpe(M[best["variant"]], trial_sharpes)
    pbo = pbo_cscv(M, n_splits=10)
    half_life = decay_half_life(positions[best["variant"]])
    edge_supported = dsr > 0.95 and pbo < 0.5

    result = {
        "ticker": TICKER,
        "start": START,
        "end": END,
        "observations": int(len(M)),
        "variants": len(specs),
        "cost_bps": COST_BPS,
        "best": best,
        "deflated_sharpe_p": round(dsr, 4),
        "pbo_cscv": round(pbo, 4),
        "position_half_life_days": round(half_life, 2),
        "edge_supported": bool(edge_supported),
    }
    Path(__file__).resolve().parents[1].joinpath("results.json").write_text(
        json.dumps(result, indent=2) + "\n")

    print(f"{'variant':<18}{'SR_net':>9}{'maxDD':>9}{'turn/yr':>9}")
    for row in sorted(rows, key=lambda x: -x["sharpe_net"])[:8]:
        print(f"{row['variant']:<18}{row['sharpe_net']:>9.2f}"
              f"{row['max_drawdown']:>9.2%}{row['turnover_yr']:>9.1f}")
    print(f"\nbest={best['variant']} DSR_p={dsr:.3f} PBO={pbo:.3f} "
          f"half_life={half_life:.1f}d edge_supported={edge_supported}")


if __name__ == "__main__":
    main()
