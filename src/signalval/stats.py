"""Statistics for separating real edge from overfit noise.

References:
  Bailey & Lopez de Prado (2014), The Deflated Sharpe Ratio.
  Bailey, Borwein, Lopez de Prado, Zhu (2017), The Probability of Backtest
  Overfitting (CSCV).
Sharpe ratios are per-period unless explicitly annualised.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy.stats import norm

EULER_MASCHERONI = 0.5772156649015329


def annualised_sharpe(r: pd.Series, periods: int = 252) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return 0.0
    return float(r.mean() / r.std(ddof=1) * math.sqrt(periods))


def max_drawdown(equity: pd.Series) -> float:
    return float((equity / equity.cummax() - 1.0).min())


def probabilistic_sharpe(r: pd.Series, sr_benchmark: float = 0.0) -> float:
    """P(true per-period Sharpe > sr_benchmark), corrected for skew, kurtosis and
    sample length."""
    r = r.dropna().to_numpy()
    n = len(r)
    if n < 3 or r.std(ddof=1) == 0:
        return float("nan")
    sr = r.mean() / r.std(ddof=1)
    skew = float(pd.Series(r).skew())
    kurt = float(pd.Series(r).kurtosis()) + 3.0  # pandas returns excess kurtosis
    denom = math.sqrt(1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2)
    return float(norm.cdf((sr - sr_benchmark) * math.sqrt(n - 1) / denom))


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """Expected maximum per-period Sharpe under the null across n_trials variants;
    the SR* threshold for the deflated Sharpe."""
    if n_trials < 2:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    return float(math.sqrt(sr_variance) * ((1 - EULER_MASCHERONI) * z1 + EULER_MASCHERONI * z2))


def deflated_sharpe(r: pd.Series, all_trial_sharpes: list[float]) -> float:
    """PSR evaluated against the expected-max-Sharpe implied by the number of
    variants tried. all_trial_sharpes are the per-period Sharpes of every variant."""
    sr_var = float(np.var(all_trial_sharpes, ddof=1))
    sr_star = expected_max_sharpe(sr_var, len(all_trial_sharpes))
    return probabilistic_sharpe(r, sr_benchmark=sr_star)


def pbo_cscv(perf: pd.DataFrame, n_splits: int = 10) -> float:
    """Probability of backtest overfitting via CSCV. perf is (T, N_strategies) of
    per-period returns. For every split of the rows into equal in/out halves, the
    in-sample-best strategy's out-of-sample rank is logit-transformed; PBO is the
    fraction of splits where that strategy lands in the bottom OOS half."""
    assert n_splits % 2 == 0, "n_splits must be even"
    M = perf.dropna().to_numpy()
    T, N = M.shape
    blocks = np.array_split(np.arange(T), n_splits)

    def sharpe_cols(rows: np.ndarray) -> np.ndarray:
        x = M[rows]
        sd = x.std(axis=0, ddof=1)
        sd[sd == 0] = np.nan
        return x.mean(axis=0) / sd

    logits = []
    for is_idx in itertools.combinations(range(n_splits), n_splits // 2):
        is_rows = np.concatenate([blocks[i] for i in is_idx])
        oos_rows = np.concatenate([blocks[i] for i in range(n_splits) if i not in is_idx])
        is_perf = sharpe_cols(is_rows)
        oos_perf = sharpe_cols(oos_rows)
        n_star = int(np.nanargmax(is_perf))
        rank = np.sum(oos_perf <= oos_perf[n_star]) / (N + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        logits.append(math.log(rank / (1 - rank)))

    return float(np.mean(np.array(logits) <= 0.0))


def decay_half_life(series: pd.Series) -> float:
    """Half-life in periods from the lag-1 autocorrelation: -ln(2)/ln(rho).
    inf if rho >= 1, nan if rho <= 0."""
    rho = series.dropna().autocorr(lag=1)
    if rho is None or rho <= 0:
        return float("nan")
    if rho >= 1:
        return float("inf")
    return float(-math.log(2) / math.log(rho))
