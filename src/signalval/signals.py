"""Candidate signals. Each returns a target position in [-1, 1] indexed like the
input, using information up to time t only; the backtester lags it before applying."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ts_momentum(close: pd.Series, lookback: int = 60) -> pd.Series:
    trailing = close.pct_change(lookback)
    return np.sign(trailing).fillna(0.0).clip(-1, 1)


def mean_reversion(close: pd.Series, lookback: int = 20, z_cap: float = 2.0) -> pd.Series:
    ma = close.rolling(lookback).mean()
    sd = close.rolling(lookback).std()
    z = (close - ma) / sd
    return (-z / z_cap).clip(-1, 1).fillna(0.0)


def vol_scaled_momentum(close: pd.Series, lookback: int = 60, vol_window: int = 20) -> pd.Series:
    sign = np.sign(close.pct_change(lookback))
    vol = close.pct_change().rolling(vol_window).std()
    scale = (vol.median() / vol).clip(upper=1.0)
    return (sign * scale).fillna(0.0).clip(-1, 1)


REGISTRY = {
    "ts_momentum": ts_momentum,
    "mean_reversion": mean_reversion,
    "vol_scaled_momentum": vol_scaled_momentum,
}
