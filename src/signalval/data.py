from __future__ import annotations

import pandas as pd


def load_equity(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Daily OHLCV for one equity via yfinance, indexed by date."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"no data for {ticker} in [{start}, {end}]")
    df = df.rename(columns=str.lower)
    df.index.name = "date"
    return df[["open", "high", "low", "close", "volume"]]


def returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()
