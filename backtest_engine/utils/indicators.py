"""Technical indicators: RSI and z-score."""

from __future__ import annotations

import pandas as pd


def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index.

    Uses exponential smoothing (alpha = 1/period) — Wilder's original
    method — rather than a simple rolling mean.  This makes the indicator
    more responsive at short periods (e.g. 5) used for mean-reversion.

    Parameters
    ----------
    prices:
        Adjusted close price series with a DatetimeIndex.
    period:
        Smoothing period.  Common values: 14 (trend), 5 (mean-reversion).

    Returns
    -------
    pd.Series
        RSI values in [0, 100]; NaN during the warm-up window.
    """
    if period < 2:
        raise ValueError(f"period must be >= 2, got {period}")

    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    # Wilder smoothing: EWM with alpha=1/period, adjust=False matches the
    # recursive formula RS_t = (RS_{t-1} * (period-1) + gain_t) / period.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    result = 100.0 - (100.0 / (1.0 + rs))
    # Flat periods (zero loss and zero gain) → RSI = 50 by convention.
    result = result.where(avg_loss != 0, other=100.0)
    result = result.where(avg_gain != 0, other=result)
    # First row is always NaN (no diff); leave warm-up rows as NaN.
    return result


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Rolling z-score: (value - rolling mean) / rolling std.

    Values below -2.0 indicate a statistically unusual selloff over the
    lookback window — the core signal for mean-reversion entries.

    Parameters
    ----------
    series:
        Input series (typically daily returns).
    window:
        Rolling lookback in bars (default 20 ≈ 1 trading month).

    Returns
    -------
    pd.Series
        Z-score series; NaN during warm-up.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")

    mu = series.rolling(window, min_periods=window).mean()
    sigma = series.rolling(window, min_periods=window).std()
    return (series - mu) / sigma.replace(0, float("nan"))
