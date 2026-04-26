"""SMA crossover strategy — long-only, V1."""

from __future__ import annotations

import pandas as pd


def sma_crossover_signals(
    prices: pd.DataFrame,
    fast_window: int = 50,
    slow_window: int = 200,
) -> pd.DataFrame:
    """Generate long-only SMA crossover signals.

    Returns 1 (long) when the fast SMA is strictly above the slow SMA,
    0 (flat) otherwise.  No short selling in V1.

    The function does NOT shift signals — the backtest engine applies
    ``shift(1)`` internally, so a signal computed from prices through
    day T is only executed at the open of day T+1.

    During the warm-up period (first ``slow_window - 1`` bars) the slow
    SMA is undefined (NaN).  A NaN comparison always returns False, so
    the signal is correctly 0 (stay flat) until we have enough history.

    Parameters
    ----------
    prices:
        Adjusted close prices — rows = dates, cols = tickers.
    fast_window:
        Lookback period for the fast SMA (e.g. 50 days).
    slow_window:
        Lookback period for the slow SMA (e.g. 200 days).

    Returns
    -------
    pd.DataFrame
        Same shape as ``prices``; values are 0.0 or 1.0.

    Raises
    ------
    ValueError
        If ``fast_window >= slow_window``.
    """
    if fast_window >= slow_window:
        raise ValueError(
            f"fast_window ({fast_window}) must be less than slow_window ({slow_window})"
        )

    fast = prices.rolling(fast_window, min_periods=fast_window).mean()
    slow = prices.rolling(slow_window, min_periods=slow_window).mean()

    # NaN comparisons evaluate to False in pandas, so the warm-up rows
    # automatically produce 0 without any explicit fillna call.
    return (fast > slow).astype(float)
