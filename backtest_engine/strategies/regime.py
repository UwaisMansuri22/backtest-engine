"""Market regime filters for momentum strategies.

Momentum strategies blow up in bear markets — exactly when 'winners keep
winning' stops being true.  During sustained drawdowns (2008, 2022) the
cross-sectional signal just picks the least-bad losers while the broad tide
goes out.  A simple regime filter gates the strategy: only trade when the
market is in a defined 'risk-on' state.  On risk-off days, all signals are
zeroed and cash is held instead.

Two orthogonal regime definitions are provided:

* ``trend_filter``: price > N-day SMA (canonical 200-day trend rule)
* ``volatility_regime``: rolling realised vol < threshold (VIX-analogue)

``apply_regime_filter`` combines either output with a signal DataFrame.
"""

from __future__ import annotations

import math

import pandas as pd


def trend_filter(
    benchmark_prices: pd.Series,
    sma_window: int = 200,
) -> pd.Series:
    """Return a daily boolean Series: True = risk-on (price > SMA).

    During the warm-up period (first ``sma_window`` rows) where no full
    SMA is available, the regime is set to False (risk-off) to avoid
    acting on an incomplete estimate.

    Parameters
    ----------
    benchmark_prices:
        Daily adjusted close prices for the benchmark (e.g. SPY).
        Must be a pd.Series with a DatetimeIndex.
    sma_window:
        Number of trading days in the simple moving average (default 200).

    Returns
    -------
    pd.Series[bool]
        True on days the benchmark closes above its SMA; False otherwise.
        Same index as ``benchmark_prices``.

    Raises
    ------
    ValueError
        If ``sma_window`` is less than 2.
    """
    if sma_window < 2:
        raise ValueError(f"sma_window must be >= 2, got {sma_window}")

    sma: pd.Series = (
        benchmark_prices.rolling(sma_window, min_periods=sma_window).mean()
    )
    # NaN SMA (warm-up) → False (risk-off)
    risk_on: pd.Series = (benchmark_prices > sma).fillna(False)
    return risk_on


def volatility_regime(
    returns: pd.Series,
    threshold: float = 0.20,
    lookback: int = 20,
) -> pd.Series:
    """Return a daily boolean Series: True = low-vol (risk-on).

    Computes rolling annualised realised volatility and marks a day as
    risk-on when that vol is below ``threshold``.  During the warm-up
    period the regime is set to False (risk-off).

    Parameters
    ----------
    returns:
        Daily simple returns of the benchmark or portfolio.
    threshold:
        Annualised vol threshold (decimal).  Rows below this value are
        risk-on (default 0.20 = 20 %).
    lookback:
        Rolling window in trading days for vol estimation (default 20 ≈
        1 month).

    Returns
    -------
    pd.Series[bool]
        True on low-volatility (risk-on) days; False otherwise.
        Same index as ``returns``.

    Raises
    ------
    ValueError
        If ``threshold`` is not positive or ``lookback`` is less than 2.
    """
    if threshold <= 0:
        raise ValueError(f"threshold must be positive, got {threshold}")
    if lookback < 2:
        raise ValueError(f"lookback must be >= 2, got {lookback}")

    rolling_vol: pd.Series = (
        returns.rolling(lookback, min_periods=lookback).std() * math.sqrt(252)
    )
    # NaN vol (warm-up) → False (risk-off)
    risk_on: pd.Series = (rolling_vol < threshold).fillna(False)
    return risk_on


def apply_regime_filter(
    signals: pd.DataFrame,
    regime: pd.Series,
) -> pd.DataFrame:
    """Zero out signals on risk-off days.

    Multiplies every column of ``signals`` by the boolean ``regime``
    Series (True = 1, False = 0), so positions are flat on risk-off
    days.  Index alignment is performed automatically; any date present
    in ``signals`` but missing from ``regime`` is treated as risk-off
    (NaN → 0).

    Parameters
    ----------
    signals:
        Raw trading signals {-1, 0, 1} — rows = dates, cols = tickers.
    regime:
        Daily boolean risk-on indicator with a compatible DatetimeIndex.

    Returns
    -------
    pd.DataFrame
        Filtered signals, same shape as ``signals``.
        On risk-off days every value is 0.0.
    """
    mask = regime.reindex(signals.index).fillna(False).astype(float)
    return signals.mul(mask, axis=0)
