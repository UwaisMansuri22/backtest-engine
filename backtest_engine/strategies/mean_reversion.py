"""Short-term mean-reversion strategy.

Entry rule (long only):
    RSI(5) < rsi_oversold  AND  z-score(20) < zscore_threshold

Exit rules (first condition met wins):
    1. RSI > 50  (momentum reverting to neutral)
    2. Position held for max_hold_days days  (time stop)

Regime gate:
    An optional SPY-200-SMA regime Series zeros all signals on risk-off
    days.  Passing ``regime=None`` skips the gate (always on).

No short-selling: the strategy is long-only.  On risk-off days or when
no entry fires, the signal is 0.

Lookahead-bias note
-------------------
All indicator calculations use only data available at bar close T.
Signals are NOT shifted in this module — consistent with ``momentum_signals``.
``run_backtest`` applies ``shift(1)`` internally so execution happens at
the open of day T+1.  In the live runner, signals computed from today's
close drive orders submitted today for tomorrow's open — same result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest_engine.strategies.regime import apply_regime_filter
from backtest_engine.utils.indicators import rsi, zscore


def mean_reversion_signals(
    prices: pd.DataFrame,
    rsi_period: int = 5,
    rsi_oversold: float = 35.0,
    rsi_exit: float = 50.0,
    zscore_window: int = 20,
    zscore_threshold: float = -2.0,
    max_hold_days: int = 5,
    regime: pd.Series | None = None,
) -> pd.DataFrame:
    """Generate daily mean-reversion long signals for each ticker.

    Entry: RSI(rsi_period) < rsi_oversold AND z-score(zscore_window) < zscore_threshold
    Exit:  RSI > rsi_exit  OR  held position for max_hold_days consecutive days

    Parameters
    ----------
    prices:
        Adjusted close prices — rows = dates, cols = tickers.
    rsi_period:
        RSI look-back period.  5 is used for short-term sensitivity.
    rsi_oversold:
        RSI threshold below which a ticker is considered oversold.
    rsi_exit:
        RSI level above which the long position is closed.
    zscore_window:
        Rolling window for return z-score calculation.
    zscore_threshold:
        Z-score below which a return is considered an unusual selloff.
    max_hold_days:
        Maximum consecutive days to hold before a forced exit (time stop).
    regime:
        Optional daily boolean Series (True = risk-on).  When provided,
        all signals are zeroed on risk-off days.

    Returns
    -------
    pd.DataFrame
        Daily signals in {0.0, 1.0}; same shape as ``prices``.
        1.0 = hold long; 0.0 = flat.  No shift applied — pass directly to
        ``run_backtest``; the engine applies ``shift(1)`` before execution.
    """
    returns = prices.pct_change()

    rsi_df = prices.apply(lambda col: rsi(col, period=rsi_period))
    zscore_df = returns.apply(lambda col: zscore(col, window=zscore_window))

    entry_signal = (rsi_df < rsi_oversold) & (zscore_df < zscore_threshold)

    signals = pd.DataFrame(
        np.zeros((len(prices), len(prices.columns)), dtype=float),
        index=prices.index,
        columns=prices.columns,
    )

    # Stateful per-ticker loop: track hold duration to implement the time stop.
    # Pure vectorization is not practical here because exit depends on the
    # number of *consecutive* bars held, which creates backward dependency.
    for ticker in prices.columns:
        entry = entry_signal[ticker].values
        rsi_vals = rsi_df[ticker].values
        sig = signals[ticker].values.copy()
        held = 0

        for i in range(len(prices)):
            if held > 0:
                # Exit: RSI recovered or time stop
                if rsi_vals[i] > rsi_exit or held >= max_hold_days:
                    held = 0
                else:
                    sig[i] = 1.0
                    held += 1
            elif entry[i]:
                sig[i] = 1.0
                held = 1

        signals[ticker] = sig

    if regime is not None:
        signals = apply_regime_filter(signals, regime)

    return signals
