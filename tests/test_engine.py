"""Tests for backtest_engine.backtest.engine."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine.backtest.engine import BacktestResult, run_backtest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prices(values: list[float], ticker: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {ticker: values},
        index=pd.date_range("2020-01-01", periods=len(values), freq="D"),
    )


def _signals(values: list[float], ticker: str = "A") -> pd.DataFrame:
    return pd.DataFrame(
        {ticker: values},
        index=pd.date_range("2020-01-01", periods=len(values), freq="D"),
    )


# ---------------------------------------------------------------------------
# Test 1: Buy-and-hold single asset
# ---------------------------------------------------------------------------

def test_buy_and_hold_matches_asset_return() -> None:
    """Always-long single asset: backtest return matches raw return within 0.1%.

    The engine shifts signals by 1 day, so we enter at prices[1].
    Transaction costs (7 bps total) are the only source of divergence.
    Tolerance of 10 bps (0.1%) is deliberately above 7 bps so the test
    passes even with the entry cost, confirming no other leakage.
    """
    prices = _prices([100.0, 101.0, 103.0, 102.0, 105.0])
    signals = _signals([1.0, 1.0, 1.0, 1.0, 1.0])

    result = run_backtest(
        prices, signals,
        initial_capital=10_000,
        transaction_cost_bps=5,
        slippage_bps=2,
    )

    # Entry is on day-index 1 (prices[1] = 101) because of the signal shift.
    # Compare the portfolio return from that entry to the final bar against
    # the raw asset return over the same window.
    backtest_return = float(result.equity_curve.iloc[-1] / result.equity_curve.iloc[1] - 1)
    asset_return = float(prices["A"].iloc[-1] / prices["A"].iloc[1] - 1)

    assert isinstance(result, BacktestResult)
    assert isinstance(result.equity_curve.index, pd.DatetimeIndex)
    assert abs(backtest_return - asset_return) < 0.001  # within 10 bps


# ---------------------------------------------------------------------------
# Test 2: Flat signal produces zero PnL
# ---------------------------------------------------------------------------

def test_flat_signal_zero_pnl() -> None:
    """Signals that are always 0 should produce no returns, no trades, no cost."""
    prices = _prices([100.0, 102.0, 98.0, 107.0])
    signals = _signals([0.0, 0.0, 0.0, 0.0])

    result = run_backtest(prices, signals, initial_capital=10_000)

    # Equity must be flat at initial capital throughout
    np.testing.assert_allclose(
        result.equity_curve.to_numpy(),
        np.full(len(prices), 10_000.0),
        rtol=1e-9,
    )
    assert (result.returns == 0.0).all()
    assert len(result.trades) == 0


# ---------------------------------------------------------------------------
# Test 3: Punishing transaction costs destroy portfolio value
# ---------------------------------------------------------------------------

def test_extreme_transaction_costs_destroy_value() -> None:
    """100 % per-trade cost (10_000 bps) on an active strategy wipes the portfolio.

    With constant prices, every position change produces a pure cost with no
    offsetting return.  After a handful of flips equity approaches zero,
    demonstrating that costs are charged on every trade and compound correctly.
    """
    # Alternating signals force a position flip on each bar after the shift.
    n = 10
    prices = pd.DataFrame(
        {"A": [100.0] * n},
        index=pd.date_range("2020-01-01", periods=n, freq="D"),
    )
    signals = pd.DataFrame(
        {"A": [1.0 if i % 2 == 0 else -1.0 for i in range(n)]},
        index=prices.index,
    )

    result = run_backtest(
        prices, signals,
        initial_capital=10_000,
        transaction_cost_bps=10_000,  # 100 % per unit of weight change
        slippage_bps=0,
    )

    assert result.equity_curve.iloc[-1] < result.equity_curve.iloc[0] * 0.01


# ---------------------------------------------------------------------------
# Test 4: Lookahead bias — extra shift degrades a perfect-foresight strategy
# ---------------------------------------------------------------------------

def test_extra_shift_degrades_foresight_strategy() -> None:
    """Shifting the signal by one extra day should produce worse results.

    The engine applies shift(1) internally.  A "cheating" signal that already
    encodes tomorrow's direction becomes correctly timed after the engine's shift.
    Manually pre-shifting the same signal by 1 more day misaligns it with the
    returns it was calibrated for, causing losses on an otherwise perfect strategy.

    This is the canonical lookahead-bias check: if your strategy only appears
    profitable before the engine's shift is applied, it is lookahead-biased.
    """
    # Alternating prices: up one day, down the next — perfectly predictable pattern.
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    alt = [100.0, 102.0, 100.0, 102.0, 100.0, 102.0, 100.0, 102.0]
    prices = pd.DataFrame({"A": alt}, index=dates)

    # Perfect-foresight signal: +1 before up bars, -1 before down bars.
    # sign(next_return) → shift(-1) of sign(pct_change())
    raw_rets = pd.Series(alt).pct_change()
    foresight_vals = raw_rets.shift(-1).apply(np.sign).fillna(0.0).tolist()
    foresight = pd.DataFrame({"A": foresight_vals}, index=dates)

    # Run 1: foresight signal fed directly.
    # The engine's shift(1) realigns it to the actual same-day return — profitable.
    result_aligned = run_backtest(
        prices, foresight,
        transaction_cost_bps=0, slippage_bps=0,
    )

    # Run 2: manually pre-shift by 1 extra day before passing to the engine.
    # The engine then applies another shift(1), so the signal is 2 bars late —
    # it now buys AFTER the up move has already happened.
    result_delayed = run_backtest(
        prices, foresight.shift(1).fillna(0.0),
        transaction_cost_bps=0, slippage_bps=0,
    )

    final_aligned = float(result_aligned.equity_curve.iloc[-1])
    final_delayed = float(result_delayed.equity_curve.iloc[-1])

    # The aligned (correctly timed) strategy must significantly outperform the delayed one.
    assert final_aligned > final_delayed, (
        f"Aligned {final_aligned:.2f} should beat delayed {final_delayed:.2f}"
    )
    # The aligned strategy must also be profitable (proves foresight works when timed right).
    assert final_aligned > float(result_aligned.equity_curve.iloc[0])
