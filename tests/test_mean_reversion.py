"""Tests for mean_reversion_signals, indicators, and blend_signals."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest_engine.strategies.mean_reversion import mean_reversion_signals
from backtest_engine.strategies.portfolio import blend_signals
from backtest_engine.utils.indicators import rsi, zscore

# ── helpers ──────────────────────────────────────────────────────────────────

def _make_prices(n: int = 120, tickers: list[str] | None = None) -> pd.DataFrame:
    """Deterministic price series that trends gently up."""
    if tickers is None:
        tickers = ["A", "B", "C"]
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    rng = np.random.default_rng(42)
    base = 100.0
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.0003 * (i + 1)
        noise = rng.normal(drift, 0.01, n)
        data[t] = base * np.exp(np.cumsum(noise))
    return pd.DataFrame(data, index=idx)


def _make_weights(prices: pd.DataFrame, fill: float = 0.33) -> pd.DataFrame:
    return pd.DataFrame(fill, index=prices.index, columns=prices.columns)


# ── RSI tests ─────────────────────────────────────────────────────────────────

class TestRSI:
    def test_range_0_to_100(self) -> None:
        prices = _make_prices(200, ["X"])["X"]
        r = rsi(prices, period=14)
        valid = r.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_warmup_is_nan(self) -> None:
        prices = _make_prices(50, ["X"])["X"]
        r = rsi(prices, period=14)
        # First row is always NaN (no diff)
        assert pd.isna(r.iloc[0])

    def test_constant_series_no_crash(self) -> None:
        idx = pd.date_range("2022-01-03", periods=30, freq="B")
        prices = pd.Series(100.0, index=idx)
        r = rsi(prices, period=5)
        # Constant series: no gains or losses — should not raise
        assert len(r) == 30

    def test_period_validation(self) -> None:
        prices = _make_prices(50, ["X"])["X"]
        with pytest.raises(ValueError, match="period must be"):
            rsi(prices, period=1)


# ── Z-score tests ─────────────────────────────────────────────────────────────

class TestZScore:
    def test_mean_zero_std_one(self) -> None:
        """Over a stationary window the average z-score should be ~0."""
        idx = pd.date_range("2022-01-03", periods=200, freq="B")
        rng = np.random.default_rng(0)
        series = pd.Series(rng.normal(0, 1, 200), index=idx)
        z = zscore(series, window=20)
        # Mean of z-scores should be close to 0 after warm-up
        assert abs(z.dropna().mean()) < 0.5

    def test_warmup_is_nan(self) -> None:
        idx = pd.date_range("2022-01-03", periods=50, freq="B")
        series = pd.Series(range(50), index=idx, dtype=float)
        z = zscore(series, window=20)
        assert z.isna().iloc[:19].all()

    def test_window_validation(self) -> None:
        idx = pd.date_range("2022-01-03", periods=30, freq="B")
        series = pd.Series(1.0, index=idx)
        with pytest.raises(ValueError, match="window must be"):
            zscore(series, window=1)


# ── mean_reversion_signals tests ──────────────────────────────────────────────

class TestMeanReversionSignals:
    def test_output_shape(self) -> None:
        prices = _make_prices(150)
        sig = mean_reversion_signals(prices)
        assert sig.shape == prices.shape

    def test_signals_binary(self) -> None:
        prices = _make_prices(150)
        sig = mean_reversion_signals(prices)
        unique = set(sig.values.flatten().round(6))
        assert unique.issubset({0.0, 1.0})

    def test_no_signal_before_warmup(self) -> None:
        prices = _make_prices(150)
        # zscore(20) needs min_periods=20 → first non-NaN at bar 19 (0-indexed).
        # Bars 0-18 have NaN indicators so no entry can fire.
        sig = mean_reversion_signals(prices, rsi_period=5, zscore_window=20)
        assert sig.iloc[:19].sum().sum() == 0.0

    def test_max_hold_days_respected(self) -> None:
        """No run of 1s should exceed max_hold_days."""
        prices = _make_prices(200)
        max_hold = 3
        sig = mean_reversion_signals(prices, max_hold_days=max_hold)
        for col in sig.columns:
            s = sig[col]
            run = 0
            for v in s:
                if v == 1.0:
                    run += 1
                    assert run <= max_hold, f"{col}: run of {run} exceeds max_hold_days={max_hold}"
                else:
                    run = 0

    def test_regime_filter_zeros_signals(self) -> None:
        prices = _make_prices(150)
        regime = pd.Series(False, index=prices.index)
        sig = mean_reversion_signals(prices, regime=regime)
        assert sig.sum().sum() == 0.0


# ── blend_signals tests ───────────────────────────────────────────────────────

class TestBlendSignals:
    def test_weighted_average(self) -> None:
        prices = _make_prices(100)
        w1 = _make_weights(prices, 0.1)
        w2 = _make_weights(prices, 0.05)
        # gross = 3 * (0.1*0.6 + 0.05*0.4) = 3 * 0.08 = 0.24 < 1.0 → no cap
        blended = blend_signals({"a": (w1, 0.6), "b": (w2, 0.4)})
        expected_val = 0.1 * 0.6 + 0.05 * 0.4  # = 0.08
        assert abs(blended.iloc[0, 0] - expected_val) < 1e-9

    def test_max_leverage_cap(self) -> None:
        prices = _make_prices(100)
        # Weights of 0.8 per ticker across 3 tickers → gross = 2.4 > 1.0
        w = _make_weights(prices, 0.8)
        blended = blend_signals({"s": (w, 1.0)}, max_leverage=1.0)
        gross = blended.abs().sum(axis=1)
        assert (gross <= 1.0 + 1e-9).all()

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            blend_signals({})

    def test_shape_preserved(self) -> None:
        prices = _make_prices(100)
        w = _make_weights(prices, 0.33)
        blended = blend_signals({"s": (w, 1.0)})
        assert blended.shape == prices.shape
