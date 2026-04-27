"""Tests for backtest_engine.strategies.regime."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine.strategies.regime import (
    apply_regime_filter,
    trend_filter,
    volatility_regime,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rising_prices(n: int = 300, start: float = 100.0) -> pd.Series:
    """Monotonically rising price series — always above any SMA."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.Series(start + np.arange(n, dtype=float) * 0.5, index=dates)
    return prices


def _falling_prices(n: int = 300, start: float = 100.0) -> pd.Series:
    """Monotonically falling price series — always below any SMA."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = pd.Series(start - np.arange(n, dtype=float) * 0.1, index=dates)
    return prices


def _constant_returns(n: int = 100, daily_ret: float = 0.001) -> pd.Series:
    """Constant daily returns → vol = 0."""
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(daily_ret, index=dates)


def _volatile_returns(n: int = 100, seed: int = 42) -> pd.Series:
    """High-vol daily returns (std ~ 3 % daily → ~48 % annualised)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.0, 0.03, n), index=dates)


# ---------------------------------------------------------------------------
# trend_filter
# ---------------------------------------------------------------------------

class TestTrendFilter:
    def test_warm_up_is_risk_off(self) -> None:
        prices = _rising_prices(n=300)
        result = trend_filter(prices, sma_window=200)
        # First 199 rows: SMA not yet defined → risk-off
        assert not result.iloc[:199].any()

    def test_rising_prices_are_risk_on(self) -> None:
        prices = _rising_prices(n=300)
        result = trend_filter(prices, sma_window=200)
        # After warm-up a monotonically rising series must be above its SMA
        assert result.iloc[200:].all()

    def test_falling_prices_are_risk_off(self) -> None:
        prices = _falling_prices(n=300)
        result = trend_filter(prices, sma_window=200)
        # After warm-up a monotonically falling series is below its SMA
        assert not result.iloc[200:].any()

    def test_returns_bool_series(self) -> None:
        prices = _rising_prices()
        result = trend_filter(prices)
        assert result.dtype == bool

    def test_same_index_as_input(self) -> None:
        prices = _rising_prices(n=250)
        result = trend_filter(prices, sma_window=50)
        assert result.index.equals(prices.index)

    def test_invalid_sma_window_raises(self) -> None:
        prices = _rising_prices()
        with pytest.raises(ValueError, match="sma_window"):
            trend_filter(prices, sma_window=1)


# ---------------------------------------------------------------------------
# volatility_regime
# ---------------------------------------------------------------------------

class TestVolatilityRegime:
    def test_warm_up_is_risk_off(self) -> None:
        rets = _constant_returns(n=100)
        lookback = 20
        result = volatility_regime(rets, lookback=lookback)
        # rolling(N, min_periods=N): first N-1 entries are NaN → risk-off
        assert not result.iloc[: lookback - 1].any()

    def test_zero_vol_is_risk_on(self) -> None:
        rets = _constant_returns(n=100, daily_ret=0.0)
        result = volatility_regime(rets, threshold=0.20, lookback=20)
        # Constant returns → vol = 0 < any positive threshold
        assert result.iloc[20:].all()

    def test_high_vol_is_risk_off(self) -> None:
        rets = _volatile_returns(n=200)
        result = volatility_regime(rets, threshold=0.10, lookback=20)
        # ~48 % annualised vol >> 10 % threshold → mostly risk-off post warmup
        post = result.iloc[25:]
        assert post.mean() < 0.20

    def test_returns_bool_series(self) -> None:
        rets = _constant_returns()
        result = volatility_regime(rets)
        assert result.dtype == bool

    def test_same_index_as_input(self) -> None:
        rets = _constant_returns(n=80)
        result = volatility_regime(rets, lookback=20)
        assert result.index.equals(rets.index)

    def test_invalid_threshold_raises(self) -> None:
        rets = _constant_returns()
        with pytest.raises(ValueError, match="threshold"):
            volatility_regime(rets, threshold=0.0)

    def test_invalid_lookback_raises(self) -> None:
        rets = _constant_returns()
        with pytest.raises(ValueError, match="lookback"):
            volatility_regime(rets, lookback=1)


# ---------------------------------------------------------------------------
# apply_regime_filter
# ---------------------------------------------------------------------------

class TestApplyRegimeFilter:
    def _signals(self, n: int = 50) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.DataFrame(
            {"A": 1.0, "B": -1.0},
            index=dates,
        )

    def _all_risk_on(self, n: int = 50) -> pd.Series:
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.Series(True, index=dates)

    def _all_risk_off(self, n: int = 50) -> pd.Series:
        dates = pd.date_range("2020-01-01", periods=n, freq="B")
        return pd.Series(False, index=dates)

    def test_risk_on_passes_signals_through(self) -> None:
        signals = self._signals()
        regime = self._all_risk_on()
        filtered = apply_regime_filter(signals, regime)
        pd.testing.assert_frame_equal(filtered, signals.astype(float))

    def test_risk_off_zeros_all_signals(self) -> None:
        signals = self._signals()
        regime = self._all_risk_off()
        filtered = apply_regime_filter(signals, regime)
        assert (filtered == 0.0).all(axis=None)

    def test_partial_regime_zeroes_correct_rows(self) -> None:
        n = 20
        signals = self._signals(n)
        dates = signals.index
        regime = pd.Series(True, index=dates)
        regime.iloc[:10] = False
        filtered = apply_regime_filter(signals, regime)
        assert (filtered.iloc[:10] == 0.0).all(axis=None)
        assert (filtered.iloc[10:] != 0.0).all(axis=None)

    def test_output_shape_matches_signals(self) -> None:
        signals = self._signals()
        regime = self._all_risk_on()
        filtered = apply_regime_filter(signals, regime)
        assert filtered.shape == signals.shape

    def test_missing_regime_dates_treated_as_risk_off(self) -> None:
        signals = self._signals(n=10)
        # Regime only covers first 5 rows
        regime = pd.Series(True, index=signals.index[:5])
        filtered = apply_regime_filter(signals, regime)
        # First 5 rows: risk-on → signals pass through
        assert (filtered.iloc[:5] != 0.0).all(axis=None)
        # Last 5 rows: NaN → risk-off → zeroed
        assert (filtered.iloc[5:] == 0.0).all(axis=None)
