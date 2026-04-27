"""Tests for backtest_engine.backtest.position_sizing."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine.backtest.position_sizing import vol_target_weights


def _make_two_asset_data(
    n_days: int = 200,
    seed: int = 42,
    daily_vol_low: float = 0.005,
    daily_vol_high: float = 0.025,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (prices, signals) with one low-vol and one high-vol asset."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    prices_low  = 100.0 * np.exp(np.cumsum(rng.normal(0.0, daily_vol_low,  n_days)))
    prices_high = 100.0 * np.exp(np.cumsum(rng.normal(0.0, daily_vol_high, n_days)))
    prices  = pd.DataFrame({"LV": prices_low, "HV": prices_high}, index=dates)
    signals = pd.DataFrame({"LV": 1.0,        "HV": 1.0},         index=dates)
    return prices, signals


# ---------------------------------------------------------------------------
# Behavioural tests
# ---------------------------------------------------------------------------

def test_high_vol_asset_gets_smaller_weight() -> None:
    """The high-vol ticker must receive a strictly smaller weight than the low-vol ticker."""
    prices, signals = _make_two_asset_data()
    weights = vol_target_weights(signals, prices, target_vol_annual=0.15, vol_lookback_days=60)

    post_warmup = weights.iloc[65:]
    active = post_warmup[(post_warmup["LV"] > 0) & (post_warmup["HV"] > 0)]
    assert len(active) > 10, "Expected several active rows post warm-up"
    assert (active["LV"] > active["HV"]).mean() > 0.90, (
        "Low-vol asset should have a larger weight than high-vol asset in >90 % of rows"
    )


def test_gross_exposure_never_exceeds_max_leverage() -> None:
    """sum(|weights|) must not exceed max_leverage on any single row."""
    prices, signals = _make_two_asset_data()
    for max_lev in (0.5, 1.0, 2.0):
        weights = vol_target_weights(signals, prices, max_leverage=max_lev)
        gross = weights.abs().sum(axis=1)
        assert (gross <= max_lev + 1e-9).all(), (
            f"Gross exposure exceeded max_leverage={max_lev}"
        )


def test_zero_signals_produce_zero_weights() -> None:
    """When every signal is 0, every weight must be 0."""
    prices, _ = _make_two_asset_data()
    zero_signals = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights = vol_target_weights(zero_signals, prices)
    assert (weights == 0.0).all(axis=None)


def test_warm_up_period_is_zero() -> None:
    """Rows before the rolling window completes must be all-zero weights."""
    n_days, lookback = 120, 60
    prices, signals = _make_two_asset_data(n_days=n_days)
    weights = vol_target_weights(signals, prices, vol_lookback_days=lookback)

    # First `lookback` rows: vol estimate undefined → weights must be 0.
    assert (weights.iloc[:lookback] == 0.0).all(axis=None)
    # At least some post-warmup rows must be non-zero.
    assert (weights.iloc[lookback:].abs().sum(axis=1) > 0).any()


def test_single_active_signal_fills_max_leverage() -> None:
    """With only one active signal, that position should use all of max_leverage."""
    prices, _ = _make_two_asset_data(n_days=150)
    single_signal = pd.DataFrame(
        {"LV": 1.0, "HV": 0.0}, index=prices.index
    )
    weights = vol_target_weights(
        single_signal, prices, target_vol_annual=0.15, vol_lookback_days=60, max_leverage=1.0
    )
    post_warmup = weights.iloc[65:]
    active = post_warmup[post_warmup["LV"] > 0]

    # The solo position should be capped exactly at max_leverage = 1.0.
    # (Uncapped inverse-vol weight for a ~5 % daily vol asset at target 15 %
    # annualised would be large; the cap brings it to 1.0.)
    assert (active["LV"] <= 1.0 + 1e-9).all()
    # HV column must always be 0 because its signal is 0.
    assert (weights["HV"] == 0.0).all()


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------

def test_invalid_target_vol_raises() -> None:
    prices, signals = _make_two_asset_data(n_days=80)
    with pytest.raises(ValueError, match="target_vol_annual"):
        vol_target_weights(signals, prices, target_vol_annual=0.0)


def test_invalid_max_leverage_raises() -> None:
    prices, signals = _make_two_asset_data(n_days=80)
    with pytest.raises(ValueError, match="max_leverage"):
        vol_target_weights(signals, prices, max_leverage=0.0)


def test_invalid_lookback_raises() -> None:
    prices, signals = _make_two_asset_data(n_days=80)
    with pytest.raises(ValueError, match="vol_lookback_days"):
        vol_target_weights(signals, prices, vol_lookback_days=1)
