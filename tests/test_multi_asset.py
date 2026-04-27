"""Tests for backtest_engine.strategies.multi_asset."""

import numpy as np
import pandas as pd
import pytest

from backtest_engine.strategies.multi_asset import (
    SECTOR_ETFS,
    UNIVERSE,
    risk_parity_weights,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data(
    n_days: int = 200,
    n_assets: int = 3,
    seed: int = 42,
    daily_vols: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (returns, signals) for n_assets."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    if daily_vols is None:
        daily_vols = [0.01] * n_assets
    cols = [f"A{i}" for i in range(n_assets)]
    rets = pd.DataFrame(
        {col: rng.normal(0.0, vol, n_days) for col, vol in zip(cols, daily_vols, strict=True)},
        index=dates,
    )
    signals = pd.DataFrame(1.0, index=dates, columns=cols)
    return rets, signals


# ---------------------------------------------------------------------------
# UNIVERSE / SECTOR_ETFS constants
# ---------------------------------------------------------------------------

class TestUniverse:
    def test_universe_has_14_tickers(self) -> None:
        assert len(UNIVERSE) == 14

    def test_sector_etfs_has_11_tickers(self) -> None:
        assert len(SECTOR_ETFS) == 11

    def test_universe_contains_diversifiers(self) -> None:
        for ticker in ("TLT", "GLD", "IEF"):
            assert ticker in UNIVERSE

    def test_universe_contains_all_sectors(self) -> None:
        for ticker in SECTOR_ETFS:
            assert ticker in UNIVERSE

    def test_sector_etfs_are_prefix_of_universe(self) -> None:
        assert UNIVERSE[:11] == SECTOR_ETFS

    def test_no_duplicate_tickers(self) -> None:
        assert len(UNIVERSE) == len(set(UNIVERSE))


# ---------------------------------------------------------------------------
# risk_parity_weights — behavioural
# ---------------------------------------------------------------------------

class TestRiskParityWeights:
    def test_equal_vol_assets_get_equal_weights(self) -> None:
        """When all active assets have equal vol, weights must be exactly 1/n."""
        n = 3
        rng = np.random.default_rng(0)
        dates = pd.date_range("2020-01-01", periods=200, freq="B")
        # Identical return series → identical rolling vol → identical weights.
        base = rng.normal(0.0, 0.01, 200)
        cols = [f"A{i}" for i in range(n)]
        rets = pd.DataFrame({col: base for col in cols}, index=dates)
        signals = pd.DataFrame(1.0, index=dates, columns=cols)
        weights = risk_parity_weights(signals, rets, lookback_days=60)
        post = weights.iloc[65:]
        active = post[(post > 0).all(axis=1)]
        assert len(active) > 10
        expected = 1.0 / n
        assert (active - expected).abs().max().max() < 1e-9

    def test_high_vol_asset_gets_smaller_weight(self) -> None:
        """High-vol asset must have strictly smaller weight than low-vol asset."""
        rets, signals = _make_data(
            n_days=200, n_assets=2, daily_vols=[0.005, 0.025]
        )
        weights = risk_parity_weights(signals, rets, lookback_days=60)
        post = weights.iloc[65:]
        active = post[(post["A0"] > 0) & (post["A1"] > 0)]
        assert len(active) > 10
        assert (active["A0"] > active["A1"]).mean() > 0.90

    def test_weights_sum_to_one_post_warmup(self) -> None:
        """Gross exposure must equal 1.0 on every active post-warmup row."""
        rets, signals = _make_data(n_days=200, n_assets=3)
        weights = risk_parity_weights(signals, rets, lookback_days=60)
        post = weights.iloc[65:]
        active = post[post.abs().sum(axis=1) > 0]
        gross = active.abs().sum(axis=1)
        assert (gross - 1.0).abs().max() < 1e-9

    def test_warmup_rows_are_zero(self) -> None:
        """First lookback_days−1 rows must be all zero."""
        lookback = 60
        rets, signals = _make_data(n_days=200, n_assets=2)
        weights = risk_parity_weights(signals, rets, lookback_days=lookback)
        assert (weights.iloc[: lookback - 1] == 0.0).all(axis=None)

    def test_zero_signals_produce_zero_weights(self) -> None:
        """When every signal is 0, every weight must be 0."""
        rets, _ = _make_data(n_days=200, n_assets=3)
        zero_signals = pd.DataFrame(0.0, index=rets.index, columns=rets.columns)
        weights = risk_parity_weights(zero_signals, rets, lookback_days=60)
        assert (weights == 0.0).all(axis=None)

    def test_inactive_asset_has_zero_weight(self) -> None:
        """An asset with signal=0 must have weight=0 regardless of its vol."""
        rets, _ = _make_data(n_days=200, n_assets=2)
        # Only A0 is active
        signals = pd.DataFrame({"A0": 1.0, "A1": 0.0}, index=rets.index)
        weights = risk_parity_weights(signals, rets, lookback_days=60)
        assert (weights["A1"] == 0.0).all()

    def test_single_active_asset_gets_full_weight(self) -> None:
        """With one active asset, its weight must equal 1.0 post warm-up."""
        rets, _ = _make_data(n_days=200, n_assets=2)
        signals = pd.DataFrame({"A0": 1.0, "A1": 0.0}, index=rets.index)
        weights = risk_parity_weights(signals, rets, lookback_days=60)
        post = weights.iloc[65:]
        active = post[post["A0"] > 0]
        assert (active["A0"] - 1.0).abs().max() < 1e-9

    def test_short_signal_produces_negative_weight(self) -> None:
        """signal = −1 must produce a negative weight of the same magnitude."""
        rets, _ = _make_data(n_days=200, n_assets=2)
        signals = pd.DataFrame({"A0": 1.0, "A1": -1.0}, index=rets.index)
        weights = risk_parity_weights(signals, rets, lookback_days=60)
        post = weights.iloc[65:]
        active = post[(post["A0"] > 0) & (post["A1"] < 0)]
        assert len(active) > 10
        assert (active["A1"] < 0.0).all()

    def test_output_shape_matches_signals(self) -> None:
        rets, signals = _make_data(n_days=150, n_assets=4)
        weights = risk_parity_weights(signals, rets, lookback_days=60)
        assert weights.shape == signals.shape

    def test_invalid_lookback_raises(self) -> None:
        rets, signals = _make_data(n_days=80)
        with pytest.raises(ValueError, match="lookback_days"):
            risk_parity_weights(signals, rets, lookback_days=1)
