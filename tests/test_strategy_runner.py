"""Tests for monthly rebalance gate helpers in strategy_runner."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from backtest_engine.live.strategy_runner import (
    is_rebalance_day,
    position_drift_exceeds_threshold,
)

# ── is_rebalance_day ──────────────────────────────────────────────────────────

def _frozen(date_str: str) -> pd.Timestamp:
    """Pre-compute a Timestamp BEFORE any patch is active."""
    return pd.Timestamp(date_str).normalize()


class TestIsRebalanceDay:
    def test_last_trading_day_of_month_returns_true(self):
        # April 30 2026 is a Thursday; next business day is May 1 (different month)
        frozen = _frozen("2026-04-30")
        with patch("backtest_engine.live.strategy_runner.pd.Timestamp") as mock_ts:
            mock_ts.today.return_value = frozen
            assert is_rebalance_day() is True

    def test_last_business_day_before_weekend_eom_returns_true(self):
        # May 29 2026 is a Friday; next bday is June 1 (different month)
        frozen = _frozen("2026-05-29")
        with patch("backtest_engine.live.strategy_runner.pd.Timestamp") as mock_ts:
            mock_ts.today.return_value = frozen
            assert is_rebalance_day() is True

    def test_mid_month_returns_false(self):
        # May 15 2026 is a Friday; next bday is May 18 (same month)
        frozen = _frozen("2026-05-15")
        with patch("backtest_engine.live.strategy_runner.pd.Timestamp") as mock_ts:
            mock_ts.today.return_value = frozen
            assert is_rebalance_day() is False

    def test_first_of_month_returns_false(self):
        frozen = _frozen("2026-05-01")
        with patch("backtest_engine.live.strategy_runner.pd.Timestamp") as mock_ts:
            mock_ts.today.return_value = frozen
            assert is_rebalance_day() is False

    def test_second_to_last_trading_day_returns_false(self):
        # April 29 2026 (Wed); next bday is Apr 30 (same month)
        frozen = _frozen("2026-04-29")
        with patch("backtest_engine.live.strategy_runner.pd.Timestamp") as mock_ts:
            mock_ts.today.return_value = frozen
            assert is_rebalance_day() is False


# ── position_drift_exceeds_threshold ─────────────────────────────────────────

class TestPositionDriftExceedsThreshold:
    def test_no_drift_returns_false(self):
        current = {"XLE": 0.33, "XLK": 0.33, "TLT": 0.34}
        target  = {"XLE": 0.33, "XLK": 0.33, "TLT": 0.34}
        assert position_drift_exceeds_threshold(current, target) is False

    def test_small_drift_below_threshold_returns_false(self):
        current = {"XLE": 0.33, "XLK": 0.33}
        target  = {"XLE": 0.36, "XLK": 0.33}  # 3% drift < 5%
        assert position_drift_exceeds_threshold(current, target) is False

    def test_drift_exactly_at_threshold_returns_false(self):
        current = {"XLE": 0.30}
        target  = {"XLE": 0.35}  # exactly 5% — not strictly greater
        assert position_drift_exceeds_threshold(current, target) is False

    def test_drift_above_threshold_returns_true(self):
        current = {"XLE": 0.30}
        target  = {"XLE": 0.36}  # 6% drift > 5%
        assert position_drift_exceeds_threshold(current, target) is True

    def test_new_position_triggers_drift(self):
        # current has no XLK; target wants 33% — that's a new 33% position
        current = {"XLE": 0.50}
        target  = {"XLE": 0.33, "XLK": 0.33}
        assert position_drift_exceeds_threshold(current, target) is True

    def test_liquidation_target_triggers_drift(self):
        # current holds XLE 33%; target is 0% (liquidate)
        current = {"XLE": 0.33}
        target  = {}
        assert position_drift_exceeds_threshold(current, target) is True

    def test_custom_threshold(self):
        current = {"XLE": 0.30}
        target  = {"XLE": 0.36}  # 6% drift
        assert position_drift_exceeds_threshold(current, target, threshold=0.10) is False
        assert position_drift_exceeds_threshold(current, target, threshold=0.05) is True

    def test_empty_both_returns_false(self):
        assert position_drift_exceeds_threshold({}, {}) is False

    def test_multiple_symbols_only_one_drifts(self):
        current = {"XLE": 0.33, "XLK": 0.33, "TLT": 0.34}
        target  = {"XLE": 0.33, "XLK": 0.33, "TLT": 0.20}  # TLT drifted 14%
        assert position_drift_exceeds_threshold(current, target) is True
