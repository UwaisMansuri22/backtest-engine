"""Tests for backtest_engine.live.safety_checks.

All Alpaca client calls are mocked — these tests must pass without
Alpaca API keys and without a network connection.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

from backtest_engine.live.safety_checks import (
    CheckResult,
    all_passed,
    check_last_run_time,
    check_market_open,
    check_order_notional,
    check_paper_account,
    check_position_concentration,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client(*, is_paper: bool = True, market_open: bool = True) -> MagicMock:
    """Return a mock AlpacaClient satisfying _ClientProtocol."""
    c = MagicMock()
    c.is_paper = is_paper
    c.is_market_open.return_value = market_open
    return c


def _write_log(log_dir: Path, hours_ago: float) -> None:
    """Write a fake live_log file with timestamp `hours_ago` hours in the past."""
    ts = datetime.now(tz=UTC) - timedelta(hours=hours_ago)
    data = {"timestamp": ts.isoformat(), "status": "executed"}
    (log_dir / f"live_log_{ts.strftime('%Y-%m-%d')}.json").write_text(
        json.dumps(data)
    )


# ---------------------------------------------------------------------------
# check_paper_account
# ---------------------------------------------------------------------------

class TestCheckPaperAccount:
    def test_passes_when_is_paper_true(self) -> None:
        r = check_paper_account(_client(is_paper=True))
        assert r.passed

    def test_fails_when_is_paper_false(self) -> None:
        r = check_paper_account(_client(is_paper=False))
        assert not r.passed
        assert "DANGER" in r.message

    def test_result_has_correct_name(self) -> None:
        r = check_paper_account(_client())
        assert r.name == "paper_account"

    def test_str_includes_status(self) -> None:
        r = check_paper_account(_client(is_paper=True))
        assert "[PASS]" in str(r)
        r2 = check_paper_account(_client(is_paper=False))
        assert "[FAIL]" in str(r2)


# ---------------------------------------------------------------------------
# check_order_notional
# ---------------------------------------------------------------------------

class TestCheckOrderNotional:
    def test_passes_within_limit(self) -> None:
        orders = [{"notional": 5_000.0}, {"notional": 3_000.0}]   # 80 % of equity
        r = check_order_notional(orders, equity=10_000.0)
        assert r.passed

    def test_fails_above_limit(self) -> None:
        orders = [{"notional": 6_000.0}, {"notional": 5_500.0}]   # 115 % of equity
        r = check_order_notional(orders, equity=10_000.0)
        assert not r.passed
        assert "exceeds" in r.message

    def test_passes_with_empty_orders(self) -> None:
        r = check_order_notional([], equity=10_000.0)
        assert r.passed

    def test_exactly_at_limit_passes(self) -> None:
        # Exactly 110 % — should pass (limit is strict >)
        r = check_order_notional([{"notional": 11_000.0}], equity=10_000.0)
        assert r.passed

    def test_custom_max_fraction(self) -> None:
        orders = [{"notional": 8_001.0}]     # > 80 % with max_fraction=0.80
        r = check_order_notional(orders, equity=10_000.0, max_fraction=0.80)
        assert not r.passed

    def test_result_name(self) -> None:
        r = check_order_notional([], equity=1.0)
        assert r.name == "order_notional"


# ---------------------------------------------------------------------------
# check_position_concentration
# ---------------------------------------------------------------------------

class TestCheckPositionConcentration:
    def test_passes_when_all_below_limit(self) -> None:
        w = {"XLK": 0.35, "XLE": 0.35, "GLD": 0.30}
        r = check_position_concentration(w)
        assert r.passed

    def test_fails_when_single_position_too_large(self) -> None:
        w = {"XLK": 0.45, "XLE": 0.30, "GLD": 0.25}
        r = check_position_concentration(w, max_weight=0.40)
        assert not r.passed
        assert "XLK" in r.message

    def test_passes_with_empty_weights(self) -> None:
        r = check_position_concentration({})
        assert r.passed

    def test_exactly_at_limit_passes(self) -> None:
        w = {"XLK": 0.40, "XLE": 0.30, "GLD": 0.30}
        r = check_position_concentration(w, max_weight=0.40)
        assert r.passed

    def test_negative_weights_checked_by_abs(self) -> None:
        # Short position of -0.45 should fail the 40 % limit
        w = {"XLK": -0.45, "XLE": 0.30}
        r = check_position_concentration(w, max_weight=0.40)
        assert not r.passed

    def test_result_name(self) -> None:
        r = check_position_concentration({})
        assert r.name == "concentration"


# ---------------------------------------------------------------------------
# check_market_open
# ---------------------------------------------------------------------------

class TestCheckMarketOpen:
    def test_passes_when_open(self) -> None:
        r = check_market_open(_client(market_open=True))
        assert r.passed

    def test_fails_when_closed(self) -> None:
        r = check_market_open(_client(market_open=False))
        assert not r.passed
        assert "closed" in r.message.lower()

    def test_calls_is_market_open_once(self) -> None:
        c = _client(market_open=True)
        check_market_open(c)
        c.is_market_open.assert_called_once()

    def test_result_name(self) -> None:
        r = check_market_open(_client())
        assert r.name == "market_open"


# ---------------------------------------------------------------------------
# check_last_run_time
# ---------------------------------------------------------------------------

class TestCheckLastRunTime:
    def test_passes_with_no_prior_logs(self, tmp_path: Path) -> None:
        r = check_last_run_time(str(tmp_path))
        assert r.passed
        assert "first execution" in r.message.lower()

    def test_passes_when_last_run_old_enough(self, tmp_path: Path) -> None:
        _write_log(tmp_path, hours_ago=25.0)
        r = check_last_run_time(str(tmp_path), min_hours=20.0)
        assert r.passed

    def test_fails_when_last_run_too_recent(self, tmp_path: Path) -> None:
        _write_log(tmp_path, hours_ago=0.5)
        r = check_last_run_time(str(tmp_path), min_hours=20.0)
        assert not r.passed
        assert "0." in r.message   # shows hours

    def test_fails_on_corrupt_log(self, tmp_path: Path) -> None:
        (tmp_path / "live_log_2024-01-01.json").write_text("NOT JSON {{{")
        r = check_last_run_time(str(tmp_path))
        assert not r.passed

    def test_fails_on_missing_timestamp_key(self, tmp_path: Path) -> None:
        (tmp_path / "live_log_2024-01-01.json").write_text(
            json.dumps({"status": "executed"})    # no 'timestamp' key
        )
        r = check_last_run_time(str(tmp_path))
        assert not r.passed

    def test_picks_latest_log_file(self, tmp_path: Path) -> None:
        # Write an old log AND a recent one; the recent one should block re-run
        _write_log(tmp_path, hours_ago=48.0)
        _write_log(tmp_path, hours_ago=1.0)
        r = check_last_run_time(str(tmp_path), min_hours=20.0)
        assert not r.passed

    def test_result_name(self, tmp_path: Path) -> None:
        r = check_last_run_time(str(tmp_path))
        assert r.name == "last_run"


# ---------------------------------------------------------------------------
# run_all_checks + all_passed
# ---------------------------------------------------------------------------

class TestRunAllChecks:
    def test_all_pass_with_valid_inputs(self, tmp_path: Path) -> None:
        c = _client(is_paper=True, market_open=True)
        orders = [{"notional": 3_000.0}]
        weights = {"XLK": 0.35, "XLE": 0.35, "GLD": 0.30}
        results = run_all_checks(c, orders, weights, equity=10_000.0, log_dir=str(tmp_path))
        assert all_passed(results)

    def test_closed_market_blocks_all(self, tmp_path: Path) -> None:
        c = _client(market_open=False)
        results = run_all_checks(c, [], {}, equity=10_000.0, log_dir=str(tmp_path))
        assert not all_passed(results)
        failed_names = {r.name for r in results if not r.passed}
        assert "market_open" in failed_names

    def test_oversized_order_blocks_all(self, tmp_path: Path) -> None:
        c = _client(market_open=True)
        orders = [{"notional": 99_999.0}]   # >> 110 % of $1
        results = run_all_checks(c, orders, {}, equity=1.0, log_dir=str(tmp_path))
        assert not all_passed(results)
        failed_names = {r.name for r in results if not r.passed}
        assert "order_notional" in failed_names

    def test_recent_run_blocks_all(self, tmp_path: Path) -> None:
        _write_log(tmp_path, hours_ago=1.0)
        c = _client(market_open=True)
        results = run_all_checks(c, [], {}, equity=10_000.0, log_dir=str(tmp_path))
        assert not all_passed(results)
        failed_names = {r.name for r in results if not r.passed}
        assert "last_run" in failed_names

    def test_returns_five_checks(self, tmp_path: Path) -> None:
        c = _client()
        results = run_all_checks(c, [], {}, equity=10_000.0, log_dir=str(tmp_path))
        assert len(results) == 5

    def test_all_passed_empty_list(self) -> None:
        assert all_passed([])

    def test_all_passed_false_if_any_fail(self) -> None:
        results = [
            CheckResult("a", True, "ok"),
            CheckResult("b", False, "fail"),
            CheckResult("c", True, "ok"),
        ]
        assert not all_passed(results)
