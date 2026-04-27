"""Tests for backtest_engine.metrics.performance.

Each test constructs a returns Series with a known expected answer and asserts
the metric matches within floating-point tolerance.
"""

import math

import numpy as np
import pandas as pd
import pytest

from backtest_engine.metrics.performance import (
    annual_volatility,
    cagr,
    calmar_ratio,
    compare_strategies,
    max_drawdown,
    performance_report,
    profit_factor,
    sharpe_ratio,
    sortino_ratio,
    total_return,
    win_rate,
)

# ---------------------------------------------------------------------------
# total_return
# ---------------------------------------------------------------------------

def test_total_return_two_periods() -> None:
    """(1 + 0.1) × (1 + 0.1) − 1 = 0.21."""
    returns = pd.Series([0.1, 0.1])
    assert total_return(returns) == pytest.approx(0.21)


def test_total_return_recovers_to_zero() -> None:
    """A 50 % gain followed by a 33.3 % loss returns to exactly the start."""
    returns = pd.Series([0.5, -1.0 / 3.0])
    assert total_return(returns) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# cagr
# ---------------------------------------------------------------------------

def test_cagr_exactly_one_year_equals_total_return() -> None:
    """Over exactly 252 bars, CAGR = total return (no annualisation needed)."""
    r = 0.001
    returns = pd.Series([r] * 252)
    expected = (1 + r) ** 252 - 1
    assert cagr(returns) == pytest.approx(expected, rel=1e-9)


def test_cagr_half_year_annualises_correctly() -> None:
    """126 bars → CAGR squares the half-year return to project a full year."""
    r = 0.001
    returns = pd.Series([r] * 126)
    half_year = (1 + r) ** 126 - 1
    expected = (1 + half_year) ** 2 - 1
    assert cagr(returns) == pytest.approx(expected, rel=1e-9)


def test_cagr_zero_returns() -> None:
    returns = pd.Series([0.0] * 252)
    assert cagr(returns) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# annual_volatility
# ---------------------------------------------------------------------------

def test_annual_volatility_formula() -> None:
    """Function must be std(ddof=1) × √252; value verified against numpy."""
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.03, -0.03] * 10)
    expected = float(returns.std(ddof=1) * np.sqrt(252))
    assert annual_volatility(returns) == pytest.approx(expected, rel=1e-9)


def test_annual_volatility_constant_returns_is_zero() -> None:
    returns = pd.Series([0.005] * 100)
    assert annual_volatility(returns) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# sharpe_ratio
# ---------------------------------------------------------------------------

def test_sharpe_ratio_formula_zero_rf() -> None:
    """Sharpe = mean/std × √252 when risk_free_rate = 0."""
    returns = pd.Series([0.002, 0.000] * 63)  # 126 alternating returns
    expected = float(returns.mean() / returns.std(ddof=1) * np.sqrt(252))
    assert sharpe_ratio(returns) == pytest.approx(expected, rel=1e-6)


def test_sharpe_ratio_constant_returns_is_nan() -> None:
    """Zero standard deviation → Sharpe is undefined (not ∞ — it's a 0/0 form)."""
    returns = pd.Series([0.001] * 100)
    assert math.isnan(sharpe_ratio(returns))


# ---------------------------------------------------------------------------
# sortino_ratio
# ---------------------------------------------------------------------------

def test_sortino_ratio_no_downside_is_inf() -> None:
    """Never-negative returns → downside deviation = 0 → Sortino = +∞."""
    returns = pd.Series([0.001, 0.002, 0.003])
    assert sortino_ratio(returns) == math.inf


def test_sortino_ratio_formula() -> None:
    """Sortino matches manual computation of the semi-variance formula."""
    returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
    annual_excess = (1.0 + returns).prod() ** (252 / len(returns)) - 1.0
    dd = float(np.sqrt(252 * float((returns.clip(upper=0.0) ** 2).mean())))
    expected = annual_excess / dd
    assert sortino_ratio(returns) == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# max_drawdown
# ---------------------------------------------------------------------------

def test_max_drawdown_known_peak_trough() -> None:
    """Equity 1.0 → 1.2 (peak) → 0.9 (trough): drawdown = (0.9/1.2) − 1 = −25 %."""
    # +20 % takes equity to 1.2; −25 % takes it to 0.9.
    returns = pd.Series([0.20, -0.25])
    assert max_drawdown(returns) == pytest.approx(-0.25, rel=1e-9)


def test_max_drawdown_monotonically_rising_is_zero() -> None:
    """A portfolio that only ever rises has no drawdown."""
    returns = pd.Series([0.01, 0.02, 0.03, 0.005])
    assert max_drawdown(returns) == pytest.approx(0.0, abs=1e-12)


def test_max_drawdown_is_negative() -> None:
    """Drawdown must always be ≤ 0."""
    returns = pd.Series([0.01, -0.05, 0.02, -0.03, 0.04])
    assert max_drawdown(returns) <= 0.0


# ---------------------------------------------------------------------------
# calmar_ratio
# ---------------------------------------------------------------------------

def test_calmar_ratio_formula() -> None:
    """Calmar = CAGR / |max_drawdown|, verified by composing the two functions."""
    returns = pd.Series([0.002, -0.05, 0.003] * 50)
    expected = cagr(returns) / abs(max_drawdown(returns))
    assert calmar_ratio(returns) == pytest.approx(expected, rel=1e-9)


def test_calmar_ratio_no_drawdown_is_inf() -> None:
    returns = pd.Series([0.01, 0.02, 0.005])
    assert calmar_ratio(returns) == math.inf


# ---------------------------------------------------------------------------
# win_rate
# ---------------------------------------------------------------------------

def test_win_rate_known_value() -> None:
    """2 positive days out of 5 → win rate = 0.4.  Zero counts as a loss."""
    returns = pd.Series([0.01, -0.02, 0.03, 0.0, -0.01])
    assert win_rate(returns) == pytest.approx(0.4)


def test_win_rate_all_positive() -> None:
    returns = pd.Series([0.01, 0.02, 0.03])
    assert win_rate(returns) == pytest.approx(1.0)


def test_win_rate_all_negative() -> None:
    returns = pd.Series([-0.01, -0.02])
    assert win_rate(returns) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# profit_factor
# ---------------------------------------------------------------------------

def test_profit_factor_known_value() -> None:
    """gains = 0.04, losses = 0.06 → PF = 0.04 / 0.06 ≈ 0.6667."""
    returns = pd.Series([0.03, -0.02, 0.01, -0.04])
    expected = 0.04 / 0.06
    assert profit_factor(returns) == pytest.approx(expected, rel=1e-9)


def test_profit_factor_no_losses_is_inf() -> None:
    returns = pd.Series([0.01, 0.02, 0.03])
    assert profit_factor(returns) == math.inf


def test_profit_factor_no_gains_is_zero() -> None:
    returns = pd.Series([-0.01, -0.02, -0.03])
    assert profit_factor(returns) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Reporting functions (smoke tests — check they run and produce output)
# ---------------------------------------------------------------------------

def test_performance_report_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    returns = pd.Series([0.001, -0.002, 0.003, -0.001, 0.002] * 50)
    performance_report(returns, name="Test Strategy")
    captured = capsys.readouterr()
    assert "Test Strategy" in captured.out
    assert "Sharpe" in captured.out
    assert "Max Drawdown" in captured.out


def test_compare_strategies_runs_without_error(capsys: pytest.CaptureFixture[str]) -> None:
    r_a = pd.Series([0.001, -0.001] * 63)
    r_b = pd.Series([0.002, -0.002] * 63)
    compare_strategies({"Strategy A": r_a, "Strategy B": r_b})
    captured = capsys.readouterr()
    assert "Strategy A" in captured.out
    assert "Strategy B" in captured.out
