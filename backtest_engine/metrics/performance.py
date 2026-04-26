"""Performance metrics for backtested strategy returns."""

from __future__ import annotations

import math
from collections.abc import Callable

import pandas as pd

_TRADING_DAYS: int = 252


def _validate(returns: pd.Series) -> None:
    if len(returns) == 0:
        raise ValueError("returns must not be empty")


# ---------------------------------------------------------------------------
# Individual metrics
# ---------------------------------------------------------------------------

def total_return(returns: pd.Series) -> float:
    """Compound total return over the full period.

    What it tells you: total P&L as a fraction of initial capital (0.25 = +25%).
    Good values: positive; context-dependent on the length of the period.
    """
    _validate(returns)
    return float((1.0 + returns).prod() - 1.0)


def cagr(returns: pd.Series) -> float:
    """Compound Annual Growth Rate.

    What it tells you: annualised return so strategies of different lengths
    can be compared on equal footing.
    Good values: >8–10 % for equity strategies is generally considered strong.
    """
    _validate(returns)
    n = len(returns)
    total = float((1.0 + returns).prod())
    return float(total ** (_TRADING_DAYS / n) - 1.0)


def annual_volatility(returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns.

    What it tells you: how widely the strategy's returns fluctuate day-to-day,
    scaled to a yearly figure.
    Good values: lower is better; most equity strategies run 10–20 %.
    """
    _validate(returns)
    return float(returns.std(ddof=1) * math.sqrt(_TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio: excess return per unit of total volatility.

    What it tells you: how efficiently the strategy converts risk into return
    — higher is better.
    Good values: >1 is acceptable, >2 is strong, >3 is exceptional.
    """
    _validate(returns)
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / _TRADING_DAYS) - 1.0
    excess = returns - daily_rf
    std = float(excess.std(ddof=1))
    # Floating-point variance algorithms accumulate rounding error, so identical
    # returns can produce std ~1e-18 rather than exactly 0.  Use a threshold that
    # is far below any plausible real daily return std (≥ ~1e-5) but above FP noise.
    if std < 1e-12:
        return math.nan
    return float(excess.mean() / std * math.sqrt(_TRADING_DAYS))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sortino ratio: excess return per unit of downside volatility.

    What it tells you: like Sharpe but only penalises negative moves, so
    large upswings are not counted against the strategy.
    Good values: >2 is strong; will always be ≥ the Sharpe ratio.

    Formula note: downside deviation = sqrt(252 × mean(min(excess, 0)²)).
    This is the semi-variance approach — all days enter the denominator,
    but only below-target days contribute variance.  Days above the target
    are clipped to zero, which gives a stable, widely-used definition.
    """
    _validate(returns)
    daily_rf = (1.0 + risk_free_rate) ** (1.0 / _TRADING_DAYS) - 1.0
    excess = returns - daily_rf
    # Annualise using geometric compounding (correct for multi-period returns)
    annual_excess = float(
        (1.0 + excess).prod() ** (_TRADING_DAYS / len(excess)) - 1.0
    )
    downside_sq_mean = float((excess.clip(upper=0.0) ** 2).mean())
    if downside_sq_mean == 0.0:
        return math.inf
    downside_dev = math.sqrt(_TRADING_DAYS * downside_sq_mean)
    return float(annual_excess / downside_dev)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough percentage decline in portfolio value.

    What it tells you: the worst loss an investor would have suffered had
    they bought at any historical peak.
    Good values: closer to 0; −20 % to −30 % is typical for equity strategies.
    """
    _validate(returns)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    return float(drawdown.min())


def calmar_ratio(returns: pd.Series) -> float:
    """CAGR divided by the absolute maximum drawdown.

    What it tells you: annualised return earned per unit of peak-to-trough
    pain — a concise risk-adjusted summary.
    Good values: >0.5 is acceptable, >1 is strong.
    """
    _validate(returns)
    mdd = max_drawdown(returns)
    if mdd == 0.0:
        return math.inf
    return float(cagr(returns) / abs(mdd))


def win_rate(returns: pd.Series) -> float:
    """Fraction of trading days with a strictly positive return.

    What it tells you: how often the strategy makes money on a given day.
    Good values: >50 % is above random; trend-following strategies often win
    <50 % of days but profit from large, infrequent winners.
    """
    _validate(returns)
    return float((returns > 0.0).sum() / len(returns))


def profit_factor(returns: pd.Series) -> float:
    """Ratio of total gains to total losses (gross profit ÷ gross loss).

    What it tells you: for every $1 lost, how many dollars are won in aggregate.
    Good values: >1.0 is profitable, >1.5 is strong, >2.0 is exceptional.
    """
    _validate(returns)
    gains = float(returns[returns > 0.0].sum())
    losses = float(returns[returns < 0.0].sum())
    if losses == 0.0:
        return math.inf
    if gains == 0.0:
        return 0.0
    return float(gains / abs(losses))


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

# Wrappers so the metric table can hold Callable[[pd.Series], float] uniformly.
def _sharpe(r: pd.Series) -> float:
    return sharpe_ratio(r)


def _sortino(r: pd.Series) -> float:
    return sortino_ratio(r)


_METRIC_ROWS: list[tuple[str, Callable[[pd.Series], float], str]] = [
    ("Total Return",       total_return,       "+.2%"),
    ("CAGR",               cagr,               "+.2%"),
    ("Annual Volatility",  annual_volatility,  ".2%"),
    ("Sharpe Ratio",       _sharpe,            ".3f"),
    ("Sortino Ratio",      _sortino,           ".3f"),
    ("Max Drawdown",       max_drawdown,       "+.2%"),
    ("Calmar Ratio",       calmar_ratio,       ".3f"),
    ("Win Rate",           win_rate,           ".2%"),
    ("Profit Factor",      profit_factor,      ".3f"),
]

_NAME_W = 22
_VAL_W = 12


def _fmt(value: float, spec: str) -> str:
    """Format a metric value; render inf/nan gracefully."""
    if math.isnan(value):
        return "N/A"
    if math.isinf(value):
        return "∞" if value > 0 else "-∞"
    return format(value, spec)


def performance_report(returns: pd.Series, name: str = "") -> None:
    """Print a formatted single-strategy performance summary.

    Parameters
    ----------
    returns:
        Daily simple returns Series produced by the backtest engine.
    name:
        Strategy label displayed in the header.
    """
    header = f"Performance: {name}" if name else "Performance Report"
    bar = "─" * max(len(header), _NAME_W + _VAL_W + 2)
    print(f"\n{header}")
    print(bar)
    for label, fn, spec in _METRIC_ROWS:
        val = _fmt(fn(returns), spec)
        print(f"  {label:<{_NAME_W}}{val:>{_VAL_W}}")
    print()


def compare_strategies(returns_dict: dict[str, pd.Series]) -> None:
    """Print a side-by-side performance comparison of multiple strategies.

    Parameters
    ----------
    returns_dict:
        Mapping of strategy name → daily returns Series.
    """
    names = list(returns_dict.keys())
    col_w = 14
    total_w = _NAME_W + col_w * len(names) + 2
    bar = "─" * total_w

    print("\nStrategy Comparison")
    print(bar)
    print(f"  {'Metric':<{_NAME_W}}" + "".join(f"{n:>{col_w}}" for n in names))
    print(bar)
    for label, fn, spec in _METRIC_ROWS:
        row = f"  {label:<{_NAME_W}}"
        for series in returns_dict.values():
            row += f"{_fmt(fn(series), spec):>{col_w}}"
        print(row)
    print()
