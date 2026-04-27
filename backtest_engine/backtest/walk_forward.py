"""Walk-forward analysis: rolling in-sample optimisation + out-of-sample testing."""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from backtest_engine.backtest.engine import run_backtest
from backtest_engine.metrics.performance import (
    cagr,
    calmar_ratio,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)

logger = logging.getLogger(__name__)

_METRIC_FNS: dict[str, Callable[[pd.Series], float]] = {
    "sharpe_ratio": sharpe_ratio,
    "cagr": cagr,
    "calmar_ratio": calmar_ratio,
    "sortino_ratio": sortino_ratio,
    "max_drawdown": max_drawdown,
}


@dataclass
class WindowResult:
    """Results for a single train/test window."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    is_metric: float        # best IS metric (cherry-picked by grid search)
    oos_metric: float       # same metric re-evaluated on the test period
    oos_returns: pd.Series  # daily simple returns for the test period


@dataclass
class WalkForwardResult:
    """Aggregated results from a complete walk-forward run."""

    oos_returns: pd.Series       # concatenated OOS daily returns (honest)
    windows: list[WindowResult]  # per-window detail
    metric: str                  # metric used for IS optimisation
    avg_is_metric: float         # average of best IS metrics (inflated by selection)
    oos_metric: float            # metric on full concatenated OOS returns (honest)


def _safe_eval(
    fn: Callable[[pd.Series], float],
    returns: pd.Series,
    min_bars: int = 20,
) -> float:
    """Evaluate metric; return -inf on error, NaN result, or too few bars."""
    if len(returns) < min_bars:
        return -math.inf
    try:
        v = fn(returns)
        return v if math.isfinite(v) else -math.inf
    except Exception:
        return -math.inf


def walk_forward_test(
    prices: pd.DataFrame,
    strategy_fn: Callable[..., pd.DataFrame],
    param_grid: dict[str, list[Any]],
    train_years: int = 3,
    test_years: int = 1,
    metric: str = "sharpe_ratio",
    initial_capital: float = 10_000.0,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> WalkForwardResult:
    """Walk-forward validation: rolling IS optimisation followed by OOS testing.

    Algorithm
    ---------
    For window starting at cursor t:

    1. **Train** on [t, t + train_years).  Grid-search all param combinations;
       pick the combo with the best ``metric`` value.
    2. **Test** on [t + train_years, t + train_years + test_years).  Run the
       strategy with the winning params — no peeking at these dates during step 1.
    3. Advance cursor by ``test_years`` and repeat until no full window fits.

    Concatenate all test-period returns for an honest out-of-sample picture.

    Signal generation note
    ----------------------
    Signals are generated over all price history available up to the window
    boundary (not just the window itself), so that strategies with a warm-up
    period (e.g. 12-month momentum lookback) are not penalised with artificial
    all-zero periods inside a later training window.  Because all signal
    functions look only backward in time, this introduces no lookahead bias.

    Parameters
    ----------
    prices:
        Adjusted close prices — rows = dates, cols = tickers.
    strategy_fn:
        ``(prices_slice, **params) -> signals_DataFrame``.
        Example: ``lambda p, **kw: momentum_signals(p, skip_months=1, **kw)``.
    param_grid:
        Dict mapping parameter names to candidate value lists.
        Every combination (Cartesian product) is evaluated.
        Example: ``{"lookback_months": [6, 9, 12], "top_n": [2, 3, 5]}``.
    train_years:
        Length of each IS window in calendar years.
    test_years:
        Length of each OOS window in calendar years; also the step size.
    metric:
        One of ``"sharpe_ratio"``, ``"cagr"``, ``"calmar_ratio"``,
        ``"sortino_ratio"``, ``"max_drawdown"``.  Higher is always better
        (``max_drawdown`` returns negative values; larger = shallower drawdown).
    initial_capital:
        Starting capital passed to each ``run_backtest`` call.
    transaction_cost_bps:
        One-way brokerage cost in basis points.
    slippage_bps:
        One-way slippage cost in basis points.

    Returns
    -------
    WalkForwardResult
        Concatenated OOS returns, per-window detail, average IS metric,
        and overall OOS metric.

    Raises
    ------
    ValueError
        If ``metric`` is not recognised, or no complete window fits the data.
    """
    if metric not in _METRIC_FNS:
        raise ValueError(
            f"metric must be one of {list(_METRIC_FNS)}, got '{metric}'"
        )
    metric_fn = _METRIC_FNS[metric]

    param_names = list(param_grid.keys())
    param_combos: list[dict[str, Any]] = [
        dict(zip(param_names, combo, strict=True))
        for combo in itertools.product(*param_grid.values())
    ]
    n_combos = len(param_combos)

    dates = prices.index
    windows: list[WindowResult] = []
    cursor = dates[0]
    window_num = 0

    while True:
        train_end = cursor + pd.DateOffset(years=train_years)
        test_end  = train_end + pd.DateOffset(years=test_years)

        if test_end > dates[-1] + pd.DateOffset(days=1):
            break

        train_mask = (dates >= cursor) & (dates < train_end)
        test_mask  = (dates >= train_end) & (dates < test_end)
        train_prices = prices.loc[train_mask]
        test_prices  = prices.loc[test_mask]

        if len(train_prices) < 30 or len(test_prices) < 5:
            break

        window_num += 1
        logger.info(
            "Window %d | train %s→%s (%d bars) | test %s→%s (%d bars) | "
            "%d param combos",
            window_num,
            train_prices.index[0].date(), train_prices.index[-1].date(),
            len(train_prices),
            test_prices.index[0].date(), test_prices.index[-1].date(),
            len(test_prices),
            n_combos,
        )

        # ── 1. In-sample grid search ───────────────────────────────────────────
        history_train = prices.loc[prices.index < train_end]
        best_is: float = -math.inf
        best_params: dict[str, Any] = param_combos[0]

        for params in param_combos:
            try:
                sigs = strategy_fn(history_train, **params)
                train_sigs = sigs.reindex(train_prices.index).fillna(0.0)
                res = run_backtest(
                    train_prices, train_sigs,
                    initial_capital=initial_capital,
                    transaction_cost_bps=transaction_cost_bps,
                    slippage_bps=slippage_bps,
                )
                val = _safe_eval(metric_fn, res.returns)
            except Exception:
                val = -math.inf

            if val > best_is:
                best_is = val
                best_params = params

        logger.info(
            "  Best IS params: %s  IS %s=%.3f", best_params, metric, best_is
        )

        # ── 2. Out-of-sample test with winning params ──────────────────────────
        # Signals are generated over history up to test_end so the momentum
        # lookback has full context at the start of the test period.
        try:
            history_test = prices.loc[prices.index < test_end]
            oos_sigs = strategy_fn(history_test, **best_params)
            oos_sigs = oos_sigs.reindex(test_prices.index).fillna(0.0)
            oos_res = run_backtest(
                test_prices, oos_sigs,
                initial_capital=initial_capital,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
            )
            oos_rets = oos_res.returns
            oos_val = _safe_eval(metric_fn, oos_rets)
        except Exception:
            oos_rets = pd.Series(dtype=float, name="returns")
            oos_val = math.nan

        logger.info("  OOS %s=%.3f", metric, oos_val)

        windows.append(WindowResult(
            train_start=cursor,
            train_end=train_end,
            test_start=train_end,
            test_end=test_end,
            best_params=best_params,
            is_metric=best_is,
            oos_metric=oos_val,
            oos_returns=oos_rets,
        ))

        cursor += pd.DateOffset(years=test_years)

    if not windows:
        span_years = (dates[-1] - dates[0]).days / 365
        raise ValueError(
            f"No complete walk-forward windows fit in the price history. "
            f"Data spans {dates[0].date()}–{dates[-1].date()} "
            f"({span_years:.1f} yr); need ≥ {train_years + test_years} yr."
        )

    # ── 3. Aggregate ───────────────────────────────────────────────────────────
    valid_oos = [w.oos_returns for w in windows if not w.oos_returns.empty]
    oos_all = pd.concat(valid_oos)
    oos_all = oos_all[~oos_all.index.duplicated(keep="first")].sort_index()
    oos_all.name = "oos_returns"

    valid_is_vals = [w.is_metric for w in windows if math.isfinite(w.is_metric)]
    avg_is = float(sum(valid_is_vals) / len(valid_is_vals)) if valid_is_vals else math.nan
    overall_oos = _safe_eval(metric_fn, oos_all)

    return WalkForwardResult(
        oos_returns=oos_all,
        windows=windows,
        metric=metric,
        avg_is_metric=avg_is,
        oos_metric=overall_oos,
    )
