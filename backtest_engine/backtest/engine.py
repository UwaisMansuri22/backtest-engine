"""Vectorized daily-bar backtest engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """All outputs from a single backtest run.

    ``stats`` is intentionally empty here; the metrics module fills it.
    """

    equity_curve: pd.Series    # portfolio value at close of each day
    returns: pd.Series         # daily simple returns
    positions: pd.DataFrame    # weight per (day, ticker) after normalization
    trades: pd.DataFrame       # one row per position-change event
    stats: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _equal_weight(signals: pd.DataFrame) -> pd.DataFrame:
    """Scale {-1, 0, 1} signals so the sum of absolute weights = 1 per row.

    With N active signals, each gets 1/N of capital.  This keeps gross
    exposure ≤ 100% regardless of how many tickers fire on a given day.
    """
    n_active = signals.abs().sum(axis=1).replace(0.0, np.nan)
    return signals.div(n_active, axis=0).fillna(0.0)


def _build_trade_log(
    weights: pd.DataFrame,
    prices: pd.DataFrame,
    equity: pd.Series,
    total_cost_bps: float,
) -> pd.DataFrame:
    """Return a fully-vectorized record of every position-change event."""
    prev_w = weights.shift(1).fillna(0.0)
    delta = weights - prev_w

    # Stack to long (date, ticker) Series; discard unchanged positions.
    delta_long = delta.stack()
    delta_long = delta_long[delta_long.abs() > 1e-9].rename("weight_change")

    if delta_long.empty:
        return pd.DataFrame(
            columns=["ticker", "weight_change", "price", "notional_usd", "cost_usd"]
        )

    idx = delta_long.index          # MultiIndex (date, ticker)
    dates = idx.get_level_values(0)

    # Use prior-day closing equity as the notional base so that the cost
    # recorded here matches the cost subtracted in the main return calc.
    prev_equity = equity.shift(1).fillna(float(equity.iloc[0]))
    notional = delta_long.abs().to_numpy() * prev_equity.reindex(dates).to_numpy()
    price_vals = prices.stack().reindex(idx).to_numpy()

    return pd.DataFrame(
        {
            "ticker": idx.get_level_values(1),
            "weight_change": delta_long.to_numpy(),
            "price": price_vals,
            "notional_usd": notional,
            "cost_usd": notional * total_cost_bps / 10_000,
        },
        index=dates,
    ).rename_axis("date")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_backtest(
    prices: pd.DataFrame,
    signals: pd.DataFrame,
    initial_capital: float = 10_000.0,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
) -> BacktestResult:
    """Run a vectorized daily-bar backtest.

    Parameters
    ----------
    prices:
        Adjusted close prices — rows = dates, cols = tickers.
        Must be free of NaN (use ``load_prices`` from the data module).
    signals:
        Trade signals, same shape as ``prices``; values in {-1, 0, 1}.
        ``+1`` = long, ``-1`` = short, ``0`` = flat.
    initial_capital:
        Starting portfolio value in dollars.
    transaction_cost_bps:
        One-way brokerage cost in basis points (1 bp = 0.01 %).
    slippage_bps:
        One-way slippage cost in basis points.

    Returns
    -------
    BacktestResult
        Equity curve, returns, positions, trade log, and an empty ``stats``
        dict ready for the metrics module to populate.
    """
    if prices.shape != signals.shape:
        raise ValueError(
            f"prices {prices.shape} and signals {signals.shape} must have the same shape"
        )
    if not prices.index.equals(signals.index):
        raise ValueError("prices and signals must share the same DatetimeIndex")

    total_bps = transaction_cost_bps + slippage_bps

    # ------------------------------------------------------------------
    # 1. LOOKAHEAD-BIAS PREVENTION
    # A signal computed on day T is based on prices known at EOD T.
    # It can only be executed at the earliest at the open of day T+1.
    # shift(1) enforces this one-bar delay: the strategy never "buys
    # at yesterday's close using today's information."
    # fillna(0) means we start the period in cash with no open position.
    # ------------------------------------------------------------------
    executed = signals.shift(1).fillna(0.0)

    # ------------------------------------------------------------------
    # 2. EQUAL-WEIGHT NORMALIZATION
    # Each active signal receives 1/N of capital (N active on that day).
    # Without this, adding more tickers to a signal silently levers up
    # the portfolio (e.g. 5 signals of weight 1 each = 5× leverage).
    # ------------------------------------------------------------------
    weights = _equal_weight(executed)

    # ------------------------------------------------------------------
    # 3. PER-ASSET LOG RETURNS
    # log(P_t / P_{t-1}) is time-additive: the sum of daily log returns
    # equals the total log return over the period, making compounding
    # exact when we use exp(cumsum(...)) for the equity curve.
    # Day 0 is NaN (no prior bar).  With weights = 0 on day 0 the product
    # 0 × NaN = NaN collapses to 0 under sum(skipna=True).
    # ------------------------------------------------------------------
    log_rets: pd.DataFrame = np.log(prices / prices.shift(1))

    # ------------------------------------------------------------------
    # 4. GROSS PORTFOLIO LOG RETURN
    # r_portfolio ≈ Σ w_i · r_i  (exact for simple returns; for log
    # returns this is an approximation accurate to <0.01 % on daily bars).
    # ------------------------------------------------------------------
    gross: pd.Series = (weights * log_rets).sum(axis=1, skipna=True)

    # ------------------------------------------------------------------
    # 5. TRANSACTION COSTS — CHARGED ON POSITION CHANGES ONLY
    # A stock held unchanged for 100 days incurs no additional commission.
    # Cost accumulates only when the weight vector changes.
    # |Δweight| ∈ [0, 2]: 0 = no trade; 1 = full entry/exit; 2 = full flip.
    # We express cost as a fraction of the portfolio so it scales correctly.
    # ------------------------------------------------------------------
    prev_weights = weights.shift(1).fillna(0.0)
    delta_w: pd.Series = (weights - prev_weights).abs().sum(axis=1)
    cost: pd.Series = delta_w * total_bps / 10_000

    # ------------------------------------------------------------------
    # 6. NET LOG RETURN → EQUITY CURVE
    # exp(cumsum(log_returns)) is mathematically identical to
    # cumprod(1 + simple_returns) but avoids floating-point drift from
    # multiplying thousands of near-unity factors over many years.
    # ------------------------------------------------------------------
    net: pd.Series = gross - cost
    equity_curve = pd.Series(
        initial_capital * np.exp(net.cumsum().to_numpy()),
        index=prices.index,
        name="equity",
    )

    # Day 0: weights = 0, log_rets = NaN → net = 0 → equity = initial_capital. ✓
    # No manual override needed; the math handles it.

    returns = equity_curve.pct_change().fillna(0.0)
    returns.name = "returns"

    trades = _build_trade_log(weights, prices, equity_curve, total_bps)

    logger.info(
        "Backtest complete | %d bars | %d trade events | final equity $%.2f",
        len(prices),
        len(trades),
        float(equity_curve.iloc[-1]),
    )

    return BacktestResult(
        equity_curve=equity_curve,
        returns=returns,
        positions=weights,
        trades=trades,
        stats={},
    )
