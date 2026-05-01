"""Daily strategy runner: fetch → signal → size → check → execute.

Champion strategy: multi-asset risk parity + SPY-200-SMA regime filter.
    Universe   : 14 assets — 11 S&P 500 sector ETFs + TLT + GLD + IEF
    Signal     : 12-1 cross-sectional momentum, top 3
    Regime     : SPY must be above 200-day SMA; else hold cash
    Sizing     : risk-parity (each active asset contributes equal risk)

In ``dry_run=True`` mode:
  - If Alpaca keys are available: full pipeline except order submission.
  - If keys are missing: prices fetched via yfinance, $100 k dummy account.

Every run appends a JSON log to ``results/live_log_YYYY-MM-DD.json``.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtest_engine.strategies.momentum import momentum_signals
from backtest_engine.strategies.multi_asset import UNIVERSE, risk_parity_weights
from backtest_engine.strategies.regime import apply_regime_filter, trend_filter

logger = logging.getLogger(__name__)

_LOG_DIR = Path("results")
_DUMMY_EQUITY = 100_000.0   # used in dry-run when no Alpaca keys are found


# ── Internal helpers ───────────────────────────────────────────────────────

def _fetch_prices_yfinance(days: int = 420) -> pd.DataFrame:
    """Fallback price fetch via yfinance (used in no-key dry-run mode)."""
    from backtest_engine.data.loader import load_prices

    end = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    # Generous start to cover momentum warm-up + SMA-200 + vol lookback
    from datetime import timedelta

    start_dt = datetime.now(tz=UTC) - timedelta(days=days * 2)
    start = start_dt.strftime("%Y-%m-%d")
    tickers = UNIVERSE + ["SPY"]
    logger.info("Fetching prices via yfinance (%d tickers, ~%d days)", len(tickers), days)
    return load_prices(tickers, start, end, cache=True)


def _build_client() -> Any:
    """Return AlpacaClient or None if keys are missing."""
    try:
        from backtest_engine.live.alpaca_client import AlpacaClient

        return AlpacaClient()
    except OSError:
        return None


def _save_log(log: dict[str, Any]) -> None:
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    log_path = _LOG_DIR / f"live_log_{date_str}.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    logger.info("Log saved → %s", log_path)


def _weights_to_dict(weights_series: pd.Series) -> dict[str, float]:
    return {
        str(k): round(float(v), 4)
        for k, v in weights_series.items()
        if abs(float(v)) > 1e-9
    }


# ── Public entry point ─────────────────────────────────────────────────────

def run_daily_strategy(dry_run: bool = False) -> dict[str, Any]:
    """Run the daily multi-asset risk-parity strategy.

    Steps
    -----
    1. Fetch prices (Alpaca or yfinance fallback).
    2. Apply SPY-200-SMA regime filter.
    3. Generate 12-1 momentum signals (top 3 of 14 assets).
    4. Compute risk-parity target weights.
    5. Read current Alpaca positions.
    6. Compute delta (target − current) and build order list.
    7. Run safety checks.
    8. Submit orders (skipped in dry-run or if safety checks fail).
    9. Log everything to ``results/live_log_YYYY-MM-DD.json``.
    10. Return summary dict.

    Parameters
    ----------
    dry_run:
        If True, log intended trades but submit nothing.
        If Alpaca keys are absent, automatically operates in dry-run
        fallback mode using yfinance prices and a dummy account.
    """
    log: dict[str, Any] = {
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "dry_run": dry_run,
        "status": "started",
    }

    try:
        # ── 1. Connect to Alpaca (or fall back) ────────────────────────────
        client = _build_client()
        no_key_fallback = client is None

        if no_key_fallback:
            if not dry_run:
                raise OSError(
                    "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set for live trading. "
                    "Use --dry-run to test without keys."
                )
            logger.warning(
                "Alpaca keys not found — dry-run fallback: "
                "prices via yfinance, dummy account $%s",
                f"{_DUMMY_EQUITY:,.0f}",
            )
            log["mode"] = "dry_run_no_keys"

        # ── 2. Fetch prices (≥ 400 trading days for SMA-200 + momentum) ───
        if client is not None:
            logger.info("Fetching prices from Alpaca (%d tickers)", len(UNIVERSE) + 1)
            prices_all = client.get_bars(UNIVERSE + ["SPY"], days=420)
        else:
            prices_all = _fetch_prices_yfinance(days=420)

        spy = prices_all["SPY"]
        prices_14 = prices_all[UNIVERSE]
        log["price_rows"] = len(prices_14)
        log["price_cols"] = list(prices_14.columns)

        # ── 3. Regime filter ────────────────────────────────────────────────
        regime = trend_filter(spy, sma_window=200)
        current_regime_on = bool(regime.iloc[-1])
        log["regime_risk_on"] = current_regime_on
        logger.info(
            "Regime: %s (SPY %s 200-SMA)",
            "RISK-ON" if current_regime_on else "RISK-OFF",
            "above" if current_regime_on else "below",
        )

        # ── 4. Momentum signals + regime filter ─────────────────────────────
        signals_raw = momentum_signals(
            prices_14, lookback_months=12, skip_months=1, top_n=3
        )
        signals = apply_regime_filter(signals_raw, regime)
        today_signals = signals.iloc[-1]
        active = [t for t in UNIVERSE if today_signals.get(t, 0.0) != 0.0]
        log["active_assets"] = active
        logger.info("Active assets: %s", active if active else "none (risk-off or warm-up)")

        # ── 5. Risk-parity weights ──────────────────────────────────────────
        rets = prices_14.pct_change()
        raw_weights = risk_parity_weights(signals, rets, lookback_days=60)
        target_w = raw_weights.iloc[-1]   # unshifted: today's signal → today's target
        target_weights = _weights_to_dict(target_w)
        log["target_weights"] = target_weights
        logger.info("Target weights: %s", {k: f"{v:.1%}" for k, v in target_weights.items()})

        # ── 6. Current positions ────────────────────────────────────────────
        if client is not None:
            account = client.get_account()
            equity = float(account.equity)
            positions = client.get_positions()
            current_weights: dict[str, float] = {
                str(p.symbol): float(p.market_value) / equity
                for p in positions
            }
            current_qty: dict[str, float] = {
                str(p.symbol): float(p.qty)
                for p in positions
            }
        else:
            equity = _DUMMY_EQUITY
            current_weights = {}
            current_qty = {}

        log["account_equity"] = round(equity, 2)
        log["current_weights"] = {k: round(v, 4) for k, v in current_weights.items()}
        logger.info(
            "Account equity: $%s  |  Open positions: %d", f"{equity:,.2f}", len(current_weights)
        )

        # ── 7. Build order list ─────────────────────────────────────────────
        orders: list[dict[str, Any]] = []
        for symbol in UNIVERSE:
            target = target_weights.get(symbol, 0.0)
            current = current_weights.get(symbol, 0.0)
            delta = target - current
            notional = abs(delta) * equity

            if notional < 10.0:
                continue  # skip noise trades

            price_val = float(prices_14[symbol].iloc[-1])
            qty = notional / price_val

            # Cap sell qty at the actual held quantity to avoid "insufficient qty"
            # errors caused by floating-point drift between computed and held shares.
            if delta < 0:
                qty = min(qty, current_qty.get(symbol, qty))

            orders.append(
                {
                    "symbol": symbol,
                    "side": "buy" if delta > 0 else "sell",
                    "qty": round(qty, 6),
                    "notional": round(notional, 2),
                    "delta_weight": round(delta, 4),
                    "price": round(price_val, 4),
                }
            )

        log["orders"] = orders
        if orders:
            for o in orders:
                logger.info(
                    "  %s  %-6s  %.4f shares  ($%.2f)",
                    o["side"].upper(),
                    o["symbol"],
                    o["qty"],
                    o["notional"],
                )
        else:
            logger.info("No orders needed — portfolio already at target")

        # ── 8. Safety checks ────────────────────────────────────────────────
        from backtest_engine.live.safety_checks import all_passed, run_all_checks

        # In no-key dry-run mode use a stub client that always passes.
        check_client = client if client is not None else _DryRunStubClient()
        checks = run_all_checks(
            client=check_client,
            orders=orders,
            target_weights=target_weights,
            equity=equity,
            log_dir=str(_LOG_DIR),
        )
        log["safety_checks"] = {
            c.name: ("PASS" if c.passed else f"FAIL: {c.message}") for c in checks
        }

        # ── 9. Submit or skip ───────────────────────────────────────────────
        if not all_passed(checks):
            failed = [c for c in checks if not c.passed]
            log["status"] = "aborted"
            log["abort_reason"] = [str(c) for c in failed]
            logger.error(
                "Safety check(s) failed — no orders submitted:\n%s",
                "\n".join(f"  {c}" for c in failed),
            )

        elif dry_run or no_key_fallback:
            log["status"] = "dry_run"
            logger.info("DRY RUN — %d order(s) logged, none submitted", len(orders))

        else:
            submitted: list[dict[str, str]] = []
            for order in orders:
                result = client.submit_order(
                    order["symbol"], order["qty"], order["side"]
                )
                order_id = str(result.id)
                filled_at = (
                    result.filled_at.isoformat()
                    if result.filled_at is not None
                    else datetime.now(tz=UTC).isoformat()
                )
                submitted.append({
                    "symbol": order["symbol"],
                    "order_id": order_id,
                    "filled_at": filled_at,
                })
                logger.info(
                    "Submitted %s %s %.4f shares  → order_id=%s  filled_at=%s",
                    order["side"],
                    order["symbol"],
                    order["qty"],
                    order_id,
                    filled_at,
                )
            log["submitted_orders"] = submitted
            log["status"] = "executed"

    except Exception as exc:
        log["status"] = "error"
        log["error"] = str(exc)
        logger.exception("run_daily_strategy failed: %s", exc)
        raise

    finally:
        _save_log(log)

    return log


# ── Stub client for no-key dry-run mode ────────────────────────────────────

class _DryRunStubClient:
    """Minimal client stub that satisfies _ClientProtocol with safe defaults.

    Used only when Alpaca keys are absent and ``dry_run=True``.
    Safety checks that would normally call the API are marked as skipped.
    """

    is_paper: bool = True

    def is_market_open(self) -> bool:
        # Always return True in dry-run — the market-open check is a live guard.
        return True
