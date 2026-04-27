#!/usr/bin/env python3
"""Daily paper-trading entry point.

Usage
-----
    # Full dry-run (no orders submitted, works without Alpaca keys):
    uv run python scripts/run_live.py --dry-run

    # Live paper-trading run (requires ALPACA_API_KEY + ALPACA_SECRET_KEY):
    uv run python scripts/run_live.py

    # Or with a .env file in the repo root:
    uv run python scripts/run_live.py --dry-run   # dotenv loaded automatically

Typical cron schedule (weekdays at 15:45 ET, 15 min before close):
    45 19 * * 1-5  cd /path/to/backtest-engine && uv run python scripts/run_live.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running as `python scripts/run_live.py` from any working directory.
_repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_repo_root))

# Load .env before importing project modules (dotenv is a no-op if .env absent).
from dotenv import load_dotenv  # noqa: E402

load_dotenv(_repo_root / ".env")

from backtest_engine.live.strategy_runner import run_daily_strategy  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_live")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Daily multi-asset risk-parity paper-trading runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/run_live.py --dry-run   # test without Alpaca keys\n"
            "  python scripts/run_live.py             # live paper-trading run\n"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Log intended trades without submitting orders. "
            "Works without Alpaca keys (uses yfinance + $100k dummy account)."
        ),
    )
    args = parser.parse_args()

    if args.dry_run:
        logger.info("=" * 60)
        logger.info("DRY RUN MODE — no orders will be submitted")
        logger.info("=" * 60)
    else:
        logger.info("=" * 60)
        logger.info("LIVE PAPER-TRADING RUN")
        logger.info("=" * 60)

    try:
        summary = run_daily_strategy(dry_run=args.dry_run)
    except OSError as exc:
        logger.error("%s", exc)
        logger.error(
            "Hint: copy .env.example → .env, fill in your paper-trading keys, "
            "then retry.  Or use --dry-run to test without keys."
        )
        return 1
    except Exception:
        logger.exception("Fatal error in run_live — check results/live_log_*.json")
        return 1

    status = summary.get("status", "unknown")
    n_orders = len(summary.get("orders", []))
    equity = summary.get("account_equity", 0.0)
    regime = "RISK-ON" if summary.get("regime_risk_on") else "RISK-OFF"
    active = summary.get("active_assets", [])

    logger.info("-" * 60)
    logger.info("Status  : %s", status.upper())
    logger.info("Regime  : %s", regime)
    logger.info("Active  : %s", active if active else "none")
    logger.info("Orders  : %d", n_orders)
    logger.info("Equity  : $%s", f"{equity:,.2f}")
    logger.info("-" * 60)

    return 0 if status != "error" else 1


if __name__ == "__main__":
    sys.exit(main())
