"""Pre-flight safety checks that must ALL pass before any order is submitted.

A single failure causes the strategy runner to log loudly and exit
without trading.  Checks are intentionally conservative — the cost of
a missed trade is far lower than the cost of a runaway bug submitting
outsized orders to a live account.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


# ── Return type ────────────────────────────────────────────────────────────

@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.message}"


# ── Client interface (satisfied by AlpacaClient and test mocks) ────────────

class _ClientProtocol(Protocol):
    is_paper: bool

    def is_market_open(self) -> bool: ...


# ── Individual checks ──────────────────────────────────────────────────────

def check_paper_account(client: _ClientProtocol) -> CheckResult:
    """Verify the client is configured for paper trading, not live."""
    if not client.is_paper:
        return CheckResult(
            name="paper_account",
            passed=False,
            message="DANGER: client is NOT configured for paper trading — aborting",
        )
    return CheckResult(
        name="paper_account",
        passed=True,
        message="Paper trading confirmed",
    )


def check_order_notional(
    orders: list[dict[str, Any]],
    equity: float,
    max_fraction: float = 1.10,
) -> CheckResult:
    """Total order notional must not exceed ``max_fraction × equity``.

    Guards against sizing bugs that would over-allocate the account.
    Default cap is 110 % of equity — allows for small rounding overflows
    but catches anything obviously broken.
    """
    total = sum(float(o.get("notional", 0.0)) for o in orders)
    limit = equity * max_fraction
    if total > limit:
        return CheckResult(
            name="order_notional",
            passed=False,
            message=(
                f"Total notional ${total:,.2f} exceeds "
                f"{max_fraction * 100:.0f}% limit (${limit:,.2f}) — aborting"
            ),
        )
    return CheckResult(
        name="order_notional",
        passed=True,
        message=f"Total notional ${total:,.2f} within ${limit:,.2f} limit",
    )


def check_position_concentration(
    target_weights: dict[str, float],
    max_weight: float = 0.40,
) -> CheckResult:
    """No single target position may exceed ``max_weight`` of equity.

    A 3-asset risk-parity portfolio with equal vols gives ≈ 33 % each.
    The 40 % cap catches the case where one position dominates due to a
    low-vol outlier inflating its inverse-vol weight.
    """
    if not target_weights:
        return CheckResult(
            name="concentration",
            passed=True,
            message="No positions — nothing to check",
        )
    max_pos = max(abs(w) for w in target_weights.values())
    if max_pos > max_weight:
        worst = max(target_weights, key=lambda k: abs(target_weights[k]))
        return CheckResult(
            name="concentration",
            passed=False,
            message=(
                f"{worst} weight {max_pos:.1%} exceeds "
                f"{max_weight:.0%} single-position limit — aborting"
            ),
        )
    return CheckResult(
        name="concentration",
        passed=True,
        message=f"Max position {max_pos:.1%} within {max_weight:.0%} limit",
    )


def check_market_open(client: _ClientProtocol) -> CheckResult:
    """Verify the US equity market is currently open.

    Submitting market orders while the market is closed creates GTC-style
    exposure that may fill at an unexpected price the next morning.
    """
    if not client.is_market_open():
        return CheckResult(
            name="market_open",
            passed=False,
            message="Market is closed — submit orders only during trading hours",
        )
    return CheckResult(
        name="market_open",
        passed=True,
        message="Market is open",
    )


def check_last_run_time(
    log_dir: str,
    min_hours: float = 20.0,
) -> CheckResult:
    """Prevent double-runs: the most recent run must be ≥ ``min_hours`` ago.

    Protects against running the script twice in the same trading session
    (e.g. via a mis-fired cron job), which would double-up on positions.
    """
    log_path = Path(log_dir)
    log_files = sorted(log_path.glob("live_log_*.json"))
    if not log_files:
        return CheckResult(
            name="last_run",
            passed=True,
            message="No prior runs found — first execution",
        )

    latest = log_files[-1]
    try:
        with open(latest) as f:
            data = json.load(f)
        last_ts = datetime.fromisoformat(str(data["timestamp"]))
        if last_ts.tzinfo is None:
            last_ts = last_ts.replace(tzinfo=UTC)
        hours_since = (datetime.now(tz=UTC) - last_ts).total_seconds() / 3600
    except (KeyError, ValueError, OSError) as exc:
        return CheckResult(
            name="last_run",
            passed=False,
            message=f"Could not parse last-run timestamp from {latest.name}: {exc}",
        )

    if hours_since < min_hours:
        return CheckResult(
            name="last_run",
            passed=False,
            message=(
                f"Last run was {hours_since:.1f}h ago "
                f"(minimum {min_hours:.0f}h required to prevent double-runs)"
            ),
        )
    return CheckResult(
        name="last_run",
        passed=True,
        message=f"Last run was {hours_since:.1f}h ago",
    )


# ── Aggregate ──────────────────────────────────────────────────────────────

def run_all_checks(
    client: _ClientProtocol,
    orders: list[dict[str, Any]],
    target_weights: dict[str, float],
    equity: float,
    log_dir: str = "results",
) -> list[CheckResult]:
    """Run every safety check and return the full list of results.

    Logs PASS / FAIL for each check.  Callers should call ``all_passed()``
    on the result before proceeding to order submission.
    """
    results = [
        check_paper_account(client),
        check_order_notional(orders, equity),
        check_position_concentration(target_weights),
        check_market_open(client),
        check_last_run_time(log_dir),
    ]
    for r in results:
        if r.passed:
            logger.info("%s", r)
        else:
            logger.error("%s", r)
    return results


def all_passed(results: list[CheckResult]) -> bool:
    """Return True only if every check passed."""
    return all(r.passed for r in results)
