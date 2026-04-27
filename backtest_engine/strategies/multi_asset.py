"""Multi-asset momentum strategy: equity sectors + long-duration bonds + gold.

A pure-stock portfolio loses 30–50 % in crashes.  Adding uncorrelated assets —
long-term Treasuries (TLT) and gold (GLD) — provides "crash insurance" because
they often rise when equities fall (flight-to-safety bid).  The risk-parity
insight: size each asset inversely by its volatility so every holding contributes
equal risk.  A low-vol bond needs a larger dollar weight than a high-vol sector
ETF to match its risk contribution.

This module provides:
- ``SECTOR_ETFS``: the 11 S&P 500 sector ETFs used throughout this project.
- ``UNIVERSE``: 14-asset list adding TLT, GLD, IEF to the sector universe.
- ``risk_parity_weights``: normalised inverse-vol sizing; each active asset
  contributes equal risk (assuming zero cross-correlation).
"""

from __future__ import annotations

import math

import pandas as pd

SECTOR_ETFS: list[str] = [
    "XLK", "XLF", "XLV", "XLE", "XLY",
    "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC",
]

UNIVERSE: list[str] = SECTOR_ETFS + [
    "TLT",   # iShares 20+ Year Treasury Bond ETF  — flight-to-safety, rate-cut beneficiary
    "GLD",   # SPDR Gold Shares                    — inflation hedge, crisis refuge
    "IEF",   # iShares 7-10 Year Treasury Bond ETF — intermediate duration, lower vol than TLT
]


def risk_parity_weights(
    signals: pd.DataFrame,
    returns: pd.DataFrame,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """Each active asset contributes equal risk to the portfolio.

    Returns *unshifted* weights — the engine applies ``shift(1)`` before
    execution so vol estimated through day T drives positions on day T+1.

    Algorithm
    ---------
    1. Rolling ``lookback_days``-day annualised vol per asset.
    2. Inverse-vol weight for each active position (signal != 0):
       ``inv_vol_i = 1 / σ_i``.  Inactive positions receive weight 0.
    3. Normalise each row: ``w_i = inv_vol_i / Σⱼ inv_vol_j``,
       so ``Σ|w_i| = 1`` (fully invested) on every active row.
    4. Multiply by signal direction so long/short/flat signs are preserved.
    5. Zero out warm-up rows (vol estimate not yet available).

    Under zero cross-correlation this achieves true equal risk contribution:
    ``w_i × σ_i = constant`` for all active i.

    Parameters
    ----------
    signals:
        Binary signals {-1, 0, 1} — rows = dates, cols = tickers.
        ``+1`` = long, ``-1`` = short, ``0`` = flat.
    returns:
        Daily simple returns — same shape as ``signals``.
    lookback_days:
        Rolling window for realised-vol estimation (default 60 trading days).

    Returns
    -------
    pd.DataFrame
        Continuous weights, same shape as ``signals``.
        On active rows: ``weights.abs().sum(axis=1) == 1``.
        Warm-up rows are all zero.

    Raises
    ------
    ValueError
        If ``lookback_days`` is less than 2.
    """
    if lookback_days < 2:
        raise ValueError(f"lookback_days must be >= 2, got {lookback_days}")

    # 1. Rolling annualised vol; NaN during warm-up (enforced by min_periods).
    asset_vol: pd.DataFrame = (
        returns.rolling(lookback_days, min_periods=lookback_days).std()
        * math.sqrt(252)
    )

    # 2. Inverse-vol for active positions; 0 for inactive, NaN during warm-up.
    active_mask: pd.DataFrame = signals.abs() > 0
    inv_vol: pd.DataFrame = pd.DataFrame(
        0.0, index=signals.index, columns=signals.columns
    ).where(~active_mask, other=1.0 / asset_vol)

    # 3. Normalise rows → Σ w_i = 1 on active rows.
    #    replace(0) lets fillna(0) handle all-flat rows cleanly.
    row_sum: pd.Series = inv_vol.sum(axis=1).replace(0.0, float("nan"))
    normalised: pd.DataFrame = inv_vol.div(row_sum, axis=0)

    # 4 & 5. Apply signal direction; NaN (warm-up / all-flat) → 0.
    return (normalised * signals).fillna(0.0)
