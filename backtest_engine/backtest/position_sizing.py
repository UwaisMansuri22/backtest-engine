"""
Volatility-targeted position sizing.

Equal-dollar weighting is naive. If I put $1000 in a calm utility stock and
$1000 in a volatile tech stock, the tech stock dominates my P&L — its swings
drown out the utility's. Vol targeting fixes this: scale each position
INVERSELY to its recent volatility, so each contributes roughly equal RISK.
This single change typically improves Sharpe by 0.2–0.5 with no signal change.
"""

from __future__ import annotations

import math

import pandas as pd


def vol_target_weights(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    target_vol_annual: float = 0.15,
    vol_lookback_days: int = 60,
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """Compute inverse-volatility weights from binary signals.

    Returns *unshifted* weights — the engine applies ``shift(1)`` before
    execution so that vol estimated through day T drives positions on day T+1.

    Algorithm
    ---------
    1. Daily simple returns from prices.
    2. Rolling ``vol_lookback_days``-day annualised vol per ticker:
       ``returns.rolling(N).std() * sqrt(252)``.
    3. Per-ticker raw weight = ``target_vol_annual / asset_vol``.
       This makes the marginal vol contribution of each active position equal.
    4. Multiply by signal direction so long/short/flat signs are preserved.
    5. Scale the entire row down proportionally if total gross exposure would
       exceed ``max_leverage`` (never levers up, only levers down).
    6. Zero out rows in the warm-up period (no vol estimate yet).

    Parameters
    ----------
    signals:
        Binary signals {-1, 0, 1} — rows = dates, cols = tickers.
        ``+1`` = long, ``-1`` = short, ``0`` = flat.
    prices:
        Adjusted close prices — same shape as ``signals``.
    target_vol_annual:
        Desired annualised volatility contribution per active position,
        as a decimal (default 0.15 = 15 %).
    vol_lookback_days:
        Rolling window for realised vol estimation (default 60 trading days
        ≈ 3 months).
    max_leverage:
        Maximum total gross exposure, i.e. ``sum(|weights|) <= max_leverage``
        on every row (default 1.0 = fully invested, no leverage).

    Returns
    -------
    pd.DataFrame
        Continuous weights, same shape as ``signals``.
        ``weights.abs().sum(axis=1) <= max_leverage`` on every row.
        Rows in the vol warm-up period are all zero.

    Raises
    ------
    ValueError
        If ``target_vol_annual``, ``vol_lookback_days``, or ``max_leverage``
        are out of range.
    """
    if target_vol_annual <= 0:
        raise ValueError(
            f"target_vol_annual must be positive, got {target_vol_annual}"
        )
    if vol_lookback_days < 2:
        raise ValueError(
            f"vol_lookback_days must be >= 2, got {vol_lookback_days}"
        )
    if max_leverage <= 0:
        raise ValueError(
            f"max_leverage must be positive, got {max_leverage}"
        )

    # 1. Daily simple returns (row 0 is NaN — no prior price).
    daily_rets = prices.pct_change()

    # 2. Rolling annualised vol per ticker.
    #    min_periods=vol_lookback_days forces NaN until the window is full,
    #    preventing underestimated vol from partial windows driving oversized bets.
    asset_vol: pd.DataFrame = (
        daily_rets.rolling(vol_lookback_days, min_periods=vol_lookback_days).std()
        * math.sqrt(252)
    )

    # 3 & 4. Inverse-vol weight × signal direction.
    #    Where signal == 0: weight == 0 (flat position).
    #    Where vol is NaN (warm-up): weight is NaN → zeroed in step 6.
    #    A near-zero vol would inflate weights enormously; the max_leverage cap
    #    in step 5 prevents runaway leverage in that case.
    raw_weights: pd.DataFrame = (target_vol_annual / asset_vol) * signals

    # 5. Proportional scale-down so gross exposure <= max_leverage.
    #    replace(0) → NaN lets fillna(0) handle the all-flat-row case cleanly.
    gross_exp: pd.Series = raw_weights.abs().sum(axis=1)
    scale: pd.Series = (
        (max_leverage / gross_exp.replace(0.0, float("nan")))
        .clip(upper=1.0)
        .fillna(0.0)
    )
    weights: pd.DataFrame = raw_weights.mul(scale, axis=0)

    # 6. Zero remaining NaN (warm-up rows and any residual NaN from bad prices).
    return weights.fillna(0.0)
