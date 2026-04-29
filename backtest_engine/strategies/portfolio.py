"""Multi-strategy portfolio blender.

``blend_signals`` combines any number of strategy weight DataFrames into a
single target-weight frame.  Each strategy contributes proportionally to its
blend_weight; the final frame is scaled to ``max_leverage`` gross exposure.

Usage
-----
    from backtest_engine.strategies.portfolio import blend_signals
    from backtest_engine.strategies.momentum import momentum_signals
    from backtest_engine.strategies.mean_reversion import mean_reversion_signals
    from backtest_engine.strategies.multi_asset import risk_parity_weights

    mom_sig  = momentum_signals(prices_14)
    mom_w    = risk_parity_weights(mom_sig, prices_14.pct_change())

    mr_sig   = mean_reversion_signals(prices_14, regime=regime_series)
    # mean_reversion_signals already applies shift(1) internally
    mr_w     = risk_parity_weights(mr_sig, prices_14.pct_change())

    blended  = blend_signals({"momentum": (mom_w, 0.6), "mr": (mr_w, 0.4)})
"""

from __future__ import annotations

import pandas as pd


def blend_signals(
    strategy_weights: dict[str, tuple[pd.DataFrame, float]],
    max_leverage: float = 1.0,
) -> pd.DataFrame:
    """Blend per-strategy weight DataFrames into a single weight frame.

    Each strategy contributes ``blend_weight * weights`` to the combined
    portfolio.  The blended frame is then scaled so that gross exposure
    (sum of absolute weights) never exceeds ``max_leverage``.

    Blend weights do NOT need to sum to 1 — they are normalised internally.
    A strategy with blend_weight=0.6 receives 60 % / (0.6+0.4) = 60 % of
    total capital, which is intuitive when weights sum to 1.

    Parameters
    ----------
    strategy_weights:
        Mapping of strategy name → (weights_DataFrame, blend_weight).
        All DataFrames must share the same index and columns.
    max_leverage:
        Maximum gross exposure cap (default 1.0 = fully invested, no leverage).

    Returns
    -------
    pd.DataFrame
        Combined weight DataFrame, same shape as the inputs.
        ``abs().sum(axis=1) <= max_leverage`` on every row.

    Raises
    ------
    ValueError
        If ``strategy_weights`` is empty, or DataFrames have mismatched shapes.
    """
    if not strategy_weights:
        raise ValueError("strategy_weights must contain at least one entry")

    items = list(strategy_weights.values())
    ref_df, _ = items[0]

    for name, (df, _) in strategy_weights.items():
        if not df.index.equals(ref_df.index) or not df.columns.equals(ref_df.columns):
            raise ValueError(
                f"Strategy '{name}' has a different index/columns than the first strategy"
            )

    total_blend = sum(w for _, w in strategy_weights.values())
    if total_blend <= 0:
        raise ValueError("Sum of blend weights must be positive")

    combined = pd.DataFrame(0.0, index=ref_df.index, columns=ref_df.columns)
    for _, (df, blend_w) in strategy_weights.items():
        combined += df * (blend_w / total_blend)

    # Cap gross exposure at max_leverage.
    gross = combined.abs().sum(axis=1).replace(0.0, float("nan"))
    over = gross > max_leverage
    if over.any():
        combined.loc[over] = combined.loc[over].div(
            gross[over], axis=0
        ).mul(max_leverage)

    return combined.fillna(0.0)
