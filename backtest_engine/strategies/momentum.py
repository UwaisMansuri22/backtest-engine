"""Cross-sectional momentum strategy (Jegadeesh & Titman 1993)."""

from __future__ import annotations

import pandas as pd


def momentum_signals(
    prices: pd.DataFrame,
    lookback_months: int = 12,
    skip_months: int = 1,
    top_n: int = 3,
) -> pd.DataFrame:
    """Generate cross-sectional momentum signals with monthly rebalancing.

    At each month-end, ranks all assets by their formation-period return
    (from ``lookback_months`` ago to ``skip_months`` ago) and goes long the
    top ``top_n`` assets for the following calendar month.

    The ``skip_months=1`` exclusion — the "12-1" convention
    ---------------------------------------------------------
    The most recent month is deliberately excluded from the formation window.
    A well-documented short-term reversal effect (Jegadeesh 1990, Lehmann
    1990) causes last month's winners to mean-revert over the *next* month.
    Including it would contaminate the medium-term momentum signal with a
    short-term contrarian one, degrading net performance.  "12-1" means:
    rank by the 12-month return, but skip (exclude) the last 1 month.
    Concretely, at month-end T the formation return is:
        price(T − skip_months) / price(T − lookback_months) − 1

    Lookahead-bias note
    -------------------
    Signals are computed entirely from prices observable at month-end T.
    The engine applies ``shift(1)`` before trading, so execution happens
    on day T+1 — no future information is used.

    Parameters
    ----------
    prices:
        Adjusted close prices — rows = dates, cols = tickers.
    lookback_months:
        Total lookback horizon (default 12).
    skip_months:
        Most-recent months excluded from the formation window (default 1).
    top_n:
        Number of top-ranked assets to hold long each month (default 3).

    Returns
    -------
    pd.DataFrame
        Daily signals in {0.0, 1.0}; same shape as ``prices``.
        The engine's equal-weight normalisation converts these to 1/top_n
        weights for the selected assets.

    Raises
    ------
    ValueError
        If ``top_n`` exceeds the number of tickers in ``prices``.
    """
    n_tickers = len(prices.columns)
    if top_n > n_tickers:
        raise ValueError(
            f"top_n ({top_n}) cannot exceed number of tickers ({n_tickers})"
        )

    # ── 1. Resample to month-end ────────────────────────────────────────────
    # 'ME' = Month End frequency (pandas 2.2+ name; replaces deprecated 'M').
    # .last() captures the final closing price within each calendar month.
    monthly: pd.DataFrame = prices.resample("ME").last()

    # ── 2. Formation return: T−lookback → T−skip ────────────────────────────
    # At month T:  formation = price(T−skip) / price(T−lookback) − 1
    # Default:     formation = price(T−1)    / price(T−12)       − 1
    # This is the return over an 11-month window starting 12 months ago
    # and ending 1 month ago, conforming to the "12-1" J&T specification.
    formation: pd.DataFrame = (
        monthly.shift(skip_months) / monthly.shift(lookback_months) - 1.0
    )

    # ── 3. Cross-sectional rank → top-N selection ───────────────────────────
    # rank(ascending=False): rank 1 = highest return = strongest momentum.
    # method='first' breaks ties by column order (deterministic, avoids bias).
    # NaN formation values naturally rank last because pandas rank() skips NaN
    # by default, but we also zero-out entire months below for safety.
    ranks: pd.DataFrame = formation.rank(axis=1, ascending=False, method="first")
    monthly_signals: pd.DataFrame = (ranks <= top_n).astype(float)

    # Zero out any month where at least one ticker has no formation data.
    # This covers the warm-up period (first lookback_months months) and any
    # gaps in the price history introduced by recently-launched tickers.
    valid_months: pd.Series = formation.notna().all(axis=1)
    monthly_signals = monthly_signals.where(valid_months, other=0.0)

    # ── 4. Broadcast month-end signals to daily frequency ──────────────────
    # reindex with method='ffill' fills every trading day between two
    # consecutive month-ends with the earlier month-end's signal, so the
    # position is held constant until the next monthly rebalance.
    daily: pd.DataFrame = monthly_signals.reindex(prices.index, method="ffill")
    return daily.fillna(0.0)
