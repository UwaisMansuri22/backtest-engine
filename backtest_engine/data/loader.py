"""Adjusted close price loader backed by yfinance with local Parquet caching."""

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Relative to the project root; created automatically on first use.
CACHE_DIR = Path("data_cache")


def _cache_path(ticker: str, start: str, end: str) -> Path:
    return CACHE_DIR / f"{ticker}_{start}_{end}.parquet"


def _fetch_single(ticker: str, start: str, end: str) -> pd.Series:
    """Download adjusted close prices for one ticker from yfinance.

    auto_adjust=True applies historical split and dividend adjustments so that
    every price is expressed in today's equivalent dollars. Without it, a
    2-for-1 split would show prices halving overnight, making pre-split returns
    appear twice as large and completely distorting strategy performance.
    """
    raw: pd.DataFrame = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        return pd.Series(dtype=float, name=ticker)

    close = raw["Close"]
    # yfinance ≥ 0.2.37 returns MultiIndex columns even for a single ticker:
    #   raw.columns = MultiIndex([('Close', 'SPY'), ...], names=['Price', 'Ticker'])
    # raw["Close"] therefore gives a single-column DataFrame, not a Series.
    # Older yfinance returns a flat Series directly.  Handle both forms here.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.rename(ticker)


def load_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache: bool = True,
) -> pd.DataFrame:
    """Load adjusted close prices for a basket of tickers.

    Parameters
    ----------
    tickers:
        Ticker symbols, e.g. ``["AAPL", "MSFT"]``.
    start:
        Inclusive start date string, e.g. ``"2020-01-01"``.
    end:
        Exclusive end date string, e.g. ``"2023-01-01"``.
    cache:
        When ``True``, persist each ticker's data to
        ``data_cache/{ticker}_{start}_{end}.parquet`` and reload from there
        on subsequent calls with the same parameters. This avoids redundant
        network requests and survives yfinance rate limits during development.

    Returns
    -------
    pd.DataFrame
        ``DatetimeIndex × tickers`` DataFrame of adjusted close prices.
        Rows where **any** ticker has a ``NaN`` value are dropped rather than
        forward-filled: injecting a synthetic price for a day a stock didn't
        trade is fabricated data and would silently corrupt backtest returns.

    Raises
    ------
    ValueError
        If no data could be retrieved for any of the requested tickers.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    series: list[pd.Series] = []

    for ticker in tickers:
        path = _cache_path(ticker, start, end)

        if cache and path.exists():
            logger.info("Cache hit: %s", ticker)
            loaded = pd.read_parquet(path)
            s: pd.Series = loaded.iloc[:, 0].rename(ticker)
        else:
            logger.info("Fetching %s from yfinance", ticker)
            s = _fetch_single(ticker, start, end)
            if s.empty:
                logger.warning("No data returned for %s — skipping", ticker)
                continue
            if cache:
                s.to_frame().to_parquet(path)

        series.append(s)

    if not series:
        raise ValueError(f"No data retrieved for any of: {tickers}")

    df = pd.concat(series, axis=1)
    df.index = pd.to_datetime(df.index)

    n_before = len(df)
    df = df.dropna(how="any")
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info(
            "Dropped %d rows containing NaN "
            "(misaligned trading calendars or missing data).",
            n_dropped,
        )

    return df
