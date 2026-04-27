"""Thin wrapper around alpaca-py for paper-trading operations.

API keys are read exclusively from environment variables — never hardcoded:
    ALPACA_API_KEY    — Alpaca paper-trading API key
    ALPACA_SECRET_KEY — Alpaca paper-trading secret key

This client ALWAYS targets the paper endpoint.  The ``is_paper = True``
class attribute exists so safety checks can verify it without calling the
API, and so test mocks can satisfy the same interface.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.models.bars import BarSet
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.models import Clock, Order, Position, TradeAccount
from alpaca.trading.requests import MarketOrderRequest

PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class AlpacaClient:
    """Paper-trading client — always uses paper endpoint, never live.

    Instantiation reads ``ALPACA_API_KEY`` and ``ALPACA_SECRET_KEY`` from
    the environment (or a ``.env`` file loaded by the caller via dotenv).
    """

    is_paper: bool = True

    def __init__(self) -> None:
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise OSError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in the environment. "
                "Copy .env.example → .env and fill in your paper-trading keys, "
                "then run: source .env  (or let python-dotenv load it automatically)."
            )
        self._trading: TradingClient = TradingClient(api_key, secret_key, paper=True)
        self._data: StockHistoricalDataClient = StockHistoricalDataClient(
            api_key, secret_key
        )

    # ── Account / positions ───────────────────────────────────────────────

    def get_account(self) -> TradeAccount:
        """Return paper-account details including current equity."""
        result = self._trading.get_account()
        return result  # type: ignore[return-value]

    def get_positions(self) -> list[Position]:
        """Return all open positions in the paper account."""
        result = self._trading.get_all_positions()
        return result  # type: ignore[return-value]

    # ── Orders ────────────────────────────────────────────────────────────

    def submit_order(self, symbol: str, qty: float, side: str) -> Order:
        """Submit a fractional DAY market order.

        Parameters
        ----------
        symbol: Ticker symbol (e.g. ``"XLK"``).
        qty:    Fractional share quantity (positive float).
        side:   ``"buy"`` or ``"sell"``.
        """
        request = MarketOrderRequest(
            symbol=symbol,
            qty=round(qty, 6),
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        result = self._trading.submit_order(request)
        return result  # type: ignore[return-value]

    # ── Market data ───────────────────────────────────────────────────────

    def get_bars(self, symbols: list[str], days: int = 400) -> pd.DataFrame:
        """Return adjusted close prices (rows = trading dates, cols = symbols).

        Fetches at least ``days`` trading days via the free IEX data feed
        with full split-and-dividend adjustment.

        Parameters
        ----------
        symbols: List of ticker symbols.
        days:    Minimum number of trading-day rows to return.
        """
        end = datetime.now(tz=UTC)
        # 2× calendar-day buffer for weekends, holidays, and the IEX calendar.
        start = end - timedelta(days=days * 2)

        request = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX,
        )
        raw: Any = self._data.get_stock_bars(request)
        bars: BarSet = raw
        df: pd.DataFrame = bars.df

        # bars.df MultiIndex: level 0 = symbol, level 1 = timestamp
        # Pivot so rows = dates, cols = symbols
        close: pd.DataFrame = df["close"].unstack(level=0)
        close.index = pd.DatetimeIndex(
            pd.to_datetime([str(ts)[:10] for ts in close.index])
        )
        close = close.dropna(how="all").iloc[-days:]
        return close

    # ── Clock ─────────────────────────────────────────────────────────────

    def is_market_open(self) -> bool:
        """Return True if the US equity market is currently open for trading."""
        raw: Any = self._trading.get_clock()
        clock: Clock = raw
        return bool(clock.is_open)
