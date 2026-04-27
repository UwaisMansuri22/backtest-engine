"""Tests for backtest_engine.data.loader."""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import backtest_engine.data.loader as loader_module
from backtest_engine.data.loader import load_prices

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_yf_response(dates: list[str], prices: list[float]) -> pd.DataFrame:
    """Minimal yfinance-style flat DataFrame (auto_adjust=True, single ticker)."""
    idx = pd.DatetimeIndex(dates, name="Date")
    return pd.DataFrame(
        {"Close": prices, "Volume": [1_000_000] * len(dates)},
        index=idx,
    )


@pytest.fixture()
def cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect CACHE_DIR to a throwaway temp directory for each test."""
    monkeypatch.setattr(loader_module, "CACHE_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_ticker_shape_and_types(cache_dir: Path) -> None:
    """load_prices returns a correctly shaped DataFrame for one ticker."""
    dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    mock_df = _mock_yf_response(dates, [300.0, 302.0, 305.0])

    with patch("yfinance.download", return_value=mock_df):
        result = load_prices(["AAPL"], "2020-01-01", "2020-01-10", cache=False)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["AAPL"]
    assert len(result) == 3
    assert isinstance(result.index, pd.DatetimeIndex)
    assert result["AAPL"].iloc[0] == pytest.approx(300.0)


def test_multi_ticker_aligns_on_common_dates(cache_dir: Path) -> None:
    """load_prices aligns multiple tickers and drops rows missing any ticker."""
    # AAPL has 3 trading days; MSFT is missing the middle day.
    aapl_dates = ["2020-01-02", "2020-01-03", "2020-01-06"]
    msft_dates = ["2020-01-02", "2020-01-06"]  # gap on 2020-01-03

    aapl_df = _mock_yf_response(aapl_dates, [300.0, 302.0, 305.0])
    msft_df = _mock_yf_response(msft_dates, [150.0, 155.0])

    responses = {"AAPL": aapl_df, "MSFT": msft_df}

    with patch(
        "yfinance.download",
        side_effect=lambda ticker, **kwargs: responses[ticker],
    ):
        result = load_prices(
            ["AAPL", "MSFT"], "2020-01-01", "2020-01-10", cache=False
        )

    assert list(result.columns) == ["AAPL", "MSFT"]
    # The row where MSFT is NaN (2020-01-03) must be dropped.
    assert len(result) == 2
    assert result.index.tolist() == pd.DatetimeIndex(
        ["2020-01-02", "2020-01-06"]
    ).tolist()


def test_cache_hit_skips_yfinance(cache_dir: Path) -> None:
    """Second call with identical params reads from cache, not yfinance."""
    dates = ["2020-01-02", "2020-01-03"]
    mock_df = _mock_yf_response(dates, [300.0, 302.0])

    with patch("yfinance.download", return_value=mock_df) as mock_dl:
        # First call: cache miss → yfinance is called.
        first = load_prices(["AAPL"], "2020-01-01", "2020-01-10", cache=True)
        assert mock_dl.call_count == 1

        # Second call: cache hit → yfinance must NOT be called again.
        second = load_prices(["AAPL"], "2020-01-01", "2020-01-10", cache=True)
        assert mock_dl.call_count == 1  # still 1

    # Both calls should return identical data.
    pd.testing.assert_frame_equal(first, second)


def test_missing_ticker_is_skipped_and_raises_when_all_missing(
    cache_dir: Path,
) -> None:
    """An empty yfinance response for a ticker is skipped; all-missing raises."""
    empty_df = pd.DataFrame()

    with patch("yfinance.download", return_value=empty_df):
        with pytest.raises(ValueError, match="No data retrieved"):
            load_prices(["INVALID_XYZ"], "2020-01-01", "2020-01-10", cache=False)
