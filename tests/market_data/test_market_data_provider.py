import pandas as pd
import pytest
from unittest.mock import MagicMock

import yfinance
from stock_analysis.market_data import MarketDataProvider, YFinanceProvider


_DATES = pd.date_range("2024-01-01", periods=5, freq="B")
_OHLCV = pd.DataFrame(
    {
        "Open": [100.0] * 5,
        "High": [105.0] * 5,
        "Low": [99.0] * 5,
        "Close": [103.0] * 5,
        "Volume": [1_000_000] * 5,
    },
    index=_DATES,
)
_FINANCIALS = pd.DataFrame(
    {"2024-12-31": [1e9, 2e8]},
    index=["Total Revenue", "Net Income"],
)
_EARNINGS = pd.DataFrame(
    {"EPS Estimate": [1.5, 1.2], "Reported EPS": [1.6, 1.1]},
    index=_DATES[:2],
)


_MISSING = object()


def _mock_ticker(history=_MISSING, earnings_dates=_MISSING):
    m = MagicMock()
    m.history.return_value = _OHLCV if history is _MISSING else history
    m.earnings_dates = _EARNINGS if earnings_dates is _MISSING else earnings_dates
    return m


def test_yfinance_provider_satisfies_protocol():
    assert isinstance(YFinanceProvider(), MarketDataProvider)


def test_get_history_by_period(monkeypatch):
    mock = _mock_ticker()
    monkeypatch.setattr(yfinance, "Ticker", lambda sym: mock)

    result = YFinanceProvider().get_history("AAPL", period="1y")

    mock.history.assert_called_once_with(period="1y")
    pd.testing.assert_frame_equal(result, _OHLCV)


def test_get_history_by_start_end(monkeypatch):
    mock = _mock_ticker()
    monkeypatch.setattr(yfinance, "Ticker", lambda sym: mock)

    result = YFinanceProvider().get_history("AAPL", start="2024-01-01", end="2024-06-01")

    mock.history.assert_called_once_with(start="2024-01-01", end="2024-06-01")
    pd.testing.assert_frame_equal(result, _OHLCV)


def test_get_financials_delegates_to_composite():
    mock_composite = MagicMock()
    mock_composite.get_annual_financials.return_value = _FINANCIALS

    result = YFinanceProvider(fundamentals_provider=mock_composite).get_financials("AAPL")

    mock_composite.get_annual_financials.assert_called_once_with("AAPL")
    pd.testing.assert_frame_equal(result, _FINANCIALS)


def test_get_earnings_returns_dataframe(monkeypatch):
    mock = _mock_ticker()
    monkeypatch.setattr(yfinance, "Ticker", lambda sym: mock)

    result = YFinanceProvider().get_earnings("AAPL")

    assert isinstance(result, pd.DataFrame)
    assert not result.empty


def test_get_earnings_returns_empty_on_none(monkeypatch):
    mock = _mock_ticker(earnings_dates=None)
    monkeypatch.setattr(yfinance, "Ticker", lambda sym: mock)

    result = YFinanceProvider().get_earnings("AAPL")

    assert isinstance(result, pd.DataFrame)
    assert result.empty
