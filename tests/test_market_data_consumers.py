"""Smoke tests for market-data consumer migrations (issue #10 US-004)."""

from unittest.mock import MagicMock

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

import stock_analysis.analyze_stock_data as analyze_mod
import stock_analysis.sector_analysis as sector_mod
import stock_analysis.sp500_analysis as sp500_mod
import stock_analysis.stock_finder as finder_mod


_DATES = pd.date_range("2024-01-01", periods=15, freq="B")
_OHLCV = pd.DataFrame(
    {
        "Open": [100.0] * 15,
        "High": [105.0] * 15,
        "Low": [99.0] * 15,
        "Close": [103.0 + i for i in range(15)],
        "Volume": [1_000_000] * 15,
        "Dividends": [0.0] * 15,
        "Stock Splits": [0.0] * 15,
    },
    index=_DATES,
)


class _MockProvider:
    def get_history(self, ticker, period="1y", start=None, end=None):
        return _OHLCV.copy()

    def get_financials(self, ticker):
        return pd.DataFrame()

    def get_earnings(self, ticker):
        return pd.DataFrame()

    def get_tickers_obj(self, symbols):
        return MagicMock()

    def get_market_tickers_obj(self, market):
        m = MagicMock()
        m.info = {}
        return m

    def get_market_stocks(self, market):
        m = MagicMock()
        m.tickers = {}
        return m


def test_sp500_analysis_uses_injected_provider(monkeypatch):
    mock = _MockProvider()
    monkeypatch.setattr(sp500_mod, "_default_provider", mock)
    monkeypatch.setattr(sp500_mod.go.Figure, "show", lambda self: None)
    monkeypatch.setattr(sp500_mod.plt, "show", lambda: None)

    result = sp500_mod.sp500_analysis(provider=mock)

    assert result is not None
    assert "return" in result.columns


def test_find_undervalued_sectors_sector_module(monkeypatch, capsys):
    mock = _MockProvider()
    sector_mod.find_undervalued_sectors(marketTicker="^GSPC", provider=mock)
    assert "{}" in capsys.readouterr().out


def test_find_stocks_by_method_provider_injection(monkeypatch):
    mock = _MockProvider()
    tickers_mock = MagicMock()
    mock.get_tickers_obj = MagicMock(return_value=tickers_mock)
    monkeypatch.setattr(
        finder_mod,
        "find_high_dividend_stocks",
        lambda **kwargs: [],
    )

    finder_mod.find_stocks_by_method(
        method="high_dividend",
        stockTickers=["AAPL"],
        provider=mock,
    )
    mock.get_tickers_obj.assert_called_once_with(["AAPL"])


def test_analyze_stock_vs_sp500_provider_injection():
    mock = _MockProvider()
    stock_df = _OHLCV.copy()

    result = analyze_mod.analyze_stock_vs_sp500(stock_df, provider=mock)

    assert result is not None
    assert "Beta" in result
