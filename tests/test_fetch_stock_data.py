import pandas as pd
import pytest

import stock_analysis.fetch_stock_data as fsd_mod
from stock_analysis.fetch_stock_data import fetch_stock_data


def _make_hist():
    return pd.DataFrame(
        {'Open': [100.0], 'High': [110.0], 'Low': [95.0], 'Close': [105.0], 'Volume': [1000]},
        index=pd.to_datetime(['2025-01-15']),
    )


class _MockTicker:
    def __init__(self, hist):
        self._hist = hist
        self.dividends = pd.Series(dtype=float)
        self.earnings_dates = None

    def history(self, period=None, start=None, end=None):
        return self._hist


class _MockTickerWithInfo:
    info = {
        'marketCap': 3e12,
        'longName': 'Apple Inc.',
        'sector': 'Technology',
        'forwardPE': 30.0,
        'trailingPE': 32.0,
        'priceToBook': 45.0,
        'profitMargins': 0.26,
        'operatingMargins': 0.30,
        'priceToSales': 8.0,
    }


class _MockTickersObj:
    def __init__(self):
        self.tickers = {
            'AAPL': _MockTickerWithInfo(),
        }


class _MockProvider:
    def __init__(self, hist=None):
        self._hist = hist if hist is not None else _make_hist()
        self._get_raw_ticker_calls = []

    def get_raw_ticker(self, ticker):
        self._get_raw_ticker_calls.append(ticker)
        return _MockTicker(self._hist)

    def get_tickers_obj(self, symbols):
        return _MockTickersObj()


def test_fetch_stock_data_provider_injection():
    mock = _MockProvider()
    df, events = fetch_stock_data('AAPL', provider=mock)
    assert df is not None
    assert set(['Open', 'High', 'Low', 'Close']).issubset(df.columns)
    assert len(df) == 1
    assert events is not None
    assert mock._get_raw_ticker_calls == ['AAPL']


def test_fetch_stock_data_default_provider_patch(monkeypatch):
    mock = _MockProvider()
    monkeypatch.setattr(fsd_mod, '_default_provider', mock)
    df, events = fetch_stock_data('MSFT')
    assert df is not None
    assert mock._get_raw_ticker_calls == ['MSFT']


def test_fetch_stock_data_returns_none_on_empty(monkeypatch):
    class _EmptyProvider:
        def get_raw_ticker(self, ticker):
            return _MockTicker(pd.DataFrame())

    df, events = fetch_stock_data('INVALID', retry_count=1, provider=_EmptyProvider())
    assert df is None
    assert events is None


def test_fetch_stocks_data_provider_injection():
    from stock_analysis.fetch_stock_data import fetch_stocks_data
    mock = _MockProvider()
    result = fetch_stocks_data(['AAPL'], provider=mock)
    assert isinstance(result, pd.DataFrame)


def test_fetch_stocks_data_alt_provider_injection():
    from stock_analysis.fetch_stock_data import fetch_stocks_data_alt
    mock = _MockProvider()
    result = fetch_stocks_data_alt(['AAPL'], provider=mock)
    assert isinstance(result, pd.DataFrame)
