import pandas as pd
import pytest

from stock_analysis.ai.fundamental_analysis.valuation_methods import StockValuation


def _make_financials():
    return pd.DataFrame(
        {'2024': [1e9, 5e9, 800e6, 1.2e9, 500e6]},
        index=['Net Income', 'Total Revenue', 'EBITDA', 'EBIT', 'Gross Profit'],
    )


def _make_balance_sheet():
    return pd.DataFrame(
        {'2024': [10e9, 6e9, 3e9, 2e9, 1e9, 500e6, 2e9, 4e9]},
        index=[
            'Total Assets',
            'Total Liabilities Net Minority Interest',
            'Current Assets',
            'Current Liabilities',
            'Total Debt',
            'Cash And Cash Equivalents',
            'Stockholders Equity',
            'Operating Income',
        ],
    )


def _make_cashflow():
    return pd.DataFrame(
        {'2024': [600e6]},
        index=['Free Cash Flow'],
    )


class _MockRawTicker:
    def __init__(self):
        self.info = {
            'currentPrice': 150.0,
            'sharesOutstanding': 1e9,
            'marketCap': 150e9,
            'forwardEps': 6.0,
            'earningsGrowth': 0.10,
            'dividendYield': 0.01,
            'priceToBook': 1.2,
        }
        self.financials = _make_financials()
        self.quarterly_financials = _make_financials()
        self.balance_sheet = _make_balance_sheet()
        self.quarterly_balance_sheet = _make_balance_sheet()
        self.cashflow = _make_cashflow()
        self.quarterly_cashflow = _make_cashflow()


class _MockProvider:
    def __init__(self):
        self._ticker = _MockRawTicker()
        self.calls = []

    def get_raw_ticker(self, ticker):
        self.calls.append(ticker)
        return self._ticker


def test_stock_valuation_uses_provider():
    mock = _MockProvider()
    sv = StockValuation('AAPL', provider=mock)
    assert mock.calls == ['AAPL']
    assert sv.current_price == 150.0
    assert sv.market_cap == 150e9


def test_stock_valuation_book_value():
    sv = StockValuation('AAPL', provider=_MockProvider())
    result = sv.book_value_valuation()
    assert 'book_value' in result
    assert result['book_value'] == pytest.approx(10e9 - 6e9)


def test_stock_valuation_default_provider_not_called_when_injected(monkeypatch):
    import stock_analysis.ai.fundamental_analysis.valuation_methods as vm_mod

    called = []

    class _TrackingProvider:
        def get_raw_ticker(self, ticker):
            called.append('tracking')
            return _MockRawTicker()

    sv = StockValuation('AAPL', provider=_TrackingProvider())
    assert called == ['tracking']
