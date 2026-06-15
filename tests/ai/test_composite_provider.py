import pandas as pd
import pytest
import yfinance
from unittest.mock import MagicMock

from stock_analysis.ai.fundamentals import CompositeProvider, FundamentalsProvider


PERIODS = pd.to_datetime(["2024-12-31", "2023-12-31"])
LINE_ITEMS = ["Total Revenue", "Net Income"]


def _sample_df():
    return pd.DataFrame(
        {p: [100.0, 20.0] for p in PERIODS},
        index=LINE_ITEMS,
    )


def _primary(annual=None, quarterly=None):
    m = MagicMock()
    m.get_annual_financials.return_value = annual if annual is not None else _sample_df()
    m.get_quarterly_financials.return_value = quarterly if quarterly is not None else _sample_df()
    return m


def _fallback(annual=None, quarterly=None):
    m = MagicMock()
    m.get_annual_financials.return_value = annual if annual is not None else _sample_df()
    m.get_quarterly_financials.return_value = quarterly if quarterly is not None else _sample_df()
    return m


def test_annual_primary_returned_when_non_empty():
    pri = _primary(annual=_sample_df())
    fb = _fallback()
    result = CompositeProvider(primary=pri, fallback=fb).get_annual_financials("AAPL")
    assert not result.empty
    fb.get_annual_financials.assert_not_called()


def test_annual_fallback_invoked_when_primary_empty():
    pri = _primary(annual=pd.DataFrame())
    expected = _sample_df()
    fb = _fallback(annual=expected)
    result = CompositeProvider(primary=pri, fallback=fb).get_annual_financials("AAPL")
    assert not result.empty
    fb.get_annual_financials.assert_called_once_with("AAPL")


def test_quarterly_primary_returned_when_non_empty():
    pri = _primary(quarterly=_sample_df())
    fb = _fallback()
    result = CompositeProvider(primary=pri, fallback=fb).get_quarterly_financials("AAPL")
    assert not result.empty
    fb.get_quarterly_financials.assert_not_called()


def test_quarterly_fallback_invoked_when_primary_empty():
    pri = _primary(quarterly=pd.DataFrame())
    expected = _sample_df()
    fb = _fallback(quarterly=expected)
    result = CompositeProvider(primary=pri, fallback=fb).get_quarterly_financials("AAPL")
    assert not result.empty
    fb.get_quarterly_financials.assert_called_once_with("AAPL")


def test_composite_provider_satisfies_protocol():
    pri = _primary()
    fb = _fallback()
    assert isinstance(CompositeProvider(primary=pri, fallback=fb), FundamentalsProvider)


def test_get_ticker_financials_uses_composite(monkeypatch):
    """get_ticker_financials delegates to the module-level composite provider."""
    import stock_analysis.ai.helpers as helpers_mod

    expected = _sample_df()
    mock_provider = MagicMock()
    mock_provider.get_annual_financials.return_value = expected
    monkeypatch.setattr(helpers_mod, "_composite_provider", mock_provider)

    from stock_analysis.ai.helpers import get_ticker_financials
    result = get_ticker_financials("AAPL")
    mock_provider.get_annual_financials.assert_called_once_with("AAPL")
    pd.testing.assert_frame_equal(result, expected)


def test_get_ticker_quarterly_financials_uses_composite(monkeypatch):
    import stock_analysis.ai.helpers as helpers_mod

    expected = _sample_df()
    mock_provider = MagicMock()
    mock_provider.get_quarterly_financials.return_value = expected
    monkeypatch.setattr(helpers_mod, "_composite_provider", mock_provider)

    from stock_analysis.ai.helpers import get_ticker_quarterly_financials
    result = get_ticker_quarterly_financials("AAPL")
    mock_provider.get_quarterly_financials.assert_called_once_with("AAPL")
    pd.testing.assert_frame_equal(result, expected)


def test_get_quarterly_balance_sheet_uses_yfinance(monkeypatch):
    mock_ticker = MagicMock()
    mock_ticker.quarterly_balance_sheet = _sample_df()
    monkeypatch.setattr(yfinance, "Ticker", lambda sym: mock_ticker)

    from stock_analysis.ai.helpers import get_quarterly_balance_sheet
    result = get_quarterly_balance_sheet("AAPL")
    assert not result.empty
