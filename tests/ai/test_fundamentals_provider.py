import pandas as pd
import pytest
import yfinance
from unittest.mock import MagicMock

from stock_analysis.ai.fundamentals import FundamentalsProvider, YFinanceAdapter


PERIODS = pd.to_datetime(["2024-12-31", "2023-12-31", "2022-12-31", "2021-12-31"])
LINE_ITEMS = [
    "Total Revenue", "Gross Profit", "Net Income", "EBITDA", "EBIT",
    "Operating Income", "Cost Of Revenue", "Depreciation",
]


def _make_df(scale=1.0):
    return pd.DataFrame(
        {p: [float(i) * scale for i in range(len(LINE_ITEMS))] for p in PERIODS},
        index=LINE_ITEMS,
    )


@pytest.fixture
def mock_yf(monkeypatch):
    annual = _make_df(1.0)
    quarterly = _make_df(0.25)

    mock_ticker = MagicMock()
    mock_ticker.financials = annual
    mock_ticker.quarterly_financials = quarterly

    monkeypatch.setattr(yfinance, "Ticker", lambda sym: mock_ticker)
    return annual, quarterly


def test_adapter_returns_annual_financials(mock_yf):
    annual, _ = mock_yf
    result = YFinanceAdapter().get_annual_financials("AAPL")
    pd.testing.assert_frame_equal(result, annual)


def test_adapter_returns_quarterly_financials(mock_yf):
    _, quarterly = mock_yf
    result = YFinanceAdapter().get_quarterly_financials("AAPL")
    pd.testing.assert_frame_equal(result, quarterly)


def test_adapter_satisfies_protocol():
    assert isinstance(YFinanceAdapter(), FundamentalsProvider)


def test_dataframe_orientation(mock_yf):
    """Line items as index, periods as columns."""
    annual, _ = mock_yf
    result = YFinanceAdapter().get_annual_financials("AAPL")
    assert list(result.index) == LINE_ITEMS
    assert list(result.columns) == list(PERIODS)
