"""P2 regression: empty Yahoo mock triggers EDGAR fallback (US-005)."""
import pandas as pd
import pytest
import yfinance
from unittest.mock import MagicMock

from stock_analysis.ai.fundamentals import CompositeProvider, EdgarAdapter, YFinanceAdapter
from stock_analysis.storage.cache_store import SqlCacheStore
from stock_analysis.storage.schema import create_engine_for_url, init_db

REQUIRED_LINE_ITEMS = [
    "Total Revenue",
    "Net Income",
    "Gross Profit",
    "Operating Income",
    "Cost Of Revenue",
    "Depreciation",
    "EBIT",
    "EBITDA",
]


@pytest.fixture
def store():
    engine = create_engine_for_url("sqlite://")
    init_db(engine)
    return SqlCacheStore(engine)


def _make_stitched_df(periods: list[str]) -> pd.DataFrame:
    rows = [
        ("Revenue", "us-gaap:Revenues", "Revenue", 1),
        ("Net Income", "us-gaap:NetIncomeLoss", "NetIncome", 1),
        ("Gross Profit", "us-gaap:GrossProfit", "GrossProfit", 1),
        ("Operating Income", "us-gaap:OperatingIncomeLoss", "OperatingIncomeLoss", 1),
        ("Cost Of Revenue", "us-gaap:CostOfRevenue", "CostOfRevenue", 1),
        ("Depreciation", "us-gaap:Depreciation", "DepreciationExpense", -1),
    ]
    base = [383_000, 97_000, 170_000, 115_000, 213_000, 11_000]
    data: dict = {
        "label": [r[0] for r in rows],
        "concept": [r[1] for r in rows],
        "standard_concept": [r[2] for r in rows],
        "preferred_sign": [r[3] for r in rows],
    }
    for i, period in enumerate(periods):
        multiplier = 1.0 + i * 0.05
        data[period] = [int(v * multiplier) for v in base]
    return pd.DataFrame(data)


def _make_mock_edgar(stitched_df: pd.DataFrame):
    mock_stmt = MagicMock()
    mock_stmt.to_dataframe.return_value = stitched_df
    mock_multi = MagicMock()
    mock_multi.income_statement.return_value = mock_stmt

    mock_filings = MagicMock()
    mock_filings.empty = False
    mock_filings.head.return_value = mock_filings
    mock_company = MagicMock()
    mock_company.get_filings.return_value = mock_filings

    return mock_company, mock_multi


def _patch_yfinance_empty(monkeypatch):
    mock_ticker = MagicMock()
    mock_ticker.financials = pd.DataFrame()
    mock_ticker.quarterly_financials = pd.DataFrame()
    monkeypatch.setattr(yfinance, "Ticker", lambda sym: mock_ticker)


def test_empty_yahoo_annual_triggers_edgar_fallback(store, monkeypatch):
    """YFinance returns empty annual → composite falls back to EDGAR → result has required line items."""
    _patch_yfinance_empty(monkeypatch)
    periods = ["2024-09-30", "2023-09-30", "2022-09-30", "2021-09-30"]
    mock_company, mock_multi = _make_mock_edgar(_make_stitched_df(periods))
    monkeypatch.setattr("edgar.Company", MagicMock(return_value=mock_company))
    monkeypatch.setattr("edgar.MultiFinancials.extract", staticmethod(lambda f: mock_multi))

    provider = CompositeProvider(
        primary=YFinanceAdapter(),
        fallback=EdgarAdapter(cache_store=store),
    )
    result = provider.get_annual_financials("AAPL")

    assert not result.empty
    for item in REQUIRED_LINE_ITEMS:
        assert item in result.index, f"Missing required line item: {item}"


def test_empty_yahoo_quarterly_triggers_edgar_fallback(store, monkeypatch):
    """YFinance returns empty quarterly → composite falls back to EDGAR → result has required line items."""
    _patch_yfinance_empty(monkeypatch)
    periods = ["2024-09-30", "2024-06-30", "2024-03-31", "2023-12-31"]
    mock_company, mock_multi = _make_mock_edgar(_make_stitched_df(periods))
    monkeypatch.setattr("edgar.Company", MagicMock(return_value=mock_company))
    monkeypatch.setattr("edgar.MultiFinancials.extract", staticmethod(lambda f: mock_multi))

    provider = CompositeProvider(
        primary=YFinanceAdapter(),
        fallback=EdgarAdapter(cache_store=store),
    )
    result = provider.get_quarterly_financials("AAPL")

    assert not result.empty
    for item in REQUIRED_LINE_ITEMS:
        assert item in result.index, f"Missing required line item: {item}"


def test_cache_key_written_on_miss(store, monkeypatch):
    """First call writes edgar:AAPL:annual to cache."""
    _patch_yfinance_empty(monkeypatch)
    periods = ["2024-09-30", "2023-09-30"]
    mock_company, mock_multi = _make_mock_edgar(_make_stitched_df(periods))
    mock_company_cls = MagicMock(return_value=mock_company)
    monkeypatch.setattr("edgar.Company", mock_company_cls)
    monkeypatch.setattr("edgar.MultiFinancials.extract", staticmethod(lambda f: mock_multi))

    edgar_adapter = EdgarAdapter(cache_store=store)
    provider = CompositeProvider(primary=YFinanceAdapter(), fallback=edgar_adapter)
    provider.get_annual_financials("AAPL")

    assert store.get("edgar:AAPL:annual") is not None


def test_cache_read_on_subsequent_call(store, monkeypatch):
    """Second call reads from cache; EDGAR not queried again."""
    _patch_yfinance_empty(monkeypatch)
    periods = ["2024-09-30", "2023-09-30"]
    mock_company, mock_multi = _make_mock_edgar(_make_stitched_df(periods))
    mock_company_cls = MagicMock(return_value=mock_company)
    monkeypatch.setattr("edgar.Company", mock_company_cls)
    monkeypatch.setattr("edgar.MultiFinancials.extract", staticmethod(lambda f: mock_multi))

    edgar_adapter = EdgarAdapter(cache_store=store)
    provider = CompositeProvider(primary=YFinanceAdapter(), fallback=edgar_adapter)

    result1 = provider.get_annual_financials("AAPL")
    assert mock_company_cls.call_count == 1

    result2 = provider.get_annual_financials("AAPL")
    assert mock_company_cls.call_count == 1, "EDGAR queried on cache hit — should be skipped"
    pd.testing.assert_frame_equal(result1, result2, check_dtype=False)
