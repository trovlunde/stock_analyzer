import pandas as pd
import pytest
from unittest.mock import MagicMock, call

from stock_analysis.ai.fundamentals import EdgarAdapter
from stock_analysis.storage.cache_store import SqlCacheStore
from stock_analysis.storage.schema import create_engine_for_url, init_db


@pytest.fixture
def store():
    engine = create_engine_for_url("sqlite://")
    init_db(engine)
    return SqlCacheStore(engine)


def _make_stitched_df(periods: list[str]) -> pd.DataFrame:
    rows = [
        ("Revenue", "us-gaap:Revenues", "Revenue", 1),
        ("Net Income", "us-gaap:NetIncomeLoss", "NetIncome", 1),
        ("Operating Income", "us-gaap:OperatingIncomeLoss", "OperatingIncomeLoss", 1),
        ("Depreciation Expense", "us-gaap:Depreciation", "DepreciationExpense", -1),
    ]
    base = [100_000, 20_000, 25_000, 5_000]
    data: dict = {
        "label": [r[0] for r in rows],
        "concept": [r[1] for r in rows],
        "standard_concept": [r[2] for r in rows],
        "preferred_sign": [r[3] for r in rows],
    }
    for i, period in enumerate(periods):
        multiplier = 1.0 + i * 0.1
        data[period] = [int(v * multiplier) for v in base]
    return pd.DataFrame(data)


def _make_mock_multi(stitched_df: pd.DataFrame) -> MagicMock:
    mock_stmt = MagicMock()
    mock_stmt.to_dataframe.return_value = stitched_df
    mock_multi = MagicMock()
    mock_multi.income_statement.return_value = mock_stmt
    return mock_multi


def _make_mock_company(mock_multi: MagicMock) -> tuple[MagicMock, MagicMock]:
    mock_filings = MagicMock()
    mock_filings.empty = False
    mock_filings.head.return_value = mock_filings
    mock_company = MagicMock()
    mock_company.get_filings.return_value = mock_filings
    return mock_company, mock_filings


class TestEdgarAdapterCache:
    def test_annual_cache_miss_fetches_and_puts(self, store, monkeypatch):
        """Cache miss → EDGAR fetch → store.put() → return DataFrame."""
        periods = ["2023-09-30", "2022-09-30"]
        mock_multi = _make_mock_multi(_make_stitched_df(periods))
        mock_company, _ = _make_mock_company(mock_multi)
        monkeypatch.setattr("edgar.Company", MagicMock(return_value=mock_company))
        monkeypatch.setattr("edgar.MultiFinancials.extract", staticmethod(lambda f: mock_multi))

        adapter = EdgarAdapter(cache_store=store)
        df = adapter.get_annual_financials("AAPL")

        assert not df.empty
        assert "Total Revenue" in df.index
        cached = store.get("edgar:AAPL:annual")
        assert cached is not None
        pd.testing.assert_frame_equal(df, cached, check_dtype=False)

    def test_annual_cache_hit_skips_edgar(self, store, monkeypatch):
        """Cache hit → return DataFrame without touching EDGAR."""
        periods = ["2023-09-30", "2022-09-30"]
        mock_multi = _make_mock_multi(_make_stitched_df(periods))
        mock_company, _ = _make_mock_company(mock_multi)
        mock_company_cls = MagicMock(return_value=mock_company)
        monkeypatch.setattr("edgar.Company", mock_company_cls)
        monkeypatch.setattr("edgar.MultiFinancials.extract", staticmethod(lambda f: mock_multi))

        adapter = EdgarAdapter(cache_store=store)
        # First call — populates cache
        df_first = adapter.get_annual_financials("AAPL")
        assert mock_company_cls.call_count == 1

        # Second call — should hit cache, not call edgar.Company again
        df_second = adapter.get_annual_financials("AAPL")
        assert mock_company_cls.call_count == 1
        pd.testing.assert_frame_equal(df_first, df_second, check_dtype=False)

    def test_quarterly_cache_miss_fetches_and_puts(self, store, monkeypatch):
        """Quarterly: cache miss → EDGAR fetch → store.put()."""
        periods = ["2023-09-30", "2023-06-30", "2023-03-31"]
        mock_multi = _make_mock_multi(_make_stitched_df(periods))
        mock_company, _ = _make_mock_company(mock_multi)
        monkeypatch.setattr("edgar.Company", MagicMock(return_value=mock_company))
        monkeypatch.setattr("edgar.MultiFinancials.extract", staticmethod(lambda f: mock_multi))

        adapter = EdgarAdapter(cache_store=store)
        df = adapter.get_quarterly_financials("MSFT")

        assert not df.empty
        cached = store.get("edgar:MSFT:quarterly")
        assert cached is not None
        pd.testing.assert_frame_equal(df, cached, check_dtype=False)

    def test_quarterly_cache_hit_skips_edgar(self, store, monkeypatch):
        """Quarterly: cache hit → no EDGAR call."""
        periods = ["2023-09-30", "2023-06-30"]
        mock_multi = _make_mock_multi(_make_stitched_df(periods))
        mock_company, _ = _make_mock_company(mock_multi)
        mock_company_cls = MagicMock(return_value=mock_company)
        monkeypatch.setattr("edgar.Company", mock_company_cls)
        monkeypatch.setattr("edgar.MultiFinancials.extract", staticmethod(lambda f: mock_multi))

        adapter = EdgarAdapter(cache_store=store)
        adapter.get_quarterly_financials("MSFT")
        assert mock_company_cls.call_count == 1

        adapter.get_quarterly_financials("MSFT")
        assert mock_company_cls.call_count == 1

    def test_cache_key_format_annual(self, store, monkeypatch):
        """Cache key for annual is edgar:{ticker}:annual."""
        periods = ["2023-09-30"]
        mock_multi = _make_mock_multi(_make_stitched_df(periods))
        mock_company, _ = _make_mock_company(mock_multi)
        monkeypatch.setattr("edgar.Company", MagicMock(return_value=mock_company))
        monkeypatch.setattr("edgar.MultiFinancials.extract", staticmethod(lambda f: mock_multi))

        adapter = EdgarAdapter(cache_store=store)
        adapter.get_annual_financials("TSLA")

        assert store.get("edgar:TSLA:annual") is not None
        assert store.get("edgar:TSLA:quarterly") is None

    def test_cache_key_format_quarterly(self, store, monkeypatch):
        """Cache key for quarterly is edgar:{ticker}:quarterly."""
        periods = ["2023-09-30"]
        mock_multi = _make_mock_multi(_make_stitched_df(periods))
        mock_company, _ = _make_mock_company(mock_multi)
        monkeypatch.setattr("edgar.Company", MagicMock(return_value=mock_company))
        monkeypatch.setattr("edgar.MultiFinancials.extract", staticmethod(lambda f: mock_multi))

        adapter = EdgarAdapter(cache_store=store)
        adapter.get_quarterly_financials("TSLA")

        assert store.get("edgar:TSLA:quarterly") is not None
        assert store.get("edgar:TSLA:annual") is None

    def test_empty_result_not_cached(self, store, monkeypatch):
        """Empty DataFrame from EDGAR is not written to cache."""
        mock_filings = MagicMock()
        mock_filings.empty = True
        mock_company = MagicMock()
        mock_company.get_filings.return_value = mock_filings
        monkeypatch.setattr("edgar.Company", MagicMock(return_value=mock_company))

        adapter = EdgarAdapter(cache_store=store)
        df = adapter.get_annual_financials("EMPTY")

        assert df.empty
        assert store.get("edgar:EMPTY:annual") is None
