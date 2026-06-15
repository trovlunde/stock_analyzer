import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from stock_analysis.ai.fundamentals import EdgarAdapter, FundamentalsProvider
from stock_analysis.storage.cache_store import SqlCacheStore
from stock_analysis.storage.schema import create_engine_for_url, init_db


def _make_in_memory_store() -> SqlCacheStore:
    engine = create_engine_for_url("sqlite://")
    init_db(engine)
    return SqlCacheStore(engine)


def _make_stitched_df(periods: list[str]) -> pd.DataFrame:
    """Build a DataFrame matching StitchedStatement.to_dataframe() output."""
    rows = [
        ("Revenue", "us-gaap:Revenues", "Revenue", 1),
        ("Net Income", "us-gaap:NetIncomeLoss", "NetIncome", 1),
        ("Gross Profit", "us-gaap:GrossProfit", "GrossProfit", 1),
        ("Operating Income", "us-gaap:OperatingIncomeLoss", "OperatingIncomeLoss", 1),
        ("Cost of Revenue", "us-gaap:CostOfGoodsSold", "CostOfGoodsAndServicesSold", -1),
        ("Depreciation Expense", "us-gaap:Depreciation", "DepreciationExpense", -1),
    ]
    base = [100_000, 20_000, 40_000, 25_000, 60_000, 5_000]
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


def _mock_edgar(monkeypatch, form: str, stitched_df: pd.DataFrame) -> None:
    """Patch edgar.Company so no SEC HTTP calls happen."""
    mock_stmt = MagicMock()
    mock_stmt.to_dataframe.return_value = stitched_df

    mock_multi = MagicMock()
    mock_multi.income_statement.return_value = mock_stmt

    mock_filings = MagicMock()
    mock_filings.empty = False
    mock_filings.head.return_value = mock_filings

    mock_company = MagicMock()
    mock_company.get_filings.return_value = mock_filings

    mock_company_cls = MagicMock(return_value=mock_company)

    monkeypatch.setattr("edgar.Company", mock_company_cls)
    monkeypatch.setattr("edgar.MultiFinancials.extract", staticmethod(lambda f: mock_multi))
    monkeypatch.setattr(
        "stock_analysis.ai.fundamentals.edgar_adapter._get_cache_store",
        _make_in_memory_store,
    )


class TestEdgarAdapter:
    def test_annual_financials_returns_mapped_df(self, monkeypatch):
        periods = ["2023-09-30", "2022-09-30", "2021-09-30"]
        _mock_edgar(monkeypatch, "10-K", _make_stitched_df(periods))

        adapter = EdgarAdapter()
        df = adapter.get_annual_financials("AAPL")

        assert not df.empty
        assert "Total Revenue" in df.index
        assert "Net Income" in df.index
        assert "Gross Profit" in df.index
        assert "Operating Income" in df.index
        assert "Cost Of Revenue" in df.index
        assert "Depreciation" in df.index

    def test_annual_financials_derives_ebit_ebitda(self, monkeypatch):
        periods = ["2023-09-30", "2022-09-30"]
        _mock_edgar(monkeypatch, "10-K", _make_stitched_df(periods))

        adapter = EdgarAdapter()
        df = adapter.get_annual_financials("AAPL")

        assert "EBIT" in df.index
        assert "EBITDA" in df.index
        # EBIT == Operating Income
        assert (df.loc["EBIT"] == df.loc["Operating Income"]).all()
        # EBITDA == Operating Income + Depreciation
        expected_ebitda = df.loc["Operating Income"] + df.loc["Depreciation"]
        pd.testing.assert_series_equal(df.loc["EBITDA"], expected_ebitda, check_names=False)

    def test_quarterly_financials_returns_mapped_df(self, monkeypatch):
        periods = ["2023-09-30", "2023-06-30", "2023-03-31", "2022-12-31"]
        _mock_edgar(monkeypatch, "10-Q", _make_stitched_df(periods))

        adapter = EdgarAdapter()
        df = adapter.get_quarterly_financials("AAPL")

        assert not df.empty
        assert "Total Revenue" in df.index

    def test_columns_are_timestamps(self, monkeypatch):
        periods = ["2023-09-30", "2022-09-30"]
        _mock_edgar(monkeypatch, "10-K", _make_stitched_df(periods))

        adapter = EdgarAdapter()
        df = adapter.get_annual_financials("AAPL")

        assert all(isinstance(c, pd.Timestamp) for c in df.columns)

    def test_orientation_line_items_as_index(self, monkeypatch):
        periods = ["2023-09-30"]
        _mock_edgar(monkeypatch, "10-K", _make_stitched_df(periods))

        adapter = EdgarAdapter()
        df = adapter.get_annual_financials("AAPL")

        # index = line items, columns = period timestamps
        assert isinstance(df.index[0], str)
        assert isinstance(df.columns[0], pd.Timestamp)

    def test_satisfies_provider_protocol(self):
        adapter = EdgarAdapter()
        assert isinstance(adapter, FundamentalsProvider)

    def test_empty_filings_returns_empty_df(self, monkeypatch):
        mock_filings = MagicMock()
        mock_filings.empty = True

        mock_company = MagicMock()
        mock_company.get_filings.return_value = mock_filings
        monkeypatch.setattr("edgar.Company", MagicMock(return_value=mock_company))
        monkeypatch.setattr(
            "stock_analysis.ai.fundamentals.edgar_adapter._get_cache_store",
            _make_in_memory_store,
        )

        adapter = EdgarAdapter()
        df = adapter.get_annual_financials("UNKNOWN")
        assert df.empty

    def test_company_not_found_returns_empty_df(self, monkeypatch):
        import edgar as _edgar

        monkeypatch.setattr(
            "edgar.Company",
            MagicMock(side_effect=_edgar.CompanyNotFoundError("UNKNOWN")),
        )
        monkeypatch.setattr(
            "stock_analysis.ai.fundamentals.edgar_adapter._get_cache_store",
            _make_in_memory_store,
        )

        adapter = EdgarAdapter()
        df = adapter.get_annual_financials("UNKNOWN")
        assert df.empty
