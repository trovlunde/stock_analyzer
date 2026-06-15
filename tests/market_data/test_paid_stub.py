import pandas as pd
import pytest

from stock_analysis.market_data import PaidMarketDataStub


def test_paid_stub_raises_not_implemented_on_history():
    stub = PaidMarketDataStub()
    with pytest.raises(NotImplementedError, match="FMP, Polygon, Stooq"):
        stub.get_history("AAPL")


def test_paid_stub_raises_not_implemented_on_financials():
    stub = PaidMarketDataStub()
    with pytest.raises(NotImplementedError, match="FMP, Polygon, Stooq"):
        stub.get_financials("AAPL")


def test_paid_stub_raises_not_implemented_on_earnings():
    stub = PaidMarketDataStub()
    with pytest.raises(NotImplementedError, match="FMP, Polygon, Stooq"):
        stub.get_earnings("AAPL")


def test_get_sp500_data_delegates_to_provider():
    from stock_analysis.ai import helpers as helpers_mod

    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    expected = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=dates)

    class _Provider:
        def get_history(self, ticker, period="20y"):
            assert ticker == "^GSPC"
            assert period == "20y"
            return expected

    result = helpers_mod.get_sp500_data(provider=_Provider())
    pd.testing.assert_frame_equal(result, expected)
