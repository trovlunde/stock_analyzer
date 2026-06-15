import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from stock_analysis.ai.helpers import get_index_data, get_ticker_data, get_ticker_financials


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


def _mock_provider(history=None, financials=None):
    m = MagicMock()
    m.get_history.return_value = _OHLCV if history is None else history
    m.get_financials.return_value = pd.DataFrame() if financials is None else financials
    return m


def _mock_store(cached=None):
    store = MagicMock()
    store.get.return_value = cached
    return store


def test_get_index_data_cache_hit_skips_provider():
    provider = _mock_provider()
    with patch("stock_analysis.ai.helpers.get_cache_store") as mock_store_fn:
        mock_store_fn.return_value = _mock_store(cached=_OHLCV)
        result = get_index_data("^GSPC", provider=provider)

    provider.get_history.assert_not_called()
    pd.testing.assert_frame_equal(result, _OHLCV)


def test_get_index_data_cache_miss_calls_provider_and_stores():
    provider = _mock_provider()
    with patch("stock_analysis.ai.helpers.get_cache_store") as mock_store_fn:
        store = _mock_store(cached=None)
        mock_store_fn.return_value = store
        result = get_index_data("^GSPC", provider=provider)

    provider.get_history.assert_called()
    store.put.assert_called_once()
    pd.testing.assert_frame_equal(result, _OHLCV)


def test_get_index_data_injects_custom_provider():
    custom = _mock_provider()
    default = _mock_provider()
    with patch("stock_analysis.ai.helpers.get_cache_store") as mock_store_fn:
        mock_store_fn.return_value = _mock_store(cached=None)
        with patch("stock_analysis.ai.helpers._default_provider", default):
            get_index_data("AAPL", provider=custom)

    custom.get_history.assert_called()
    default.get_history.assert_not_called()


def test_get_index_data_uses_default_provider_when_none():
    default = _mock_provider()
    with patch("stock_analysis.ai.helpers.get_cache_store") as mock_store_fn:
        mock_store_fn.return_value = _mock_store(cached=None)
        with patch("stock_analysis.ai.helpers._default_provider", default):
            get_index_data("AAPL")

    default.get_history.assert_called()


def test_get_ticker_data_calls_provider_with_20y():
    provider = _mock_provider()
    result = get_ticker_data("AAPL", provider=provider)

    provider.get_history.assert_called_once_with("AAPL", period="20y")
    pd.testing.assert_frame_equal(result, _OHLCV)


def test_get_ticker_financials_injects_provider():
    financials = pd.DataFrame({"Revenue": [1e9]})
    provider = _mock_provider(financials=financials)

    result = get_ticker_financials("AAPL", provider=provider)

    provider.get_financials.assert_called_once_with("AAPL")
    pd.testing.assert_frame_equal(result, financials)
