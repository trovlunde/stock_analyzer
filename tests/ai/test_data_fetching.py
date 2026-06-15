from datetime import date, timedelta

import pandas as pd
import pytest

from stock_analysis.ai.finviz_classifier.data_fetching import (
    _cache_entry_usable,
    compute_returns_from_history,
    deserialize_daily_returns,
    fetch_ticker_history,
    serialize_daily_returns,
)


def _make_hist(rows):
    """Build a minimal OHLCV history DataFrame."""
    index = pd.to_datetime([r[0] for r in rows])
    return pd.DataFrame(
        {
            'Open': [r[1] for r in rows],
            'High': [r[2] for r in rows],
            'Low': [r[3] for r in rows],
            'Close': [r[4] for r in rows],
        },
        index=index,
    )


def test_serialize_daily_returns_roundtrip():
    daily_returns = {
        0: {'close_return': 0.01, 'high_return': 0.02, 'date': date(2025, 1, 15)},
        1: {'close_return': 0.03, 'high_return': 0.04, 'date': date(2025, 1, 16)},
    }
    restored = deserialize_daily_returns(serialize_daily_returns(daily_returns))
    assert restored == daily_returns


def test_compute_returns_from_history():
    signal_date = date(2025, 1, 15)
    hist = _make_hist([
        ('2025-01-14', 90, 91, 89, 90),
        ('2025-01-15', 100, 110, 95, 105),
        ('2025-01-16', 105, 112, 104, 108),
        ('2025-01-17', 108, 109, 107, 108),
        ('2025-01-21', 108, 115, 108, 114),
        ('2025-01-22', 114, 120, 113, 118),
        ('2025-01-23', 118, 119, 117, 118),
        ('2025-01-24', 118, 122, 118, 121),
    ])

    standard, detailed, is_complete = compute_returns_from_history(
        'TEST', signal_date, hist)

    assert standard is not None
    assert standard['Ticker'] == 'TEST'
    assert standard['next_day_return'] == pytest.approx(0.05)
    assert standard['next_day_high_return'] == pytest.approx(0.10)
    assert detailed['daily_returns'][0]['close_return'] == pytest.approx(0.05)
    assert detailed['daily_returns'][0]['high_return'] == pytest.approx(0.10)
    assert is_complete is True


def test_cache_entry_usable_requires_daily_returns_json():
    from datetime import datetime

    row = pd.DataFrame([{
        'Ticker': 'AAPL',
        'timestamp': pd.Timestamp('2025-01-15'),
        'is_complete': True,
        'daily_returns_json': None,
    }])
    assert _cache_entry_usable(
        row, pd.Timestamp('2025-01-15'), datetime(2025, 1, 20)) is False

    row.iloc[0, row.columns.get_loc('daily_returns_json')] = (
        serialize_daily_returns({
            0: {'close_return': 0.01, 'high_return': 0.02, 'date': date(2025, 1, 15)},
        }))
    assert _cache_entry_usable(
        row, pd.Timestamp('2025-01-15'), datetime(2025, 1, 20)) is True


def test_fetch_ticker_history_provider_injection():
    hist = _make_hist([
        ('2025-01-15', 100, 110, 95, 105),
        ('2025-01-16', 105, 112, 104, 108),
    ])

    class _MockTicker:
        def __init__(self, df):
            self._df = df
            self.calls = []

        def history(self, start=None, end=None, interval=None):
            self.calls.append((start, end, interval))
            return self._df

    mock_ticker = _MockTicker(hist)

    class _MockProvider:
        def get_raw_ticker(self, ticker):
            return mock_ticker

    start = date(2025, 1, 15)
    end = date(2025, 1, 16)
    result = fetch_ticker_history('AAPL', start, end, provider=_MockProvider())

    assert result is not None
    assert set(['Open', 'High', 'Low', 'Close']).issubset(result.columns)
    assert mock_ticker.calls[0][2] == '1d'
    assert mock_ticker.calls[0][1] == end + timedelta(days=1)
