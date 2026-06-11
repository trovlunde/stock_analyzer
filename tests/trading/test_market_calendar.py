from datetime import date

from stock_analysis.trading.market_calendar import (
    is_trading_day,
    next_trading_day_after,
)


def test_weekend_not_trading_day():
    assert is_trading_day(date(2024, 6, 1)) is False  # Saturday


def test_weekday_trading_day():
    assert is_trading_day(date(2024, 6, 3)) is True  # Monday


def test_next_trading_day_after_friday():
    assert next_trading_day_after(date(2024, 6, 7)) == date(2024, 6, 10)
