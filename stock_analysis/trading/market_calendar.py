from datetime import date, datetime, timedelta, timezone

import pandas_market_calendars as mcal


def _nyse_calendar():
    return mcal.get_calendar("NYSE")


def is_trading_day(day: date) -> bool:
    schedule = _nyse_calendar().schedule(start_date=day, end_date=day)
    return not schedule.empty


def last_trading_day_on_or_before(day: date) -> date | None:
    start = day - timedelta(days=10)
    schedule = _nyse_calendar().schedule(start_date=start, end_date=day)
    if schedule.empty:
        return None
    return schedule.index[-1].date()


def next_trading_day_after(day: date) -> date | None:
    end = day + timedelta(days=10)
    schedule = _nyse_calendar().schedule(start_date=day + timedelta(days=1), end_date=end)
    if schedule.empty:
        return None
    return schedule.index[0].date()


def should_run_today(now: datetime | None = None) -> bool:
    """Run after US close only on days NYSE was open."""
    current = (now or datetime.now(timezone.utc)).date()
    return is_trading_day(current)
