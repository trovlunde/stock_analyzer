import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

from stock_analysis.jobs import scheduler

_TZ = ZoneInfo("America/New_York")


def test_next_run_at_same_day_before_close(monkeypatch):
    monkeypatch.setattr(scheduler, "is_trading_day", lambda d: True)
    now = datetime(2024, 6, 3, 10, 0, tzinfo=_TZ)  # Monday, before 16:05

    target = scheduler.next_run_at(now, run_after_close=True)

    assert target == datetime(2024, 6, 3, 16, 5, tzinfo=_TZ)


def test_next_run_at_rolls_to_next_day_after_close(monkeypatch):
    monkeypatch.setattr(scheduler, "is_trading_day", lambda d: True)
    now = datetime(2024, 6, 3, 17, 0, tzinfo=_TZ)  # after today's 16:05 slot

    target = scheduler.next_run_at(now, run_after_close=True)

    assert target == datetime(2024, 6, 4, 16, 5, tzinfo=_TZ)


def test_next_run_at_skips_non_trading_days(monkeypatch):
    # Only the 3rd calendar day checked is a trading day.
    calls = []

    def fake_is_trading_day(d):
        calls.append(d)
        return len(calls) >= 3

    monkeypatch.setattr(scheduler, "is_trading_day", fake_is_trading_day)
    now = datetime(2024, 6, 3, 10, 0, tzinfo=_TZ)

    target = scheduler.next_run_at(now, run_after_close=True)

    assert len(calls) == 3
    assert target.date() == datetime(2024, 6, 5).date()


def test_run_forever_triggers_run_daily_once_then_stops(monkeypatch, tmp_path):
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        "portfolio:\n  name: x\nschedule:\n  timezone: America/New_York\n  run_after_close: true\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        scheduler,
        "load_portfolio_config",
        lambda path: MagicMock(schedule={"timezone": "America/New_York", "run_after_close": True}),
    )
    # Target already in the past -> run_daily fires immediately, no waiting.
    monkeypatch.setattr(
        scheduler, "next_run_at", lambda now, run_after_close: now
    )

    stop_event = threading.Event()
    mock_run_daily = MagicMock(side_effect=lambda *a, **kw: stop_event.set())
    monkeypatch.setattr(scheduler, "run_daily", mock_run_daily)
    monkeypatch.setattr(scheduler, "EmailNotifier", MagicMock())
    monkeypatch.setattr(scheduler, "load_smtp_config_from_env", lambda: None)

    heartbeat_path = tmp_path / "heartbeat"
    scheduler.run_forever(
        str(config_path), heartbeat_path=heartbeat_path, stop_event=stop_event
    )

    assert mock_run_daily.call_count == 1
    assert heartbeat_path.exists()


def test_run_forever_refreshes_heartbeat_during_long_run(monkeypatch, tmp_path):
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text("portfolio:\n  name: x\n", encoding="utf-8")
    monkeypatch.setattr(
        scheduler, "load_portfolio_config", lambda path: MagicMock(schedule={})
    )
    monkeypatch.setattr(scheduler, "next_run_at", lambda now, run_after_close: now)
    monkeypatch.setattr(scheduler, "EmailNotifier", MagicMock())
    monkeypatch.setattr(scheduler, "load_smtp_config_from_env", lambda: None)

    heartbeat_path = tmp_path / "heartbeat"
    stop_event = threading.Event()

    def _slow_run_daily(*args, **kwargs):
        import time

        initial = heartbeat_path.read_text(encoding="utf-8")
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if heartbeat_path.read_text(encoding="utf-8") != initial:
                break
            time.sleep(0.02)
        else:
            pytest.fail("heartbeat was never refreshed during the long-running call")
        stop_event.set()

    monkeypatch.setattr(scheduler, "run_daily", _slow_run_daily)

    scheduler.run_forever(
        str(config_path),
        heartbeat_path=heartbeat_path,
        heartbeat_interval_seconds=0.05,
        stop_event=stop_event,
    )


def test_run_forever_swallows_run_daily_exception(monkeypatch, tmp_path):
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text("portfolio:\n  name: x\n", encoding="utf-8")
    monkeypatch.setattr(
        scheduler, "load_portfolio_config", lambda path: MagicMock(schedule={})
    )
    monkeypatch.setattr(scheduler, "next_run_at", lambda now, run_after_close: now)

    stop_event = threading.Event()

    def _raise_then_stop(*args, **kwargs):
        stop_event.set()
        raise RuntimeError("boom")

    mock_run_daily = MagicMock(side_effect=_raise_then_stop)
    mock_notifier_instance = MagicMock()
    monkeypatch.setattr(scheduler, "run_daily", mock_run_daily)
    monkeypatch.setattr(scheduler, "EmailNotifier", lambda config: mock_notifier_instance)
    monkeypatch.setattr(scheduler, "load_smtp_config_from_env", lambda: None)

    # Must not raise even though run_daily raises.
    scheduler.run_forever(
        str(config_path), heartbeat_path=tmp_path / "heartbeat", stop_event=stop_event
    )

    mock_notifier_instance.send_error_alert.assert_called_once()


def test_run_forever_runs_every_config_even_if_one_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(
        scheduler, "load_portfolio_config", lambda path: MagicMock(schedule={})
    )
    monkeypatch.setattr(scheduler, "next_run_at", lambda now, run_after_close: now)

    stop_event = threading.Event()
    ran = []

    def _run(path, **kwargs):
        ran.append(path)
        if path == "first.yaml":
            raise RuntimeError("boom")
        stop_event.set()

    notifier = MagicMock()
    monkeypatch.setattr(scheduler, "run_daily", _run)
    monkeypatch.setattr(scheduler, "EmailNotifier", lambda config: notifier)
    monkeypatch.setattr(scheduler, "load_smtp_config_from_env", lambda: None)

    scheduler.run_forever(
        ["first.yaml", "second.yaml"],
        heartbeat_path=tmp_path / "heartbeat",
        stop_event=stop_event,
    )

    assert ran == ["first.yaml", "second.yaml"]
    notifier.send_error_alert.assert_called_once()
    assert "first.yaml" in notifier.send_error_alert.call_args.args[0]


def test_run_forever_skips_and_alerts_on_a_slot_missed_by_too_much(monkeypatch, tmp_path):
    """E.g. the host slept through 16:05 and woke the next morning — running then
    would trade on a stale session (and a session date already rolled over in UTC)."""
    monkeypatch.setattr(
        scheduler, "load_portfolio_config", lambda path: MagicMock(schedule={})
    )
    stop_event = threading.Event()
    targets = iter(["missed", "future"])

    def _next_run_at(now, run_after_close):
        if next(targets) == "missed":
            return now - scheduler.MAX_RUN_LATENESS - timedelta(minutes=1)
        stop_event.set()
        return now + timedelta(days=1)

    monkeypatch.setattr(scheduler, "next_run_at", _next_run_at)
    mock_run_daily = MagicMock()
    mock_notifier_instance = MagicMock()
    monkeypatch.setattr(scheduler, "run_daily", mock_run_daily)
    monkeypatch.setattr(scheduler, "EmailNotifier", lambda config: mock_notifier_instance)
    monkeypatch.setattr(scheduler, "load_smtp_config_from_env", lambda: None)

    scheduler.run_forever(
        "unused.yaml", heartbeat_path=tmp_path / "heartbeat", stop_event=stop_event
    )

    mock_run_daily.assert_not_called()
    mock_notifier_instance.send_error_alert.assert_called_once()


def test_run_forever_exits_immediately_when_stop_already_set(monkeypatch, tmp_path):
    monkeypatch.setattr(scheduler, "EmailNotifier", MagicMock())
    monkeypatch.setattr(scheduler, "load_smtp_config_from_env", lambda: None)
    mock_run_daily = MagicMock()
    monkeypatch.setattr(scheduler, "run_daily", mock_run_daily)

    stop_event = threading.Event()
    stop_event.set()

    scheduler.run_forever(
        "unused.yaml", heartbeat_path=tmp_path / "heartbeat", stop_event=stop_event
    )

    mock_run_daily.assert_not_called()
