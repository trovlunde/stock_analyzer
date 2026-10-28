import argparse
import contextlib
import logging
import signal
import sys
import threading
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from stock_analysis.jobs.daily import run_daily
from stock_analysis.notifications import EmailNotifier, load_smtp_config_from_env
from stock_analysis.trading.config import load_portfolio_config
from stock_analysis.trading.market_calendar import is_trading_day

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "config/portfolio.yaml"
DEFAULT_POLL_SECONDS = 60
DEFAULT_HEARTBEAT_INTERVAL_SECONDS = 20.0
HEARTBEAT_PATH = Path("data/heartbeat")


def next_run_at(now: datetime, run_after_close: bool) -> datetime:
    """Next scheduled trigger strictly after `now`, skipping non-trading days. `now` must be tz-aware."""
    hour, minute = (16, 5) if run_after_close else (9, 35)
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    candidate = target if now < target else target + timedelta(days=1)
    while not is_trading_day(candidate.date()):
        candidate += timedelta(days=1)
    return candidate


def _write_heartbeat(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(datetime.now().isoformat(), encoding="utf-8")


@contextlib.contextmanager
def _heartbeat_during(path: Path, interval_seconds: float):
    """Keep touching the heartbeat file while a long-running call executes, so a
    healthcheck watching the heartbeat's age doesn't flag a working run as hung."""
    stop = threading.Event()

    def _loop():
        while not stop.wait(interval_seconds):
            _write_heartbeat(path)

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=5)


def run_forever(
    config_path: str = DEFAULT_CONFIG,
    poll_seconds: int = DEFAULT_POLL_SECONDS,
    heartbeat_path: Path = HEARTBEAT_PATH,
    heartbeat_interval_seconds: float = DEFAULT_HEARTBEAT_INTERVAL_SECONDS,
    stop_event: threading.Event | None = None,
) -> None:
    """Run the daily trading job once per NYSE trading day, forever, until stopped.

    No auto-retry within the same day on failure — a failed run alerts and waits
    for the next scheduled slot, rather than risking a crash-and-retry loop.
    """
    stop_event = stop_event or threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_event.set())
    signal.signal(signal.SIGINT, lambda signum, frame: stop_event.set())

    notifier = EmailNotifier(load_smtp_config_from_env())

    while not stop_event.is_set():
        settings = load_portfolio_config(config_path)
        tz = ZoneInfo(settings.schedule.get("timezone", "America/New_York"))
        run_after_close = settings.schedule.get("run_after_close", True)

        target = next_run_at(datetime.now(tz), run_after_close)
        logger.info("Next scheduled run at %s", target.isoformat())

        while not stop_event.is_set() and datetime.now(tz) < target:
            _write_heartbeat(heartbeat_path)
            stop_event.wait(min(poll_seconds, 30))

        if stop_event.is_set():
            break

        _write_heartbeat(heartbeat_path)
        try:
            with _heartbeat_during(heartbeat_path, heartbeat_interval_seconds):
                run_daily(config_path, dry_run=False, notifier=notifier)
        except Exception as exc:
            logger.exception("Scheduled run failed")
            notifier.send_error_alert("Scheduled trading run failed", exc)

    logger.info("Scheduler stopped")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Long-running scheduler that triggers the daily trading job"
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to portfolio YAML config")
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=DEFAULT_POLL_SECONDS,
        help="Sleep interval while waiting for the next scheduled run",
    )
    args = parser.parse_args(argv)
    run_forever(args.config, poll_seconds=args.poll_seconds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
