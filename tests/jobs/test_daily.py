from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import select, update

from stock_analysis.jobs import daily
from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.storage.schema import create_engine_for_url, init_db, paper_runs, paper_signals
from stock_analysis.trading.paper_broker import PaperBroker
from stock_analysis.trading.signals import (
    SignalResult,
    register_signal_strategy,
    unregister_signal_strategy,
)


def _backdate_run_started_at(engine, run_id: int, hours_ago: float) -> None:
    started_at = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    with engine.begin() as conn:
        conn.execute(update(paper_runs).where(paper_runs.c.id == run_id).values(started_at=started_at))

CONFIG_TEMPLATE = """
portfolio:
  name: test-portfolio
  enabled: true
  initial_cash: 10000
  max_positions: 10
  position_size_pct: 0.10
  long_only: true

schedule:
  timezone: America/New_York
  run_after_close: true

costs:
  model: equity
  commission_per_trade: 1.00
  slippage_pct: 0.0005

defaults:
  strategy: {strategy}
  ma_short: 20
  ma_long: 50

positions:
{positions}
"""


class _FakeProvider:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self._frames = frames

    def get_history(self, ticker, period="1y", start=None, end=None):
        return self._frames.get(ticker, pd.DataFrame())

    def get_financials(self, ticker):
        return pd.DataFrame()

    def get_earnings(self, ticker):
        return pd.DataFrame()


class _HoldStrategy:
    def evaluate(self, df, ticker, params):
        return SignalResult(action="HOLD", reason="test hold", metadata={})


class _BuyStrategy:
    def evaluate(self, df, ticker, params):
        return SignalResult(action="BUY", reason="test buy", metadata={})


class _RaisingStrategy:
    def evaluate(self, df, ticker, params):
        raise RuntimeError("boom")


class _SellStrategy:
    def evaluate(self, df, ticker, params):
        return SignalResult(action="SELL", reason="test sell", metadata={})


def _ohlcv(n=5, price=100.0):
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": [price] * n,
            "High": [price * 1.01] * n,
            "Low": [price * 0.99] * n,
            "Close": [price] * n,
            "Volume": [1_000_000] * n,
        },
        index=index,
    )


@pytest.fixture
def engine(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return engine


@pytest.fixture
def write_config(tmp_path):
    def _write(strategy="hold_test", tickers=("AAPL",)):
        positions = "\n".join(f"  - ticker: {t}" for t in tickers)
        config_path = tmp_path / "portfolio.yaml"
        config_path.write_text(
            CONFIG_TEMPLATE.format(strategy=strategy, positions=positions),
            encoding="utf-8",
        )
        return str(config_path)

    return _write


@pytest.fixture(autouse=True)
def _register_test_strategies():
    register_signal_strategy("hold_test", _HoldStrategy())
    register_signal_strategy("buy_test", _BuyStrategy())
    register_signal_strategy("raising_test", _RaisingStrategy())
    yield
    for name in ("hold_test", "buy_test", "raising_test"):
        unregister_signal_strategy(name)


@pytest.fixture(autouse=True)
def _always_trading_day(monkeypatch):
    monkeypatch.setattr(daily, "is_trading_day", lambda d: True)
    monkeypatch.setattr(daily, "next_trading_day_after", lambda d: date(2024, 1, 2))


def test_run_daily_skips_on_non_trading_day(monkeypatch, write_config, engine, tmp_path):
    monkeypatch.setattr(daily, "is_trading_day", lambda d: False)
    monkeypatch.chdir(tmp_path)
    config_path = write_config()

    result = daily.run_daily(
        config_path, provider=_FakeProvider({}), engine=engine
    )

    assert result == 0
    report_path = tmp_path / "data" / "reports"
    files = list(report_path.glob("*.md"))
    assert len(files) == 1
    assert "NYSE closed" in files[0].read_text(encoding="utf-8")

    store = PortfolioStore(engine)
    assert store.get_portfolio_by_name("test-portfolio") is None


def test_run_daily_dry_run_does_not_mutate_state(write_config, engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = write_config(strategy="buy_test")
    provider = _FakeProvider({"AAPL": _ohlcv()})

    result = daily.run_daily(config_path, dry_run=True, provider=provider, engine=engine)

    assert result == 0
    store = PortfolioStore(engine)
    assert store.get_portfolio_by_name("test-portfolio") is None


def test_run_daily_dry_run_reflects_already_tripped_kill_switch(
    write_config, engine, tmp_path, monkeypatch
):
    """A dry-run preview must show the same BUY-blocking an operator would see on a
    real run — and must not itself clear or alter the persisted trip."""
    monkeypatch.chdir(tmp_path)
    config_path = write_config(strategy="buy_test")
    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("test-portfolio", 10_000)
    store.trip_kill_switch(state.id, "manually tripped for test")
    provider = _FakeProvider({"AAPL": _ohlcv()})

    result = daily.run_daily(config_path, dry_run=True, provider=provider, engine=engine)

    assert result == 0
    report_path = tmp_path / "data" / "reports" / f"{daily._session_date().isoformat()}.md"
    content = report_path.read_text(encoding="utf-8")
    assert "kill switch" in content.lower()
    tripped_at, reason = store.get_kill_switch_state(state.id)
    assert tripped_at is not None
    assert reason == "manually tripped for test"


def test_run_daily_normal_run_queues_buy_order(write_config, engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = write_config(strategy="buy_test")
    provider = _FakeProvider({"AAPL": _ohlcv()})

    result = daily.run_daily(config_path, provider=provider, engine=engine)

    assert result == 0
    store = PortfolioStore(engine)
    state = store.get_portfolio_by_name("test-portfolio")
    assert state is not None
    pending = store.get_pending_orders(state.id)
    assert len(pending) == 1
    assert pending[0].ticker == "AAPL"
    assert pending[0].side == "BUY"


def test_run_daily_missing_price_data_skips_ticker(write_config, engine, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config_path = write_config(strategy="buy_test", tickers=("AAPL", "MSFT"))
    # MSFT has no data at all — provider returns an empty frame for it.
    provider = _FakeProvider({"AAPL": _ohlcv()})

    result = daily.run_daily(config_path, provider=provider, engine=engine)

    assert result == 0
    store = PortfolioStore(engine)
    state = store.get_portfolio_by_name("test-portfolio")
    pending = store.get_pending_orders(state.id)
    assert [o.ticker for o in pending] == ["AAPL"]


def test_run_daily_strategy_error_skips_only_that_ticker(
    write_config, engine, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(
            strategy="buy_test",
            positions="  - ticker: AAPL\n  - ticker: MSFT\n    strategy: raising_test\n",
        ),
        encoding="utf-8",
    )
    provider = _FakeProvider({"AAPL": _ohlcv(), "MSFT": _ohlcv()})

    result = daily.run_daily(str(config_path), provider=provider, engine=engine)

    assert result == 0
    store = PortfolioStore(engine)
    state = store.get_portfolio_by_name("test-portfolio")
    pending = store.get_pending_orders(state.id)
    assert [o.ticker for o in pending] == ["AAPL"]


def test_run_daily_broker_error_after_decision_skips_only_that_ticker(
    write_config, engine, tmp_path, monkeypatch
):
    """A failure after decide() (e.g. in broker.queue_order) must not abort other tickers
    or leave the run half-finished — the try/except around the per-ticker body must cover
    order submission too, not just signal evaluation."""
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(
            strategy="buy_test",
            positions="  - ticker: AAPL\n  - ticker: MSFT\n",
        ),
        encoding="utf-8",
    )
    provider = _FakeProvider({"AAPL": _ohlcv(), "MSFT": _ohlcv()})

    original_queue_order = PaperBroker.queue_order

    def flaky_queue_order(self, state, run_id, intent, fill_date, dry_run):
        if intent.ticker == "MSFT":
            raise RuntimeError("broker boom")
        return original_queue_order(
            self, state, run_id, intent, fill_date=fill_date, dry_run=dry_run
        )

    monkeypatch.setattr(PaperBroker, "queue_order", flaky_queue_order)

    result = daily.run_daily(str(config_path), provider=provider, engine=engine)

    assert result == 0
    store = PortfolioStore(engine)
    state = store.get_portfolio_by_name("test-portfolio")
    assert state is not None
    # The run must have completed (not aborted) — the portfolio row exists and
    # the surviving ticker's order was still queued.
    pending = store.get_pending_orders(state.id)
    assert [o.ticker for o in pending] == ["AAPL"]


def test_run_daily_all_tickers_failed_sends_error_alert(
    write_config, engine, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(
            strategy="raising_test",
            positions="  - ticker: AAPL\n  - ticker: MSFT\n",
        ),
        encoding="utf-8",
    )
    provider = _FakeProvider({"AAPL": _ohlcv(), "MSFT": _ohlcv()})
    notifier = MagicMock()

    result = daily.run_daily(
        str(config_path), provider=provider, engine=engine, notifier=notifier
    )

    assert result == 0
    notifier.send_error_alert.assert_called_once()
    subject = notifier.send_error_alert.call_args[0][0]
    assert "all tickers failed" in subject.lower()


def test_run_daily_kill_switch_trips_on_daily_loss_and_blocks_buy(
    engine, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(strategy="buy_test", positions="  - ticker: AAPL\n")
        + "\nrisk:\n  max_daily_loss_pct: 0.05\n",
        encoding="utf-8",
    )
    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("test-portfolio", 10_000)
    seed_run = store.start_run(state.id, dry_run=False)
    store.finish_run(
        seed_run, "completed", {"session_date": "2024-01-01", "portfolio_value": 10_000}
    )
    store.update_cash(state.id, 9_000)  # simulate a 10% loss already realized

    provider = _FakeProvider({"AAPL": _ohlcv()})
    notifier = MagicMock()

    result = daily.run_daily(
        str(config_path), provider=provider, engine=engine, notifier=notifier
    )

    assert result == 0
    tripped_at, reason = store.get_kill_switch_state(state.id)
    assert tripped_at is not None
    assert "daily loss" in reason
    assert store.get_pending_orders(state.id) == []  # BUY signal fired but was blocked
    notifier.send_error_alert.assert_called_once()


def test_run_daily_respects_already_tripped_kill_switch_without_realerting(
    write_config, engine, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = write_config(strategy="buy_test")
    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("test-portfolio", 10_000)
    store.trip_kill_switch(state.id, "manually tripped for test")

    provider = _FakeProvider({"AAPL": _ohlcv()})
    notifier = MagicMock()

    result = daily.run_daily(
        str(config_path), provider=provider, engine=engine, notifier=notifier
    )

    assert result == 0
    assert store.get_pending_orders(state.id) == []
    notifier.send_error_alert.assert_not_called()
    _tripped_at, reason = store.get_kill_switch_state(state.id)
    assert reason == "manually tripped for test"


def test_run_daily_detects_unclean_shutdown_and_trips_kill_switch(
    write_config, engine, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = write_config(strategy="buy_test")
    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("test-portfolio", 10_000)
    stale_run_id = store.start_run(state.id, dry_run=False)  # simulates a crashed run
    _backdate_run_started_at(engine, stale_run_id, hours_ago=7)

    provider = _FakeProvider({"AAPL": _ohlcv()})
    notifier = MagicMock()

    result = daily.run_daily(
        str(config_path), provider=provider, engine=engine, notifier=notifier
    )

    assert result == 0
    assert store.get_running_run_ids(state.id) == []
    tripped_at, reason = store.get_kill_switch_state(state.id)
    assert tripped_at is not None
    assert "unclean shutdown" in reason
    assert store.get_pending_orders(state.id) == []
    notifier.send_error_alert.assert_called_once()


def test_run_daily_does_not_flag_a_recently_started_run_as_orphaned(
    write_config, engine, tmp_path, monkeypatch
):
    """A still-running row from moments ago (e.g. an overlapping deploy) must not be
    mistaken for a crash — only a run stale well beyond normal duration should trip."""
    monkeypatch.chdir(tmp_path)
    config_path = write_config(strategy="buy_test")
    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("test-portfolio", 10_000)
    store.start_run(state.id, dry_run=False)  # started "just now" — not stale

    provider = _FakeProvider({"AAPL": _ohlcv()})
    notifier = MagicMock()

    result = daily.run_daily(
        str(config_path), provider=provider, engine=engine, notifier=notifier
    )

    assert result == 0
    tripped_at, _reason = store.get_kill_switch_state(state.id)
    assert tripped_at is None
    notifier.send_error_alert.assert_not_called()


def _signals_for_run(engine, run_id: int) -> list[tuple[str, str]]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(paper_signals.c.reason).where(paper_signals.c.run_id == run_id)
        ).all()
    return [row[0] for row in rows]


def test_run_daily_multi_strategy_records_each_signal_and_the_combined_result(
    tmp_path, engine, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(
            strategy="buy_test",  # unused — position below overrides with `strategies`
            positions="  - ticker: AAPL\n    strategies: [buy_test, hold_test]\n",
        ),
        encoding="utf-8",
    )
    provider = _FakeProvider({"AAPL": _ohlcv()})

    result = daily.run_daily(str(config_path), provider=provider, engine=engine)

    assert result == 0
    store = PortfolioStore(engine)
    state = store.get_portfolio_by_name("test-portfolio")

    reasons = _signals_for_run(engine, run_id=1)
    assert any(r.startswith("[buy_test]") for r in reasons)
    assert any(r.startswith("[hold_test]") for r in reasons)
    assert any(r.startswith("[combined:consensus]") for r in reasons)
    assert len(reasons) == 3

    # buy_test is the only actionable signal (hold_test abstains) -> consensus
    # passes it through, so the order still gets queued.
    pending = store.get_pending_orders(state.id)
    assert [o.ticker for o in pending] == ["AAPL"]


def test_run_daily_multi_strategy_conflict_blocks_the_order(tmp_path, engine, monkeypatch):
    register_signal_strategy("sell_test", _SellStrategy())
    try:
        monkeypatch.chdir(tmp_path)
        config_path = tmp_path / "portfolio.yaml"
        config_path.write_text(
            CONFIG_TEMPLATE.format(
                strategy="buy_test",
                positions="  - ticker: AAPL\n    strategies: [buy_test, sell_test]\n",
            ),
            encoding="utf-8",
        )
        provider = _FakeProvider({"AAPL": _ohlcv()})

        result = daily.run_daily(str(config_path), provider=provider, engine=engine)

        assert result == 0
        store = PortfolioStore(engine)
        state = store.get_portfolio_by_name("test-portfolio")
        assert store.get_pending_orders(state.id) == []

        reasons = _signals_for_run(engine, run_id=1)
        assert any("conflict" in r for r in reasons if r.startswith("[combined:"))
    finally:
        unregister_signal_strategy("sell_test")


def test_run_daily_multi_strategy_kill_switch_still_blocks_combined_buy(
    write_config, engine, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        CONFIG_TEMPLATE.format(
            strategy="buy_test",
            positions="  - ticker: AAPL\n    strategies: [buy_test, hold_test]\n",
        ),
        encoding="utf-8",
    )
    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("test-portfolio", 10_000)
    store.trip_kill_switch(state.id, "manually tripped for test")

    provider = _FakeProvider({"AAPL": _ohlcv()})

    result = daily.run_daily(str(config_path), provider=provider, engine=engine)

    assert result == 0
    assert store.get_pending_orders(state.id) == []
