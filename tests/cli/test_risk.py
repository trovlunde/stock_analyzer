import pytest

from stock_analysis.cli import risk as risk_cli
from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.storage.schema import create_engine_for_url, init_db


@pytest.fixture
def engine(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return engine


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "portfolio.yaml"
    path.write_text(
        "portfolio:\n  name: test-portfolio\npositions:\n  - ticker: AAPL\n",
        encoding="utf-8",
    )
    return str(path)


@pytest.fixture(autouse=True)
def _patch_engine(monkeypatch, engine):
    monkeypatch.setattr(risk_cli, "get_db_engine", lambda: engine)


def test_status_reports_clear_when_untripped(config_path, engine, capsys):
    PortfolioStore(engine).get_or_create_portfolio("test-portfolio", 10_000)

    result = risk_cli.status(config_path)

    assert result == 0
    assert "clear" in capsys.readouterr().out.lower()


def test_status_reports_tripped_state(config_path, engine, capsys):
    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("test-portfolio", 10_000)
    store.trip_kill_switch(state.id, "daily loss breach")

    result = risk_cli.status(config_path)

    out = capsys.readouterr().out.lower()
    assert result == 0
    assert "tripped" in out
    assert "daily loss breach" in out


def test_status_handles_missing_portfolio(config_path, capsys):
    result = risk_cli.status(config_path)

    assert result == 0
    assert "no portfolio" in capsys.readouterr().out.lower()


def test_clear_resets_tripped_state(config_path, engine, capsys):
    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("test-portfolio", 10_000)
    store.trip_kill_switch(state.id, "daily loss breach")

    result = risk_cli.clear(config_path, reason="reviewed, resuming")

    assert result == 0
    tripped_at, reason = store.get_kill_switch_state(state.id)
    assert tripped_at is None
    assert reason is None
    assert "cleared" in capsys.readouterr().out.lower()


def test_clear_is_a_no_op_when_already_clear(config_path, engine, capsys):
    PortfolioStore(engine).get_or_create_portfolio("test-portfolio", 10_000)

    result = risk_cli.clear(config_path, reason="just checking")

    assert result == 0
    assert "already clear" in capsys.readouterr().out.lower()


def test_clear_fails_for_missing_portfolio(config_path, capsys):
    result = risk_cli.clear(config_path, reason="n/a")

    assert result == 1
    assert "no portfolio" in capsys.readouterr().out.lower()


def test_main_dispatches_status_and_clear(config_path, engine):
    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("test-portfolio", 10_000)
    store.trip_kill_switch(state.id, "x")

    assert risk_cli.main(["--config", config_path, "status"]) == 0
    assert risk_cli.main(["--config", config_path, "clear", "--reason", "ok"]) == 0

    tripped_at, _ = store.get_kill_switch_state(state.id)
    assert tripped_at is None


def test_main_dispatches_reconcile(config_path, engine):
    PortfolioStore(engine).get_or_create_portfolio("test-portfolio", 10_000)
    assert risk_cli.main(["--config", config_path, "reconcile"]) == 1  # paper broker, nothing to do


def test_reconcile_fails_for_missing_portfolio(config_path, capsys):
    result = risk_cli.reconcile(config_path, accept_broker_truth=False)

    assert result == 1
    assert "no portfolio" in capsys.readouterr().out.lower()


def test_reconcile_with_paper_broker_has_nothing_to_reconcile(config_path, engine, capsys):
    PortfolioStore(engine).get_or_create_portfolio("test-portfolio", 10_000)

    result = risk_cli.reconcile(config_path, accept_broker_truth=False)

    assert result == 1
    assert "nothing to reconcile" in capsys.readouterr().out.lower()


def _mock_alpaca_broker(monkeypatch, snapshot):
    from unittest.mock import MagicMock

    fake_broker = MagicMock()
    fake_broker.get_account_snapshot.return_value = snapshot
    monkeypatch.setattr(
        "stock_analysis.trading.broker_config.connect_broker_for_inspection",
        lambda *a, **kw: fake_broker,
    )
    return fake_broker


def test_reconcile_works_even_when_kill_switch_is_tripped_and_live_requested(
    config_path, engine, capsys, monkeypatch
):
    """Regression: reconcile must never route through the live-trading gate (which
    requires the kill switch to be clear) — that's exactly the chicken-and-egg
    lockout this CLI exists to break out of."""
    from stock_analysis.trading.broker import AccountSnapshot

    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("test-portfolio", 10_000)
    store.trip_kill_switch(state.id, "first live run reconciliation mismatch")
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "whatever")
    _mock_alpaca_broker(
        monkeypatch, AccountSnapshot(cash=50_000.0, positions={}, source="alpaca")
    )

    result = risk_cli.reconcile(config_path, accept_broker_truth=False)  # must not raise

    assert result == 0
    assert "50,000.00" in capsys.readouterr().out


def test_reconcile_preview_does_not_change_local_state(config_path, engine, capsys, monkeypatch):
    from stock_analysis.trading.broker import AccountSnapshot

    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("test-portfolio", 10_000)
    store.upsert_position(state.id, "AAPL", 5, 100.0)
    _mock_alpaca_broker(
        monkeypatch, AccountSnapshot(cash=9_000.0, positions={"AAPL": 5, "MSFT": 3}, source="alpaca")
    )

    result = risk_cli.reconcile(config_path, accept_broker_truth=False)

    assert result == 0
    out = capsys.readouterr().out
    assert "--accept-broker-truth" in out
    reloaded = store.get_portfolio_by_name("test-portfolio")
    assert reloaded.cash == 10_000  # unchanged
    assert {p.ticker for p in reloaded.positions} == {"AAPL"}  # unchanged


def test_reconcile_with_accept_flag_applies_broker_truth(config_path, engine, capsys, monkeypatch):
    from stock_analysis.trading.broker import AccountSnapshot

    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("test-portfolio", 10_000)
    store.upsert_position(state.id, "AAPL", 5, 100.0)  # broker will disagree (10 shares)
    store.upsert_position(state.id, "GOOGL", 2, 50.0)  # broker has no record of this -> removed
    _mock_alpaca_broker(
        monkeypatch,
        AccountSnapshot(cash=8_500.0, positions={"AAPL": 10, "MSFT": 3}, source="alpaca"),
    )

    result = risk_cli.reconcile(config_path, accept_broker_truth=True)

    assert result == 0
    reloaded = store.get_portfolio_by_name("test-portfolio")
    assert reloaded.cash == 8_500.0
    by_ticker = {p.ticker: p for p in reloaded.positions}
    assert set(by_ticker) == {"AAPL", "MSFT"}
    assert by_ticker["AAPL"].shares == 10
    assert by_ticker["AAPL"].avg_cost == 100.0  # kept — broker snapshot has no cost basis for it
    assert by_ticker["MSFT"].shares == 3
    assert by_ticker["MSFT"].avg_cost == 0.0  # new position, no broker cost basis either
    out = capsys.readouterr().out.lower()
    assert "reconciled" in out
    assert "clear" in out  # points the operator at the next step
    assert "warning" in out  # flags that MSFT's stop-loss is now disabled


def test_reconcile_uses_broker_reported_cost_basis_when_available(
    config_path, engine, capsys, monkeypatch
):
    from stock_analysis.trading.broker import AccountSnapshot

    store = PortfolioStore(engine)
    store.get_or_create_portfolio("test-portfolio", 10_000)
    _mock_alpaca_broker(
        monkeypatch,
        AccountSnapshot(
            cash=5_000.0,
            positions={"MSFT": 3},
            avg_costs={"MSFT": 310.25},
            source="alpaca",
        ),
    )

    result = risk_cli.reconcile(config_path, accept_broker_truth=True)

    assert result == 0
    reloaded = store.get_portfolio_by_name("test-portfolio")
    msft = next(p for p in reloaded.positions if p.ticker == "MSFT")
    assert msft.avg_cost == 310.25
    assert "warning" not in capsys.readouterr().out.lower()
