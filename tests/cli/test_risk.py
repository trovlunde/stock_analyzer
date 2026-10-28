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
