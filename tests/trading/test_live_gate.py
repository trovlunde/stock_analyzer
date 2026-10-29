import pytest

from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.storage.schema import create_engine_for_url, init_db
from stock_analysis.trading.live_gate import (
    LIVE_CONFIRM_PHRASE,
    MIN_CONSECUTIVE_CLEAN_RUNS,
    check_live_trading_gate,
    is_first_live_run,
    live_notional_ceiling_violation,
    live_portfolio_ceiling_violation,
    live_trading_requested,
)
from stock_analysis.trading.risk import RiskConfig


@pytest.fixture
def store(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return PortfolioStore(engine)


@pytest.fixture
def state(store):
    return store.get_or_create_portfolio("paper-us", 100_000)


def _settings(risk: RiskConfig):
    from stock_analysis.trading.config import PortfolioConfig, PortfolioSettings

    return PortfolioSettings(
        portfolio=PortfolioConfig(
            name="paper-us", enabled=True, initial_cash=100_000, max_positions=5,
            position_size_pct=0.1, long_only=True, stop_loss_pct=None,
        ),
        costs={}, defaults={}, positions=[], risk=risk,
    )


def _make_clean(store, state, n=MIN_CONSECUTIVE_CLEAN_RUNS):
    for _ in range(n):
        run_id = store.start_run(state.id, dry_run=False)
        store.finish_run(run_id, "completed", {"session_date": "x", "portfolio_value": 1})


def _set_all_env_for_allowed(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", LIVE_CONFIRM_PHRASE)
    monkeypatch.setenv("ALPACA_PAPER", "false")
    monkeypatch.setenv("LIVE_MAX_NOTIONAL_PER_ORDER", "1000")
    monkeypatch.setenv("LIVE_MAX_PORTFOLIO_NOTIONAL", "50000")


def test_live_trading_requested_false_by_default(monkeypatch):
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    assert live_trading_requested() is False


def test_live_trading_requested_true(monkeypatch):
    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    assert live_trading_requested() is True


def test_gate_allows_when_everything_satisfied(monkeypatch, store, state):
    _set_all_env_for_allowed(monkeypatch)
    _make_clean(store, state)
    settings = _settings(RiskConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.2))

    result = check_live_trading_gate(settings, store, state.id)

    assert result.allowed is True
    assert result.reasons == ()


def test_gate_blocks_when_not_requested(monkeypatch, store, state):
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    settings = _settings(RiskConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.2))

    result = check_live_trading_gate(settings, store, state.id)

    assert result.allowed is False
    assert any("LIVE_TRADING_ENABLED" in r for r in result.reasons)


def test_gate_blocks_on_wrong_confirm_phrase(monkeypatch, store, state):
    _set_all_env_for_allowed(monkeypatch)
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", "yes please")
    _make_clean(store, state)
    settings = _settings(RiskConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.2))

    result = check_live_trading_gate(settings, store, state.id)

    assert result.allowed is False
    assert any("LIVE_TRADING_CONFIRM" in r for r in result.reasons)


def test_gate_blocks_when_alpaca_paper_is_true(monkeypatch, store, state):
    _set_all_env_for_allowed(monkeypatch)
    monkeypatch.setenv("ALPACA_PAPER", "true")
    _make_clean(store, state)
    settings = _settings(RiskConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.2))

    result = check_live_trading_gate(settings, store, state.id)

    assert result.allowed is False
    assert any("ALPACA_PAPER" in r for r in result.reasons)


def test_gate_blocks_when_risk_thresholds_unset(monkeypatch, store, state):
    _set_all_env_for_allowed(monkeypatch)
    _make_clean(store, state)
    settings = _settings(RiskConfig())

    result = check_live_trading_gate(settings, store, state.id)

    assert result.allowed is False
    assert any("risk." in r for r in result.reasons)


def test_gate_blocks_when_kill_switch_tripped(monkeypatch, store, state):
    _set_all_env_for_allowed(monkeypatch)
    _make_clean(store, state)
    store.trip_kill_switch(state.id, "daily loss breach")
    settings = _settings(RiskConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.2))

    result = check_live_trading_gate(settings, store, state.id)

    assert result.allowed is False
    assert any("kill switch" in r for r in result.reasons)


def test_gate_blocks_when_not_enough_clean_runs(monkeypatch, store, state):
    _set_all_env_for_allowed(monkeypatch)
    _make_clean(store, state, n=MIN_CONSECUTIVE_CLEAN_RUNS - 1)
    settings = _settings(RiskConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.2))

    result = check_live_trading_gate(settings, store, state.id)

    assert result.allowed is False
    assert any("consecutive clean run" in r for r in result.reasons)


def test_gate_blocks_when_notional_ceilings_unset(monkeypatch, store, state):
    _set_all_env_for_allowed(monkeypatch)
    monkeypatch.delenv("LIVE_MAX_NOTIONAL_PER_ORDER", raising=False)
    _make_clean(store, state)
    settings = _settings(RiskConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.2))

    result = check_live_trading_gate(settings, store, state.id)

    assert result.allowed is False
    assert any("LIVE_MAX_NOTIONAL_PER_ORDER" in r for r in result.reasons)


def test_gate_blocks_when_notional_ceiling_not_a_number(monkeypatch, store, state):
    _set_all_env_for_allowed(monkeypatch)
    monkeypatch.setenv("LIVE_MAX_NOTIONAL_PER_ORDER", "not-a-number")
    _make_clean(store, state)
    settings = _settings(RiskConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.2))

    result = check_live_trading_gate(settings, store, state.id)

    assert result.allowed is False


def test_gate_collects_all_failing_reasons_not_just_first(monkeypatch, store, state):
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    monkeypatch.delenv("LIVE_TRADING_CONFIRM", raising=False)
    settings = _settings(RiskConfig())

    result = check_live_trading_gate(settings, store, state.id)

    assert len(result.reasons) >= 2


def test_is_first_live_run_true_until_confirmed(store, state):
    assert is_first_live_run(store, state.id) is True
    store.mark_live_confirmed(state.id)
    assert is_first_live_run(store, state.id) is False


def test_live_notional_ceiling_violation_within_limit(monkeypatch):
    monkeypatch.setenv("LIVE_MAX_NOTIONAL_PER_ORDER", "1000")
    assert live_notional_ceiling_violation("AAPL", shares=5, price=100.0) is None


def test_live_notional_ceiling_violation_exceeds_limit(monkeypatch):
    monkeypatch.setenv("LIVE_MAX_NOTIONAL_PER_ORDER", "100")
    reason = live_notional_ceiling_violation("AAPL", shares=5, price=100.0)
    assert reason is not None
    assert "AAPL" in reason


def test_live_notional_ceiling_violation_unset_never_blocks(monkeypatch):
    monkeypatch.delenv("LIVE_MAX_NOTIONAL_PER_ORDER", raising=False)
    assert live_notional_ceiling_violation("AAPL", shares=5, price=1_000_000.0) is None


def test_live_portfolio_ceiling_violation_within_limit(monkeypatch):
    monkeypatch.setenv("LIVE_MAX_PORTFOLIO_NOTIONAL", "50000")
    assert live_portfolio_ceiling_violation(40_000) is None


def test_live_portfolio_ceiling_violation_at_or_over_limit(monkeypatch):
    monkeypatch.setenv("LIVE_MAX_PORTFOLIO_NOTIONAL", "50000")
    assert live_portfolio_ceiling_violation(50_000) is not None
    assert live_portfolio_ceiling_violation(60_000) is not None
