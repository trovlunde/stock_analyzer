import pytest
from alpaca.common.enums import BaseURL

from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.storage.schema import create_engine_for_url, init_db
from stock_analysis.trading.alpaca_broker import AlpacaBroker
from stock_analysis.trading.broker_config import load_broker_from_env
from stock_analysis.trading.cost_models import EquityCostModel
from stock_analysis.trading.paper_broker import PaperBroker


@pytest.fixture
def store(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return PortfolioStore(engine)


def test_defaults_to_paper_sim(monkeypatch, store):
    monkeypatch.delenv("BROKER_MODE", raising=False)
    broker = load_broker_from_env(store, EquityCostModel())
    assert isinstance(broker, PaperBroker)


def test_paper_sim_explicit(monkeypatch, store):
    monkeypatch.setenv("BROKER_MODE", "paper_sim")
    broker = load_broker_from_env(store, EquityCostModel())
    assert isinstance(broker, PaperBroker)


def test_alpaca_mode_builds_alpaca_broker(monkeypatch, store):
    monkeypatch.setenv("BROKER_MODE", "alpaca")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_PAPER", "true")

    broker = load_broker_from_env(store, EquityCostModel())

    assert isinstance(broker, AlpacaBroker)


def test_alpaca_mode_always_uses_paper_endpoint_even_if_alpaca_paper_is_false(
    monkeypatch, store
):
    """Without LIVE_TRADING_ENABLED=true (and the rest of the live gate passing),
    ALPACA_PAPER=false alone must not flip the TradingClient to Alpaca's live
    endpoint — fail safe to paper."""
    monkeypatch.delenv("LIVE_TRADING_ENABLED", raising=False)
    monkeypatch.setenv("BROKER_MODE", "alpaca")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_PAPER", "false")

    broker = load_broker_from_env(store, EquityCostModel())

    assert broker.client._base_url == BaseURL.TRADING_PAPER


def test_alpaca_mode_treats_any_typo_in_alpaca_paper_as_paper(monkeypatch, store):
    monkeypatch.setenv("BROKER_MODE", "alpaca")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    monkeypatch.setenv("ALPACA_PAPER", "treu")  # typo for "true"

    broker = load_broker_from_env(store, EquityCostModel())

    assert broker.client._base_url == BaseURL.TRADING_PAPER


def test_alpaca_mode_requires_api_keys(monkeypatch, store):
    monkeypatch.setenv("BROKER_MODE", "alpaca")
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_SECRET_KEY", raising=False)

    with pytest.raises(KeyError):
        load_broker_from_env(store, EquityCostModel())


def test_unknown_mode_raises(monkeypatch, store):
    monkeypatch.setenv("BROKER_MODE", "not-a-real-mode")
    with pytest.raises(ValueError, match="Unknown BROKER_MODE"):
        load_broker_from_env(store, EquityCostModel())


def _settings_with_risk():
    from stock_analysis.trading.config import PortfolioConfig, PortfolioSettings
    from stock_analysis.trading.risk import RiskConfig

    return PortfolioSettings(
        portfolio=PortfolioConfig(
            name="paper-us", enabled=True, initial_cash=100_000, max_positions=5,
            position_size_pct=0.1, long_only=True, stop_loss_pct=None,
        ),
        costs={}, defaults={},
        positions=[],
        risk=RiskConfig(max_daily_loss_pct=0.05, max_drawdown_pct=0.2),
    )


def _set_live_env(monkeypatch):
    from stock_analysis.trading.live_gate import LIVE_CONFIRM_PHRASE

    monkeypatch.setenv("LIVE_TRADING_ENABLED", "true")
    monkeypatch.setenv("LIVE_TRADING_CONFIRM", LIVE_CONFIRM_PHRASE)
    monkeypatch.setenv("ALPACA_PAPER", "false")
    monkeypatch.setenv("LIVE_MAX_NOTIONAL_PER_ORDER", "1000")
    monkeypatch.setenv("LIVE_MAX_PORTFOLIO_NOTIONAL", "50000")


def test_live_requested_without_settings_raises(monkeypatch, store):
    monkeypatch.setenv("BROKER_MODE", "alpaca")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    _set_live_env(monkeypatch)

    with pytest.raises(ValueError, match="settings and portfolio_id"):
        load_broker_from_env(store, EquityCostModel())


def test_live_requested_but_gate_fails_raises_live_trading_blocked(monkeypatch, store):
    from stock_analysis.trading.live_gate import LiveTradingBlocked

    state = store.get_or_create_portfolio("paper-us", 100_000)
    monkeypatch.setenv("BROKER_MODE", "alpaca")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    _set_live_env(monkeypatch)
    # No clean-run history at all -> gate must refuse.

    with pytest.raises(LiveTradingBlocked):
        load_broker_from_env(
            store, EquityCostModel(), settings=_settings_with_risk(), portfolio_id=state.id
        )


def test_live_requested_and_gate_passes_builds_live_broker(monkeypatch, store):
    from stock_analysis.trading.live_gate import MIN_CONSECUTIVE_CLEAN_RUNS

    state = store.get_or_create_portfolio("paper-us", 100_000)
    for _ in range(MIN_CONSECUTIVE_CLEAN_RUNS):
        run_id = store.start_run(state.id, dry_run=False)
        store.finish_run(run_id, "completed", {"session_date": "x", "portfolio_value": 1})

    monkeypatch.setenv("BROKER_MODE", "alpaca")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    _set_live_env(monkeypatch)

    broker = load_broker_from_env(
        store, EquityCostModel(), settings=_settings_with_risk(), portfolio_id=state.id
    )

    assert isinstance(broker, AlpacaBroker)
    assert broker.client._base_url == BaseURL.TRADING_LIVE
    assert broker.live is True
    assert broker.read_only is True  # no clean live reconciliation observed yet


def test_live_broker_is_not_read_only_once_confirmed(monkeypatch, store):
    from stock_analysis.trading.live_gate import MIN_CONSECUTIVE_CLEAN_RUNS

    state = store.get_or_create_portfolio("paper-us", 100_000)
    for _ in range(MIN_CONSECUTIVE_CLEAN_RUNS):
        run_id = store.start_run(state.id, dry_run=False)
        store.finish_run(run_id, "completed", {"session_date": "x", "portfolio_value": 1})
    store.mark_live_confirmed(state.id)

    monkeypatch.setenv("BROKER_MODE", "alpaca")
    monkeypatch.setenv("ALPACA_API_KEY", "key")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "secret")
    _set_live_env(monkeypatch)

    broker = load_broker_from_env(
        store, EquityCostModel(), settings=_settings_with_risk(), portfolio_id=state.id
    )

    assert broker.read_only is False
