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
    """No live-trading path exists yet — ALPACA_PAPER=false must not flip the
    TradingClient to Alpaca's live endpoint. This is the only safety gate that
    exists before live_gate.py (a future phase) is written, so it must fail safe."""
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
