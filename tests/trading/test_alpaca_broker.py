from datetime import date
from unittest.mock import MagicMock

import pytest

from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.storage.schema import create_engine_for_url, init_db
from stock_analysis.trading.alpaca_broker import AlpacaBroker, BrokerRejected
from stock_analysis.trading.broker import Broker
from stock_analysis.trading.decision import OrderIntent


@pytest.fixture
def store(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return PortfolioStore(engine)


@pytest.fixture
def state(store):
    return store.get_or_create_portfolio("paper-us", 10_000)


def _account(trading_blocked=False, account_blocked=False, cash="10000"):
    account = MagicMock()
    account.trading_blocked = trading_blocked
    account.account_blocked = account_blocked
    account.cash = cash
    return account


def _alpaca_order(order_id="order-123", status="new"):
    order = MagicMock()
    order.id = order_id
    order.status.value = status
    return order


def test_alpaca_broker_satisfies_broker_protocol(store):
    broker = AlpacaBroker(MagicMock(), store)
    assert isinstance(broker, Broker)


def test_submit_order_passes_client_order_id(store, state):
    client = MagicMock()
    client.get_account.return_value = _account()
    client.submit_order.return_value = _alpaca_order()
    broker = AlpacaBroker(client, store)
    intent = OrderIntent(ticker="AAPL", side="BUY", shares=5, reason="test")

    broker_order_id = broker.submit_order(state, run_id=1, intent=intent, dry_run=False)

    assert broker_order_id == "order-123"
    request = client.submit_order.call_args[0][0]
    assert request.client_order_id == f"{state.name}-1-AAPL-BUY"
    assert request.symbol == "AAPL"
    assert request.qty == 5


def test_submit_order_dry_run_never_calls_client(store, state):
    client = MagicMock()
    broker = AlpacaBroker(client, store)
    intent = OrderIntent(ticker="AAPL", side="BUY", shares=5, reason="test")

    result = broker.submit_order(state, run_id=1, intent=intent, dry_run=True)

    assert result is None
    client.submit_order.assert_not_called()


def test_submit_order_does_not_resubmit_same_client_order_id(store, state):
    client = MagicMock()
    client.get_account.return_value = _account()
    client.submit_order.return_value = _alpaca_order(order_id="order-abc")
    broker = AlpacaBroker(client, store)
    intent = OrderIntent(ticker="AAPL", side="BUY", shares=5, reason="test")

    first = broker.submit_order(state, run_id=1, intent=intent, dry_run=False)
    second = broker.submit_order(state, run_id=1, intent=intent, dry_run=False)

    assert first == second == "order-abc"
    assert client.submit_order.call_count == 1


def test_submit_order_reconciles_unconfirmed_order_found_at_alpaca(store, state):
    """Simulates a crash between create_broker_order() and mark_order_submitted() —
    the local row is stuck with broker_order_id=None. On retry, Alpaca DOES have a
    record of it (the original submit actually reached Alpaca); must reconcile
    instead of silently returning None forever or blindly resubmitting."""
    client = MagicMock()
    client.get_account.return_value = _account()
    store.create_broker_order(
        state.id, run_id=1, ticker="AAPL", side="BUY", shares=5,
        signal_reason="test", client_order_id=f"{state.name}-1-AAPL-BUY",
    )
    found_order = _alpaca_order(order_id="order-found", status="accepted")
    client.get_order_by_client_id.return_value = found_order
    broker = AlpacaBroker(client, store)
    intent = OrderIntent(ticker="AAPL", side="BUY", shares=5, reason="test")

    result = broker.submit_order(state, run_id=1, intent=intent, dry_run=False)

    assert result == "order-found"
    client.submit_order.assert_not_called()
    reloaded = store.get_order_by_client_id(f"{state.name}-1-AAPL-BUY")
    assert reloaded.broker_order_id == "order-found"


def test_submit_order_resubmits_when_alpaca_has_no_record_of_unconfirmed_order(store, state):
    """Same stranded-row scenario, but Alpaca genuinely never received the original
    submission — safe to actually submit now, reusing the same client_order_id."""
    client = MagicMock()
    client.get_account.return_value = _account()
    client.get_order_by_client_id.side_effect = Exception("404 not found")
    client.submit_order.return_value = _alpaca_order(order_id="order-new")
    store.create_broker_order(
        state.id, run_id=1, ticker="AAPL", side="BUY", shares=5,
        signal_reason="test", client_order_id=f"{state.name}-1-AAPL-BUY",
    )
    broker = AlpacaBroker(client, store)
    intent = OrderIntent(ticker="AAPL", side="BUY", shares=5, reason="test")

    result = broker.submit_order(state, run_id=1, intent=intent, dry_run=False)

    assert result == "order-new"
    client.submit_order.assert_called_once()
    assert client.submit_order.call_args[0][0].client_order_id == f"{state.name}-1-AAPL-BUY"


def test_submit_order_refuses_when_account_blocked(store, state):
    client = MagicMock()
    client.get_account.return_value = _account(trading_blocked=True)
    broker = AlpacaBroker(client, store)
    intent = OrderIntent(ticker="AAPL", side="BUY", shares=5, reason="test")

    with pytest.raises(BrokerRejected):
        broker.submit_order(state, run_id=1, intent=intent, dry_run=False)

    client.submit_order.assert_not_called()


def test_sync_fills_applies_a_filled_order(store, state):
    client = MagicMock()
    client.get_account.return_value = _account()
    client.submit_order.return_value = _alpaca_order(order_id="order-xyz", status="new")
    broker = AlpacaBroker(client, store)
    intent = OrderIntent(ticker="AAPL", side="BUY", shares=5, reason="test")
    broker.submit_order(state, run_id=1, intent=intent, dry_run=False)

    filled_order = _alpaca_order(order_id="order-xyz", status="filled")
    filled_order.filled_avg_price = "101.50"
    client.get_orders.return_value = [filled_order]

    fills = broker.sync_fills(state, price_data={}, session_date=date(2024, 6, 3), dry_run=False)

    assert len(fills) == 1
    assert fills[0].fill_price == 101.50
    assert state.cash == pytest.approx(10_000 - 5 * 101.50)
    assert len(state.positions) == 1
    assert state.positions[0].ticker == "AAPL"


def test_sync_fills_second_fill_into_held_ticker_averages_correctly_in_the_db(store, state):
    client = MagicMock()
    client.get_account.return_value = _account()

    client.submit_order.return_value = _alpaca_order(order_id="order-1", status="new")
    broker = AlpacaBroker(client, store)
    first_intent = OrderIntent(ticker="AAPL", side="BUY", shares=10, reason="first")
    broker.submit_order(state, run_id=1, intent=first_intent, dry_run=False)

    filled_1 = _alpaca_order(order_id="order-1", status="filled")
    filled_1.filled_avg_price = "100.0"
    client.get_orders.return_value = [filled_1]
    broker.sync_fills(state, price_data={}, session_date=date(2024, 6, 3), dry_run=False)

    client.submit_order.return_value = _alpaca_order(order_id="order-2", status="new")
    second_intent = OrderIntent(ticker="AAPL", side="BUY", shares=10, reason="second")
    broker.submit_order(state, run_id=2, intent=second_intent, dry_run=False)

    filled_2 = _alpaca_order(order_id="order-2", status="filled")
    filled_2.filled_avg_price = "120.0"
    client.get_orders.return_value = [filled_2]
    broker.sync_fills(state, price_data={}, session_date=date(2024, 6, 4), dry_run=False)

    reloaded = store.get_portfolio_by_name(state.name)
    assert len(reloaded.positions) == 1
    assert reloaded.positions[0].shares == 20
    assert reloaded.positions[0].avg_cost == pytest.approx(110.0)


def test_sync_fills_marks_rejected_orders_without_applying_a_fill(store, state):
    client = MagicMock()
    client.get_account.return_value = _account()
    client.submit_order.return_value = _alpaca_order(order_id="order-rej", status="new")
    broker = AlpacaBroker(client, store)
    intent = OrderIntent(ticker="AAPL", side="BUY", shares=5, reason="test")
    broker.submit_order(state, run_id=1, intent=intent, dry_run=False)

    rejected_order = _alpaca_order(order_id="order-rej", status="rejected")
    client.get_orders.return_value = [rejected_order]

    fills = broker.sync_fills(state, price_data={}, session_date=date(2024, 6, 3), dry_run=False)

    assert fills == []
    assert state.cash == 10_000
    assert store.get_open_broker_orders(state.id) == []


def test_sync_fills_dry_run_is_a_no_op(store, state):
    client = MagicMock()
    broker = AlpacaBroker(client, store)

    fills = broker.sync_fills(state, price_data={}, session_date=date(2024, 6, 3), dry_run=True)

    assert fills == []
    client.get_orders.assert_not_called()


def test_get_account_snapshot_reflects_alpaca_truth(store, state):
    client = MagicMock()
    client.get_account.return_value = _account(cash="12345.67")
    position = MagicMock()
    position.symbol = "AAPL"
    position.qty = "10"
    client.get_all_positions.return_value = [position]
    broker = AlpacaBroker(client, store)

    snapshot = broker.get_account_snapshot(state)

    assert snapshot.source == "alpaca"
    assert snapshot.cash == 12345.67
    assert snapshot.positions == {"AAPL": 10.0}


def test_cancel_order_delegates_to_client(store):
    client = MagicMock()
    broker = AlpacaBroker(client, store)

    assert broker.cancel_order("order-1") is True
    client.cancel_order_by_id.assert_called_once_with("order-1")


def test_cancel_order_returns_false_on_client_error(store):
    client = MagicMock()
    client.cancel_order_by_id.side_effect = RuntimeError("network error")
    broker = AlpacaBroker(client, store)

    assert broker.cancel_order("order-1") is False
