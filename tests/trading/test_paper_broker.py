from datetime import date

import pandas as pd
import pytest

from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.storage.schema import create_engine_for_url, init_db
from stock_analysis.trading.broker import Broker
from stock_analysis.trading.cost_models import EquityCostModel
from stock_analysis.trading.decision import OrderIntent
from stock_analysis.trading.paper_broker import PaperBroker, open_price_for_date


def test_open_price_for_date():
    index = pd.to_datetime(["2024-06-03", "2024-06-04"])
    df = pd.DataFrame({"Open": [100.0, 101.0], "Close": [100.5, 101.5]}, index=index)
    assert open_price_for_date(df, date(2024, 6, 4)) == 101.0


@pytest.fixture
def broker_setup(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'broker.db'}")
    init_db(engine)
    store = PortfolioStore(engine)
    state = store.get_or_create_portfolio("paper-us", 10_000)
    broker = PaperBroker(store, EquityCostModel())
    return broker, store, state


def test_fill_pending_buy_order(broker_setup):
    broker, store, state = broker_setup
    run_id = store.start_run(state.id, dry_run=False)
    store.create_pending_order(
        state.id, run_id, "AAPL", "BUY", 10, "test", fill_date=date(2024, 6, 3)
    )

    index = pd.to_datetime(["2024-06-03"])
    price_data = {
        "AAPL": pd.DataFrame({"Open": [100.0], "Close": [101.0]}, index=index),
    }
    fills = broker.sync_fills(
        state, price_data, session_date=date(2024, 6, 3), dry_run=False
    )
    assert len(fills) == 1
    assert state.cash < 10_000
    assert len(state.positions) == 1


def test_second_fill_into_held_ticker_averages_correctly_in_the_db(broker_setup):
    """A second BUY fill into an already-held ticker must accumulate shares/avg_cost
    in the DB, not overwrite them with just the incremental fill — regression test
    for a bug where upsert_position was passed the per-fill values instead of the
    post-fill totals apply_fill_to_state already computed in memory."""
    broker, store, state = broker_setup
    run_id = store.start_run(state.id, dry_run=False)
    store.create_pending_order(
        state.id, run_id, "AAPL", "BUY", 10, "first", fill_date=date(2024, 6, 3)
    )
    price_data = {
        "AAPL": pd.DataFrame(
            {"Open": [100.0], "Close": [100.0]}, index=pd.to_datetime(["2024-06-03"])
        ),
    }
    broker.sync_fills(state, price_data, session_date=date(2024, 6, 3), dry_run=False)

    store.create_pending_order(
        state.id, run_id, "AAPL", "BUY", 10, "second", fill_date=date(2024, 6, 4)
    )
    price_data = {
        "AAPL": pd.DataFrame(
            {"Open": [120.0], "Close": [120.0]}, index=pd.to_datetime(["2024-06-04"])
        ),
    }
    broker.sync_fills(state, price_data, session_date=date(2024, 6, 4), dry_run=False)

    reloaded = store.get_portfolio_by_name("paper-us")
    assert len(reloaded.positions) == 1
    assert reloaded.positions[0].shares == 20
    assert reloaded.positions[0].avg_cost == pytest.approx(110.0)


def test_paper_broker_satisfies_broker_protocol(broker_setup):
    broker, _store, _state = broker_setup
    assert isinstance(broker, Broker)


def test_submit_order_requires_fill_date(broker_setup):
    broker, store, state = broker_setup
    run_id = store.start_run(state.id, dry_run=False)
    intent = OrderIntent(ticker="AAPL", side="BUY", shares=1, reason="test")

    with pytest.raises(ValueError, match="fill_date"):
        broker.submit_order(state, run_id, intent, dry_run=False)


def test_submit_order_queues_and_dry_run_is_a_no_op(broker_setup):
    broker, store, state = broker_setup
    run_id = store.start_run(state.id, dry_run=False)
    intent = OrderIntent(ticker="AAPL", side="BUY", shares=1, reason="test")

    assert broker.submit_order(state, run_id, intent, dry_run=True, fill_date=date(2024, 6, 3)) is None
    assert store.get_pending_orders(state.id) == []

    order_id = broker.submit_order(
        state, run_id, intent, dry_run=False, fill_date=date(2024, 6, 3)
    )
    assert order_id is not None
    assert len(store.get_pending_orders(state.id)) == 1


def test_get_account_snapshot_reflects_local_state(broker_setup):
    broker, store, state = broker_setup
    snapshot = broker.get_account_snapshot(state)
    assert snapshot.source == "local"
    assert snapshot.cash == state.cash
    assert snapshot.positions == {}


def test_cancel_order_cancels_a_pending_order(broker_setup):
    broker, store, state = broker_setup
    run_id = store.start_run(state.id, dry_run=False)
    order_id = store.create_pending_order(
        state.id, run_id, "AAPL", "BUY", 10, "test", fill_date=date(2024, 6, 3)
    )

    assert broker.cancel_order(str(order_id)) is True
    assert store.get_pending_orders(state.id) == []


def test_cancel_order_returns_false_for_unknown_order(broker_setup):
    broker, _store, _state = broker_setup
    assert broker.cancel_order("999999") is False
