from datetime import date

import pytest

from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.storage.schema import create_engine_for_url, init_db


@pytest.fixture
def store(tmp_path):
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'test.db'}")
    init_db(engine)
    return PortfolioStore(engine)


def test_create_portfolio_and_pending_order(store):
    state = store.get_or_create_portfolio("paper-us", 100_000)
    assert state.cash == 100_000
    assert state.positions == []

    run_id = store.start_run(state.id, dry_run=False)
    order_id = store.create_pending_order(
        state.id,
        run_id,
        "AAPL",
        "BUY",
        10,
        "golden cross",
        fill_date=date(2024, 6, 3),
    )
    pending = store.get_pending_orders(state.id)
    assert len(pending) == 1
    assert pending[0].id == order_id

    store.fill_order(order_id, 180.0, 1.0, 0.5)
    store.upsert_position(state.id, "AAPL", 10, 180.0)
    store.update_cash(state.id, 98_000)

    updated = store.get_portfolio_by_name("paper-us")
    assert updated.cash == 98_000
    assert len(updated.positions) == 1
    assert updated.positions[0].ticker == "AAPL"
