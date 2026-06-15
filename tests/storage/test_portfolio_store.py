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


def test_get_equity_curve(store):
    state = store.get_or_create_portfolio("paper-us", 100_000)

    # dry-run — excluded
    dry_id = store.start_run(state.id, dry_run=True)
    store.finish_run(dry_id, "ok", {"session_date": "2024-06-01", "portfolio_value": 100_500.0})

    # real run, earlier date
    run_a = store.start_run(state.id, dry_run=False)
    store.finish_run(run_a, "ok", {"session_date": "2024-06-02", "portfolio_value": 100_800.0})

    # real run, later date
    run_b = store.start_run(state.id, dry_run=False)
    store.finish_run(run_b, "ok", {"session_date": "2024-06-03", "portfolio_value": 101_000.0})

    # unfinished run — excluded (no finish_run call)
    store.start_run(state.id, dry_run=False)

    # finished run with missing summary fields — excluded
    run_c = store.start_run(state.id, dry_run=False)
    store.finish_run(run_c, "ok", {"fills": 0})

    curve = store.get_equity_curve(state.id)

    assert len(curve) == 2
    assert curve[0] == ("2024-06-02", 100_800.0)
    assert curve[1] == ("2024-06-03", 101_000.0)
