from datetime import date, datetime, timedelta, timezone

import pytest
from sqlalchemy import update

from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.storage.schema import create_engine_for_url, init_db, paper_runs


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


def test_kill_switch_round_trip(store):
    state = store.get_or_create_portfolio("paper-us", 100_000)

    tripped_at, reason = store.get_kill_switch_state(state.id)
    assert tripped_at is None
    assert reason is None

    store.trip_kill_switch(state.id, "daily loss 10.0% >= limit 5.0%")
    tripped_at, reason = store.get_kill_switch_state(state.id)
    assert tripped_at is not None
    assert reason == "daily loss 10.0% >= limit 5.0%"

    store.clear_kill_switch(state.id)
    tripped_at, reason = store.get_kill_switch_state(state.id)
    assert tripped_at is None
    assert reason is None


def test_kill_switch_state_survives_fresh_connection(tmp_path):
    db_path = tmp_path / "persist.db"
    engine_a = create_engine_for_url(f"sqlite:///{db_path}")
    init_db(engine_a)
    store_a = PortfolioStore(engine_a)
    state = store_a.get_or_create_portfolio("paper-us", 100_000)
    store_a.trip_kill_switch(state.id, "max drawdown breached")

    # Simulate a process restart — a brand new engine/connection against the same file.
    engine_b = create_engine_for_url(f"sqlite:///{db_path}")
    store_b = PortfolioStore(engine_b)
    tripped_at, reason = store_b.get_kill_switch_state(state.id)

    assert tripped_at is not None
    assert reason == "max drawdown breached"


def test_get_running_run_ids_and_abandon_run(store):
    state = store.get_or_create_portfolio("paper-us", 100_000)

    assert store.get_running_run_ids(state.id) == []

    run_id = store.start_run(state.id, dry_run=False)
    assert store.get_running_run_ids(state.id) == [run_id]

    store.abandon_run(run_id, "crashed mid-flight")
    assert store.get_running_run_ids(state.id) == []


def test_get_running_run_ids_older_than_filters_recent_runs(store):
    state = store.get_or_create_portfolio("paper-us", 100_000)
    run_id = store.start_run(state.id, dry_run=False)

    # Default (no filter) sees it regardless of age.
    assert store.get_running_run_ids(state.id) == [run_id]

    # "older than 6 hours ago" must not match a run that just started.
    six_hours_ago = datetime.now(timezone.utc) - timedelta(hours=6)
    assert store.get_running_run_ids(state.id, older_than=six_hours_ago) == []

    # Backdate it to simulate a genuinely stale/crashed run.
    with store.engine.begin() as conn:
        conn.execute(
            update(paper_runs)
            .where(paper_runs.c.id == run_id)
            .values(started_at=datetime.now(timezone.utc) - timedelta(hours=7))
        )
    assert store.get_running_run_ids(state.id, older_than=six_hours_ago) == [run_id]


def test_create_broker_order_starts_in_submitting_state(store):
    state = store.get_or_create_portfolio("paper-us", 100_000)
    run_id = store.start_run(state.id, dry_run=False)

    order_id = store.create_broker_order(
        state.id, run_id, "AAPL", "BUY", 5, "test signal", client_order_id="cid-1"
    )

    order = store.get_order_by_client_id("cid-1")
    assert order.id == order_id
    assert order.status == "SUBMITTING"
    assert order.broker_order_id is None


def test_get_order_by_client_id_returns_none_when_missing(store):
    assert store.get_order_by_client_id("does-not-exist") is None


def test_mark_order_submitted_updates_status_and_broker_id(store):
    state = store.get_or_create_portfolio("paper-us", 100_000)
    run_id = store.start_run(state.id, dry_run=False)
    order_id = store.create_broker_order(
        state.id, run_id, "AAPL", "BUY", 5, "test", client_order_id="cid-2"
    )

    store.mark_order_submitted(order_id, broker_order_id="alpaca-123", broker_status="new")

    order = store.get_order_by_client_id("cid-2")
    assert order.status == "SUBMITTED"
    assert order.broker_order_id == "alpaca-123"


def test_get_open_broker_orders_excludes_terminal_states(store):
    state = store.get_or_create_portfolio("paper-us", 100_000)
    run_id = store.start_run(state.id, dry_run=False)

    submitting_id = store.create_broker_order(
        state.id, run_id, "AAPL", "BUY", 5, "t", client_order_id="cid-a"
    )
    submitted_id = store.create_broker_order(
        state.id, run_id, "MSFT", "BUY", 3, "t", client_order_id="cid-b"
    )
    store.mark_order_submitted(submitted_id, broker_order_id="alpaca-b", broker_status="new")

    filled_id = store.create_broker_order(
        state.id, run_id, "GOOGL", "BUY", 1, "t", client_order_id="cid-c"
    )
    store.mark_order_submitted(filled_id, broker_order_id="alpaca-c", broker_status="new")
    store.fill_order(filled_id, 100.0, 0.0, 0.0, broker_status="filled")

    open_orders = {o.id for o in store.get_open_broker_orders(state.id)}
    assert open_orders == {submitting_id, submitted_id}


def test_update_broker_order_status_marks_rejected(store):
    state = store.get_or_create_portfolio("paper-us", 100_000)
    run_id = store.start_run(state.id, dry_run=False)
    order_id = store.create_broker_order(
        state.id, run_id, "AAPL", "BUY", 5, "t", client_order_id="cid-d"
    )
    store.mark_order_submitted(order_id, broker_order_id="alpaca-d", broker_status="new")

    store.update_broker_order_status(order_id, "REJECTED", "rejected")

    assert store.get_open_broker_orders(state.id) == []


def test_cancel_pending_order_cancels_submitted_order(store):
    state = store.get_or_create_portfolio("paper-us", 100_000)
    run_id = store.start_run(state.id, dry_run=False)
    order_id = store.create_broker_order(
        state.id, run_id, "AAPL", "BUY", 5, "t", client_order_id="cid-e"
    )

    assert store.cancel_pending_order(order_id) is True
    assert store.get_open_broker_orders(state.id) == []


def test_cancel_pending_order_returns_false_for_already_filled_order(store):
    state = store.get_or_create_portfolio("paper-us", 100_000)
    run_id = store.start_run(state.id, dry_run=False)
    order_id = store.create_pending_order(
        state.id, run_id, "AAPL", "BUY", 5, "t", fill_date=date(2024, 6, 3)
    )
    store.fill_order(order_id, 100.0, 1.0, 0.0)

    assert store.cancel_pending_order(order_id) is False
