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
