from datetime import datetime, timezone

import pytest
from sqlalchemy import (
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    Table,
    Text,
    insert,
    inspect,
    select,
)
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError, OperationalError

from stock_analysis.storage.schema import create_engine_for_url, init_db, run_light_migrations


def _old_shape_engine(tmp_path):
    """Build a DB with paper_portfolios as it looked before the kill-switch columns existed."""
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'old.db'}")
    old_metadata = MetaData()
    old_paper_portfolios = Table(
        "paper_portfolios",
        old_metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("name", Text, nullable=False, unique=True),
        Column("cash", Float, nullable=False),
        Column("created_at", DateTime, nullable=False),
        Column("updated_at", DateTime, nullable=False),
    )
    old_metadata.create_all(engine)
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        conn.execute(
            insert(old_paper_portfolios).values(
                name="existing-portfolio", cash=1234.5, created_at=now, updated_at=now
            )
        )
    return engine, old_paper_portfolios


def test_run_light_migrations_adds_missing_columns(tmp_path):
    engine, _ = _old_shape_engine(tmp_path)

    run_light_migrations(engine)

    inspector = inspect(engine)
    columns = {c["name"] for c in inspector.get_columns("paper_portfolios")}
    assert "kill_switch_tripped_at" in columns
    assert "kill_switch_reason" in columns


def _old_shape_orders_engine(tmp_path):
    """paper_orders as it looked before the broker-order columns existed."""
    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'old_orders.db'}")
    old_metadata = MetaData()
    old_paper_orders = Table(
        "paper_orders",
        old_metadata,
        Column("id", Integer, primary_key=True, autoincrement=True),
        Column("portfolio_id", Integer, nullable=False),
        Column("ticker", Text, nullable=False),
        Column("side", Text, nullable=False),
        Column("shares", Float, nullable=False),
        Column("status", Text, nullable=False),
        Column("created_at", DateTime, nullable=False),
    )
    old_metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(
            insert(old_paper_orders).values(
                portfolio_id=1,
                ticker="AAPL",
                side="BUY",
                shares=10,
                status="PENDING",
                created_at=datetime.now(timezone.utc),
            )
        )
    return engine, old_paper_orders


def test_run_light_migrations_adds_broker_order_columns(tmp_path):
    engine, old_table = _old_shape_orders_engine(tmp_path)

    run_light_migrations(engine)

    inspector = inspect(engine)
    columns = {c["name"] for c in inspector.get_columns("paper_orders")}
    assert {"client_order_id", "broker_order_id", "broker_status"} <= columns

    with engine.connect() as conn:
        row = conn.execute(select(old_table.c.ticker, old_table.c.status)).mappings().first()
    assert row["ticker"] == "AAPL"
    assert row["status"] == "PENDING"


def test_client_order_id_unique_index_rejects_duplicates(tmp_path):
    from sqlalchemy import insert

    from stock_analysis.storage.schema import paper_orders

    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'fresh.db'}")
    init_db(engine)

    with engine.begin() as conn:
        conn.execute(
            insert(paper_orders).values(
                portfolio_id=1,
                ticker="AAPL",
                side="BUY",
                shares=5,
                status="SUBMITTING",
                created_at=datetime.now(timezone.utc),
                client_order_id="dup-key",
            )
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                insert(paper_orders).values(
                    portfolio_id=1,
                    ticker="MSFT",
                    side="BUY",
                    shares=5,
                    status="SUBMITTING",
                    created_at=datetime.now(timezone.utc),
                    client_order_id="dup-key",
                )
            )


def test_client_order_id_unique_index_allows_multiple_nulls(tmp_path):
    from sqlalchemy import insert

    from stock_analysis.storage.schema import paper_orders

    engine = create_engine_for_url(f"sqlite:///{tmp_path / 'fresh2.db'}")
    init_db(engine)

    with engine.begin() as conn:
        for ticker in ("AAPL", "MSFT"):
            conn.execute(
                insert(paper_orders).values(
                    portfolio_id=1,
                    ticker=ticker,
                    side="BUY",
                    shares=5,
                    status="PENDING",
                    created_at=datetime.now(timezone.utc),
                    client_order_id=None,
                )
            )  # must not raise — legacy PaperBroker rows have no client_order_id


def test_run_light_migrations_adds_unique_index_to_existing_db(tmp_path):
    engine, _ = _old_shape_orders_engine(tmp_path)

    run_light_migrations(engine)
    run_light_migrations(engine)  # idempotent — must not raise on a second pass

    from sqlalchemy import insert

    from stock_analysis.storage.schema import paper_orders

    with engine.begin() as conn:
        conn.execute(
            insert(paper_orders).values(
                portfolio_id=1,
                ticker="MSFT",
                side="BUY",
                shares=5,
                status="SUBMITTING",
                created_at=datetime.now(timezone.utc),
                client_order_id="dup-key",
            )
        )

    with pytest.raises(IntegrityError):
        with engine.begin() as conn:
            conn.execute(
                insert(paper_orders).values(
                    portfolio_id=1,
                    ticker="GOOGL",
                    side="BUY",
                    shares=5,
                    status="SUBMITTING",
                    created_at=datetime.now(timezone.utc),
                    client_order_id="dup-key",
                )
            )


def test_run_light_migrations_preserves_existing_rows(tmp_path):
    engine, old_table = _old_shape_engine(tmp_path)

    run_light_migrations(engine)

    with engine.connect() as conn:
        row = conn.execute(select(old_table.c.name, old_table.c.cash)).mappings().first()
    assert row["name"] == "existing-portfolio"
    assert row["cash"] == 1234.5


def test_run_light_migrations_is_idempotent(tmp_path):
    engine, _ = _old_shape_engine(tmp_path)

    run_light_migrations(engine)
    run_light_migrations(engine)  # must not raise on the second pass

    inspector = inspect(engine)
    columns = [c["name"] for c in inspector.get_columns("paper_portfolios")]
    assert columns.count("kill_switch_tripped_at") == 1


def test_run_light_migrations_tolerates_concurrent_duplicate_column_race(
    tmp_path, monkeypatch
):
    """Simulates another process having already applied the same ALTER TABLE between
    this process's column-existence check and its own ALTER attempt."""
    engine, _ = _old_shape_engine(tmp_path)
    original_execute = Connection.execute

    def flaky_execute(self, statement, *args, **kwargs):
        if "ALTER TABLE paper_portfolios ADD COLUMN kill_switch_tripped_at" in str(statement):
            raise OperationalError(
                str(statement), {}, Exception("duplicate column name: kill_switch_tripped_at")
            )
        return original_execute(self, statement, *args, **kwargs)

    monkeypatch.setattr(Connection, "execute", flaky_execute)

    run_light_migrations(engine)  # must not raise
