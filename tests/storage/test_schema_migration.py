from datetime import datetime, timezone

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
from sqlalchemy.exc import OperationalError

from stock_analysis.storage.schema import create_engine_for_url, run_light_migrations


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
