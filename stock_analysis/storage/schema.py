from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    LargeBinary,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError

metadata = MetaData()

cache_entries = Table(
    "cache_entries",
    metadata,
    Column("key", Text, primary_key=True),
    Column("payload", LargeBinary, nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("metadata", Text, nullable=True),
)

paper_portfolios = Table(
    "paper_portfolios",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", Text, nullable=False, unique=True),
    Column("cash", Float, nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("updated_at", DateTime, nullable=False),
    Column("kill_switch_tripped_at", DateTime, nullable=True),
    Column("kill_switch_reason", Text, nullable=True),
)

paper_positions = Table(
    "paper_positions",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("portfolio_id", Integer, nullable=False),
    Column("ticker", Text, nullable=False),
    Column("shares", Float, nullable=False),
    Column("avg_cost", Float, nullable=False),
    Column("opened_at", DateTime, nullable=False),
    UniqueConstraint("portfolio_id", "ticker", name="uq_paper_positions_portfolio_ticker"),
)

paper_orders = Table(
    "paper_orders",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("portfolio_id", Integer, nullable=False),
    Column("run_id", Integer, nullable=True),
    Column("ticker", Text, nullable=False),
    Column("side", Text, nullable=False),
    Column("shares", Float, nullable=False),
    Column("status", Text, nullable=False),
    Column("signal_reason", Text, nullable=True),
    Column("fill_price", Float, nullable=True),
    Column("commission", Float, nullable=True),
    Column("slippage", Float, nullable=True),
    Column("created_at", DateTime, nullable=False),
    Column("filled_at", DateTime, nullable=True),
    Column("fill_date", Date, nullable=True),
)

paper_runs = Table(
    "paper_runs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("portfolio_id", Integer, nullable=False),
    Column("started_at", DateTime, nullable=False),
    Column("finished_at", DateTime, nullable=True),
    Column("status", Text, nullable=False),
    Column("summary_json", Text, nullable=True),
    Column("dry_run", Boolean, nullable=False, default=False),
)

paper_signals = Table(
    "paper_signals",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, nullable=False),
    Column("ticker", Text, nullable=False),
    Column("action", Text, nullable=False),
    Column("reason", Text, nullable=False),
    Column("metadata_json", Text, nullable=True),
)


def create_engine_for_url(database_url: str) -> Engine:
    engine = create_engine(database_url, future=True)
    if database_url.startswith("sqlite:"):
        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.close()
    return engine


_NEW_COLUMNS = {
    "paper_portfolios": [
        ("kill_switch_tripped_at", "DATETIME"),
        ("kill_switch_reason", "TEXT"),
    ],
}


def run_light_migrations(engine: Engine) -> None:
    """Add newly-introduced nullable columns to existing tables. SQLite supports
    ADD COLUMN without a table rewrite — this is the project's standard no-Alembic
    migration approach (see docs/database-cache-plan.md).

    The existence check and the ALTER aren't atomic across processes — two containers
    starting up against the same fresh-but-old DB file could both decide a column is
    missing. Tolerate that race rather than crash: SQLite's error message for it is
    stable ("duplicate column name"), so swallow only that one case."""
    inspector = inspect(engine)
    with engine.begin() as conn:
        for table, columns in _NEW_COLUMNS.items():
            if table not in inspector.get_table_names():
                continue
            existing = {c["name"] for c in inspector.get_columns(table)}
            for name, sqltype in columns:
                if name not in existing:
                    try:
                        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {sqltype}"))
                    except OperationalError as exc:
                        if "duplicate column name" not in str(exc).lower():
                            raise


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)
    run_light_migrations(engine)
