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
)
from sqlalchemy.engine import Engine

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
    return create_engine(database_url, future=True)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)
