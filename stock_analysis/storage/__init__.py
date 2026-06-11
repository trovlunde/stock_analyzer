import os
from pathlib import Path

from dotenv import load_dotenv

from .cache_store import SqlCacheStore
from .schema import create_engine_for_url, init_db

DEFAULT_DATABASE_URL = "sqlite:///data/stock_analysis.db"

_store: SqlCacheStore | None = None


def get_database_url() -> str:
    load_dotenv()
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def _ensure_sqlite_parent_dir(database_url: str) -> None:
    if not database_url.startswith("sqlite:///"):
        return
    if database_url.startswith("sqlite:////"):
        db_path = database_url.removeprefix("sqlite://")
    else:
        db_path = database_url.removeprefix("sqlite:///")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)


def get_db_engine():
    database_url = get_database_url()
    _ensure_sqlite_parent_dir(database_url)
    engine = create_engine_for_url(database_url)
    init_db(engine)
    return engine


def get_cache_store() -> SqlCacheStore:
    global _store
    if _store is None:
        database_url = get_database_url()
        _ensure_sqlite_parent_dir(database_url)
        engine = create_engine_for_url(database_url)
        init_db(engine)
        _store = SqlCacheStore(engine)
    return _store


def reset_cache_store() -> None:
    global _store
    _store = None
