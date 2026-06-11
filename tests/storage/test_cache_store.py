from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import insert

from stock_analysis.storage.cache_store import SqlCacheStore
from stock_analysis.storage.legacy_csv import key_to_csv_path
from stock_analysis.storage.schema import cache_entries, create_engine_for_url, init_db


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "cache.db"
    engine = create_engine_for_url(f"sqlite:///{db_path}")
    init_db(engine)
    return SqlCacheStore(engine)


def test_put_and_get_round_trip(store):
    df = pd.DataFrame({"symbol": ["AAPL", "MSFT"], "price": [100.0, 200.0]})
    store.put("market:test", df)

    result = store.get("market:test")

    pd.testing.assert_frame_equal(result, df)


def test_get_returns_none_for_missing_key(store):
    assert store.get("market:missing") is None


def test_ttl_expiry(store):
    from stock_analysis.storage.cache_store import _serialize_dataframe

    df = pd.DataFrame({"symbol": ["AAPL"], "price": [100.0]})
    with store.engine.begin() as conn:
        conn.execute(
            insert(cache_entries).values(
                key="market:ttl",
                payload=_serialize_dataframe(df),
                created_at=datetime.now(timezone.utc) - timedelta(hours=13),
                metadata=None,
            )
        )

    assert store.get("market:ttl", max_age=timedelta(hours=12)) is None


def test_validator_rejects_stale_data(store):
    dates = pd.date_range("2020-01-01", periods=3, freq="D")
    df = pd.DataFrame({"Close": [100.0, 101.0, 102.0]}, index=dates)
    store.put("index:TEST:10y", df)

    assert store.get("index:TEST:10y", validator=lambda data: False) is None


def test_delete_and_clear_prefix(store):
    store.put("market:a", pd.DataFrame({"x": [1]}))
    store.put("market:b", pd.DataFrame({"x": [2]}))
    store.put("index:TEST:1y", pd.DataFrame({"Close": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"])))

    assert store.delete("market:a") is True
    assert store.get("market:a") is None
    assert store.get("market:b") is not None

    deleted = store.clear_prefix("market:")
    assert deleted == 1
    assert store.get("market:b") is None
    assert store.get("index:TEST:1y") is not None


def test_lazy_csv_import_for_market(tmp_path, monkeypatch, store):
    monkeypatch.chdir(tmp_path)
    df = pd.DataFrame({"symbol": ["AAPL"], "price": [150.0]})
    csv_path = key_to_csv_path("market:sp500")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)

    result = store.get("market:sp500", max_age=timedelta(hours=12))

    pd.testing.assert_frame_equal(result, df)
    assert store.get("market:sp500", max_age=timedelta(hours=12)) is not None


def test_lazy_csv_import_for_index(tmp_path, monkeypatch, store):
    monkeypatch.chdir(tmp_path)
    dates = pd.date_range(end=datetime.now(timezone.utc), periods=2, freq="D")
    df = pd.DataFrame({"Close": [100.0, 101.0]}, index=dates)
    csv_path = key_to_csv_path("index:^GSPC:10y")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path)

    result = store.get(
        "index:^GSPC:10y",
        max_age=timedelta(hours=12),
        validator=lambda data: not data.empty,
    )

    assert result is not None
    assert len(result) == 2
