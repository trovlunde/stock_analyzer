import io
import json
from datetime import datetime, timedelta, timezone
from typing import Callable

import pandas as pd
from sqlalchemy import delete, insert, select
from sqlalchemy.engine import Engine

from .legacy_csv import key_to_csv_path, legacy_metadata_json, read_legacy_csv
from .schema import cache_entries


def _serialize_dataframe(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_parquet(buffer, index=True)
    return buffer.getvalue()


def _deserialize_dataframe(payload: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(payload))


def _normalize_created_at(created_at: datetime) -> datetime:
    if created_at.tzinfo is None:
        return created_at.replace(tzinfo=timezone.utc)
    return created_at.astimezone(timezone.utc)


def _is_entry_fresh(created_at: datetime, max_age: timedelta | None) -> bool:
    if max_age is None:
        return True
    return datetime.now(timezone.utc) - _normalize_created_at(created_at) <= max_age


class SqlCacheStore:
    def __init__(self, engine: Engine):
        self.engine = engine

    def get(
        self,
        key: str,
        *,
        max_age: timedelta | None = None,
        validator: Callable[[pd.DataFrame], bool] | None = None,
    ) -> pd.DataFrame | None:
        row = self._fetch_row(key)
        if row is not None:
            created_at, payload, _ = row
            if _is_entry_fresh(created_at, max_age):
                df = _deserialize_dataframe(payload)
                if validator is None or validator(df):
                    return df

        legacy_df = read_legacy_csv(key, max_age=max_age, validator=validator)
        if legacy_df is None:
            return None

        path = key_to_csv_path(key)
        metadata = None
        if path is not None:
            metadata = json.loads(legacy_metadata_json(path))
        self.put(key, legacy_df, metadata=metadata)
        return legacy_df

    def put(
        self,
        key: str,
        df: pd.DataFrame,
        *,
        metadata: dict | None = None,
    ) -> None:
        payload = _serialize_dataframe(df)
        metadata_json = json.dumps(metadata) if metadata is not None else None
        created_at = datetime.now(timezone.utc)

        with self.engine.begin() as conn:
            conn.execute(delete(cache_entries).where(cache_entries.c.key == key))
            conn.execute(
                insert(cache_entries).values(
                    key=key,
                    payload=payload,
                    created_at=created_at,
                    metadata=metadata_json,
                )
            )

    def delete(self, key: str) -> bool:
        with self.engine.begin() as conn:
            result = conn.execute(
                delete(cache_entries).where(cache_entries.c.key == key)
            )
            return result.rowcount > 0

    def clear_prefix(self, prefix: str) -> int:
        with self.engine.begin() as conn:
            result = conn.execute(
                delete(cache_entries).where(cache_entries.c.key.startswith(prefix))
            )
            return result.rowcount

    def _fetch_row(self, key: str) -> tuple[datetime, bytes, str | None] | None:
        with self.engine.connect() as conn:
            row = conn.execute(
                select(
                    cache_entries.c.created_at,
                    cache_entries.c.payload,
                    cache_entries.c.metadata,
                ).where(cache_entries.c.key == key)
            ).first()

        if row is None:
            return None

        return row.created_at, row.payload, row.metadata
