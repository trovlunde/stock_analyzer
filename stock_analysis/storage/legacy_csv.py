import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import pandas as pd


def key_to_csv_path(key: str) -> Path | None:
    if key.startswith("market:"):
        market = key.removeprefix("market:")
        return Path("data") / f"{market}_stock_data.csv"

    if key.startswith("index:"):
        parts = key.split(":", 3)
        if len(parts) < 3:
            return None
        _, ticker, period = parts[0], parts[1], parts[2]
        cache_dir = Path("data") / f"{ticker}_cache"
        if len(parts) == 4:
            start_date = parts[3]
            return cache_dir / f"{ticker}_{period}_{start_date}.csv"
        return cache_dir / f"{ticker}_{period}.csv"

    return None


def _read_csv(path: Path, key: str) -> pd.DataFrame | None:
    try:
        if key.startswith("index:"):
            return pd.read_csv(path, index_col=0, parse_dates=True)
        return pd.read_csv(path)
    except Exception:
        return None


def _file_age_ok(path: Path, max_age: timedelta | None) -> bool:
    if max_age is None:
        return True
    file_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return datetime.now(timezone.utc) - file_time <= max_age


def read_legacy_csv(
    key: str,
    *,
    max_age: timedelta | None = None,
    validator: Callable[[pd.DataFrame], bool] | None = None,
) -> pd.DataFrame | None:
    path = key_to_csv_path(key)
    if path is None or not path.exists():
        return None

    if not _file_age_ok(path, max_age):
        return None

    df = _read_csv(path, key)
    if df is None or df.empty:
        return None

    if validator is not None and not validator(df):
        return None

    return df


def legacy_metadata(path: Path) -> dict:
    file_time = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return {
        "source": "legacy_csv",
        "path": str(path),
        "imported_from_mtime": file_time.isoformat(),
    }


def legacy_metadata_json(path: Path) -> str:
    return json.dumps(legacy_metadata(path))
