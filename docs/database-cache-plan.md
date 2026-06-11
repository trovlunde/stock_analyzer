# Database cache plan — Phase 1

Replace scattered CSV/Parquet file caches with a SQLite-backed `CacheStore`, plug-and-play into existing call sites. Decisions captured from design review (June 2026).

## Decisions (locked)

| # | Topic | Decision |
|---|--------|----------|
| 1 | Scope | Phase 1: `cache_data.py` + `helpers.get_index_data()` only |
| 2 | Engine | SQLite now; Postgres later via same `DATABASE_URL` |
| 3 | Storage | Generic `cache_entries` table; Parquet bytes in `payload` |
| 4 | Read API | `get(key, max_age=..., validator=...)` — TTL in store, domain rules via validator |
| 5 | CSV migration | Lazy import on read (SQLite miss → valid legacy CSV → `put` → return) |
| 6 | Stack | SQLAlchemy Core (no ORM) |
| 7 | Config | `DATABASE_URL` env var; default `sqlite:///data/stock_analysis.db` |
| 8 | Keys | Namespaced: `market:{market}`, `index:{ticker}:{period}`, `index:{ticker}:{period}:{start_date}` |
| 9 | Docker | Mount `./data:/app/data` in `docker-compose.yml` |
| 10 | Ops API | `delete` / `clear_prefix` on store only — no HTTP/CLI in phase 1 |
| 11 | Tests | Unit tests for `CacheStore` (TTL, validator, lazy CSV, delete) |

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│  cache_data.py      │     │  helpers.py         │
│  get/save market    │     │  get_index_data()   │
└──────────┬──────────┘     └──────────┬──────────┘
           │                           │
           └───────────┬───────────────┘
                       ▼
              ┌─────────────────┐
              │  get_cache_store()│  singleton
              └────────┬────────┘
                       ▼
              ┌─────────────────┐
              │  SqlCacheStore    │
              │  put / get / del  │
              └────────┬────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
    ┌──────────────┐        ┌──────────────┐
    │ SQLite DB    │        │ legacy_csv   │
    │ cache_entries│        │ (lazy import)│
    └──────────────┘        └──────────────┘
```

### Schema

```sql
CREATE TABLE IF NOT EXISTS cache_entries (
    key         TEXT PRIMARY KEY,
    payload     BLOB NOT NULL,           -- DataFrame as Parquet bytes
    created_at  TIMESTAMP NOT NULL,
    metadata    TEXT                     -- optional JSON for inspection
);
```

### Cache key → legacy CSV mapping

| Key | Legacy path |
|-----|-------------|
| `market:sp500` | `data/sp500_stock_data.csv` |
| `index:^GSPC:10y` | `data/^GSPC_cache/^GSPC_10y.csv` |
| `index:^GSPC:10y:2020-01-01` | `data/^GSPC_cache/^GSPC_10y_2020-01-01.csv` |

### `CacheStore` interface

```python
class CacheStore(Protocol):
    def get(
        self,
        key: str,
        *,
        max_age: timedelta | None = None,
        validator: Callable[[pd.DataFrame], bool] | None = None,
    ) -> pd.DataFrame | None: ...

    def put(
        self,
        key: str,
        df: pd.DataFrame,
        *,
        metadata: dict | None = None,
    ) -> None: ...

    def delete(self, key: str) -> bool: ...

    def clear_prefix(self, prefix: str) -> int: ...
```

### Validators (call-site, not in store)

- **Market cache** (`cache_data.py`): `max_age=timedelta(hours=12)` only.
- **Index OHLCV** (`helpers.py`): `max_age=timedelta(hours=12)` plus validator that latest index date is ≤1 day old (preserves current behaviour).

---

## Phase 1 implementation steps

### 1. Dependencies

Add to `pyproject.toml`:

```toml
sqlalchemy = "^2.0"
```

Install with `uv sync`.

### 2. New module: `stock_analysis/storage/`

| File | Responsibility |
|------|----------------|
| `__init__.py` | `get_cache_store()` singleton; reads `DATABASE_URL`; calls `init_db()` once |
| `schema.py` | SQLAlchemy Core `cache_entries` table; `init_db(engine)` |
| `cache_store.py` | `SqlCacheStore` — Parquet serialize/deserialize, TTL, validator, CRUD |
| `legacy_csv.py` | Parse key → CSV path; read CSV with same index_col rules as today; apply TTL via file mtime |

### 3. Wire `cache_data.py`

Keep public API unchanged (`get_cached_stock_data`, `save_stock_data`):

```python
def get_cached_stock_data(market):
    store = get_cache_store()
    return store.get(f"market:{market}", max_age=timedelta(hours=12))

def save_stock_data(stocks_df, market):
    get_cache_store().put(f"market:{market}", stocks_df)
```

Remove direct `Path('data')` / CSV logic.

### 4. Wire `helpers.get_index_data()`

- Build key: `index:{ticker}:{period}` or `index:{ticker}:{period}:{start_date}`.
- Replace file read/write blocks with `store.get(...)` / `store.put(...)`.
- Extract `_index_data_fresh(df) -> bool` for the ≤1-day-latest-row rule.
- Keep yfinance download logic as-is on cache miss.
- Optional: replace `print` debug lines with `logging` (not required for phase 1).

### 5. Configuration

- `.env.example` (if missing): `DATABASE_URL=sqlite:///data/stock_analysis.db`
- Ensure `data/` directory is created before engine connect (SQLite relative path).
- `data/` already in `.gitignore` — DB file is not committed.

### 6. Docker

In `docker-compose.yml` under `stock-analysis`:

```yaml
volumes:
  - ./models:/app/models
  - ./data:/app/data
environment:
  - DATABASE_URL=sqlite:////app/data/stock_analysis.db
```

### 7. Tests

Add `tests/storage/test_cache_store.py` (pytest):

- Round-trip `put` → `get`
- TTL: entry older than `max_age` returns `None`
- Validator: returns `None` when validator fails
- Lazy CSV: seed a temp CSV matching key map → `get` imports and persists to DB
- `delete` / `clear_prefix`
- Use in-memory SQLite: `DATABASE_URL=sqlite:///:memory:`

### 8. README

Short section under "Data Caching": SQLite backend, `DATABASE_URL`, phase-1 scope, link to this doc.

---

## Verification checklist

- [ ] CLI `main.py` market flow uses DB (no new CSV writes for market keys).
- [ ] `get_index_data('^GSPC')` hits DB on second call within 12h.
- [ ] Existing CSV on disk is imported on first read after upgrade.
- [ ] Docker restart preserves cache (`./data` mount).
- [ ] `pytest tests/storage/` passes.

---

# Saved for later (Phase 2+)

Items explicitly deferred — not forgotten.

## Cache layers not in Phase 1

| Layer | Module | Format today | Suggested key prefix | TTL today |
|-------|--------|--------------|----------------------|-----------|
| Finviz returns | `ai/finviz_classifier/data_fetching.py` | `stock_returns_cache.parquet` | `finviz:return:{ticker}:{signal_date}` or single `finviz:returns` blob | Row-level `is_complete` |
| Technical classifier features | `ai/technical_analysis/prepare_classification_data.py` | CSV per symbol | `technical:{symbol}:{weekly}:{threshold}` | 24h |
| Fundamental classifier features | `ai/fundamental_analysis/fin_statement_classifier.py` | CSV per ticker (data + targets) | `fin_classifier:{ticker}:features` / `:targets` | 30 days |
| ML model in-memory cache | `ai/model_manager.py` | RAM dict | N/A — different concern | 60 min |

**Notes for phase 2:**

- Finviz cache uses row-level dedup and `is_complete` flags — may need a validator or a dedicated table instead of one blob key.
- Classifier caches store *computed features*, not raw OHLCV — same `CacheStore` works with different key prefixes and TTLs.
- `ModelManager` cache is about loaded pickle models, not fetched data — keep separate unless you unify all persistence.

## Infrastructure & ops (deferred)

| Item | Why deferred |
|------|----------------|
| PostgreSQL / Supabase | SQLite sufficient for hobby + single-user API; swap via `DATABASE_URL` when needed |
| `CACHE_DATABASE_URL` separate env var | Rejected; one `DATABASE_URL` is enough |
| Flask `DELETE /api/v1/cache` endpoint | Phase 1 uses store methods only; add with auth/scoping in phase 2 |
| CLI cache-clear menu option | Same as above |
| Alembic migrations | Single table + `CREATE IF NOT EXISTS` is enough until schema grows |
| Delete legacy CSV files automatically | Lazy import leaves files in place for safe rollback; document manual cleanup |
| Dual-write CSV + DB transition | Rejected in favour of lazy read migration |
| Async store (`aiosqlite`) | Flask app is sync; no benefit now |

## Code quality (optional follow-ups)

- Replace `print` debug in `helpers.get_index_data()` with `logging`.
- Consolidate duplicate cache-read blocks inside `helpers.py` (two near-identical branches for `start_date` vs not).
- Typed tables (`market_stock_data`, `index_ohlcv`) — rejected for phase 1; revisit if querying inside SQL becomes a requirement.

## Postgres migration path (when ready)

1. Provision Postgres (local Docker or Supabase).
2. Set `DATABASE_URL=postgresql://...`.
3. Same `SqlCacheStore` + SQLAlchemy Core — no call-site changes.
4. Optional: one-off script to copy `cache_entries` from SQLite export to Postgres.
5. Consider Alembic once classifier caches add tables or indexes.

## Phase 2 suggested order

1. Finviz returns parquet → `CacheStore` (highest value; already parquet-shaped).
2. Technical + fundamental classifier CSV caches.
3. Flask cache management endpoint (if API consumers need it).
4. Postgres if deploying multi-instance or need concurrent writes.
