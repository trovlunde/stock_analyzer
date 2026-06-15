# Stack & tooling improvement plan

Assessment of the stock-analysis dependency stack, data sources, and ML tooling (June 2026). Tracks incremental upgrades — not a rewrite.

## Current state (baseline)

| Area | Choice | Status |
|------|--------|--------|
| Tooling | Python 3.12+, uv, `uv.lock` | Good |
| Core stack | pandas 2.x, numpy 2.x, scikit-learn 1.9, scipy 1.17 | Current |
| Viz | Plotly 6.x (+ matplotlib/seaborn in research scripts) | Mixed |
| Portfolio math | PyPortfolioOpt, QuantStats metrics in daily reports | Good |
| Trading calendar | pandas-market-calendars | Appropriate |
| Persistence | SQLAlchemy 2 + SQLite | Fine for hobby / paper trading |
| ML evaluation | Temporal holdout, LightGBM + sklearn ensembles | Good |
| Market data | Yahoo Finance via `yfinance` + EDGAR fallback | Improved |
| API | Flask 3.x | Works; FastAPI evaluation deferred (#13) |

**Bottom line:** Phases 1–2 and MarketDataProvider (#10) are complete. Remaining stack work is Phase 3 spikes (#11–#13).

---

## Phase 1 — Do now (high value, low cost) ✅

### 1.1 Fix fundamental classifier cache return mismatch

**Problem:** `prepare_classification_data_cache()` returned inconsistent tuple lengths on cache hit vs miss.

- [x] Return `(X, y)` consistently from `prepare_classification_data_cache()`
- [x] Add a regression test that exercises cache miss → cache hit path

### 1.2 Consolidate technical-analysis libraries

- [x] Migrate movement feature prep to `pandas-ta`
- [x] Remove `ta` from `pyproject.toml`
- [x] Tests pass after migration

### 1.3 Resolve orphaned neural-network module

- [x] **Option C:** Rewrote with sklearn `MLPRegressor` (no TensorFlow)

### 1.4 Pandas `pct_change` deprecation

- [x] Pass `fill_method=None` (or explicit fill) at audited call sites

---

## Phase 2 — Try next (meaningful upgrades) ✅

### 2.1 Gradient-boosted classifiers

- [x] Add **LightGBM** in evaluate CLI
- [x] Compare on same temporal holdout as existing RF pipeline
- [x] Document when to prefer boosters vs ensemble in README

### 2.2 Portfolio analytics library

- [x] QuantStats metrics (Calmar, Sortino, rolling Sharpe) in daily reports
- [x] Integrated into `stock_analysis/trading/report.py`

### 2.3 Supplement fundamentals beyond Yahoo

- [x] SEC EDGAR integration (`edgartools`) as fallback
- [x] Fallback when Yahoo fundamentals are empty
- [x] `yfinance` remains primary price source

### 2.4 Statistical analysis

- [x] **statsmodels** in opt-in `stock_analysis/diagnostics/` module
- [x] Not wired into daily job by default

---

## Phase 3 — Consider later (if scope grows)

### 3.1 Data provider abstraction ✅

- [x] `MarketDataProvider` protocol (prices, fundamentals, corporate actions)
- [x] `YFinanceProvider` implementing current behaviour
- [ ] Optional second provider stub for FMP/Polygon/Stooq (deferred)

### 3.2 Faster batch analytics — [#11](https://github.com/trovlunde/stock_analyzer/issues/11)

Recon (June 2026): batch jobs are **network/I/O bound** (yfinance, rate limits), not pandas CPU bound. DuckDB does not fit current key-value cache without new query patterns.

- [ ] Profile one concrete bottleneck if maintainer reports slowness
- [ ] Spike doc with adopt/defer recommendation
- [ ] No mandatory Polars/DuckDB dependency unless spike shows clear win

### 3.3 API modernisation — [#13](https://github.com/trovlunde/stock_analyzer/issues/13)

- [ ] ADR: Flask vs FastAPI (9 routes today; likely stay on Flask)
- [ ] No migration unless API surface grows materially

### 3.4 Visualisation consolidation — [#12](https://github.com/trovlunde/stock_analyzer/issues/12)

Production paths (`jobs/daily`, `cli/`, `trading/`) already plotly-free of matplotlib. Research scripts still use mpl/seaborn; `plottable` requires matplotlib.

- [ ] Inventory all viz library usages (sub-issue)
- [ ] Quick wins: dead mpl code, plotly heatmaps where trivial
- [ ] Defer `sector_analysis` / deep `movement_classification` plot rewrites
- [ ] Do not remove matplotlib from `pyproject.toml` until `plottable` path resolved

---

## Explicitly deferred (not recommended now)

| Idea | Reason |
|------|--------|
| **Polars / DuckDB** (without profiling) | Batch paths are I/O bound; see #11 recon |
| **FastAPI migration** | Small Flask surface; see #13 |
| **zipline** / **backtrader** | Heavy; custom paper-trading loop is simpler |
| **Deep learning** for price prediction | High overfit risk; sklearn + good features usually better |
| **quantlib** | Steep learning curve; only if options/fixed-income math is required |
| Enterprise data APIs without budget | FMP/Polygon are better but paid; EDGAR is the free fundamentals upgrade |

---

## Related docs

- [database-cache-plan.md](./database-cache-plan.md) — SQLite cache architecture and phase-2 cache scope
- [issues/README.md](./issues/README.md) — GitHub issue index
