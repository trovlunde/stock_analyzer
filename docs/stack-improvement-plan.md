# Stack & tooling improvement plan

Assessment of the stock-analysis dependency stack, data sources, and ML tooling (June 2026). Tracks incremental upgrades — not a rewrite.

## Current state (baseline)

| Area | Choice | Status |
|------|--------|--------|
| Tooling | Python 3.12+, uv, `uv.lock` | Good |
| Core stack | pandas 2.x, numpy 2.x, scikit-learn 1.9, scipy 1.17 | Current |
| Viz | Plotly 6.x | Good |
| Portfolio math | PyPortfolioOpt | Standard for MPT |
| Trading calendar | pandas-market-calendars | Appropriate |
| Persistence | SQLAlchemy 2 + SQLite | Fine for hobby / paper trading |
| ML evaluation | Temporal holdout, custom backtests | Sound direction |
| Market data | Yahoo Finance via `yfinance` | Convenient but unreliable |
| API | Flask 3.x | Works; not the newest option |

**Bottom line:** Modern Python scientific stack with sensible ML methodology. Main gaps are data reliability, dependency hygiene, and a few known bugs — not outdated core libraries.

---

## Phase 1 — Do now (high value, low cost)

### 1.1 Fix fundamental classifier cache return mismatch

**Problem:** `prepare_classification_data_cache()` in `fin_statement_classifier.py` returns `(X, y)` on cache hit but `(X, y, returns)` on fresh compute. `train_classifier()` always unpacks two values → `too many values to unpack (expected 2)` after "Cached data for …".

**File:** `stock_analysis/ai/fundamental_analysis/fin_statement_classifier.py`

- [x] Return `(X, y)` consistently from `prepare_classification_data_cache()` (drop `returns` from the return tuple, or unpack three values everywhere callers expect it)
- [x] Add a regression test that exercises cache miss → cache hit path

### 1.2 Consolidate technical-analysis libraries

**Problem:** Both `ta` and `pandas-ta` are dependencies; different modules use different packages.

| Module | Current |
|--------|---------|
| `analyze_stock_data.py` | `pandas_ta` |
| `prepare_classification_data.py` | `pandas-ta` |

- [x] Migrate `prepare_classification_data.py` to `pandas-ta`
- [x] Remove `ta` from `pyproject.toml`
- [ ] Run tests / smoke-eval on movement classifier after migration

### 1.3 Resolve orphaned neural-network module

**Problem:** `nn_predictor.py` imports TensorFlow, Keras, and keras-tuner but they are not declared in `pyproject.toml`.

Pick one:

- [ ] **Option A:** Add `tensorflow`, `keras-tuner` (or equivalent) to dependencies and document optional GPU / install size
- [ ] **Option B:** Mark module deprecated and remove from active CLI paths
- [ ] **Option C:** Rewrite as sklearn `MLPRegressor` to avoid TensorFlow entirely

### 1.4 Pandas `pct_change` deprecation

**Problem:** `FutureWarning` on `Series.pct_change()` default `fill_method='pad'` (removed in future pandas).

**File:** `stock_analysis/ai/fundamental_analysis/fin_statement_classifier.py` (and any other call sites)

- [ ] Pass `fill_method=None` (or explicit fill before `pct_change`) everywhere `pct_change` is used

---

## Phase 2 — Try next (meaningful upgrades)

### 2.1 Gradient-boosted classifiers

sklearn ensembles (RandomForest, SVM, kNN) are fine for tabular features; tree boosters often outperform RF on structured financial features.

- [ ] Add **LightGBM** or **XGBoost** as optional classifier in `stock_analysis/cli/evaluate_models.py`
- [ ] Compare on same temporal holdout as existing RF pipeline (`stock-analysis-evaluate compare`)
- [ ] Document when to prefer boosters vs ensemble in README or CLI help

### 2.2 Portfolio analytics library

Custom helpers in `stock_analysis/ai/metrics.py` cover basics (Sharpe, max drawdown). Richer reports would help daily job output.

- [ ] Evaluate **QuantStats** or **empyrical** for Calmar, Sortino, rolling Sharpe, tear sheets
- [ ] Integrate into `stock_analysis/trading/report.py` and `data/reports/YYYY-MM-DD.md` output

### 2.3 Supplement fundamentals beyond Yahoo

`yfinance` fundamentals and earnings are inconsistent for some tickers.

- [ ] Spike **SEC EDGAR** integration (`edgartools` or `sec-edgar-downloader`) for authoritative statements
- [ ] Use as fallback when `ticker.financials` / `ticker.earnings` are empty or stale
- [ ] Keep `yfinance` as primary price source unless a paid API is adopted

### 2.4 Statistical analysis

- [ ] Add **statsmodels** for time-series diagnostics (stationarity, regression, optional ARIMA/GARCH experiments)
- [ ] Use only where it replaces ad-hoc stats — not as a blanket dependency

---

## Phase 3 — Consider later (if scope grows)

### 3.1 Data provider abstraction

- [ ] Introduce a thin `MarketDataProvider` protocol (prices, fundamentals, corporate actions)
- [ ] Implement `YFinanceProvider` (current behaviour)
- [ ] Optional second provider: FMP, Polygon, Alpha Vantage, or Stooq for historical OHLCV

### 3.2 Faster batch analytics

- [ ] **Polars** for bulk Finviz / SP500 scans where pandas becomes a bottleneck
- [ ] **DuckDB** for analytical queries over Parquet cache and SQLite without full DataFrame loads

### 3.3 API modernisation

- [ ] Evaluate **FastAPI** migration from Flask if API surface grows (OpenAPI, async, typing)
- [ ] Not urgent for current hobby scope

### 3.4 Visualisation consolidation

Currently: matplotlib, seaborn, plotly.

- [ ] Standardise on **plotly** for interactive charts
- [ ] Drop matplotlib/seaborn unless static publication plots are needed

---

## Explicitly deferred (not recommended now)

| Idea | Reason |
|------|--------|
| **zipline** / **backtrader** | Heavy; custom paper-trading loop is simpler and lookahead is easier to control |
| **Deep learning** for price prediction | High overfit risk; sklearn + good features usually better for this use case |
| **quantlib** | Steep learning curve; only if options/fixed-income math is required |
| Enterprise data APIs without budget | FMP/Polygon are better but paid; EDGAR is the free fundamentals upgrade |

---

## Related docs

- [database-cache-plan.md](./database-cache-plan.md) — SQLite cache architecture and phase-2 cache scope
