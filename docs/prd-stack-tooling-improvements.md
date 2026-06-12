# PRD: Stack & tooling improvements for stock analysis

**Status:** Tracked on GitHub  
**Source:** Stack assessment and [stack-improvement-plan.md](./stack-improvement-plan.md) (June 2026)  
**Epic:** [#1 Stack & tooling improvements](https://github.com/trovlunde/stock_analyzer/issues/1)  
**Issue index:** [docs/issues/README.md](./issues/README.md)

---

## Problem Statement

The stock-analysis project runs on a modern Python scientific stack (uv, pandas 2.x, scikit-learn, temporal holdout evaluation, SQLite-backed caching, paper trading), but several gaps limit reliability and maintainability:

- **Batch fundamental classifier training fails** for many tickers after caching (`too many values to unpack`), blocking multi-ticker training workflows.
- **Dependency hygiene is inconsistent**: two technical-analysis libraries (`ta` and `pandas-ta`), an orphaned neural-network module that imports TensorFlow without declaring it, and deprecation warnings from pandas API changes.
- **Market data is single-sourced** through Yahoo Finance (`yfinance`), which is convenient but unreliable for fundamentals, earnings, and corporate actions — visible in production logs (missing trading days, sparse financials).
- **ML and portfolio tooling lag behind low-effort upgrades**: sklearn ensembles only, hand-rolled performance metrics, no structured path to better classifiers or richer daily reports.
- **No abstraction for data providers**, making it hard to add SEC EDGAR or paid APIs later without touching many call sites.

The user wants incremental, phased improvements — not a rewrite — documented and executable as tracer-bullet work items.

## Solution

Deliver a phased improvement programme across three horizons:

1. **Phase 1 (immediate):** Fix known bugs, unify technical-analysis dependencies, resolve the orphaned NN module, and silence pandas deprecations.
2. **Phase 2 (next):** Add optional gradient-boosted classifiers to the evaluation CLI, richer portfolio analytics in daily reports, SEC EDGAR as a fundamentals fallback, and statsmodels where it replaces ad-hoc statistics.
3. **Phase 3 (later):** Introduce a market-data provider protocol, optional Polars/DuckDB for batch analytics, and consolidate visualization dependencies.

Each phase ships independently; later phases must not block Phase 1.

## User Stories

### Phase 1 — Reliability & hygiene

1. As a developer running fundamental classifier training on the SP500 watchlist, I want multi-ticker batch training to complete without unpack errors, so that I can train on the full universe reliably.
2. As a developer, I want the fundamental classification cache to return a consistent shape on cache hit and cache miss, so that callers never depend on implicit tuple length.
3. As a developer, I want a regression test for the fundamental cache round-trip (miss → write → hit → read), so that this bug cannot reappear silently.
4. As a maintainer, I want a single technical-analysis library across stock analysis and movement classification, so that indicator behaviour is consistent and dependencies are simpler.
5. As a maintainer, I want movement-classifier feature preparation migrated from `ta` to `pandas-ta`, so that we align with the rest of the codebase.
6. As a maintainer, I want the legacy `ta` package removed from project dependencies, so that `uv sync` installs fewer overlapping packages.
7. As a user running movement classification after the TA migration, I want identical or equivalent features (RSI, returns, volatility), so that model behaviour does not regress unexpectedly.
8. As a developer opening `nn_predictor`, I want either working declared dependencies or a clear deprecation path, so that I am not surprised by import errors.
9. As a maintainer, I want the neural-network predictor either wired with explicit optional dependencies, rewritten with sklearn MLP, or removed from active CLI paths, so that the documented API matches what installs.
10. As a developer, I want README and CLI docs to reflect the chosen NN strategy, so that optional heavy installs are discoverable.
11. As a developer running fundamental analysis, I want no `pct_change` FutureWarnings, so that logs stay clean ahead of pandas 3.x.
12. As a maintainer, I want all `pct_change` call sites audited for explicit `fill_method`, so that deprecation policy is applied project-wide.

### Phase 2 — Capability upgrades

13. As a quant hobbyist, I want LightGBM or XGBoost available in `stock-analysis-evaluate compare`, so that I can compare boosters against the existing RandomForest ensemble on the same temporal holdout.
14. As a quant hobbyist, I want booster evaluation to use the same holdout-months and extra-features flags as existing classifiers, so that comparisons are fair.
15. As a maintainer, I want CLI help or README guidance on when to prefer boosters vs sklearn ensembles, so that users make informed choices.
16. As a paper-trading user, I want daily markdown reports to include Calmar, Sortino, and rolling Sharpe (via QuantStats or empyrical), so that I get institutional-style metrics without hand-rolling them.
17. As a paper-trading user, I want portfolio metrics integrated into the existing daily job report output, so that I do not need a separate analytics script.
18. As a fundamental-analysis user, I want SEC EDGAR financial statements as a fallback when Yahoo fundamentals are empty, so that valuation and fin-statement classifiers work for more tickers.
19. As a maintainer, I want EDGAR access behind the same helper surface used for `get_ticker_financials`, so that call sites do not branch on provider details.
20. As a developer, I want `yfinance` to remain the primary price source in Phase 2, so that we do not introduce paid API costs prematurely.
21. As a researcher, I want statsmodels available for stationarity checks and regression diagnostics, so that I can validate features before adding them to classifiers.
22. As a maintainer, I want statsmodels used only where it replaces duplicated logic, so that dependency weight stays proportional to value.

### Phase 3 — Scale & architecture (deferred until needed)

23. As a maintainer, I want a `MarketDataProvider` protocol (prices, fundamentals, corporate actions), so that new data sources plug in without widespread refactors.
24. As a maintainer, I want a `YFinanceProvider` implementing current behaviour, so that Phase 3 is a refactor, not a behaviour change.
25. As a maintainer, I want an optional second provider interface stub for FMP/Polygon/Stooq, so that paid upgrades have a clear extension point.
26. As a developer running bulk Finviz or SP500 scans, I want Polars considered for hot paths where pandas is slow, so that batch jobs finish faster.
27. As a developer, I want DuckDB considered for querying Parquet/SQLite cache without full DataFrame loads, so that analytics on large caches stay memory-efficient.
28. As an API consumer, I want FastAPI evaluated only if the Flask surface grows materially, so that we do not migrate prematurely.
29. As a maintainer, I want visualization dependencies consolidated toward Plotly, so that matplotlib/seaborn do not linger without purpose.

### Cross-cutting

30. As a maintainer, I want each phase tracked in the stack improvement plan with checkboxes updated on completion, so that progress is visible in-repo.
31. As a developer, I want new dependencies added via `uv` and recorded in `pyproject.toml` + `uv.lock`, so that reproducibility is preserved.
32. As a developer, I want Phase 1 changes covered by pytest before Phase 2 dependencies land, so that the foundation is stable.

## Implementation Decisions

### Deep modules (testable, stable interfaces)

| Module | Responsibility | Phase |
|--------|----------------|-------|
| **Fundamental classification cache** | Encapsulate read/write/TTL for fin-statement classifier features and targets; always expose `(features, targets)` tuple | 1 |
| **Technical indicator facade** | Single entry point for RSI, MACD, and movement-classifier features using `pandas-ta`; hides library choice from callers | 1 |
| **Neural predictor policy** | Either optional TensorFlow extra, sklearn MLP replacement, or explicit deprecation shim that fails fast with a clear message | 1 |
| **Classifier evaluation backend** | Extend existing evaluate CLI with pluggable classifier factories (sklearn ensemble, LightGBM, XGBoost) sharing temporal holdout split | 2 |
| **Portfolio metrics reporter** | Wrap QuantStats/empyrical for tear-sheet metrics consumed by daily job report generator | 2 |
| **Fundamentals provider** | Protocol + YFinance implementation + EDGAR fallback for statements when Yahoo returns empty | 2 |
| **Market data provider** | Protocol for prices/fundamentals/actions; YFinance adapter; optional future paid adapters | 3 |

### Phase 1 technical decisions

- **Fundamental cache contract:** `prepare_classification_data_cache` returns exactly two DataFrames `(X, y)` on all success paths; `returns` remain internal or persisted separately if needed later.
- **TA consolidation:** Map existing `ta` indicators in movement `prepare_classification_data` to `pandas-ta` equivalents; verify column names via existing `get_features` contract.
- **NN module (recommended default):** Option C — rewrite with sklearn `MLPRegressor` to avoid TensorFlow install weight; remove keras-tuner path or gate behind optional extra if user insists on deep learning later.
- **Pandas deprecations:** Audit all `pct_change` usages; prefer `fill_method=None` explicitly.

### Phase 2 technical decisions

- **Boosters:** Add one of LightGBM or XGBoost first (LightGBM preferred for speed on tabular data); register in classifier comparison grid alongside existing ensemble.
- **Portfolio analytics:** Prefer QuantStats for tear-sheet HTML/markdown-friendly output; fall back to empyrical for single metrics if QuantStats is too heavy.
- **EDGAR:** Use `edgartools` or equivalent; fallback only when `quarterly_financials` / `financials` empty; cache EDGAR responses in existing SQLite cache store with namespaced keys.
- **statsmodels:** Introduce only for new diagnostic scripts or explicit opt-in analysis — not wired into daily job by default.

### Phase 3 technical decisions

- **Provider protocol:** Typed protocol with `get_history`, `get_financials`, `get_earnings`; no HTTP in consumers.
- **Polars/DuckDB:** Spike only after profiling identifies a concrete bottleneck (Finviz batch, cache analytics).
- **FastAPI:** ADR-style note if pursued; out of Phase 1–2 scope.

### Architectural alignment

- Respect existing **temporal holdout** evaluation pattern (train on past, test on recent) — no change to split semantics.
- Respect **SQLite cache store** (`CacheStore` protocol, namespaced keys) for any new cached fundamentals data.
- Respect **paper trading** domain: daily job, `SignalStrategy` registry, `DecisionResult` / `FillResult` — portfolio metrics augment reports only.
- Do **not** introduce zipline, backtrader, or quantlib in this PRD scope.

### Schema changes

- Phase 1: None.
- Phase 2: Optional new cache key prefix for EDGAR payloads (e.g. `edgar:{ticker}:{filing_type}`) in existing `cache_entries` table.
- Phase 3: None required for provider protocol alone.

### API contracts

- Flask API unchanged in Phase 1–2 unless NN endpoints are removed/deprecated (document in changelog).
- `stock-analysis-evaluate` gains optional `--classifier lightgbm` (or similar) in Phase 2.
- Daily job report schema gains optional metrics section (markdown headings for Calmar, Sortino, rolling Sharpe).

## Testing Decisions

### What makes a good test here

- Test **observable behaviour** (return shapes, metric values on fixture data, classifier compare output structure) — not private helpers or library import paths.
- Use **fixed fixtures** (synthetic OHLCV, mock financial statements) to avoid live `yfinance` in CI.
- Prefer **round-trip and contract tests** for caches and providers.

### Modules to test (priority)

| Priority | Module | Test focus |
|----------|--------|------------|
| P0 | Fundamental classification cache | Cache miss writes files; cache hit returns `(X, y)` with same columns; TTL respected |
| P0 | Technical indicator facade | RSI/volatility columns present and finite on synthetic OHLCV after `ta` → `pandas-ta` migration |
| P1 | Classifier evaluation backend | Booster appears in compare output; same holdout row count as sklearn baseline on fixture |
| P1 | Portfolio metrics reporter | Known return series produces expected Sharpe/Sortino within tolerance |
| P2 | Fundamentals provider (EDGAR fallback) | When Yahoo mock returns empty, EDGAR mock populates minimum statement fields |
| P2 | Neural predictor policy | Import/train path succeeds with declared deps OR raises clear deprecation error |

### Prior art in codebase

- `tests/ai/test_temporal_split.py` — chronological split contracts on synthetic prepared data.
- `tests/ai/test_variant_comparison.py` / `test_backtest_splits.py` — classifier pipeline behaviour.
- `tests/storage/test_cache_store.py` — TTL, validator, lazy import patterns for SQLite cache.
- `tests/trading/test_paper_broker.py` / `test_decision.py` — daily job domain behaviour.

### CI expectation

- All new Phase 1 tests pass with `uv run pytest` and no network access.
- Smoke manual check: `stock-analysis-evaluate compare --ticker ^GSPC --holdout-months 12` after TA migration.

## Out of Scope

- Paid market-data APIs (FMP, Polygon) unless explicitly funded in a follow-up PRD.
- zipline, backtrader, quantlib integration.
- Deep learning / TensorFlow as default classifier path (optional extra only if explicitly chosen over sklearn MLP).
- FastAPI migration (Phase 3 evaluation only).
- Full rewrite of movement classification or finviz classifier pipelines.
- Live broker integration (IKBR, etc.).
- Production deployment hardening beyond existing Docker compose setup.

## Further Notes

- **Related docs:** [stack-improvement-plan.md](./stack-improvement-plan.md), [database-cache-plan.md](./database-cache-plan.md).
- **Observed production symptom:** `Error processing {TICKER}: too many values to unpack (expected 2)` during fundamental classifier batch runs after "Cached data for …" — root cause is inconsistent cache return contract; regression test is mandatory for Phase 1 sign-off.
- **Recommended implementation order:** Phase 1.1 → 1.2 → 1.4 → 1.3 → Phase 2.1 → 2.2 → 2.3 → 2.4.
- **Triage:** Label `needs-triage` on issue creation; break into child issues per phase or per deep module after triage.
