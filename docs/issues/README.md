# Stack tooling — GitHub issues

Tracker for [PRD: Stack & tooling improvements](../prd-stack-tooling-improvements.md).

## Status (June 2026)

| # | Phase | Status | Issue | Title |
|---|-------|--------|-------|-------|
| 1 | Epic | **open** | [#1](https://github.com/trovlunde/stock_analyzer/issues/1) | Stack & tooling improvements |
| 2 | 1 | closed | [#2](https://github.com/trovlunde/stock_analyzer/issues/2) | Fix fundamental classification cache return contract |
| 3 | 1 | closed | [#3](https://github.com/trovlunde/stock_analyzer/issues/3) | Consolidate technical-analysis libraries to pandas-ta |
| 4 | 1 | closed | [#4](https://github.com/trovlunde/stock_analyzer/issues/4) | Resolve orphaned neural-network predictor module |
| 5 | 1 | closed | [#5](https://github.com/trovlunde/stock_analyzer/issues/5) | Audit pct_change for pandas 3.x deprecation |
| 6 | 2 | closed | [#6](https://github.com/trovlunde/stock_analyzer/issues/6) | Add gradient-boosted classifiers to evaluate CLI |
| 7 | 2 | closed | [#7](https://github.com/trovlunde/stock_analyzer/issues/7) | Richer portfolio metrics in daily paper-trading reports |
| 8 | 2 | closed | [#8](https://github.com/trovlunde/stock_analyzer/issues/8) | SEC EDGAR fundamentals fallback when Yahoo data is empty |
| 9 | 2 | closed | [#9](https://github.com/trovlunde/stock_analyzer/issues/9) | Add statsmodels for time-series diagnostics |
| 10 | 3 | closed | [#10](https://github.com/trovlunde/stock_analyzer/issues/10) | Introduce MarketDataProvider protocol |
| 11 | 3 | **open** | [#11](https://github.com/trovlunde/stock_analyzer/issues/11) | Spike Polars and DuckDB for batch analytics |
| 12 | 3 | **open** | [#12](https://github.com/trovlunde/stock_analyzer/issues/12) | Consolidate visualization dependencies on Plotly |
| 13 | 3 | **open** | [#13](https://github.com/trovlunde/stock_analyzer/issues/13) | Evaluate FastAPI migration for Flask API |

### #12 sub-issues (viz consolidation)

Parent [#12](https://github.com/trovlunde/stock_analyzer/issues/12) coordinates; implement on children:

| Slice | Issue | Title | Ralph? |
|-------|-------|-------|--------|
| 12a | [#24](https://github.com/trovlunde/stock_analyzer/issues/24) | Viz inventory doc | Yes |
| 12b | [#25](https://github.com/trovlunde/stock_analyzer/issues/25) | Quick wins: dead mpl + plotly heatmaps | Yes |
| 12c | [#26](https://github.com/trovlunde/stock_analyzer/issues/26) | Movement classifier diagnostic plots | Human / later |
| 12d | [#27](https://github.com/trovlunde/stock_analyzer/issues/27) | sector_analysis + plottable (defer) | Human |

**Remaining order:** #13 (ADR) → #24 → #25 → #11 (defer unless profiling needed) → close #1

Labels: `needs-triage`, `ready-for-agent`, `ready-for-human`, `phase-1` / `phase-2` / `phase-3`
