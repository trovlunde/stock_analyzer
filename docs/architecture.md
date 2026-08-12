# Architecture

Single-repo modular monolith: one Python package (`stock_analysis/`) with several entry surfaces sharing storage and market-data infrastructure.

## Entry surfaces

| Surface | Entry point | Purpose |
|---------|-------------|---------|
| Interactive CLI | `python -m stock_analysis.research.main` | Exploratory analysis, stock finder, sector views |
| Web API | `python -m stock_analysis.web.app` | Flask REST API for movement classifiers |
| Paper trading | `python -m stock_analysis.jobs.daily` | Daily signal evaluation and paper fills |
| ML evaluation | `stock-analysis-evaluate` CLI | Temporal holdout classifier comparison |
| Research scripts | `stock_analysis.ai.*` modules | Notebooks-style ML and valuation experiments |

## Layer diagram

```mermaid
flowchart TB
  subgraph surfaces [Entry surfaces]
    CLI["research.main"]
    Web["web.app"]
    Daily["jobs.daily"]
    EvalCLI["cli.evaluate_models"]
  end

  subgraph core [Shared core]
    MarketData["market_data/"]
    Storage["storage/"]
    Trading["trading/"]
  end

  subgraph research_layer [Research and ML]
    Research["research/"]
    AI["ai/"]
    Diagnostics["diagnostics/ opt-in"]
  end

  CLI --> Research
  Research --> core
  Web --> AI
  Daily --> Trading
  Daily --> Storage
  Daily --> MarketData
  EvalCLI --> AI
  EvalCLI --> Diagnostics
  AI --> MarketData
  AI --> Storage
  Research --> MarketData
```

## Import boundaries

- **`jobs/daily.py` and `trading/`** — production paper-trading path. Use `market_data`, `storage`, and `trading` only. Do not import `diagnostics/` or heavy research viz.
- **`web/app.py`** — serves ML endpoints. Imports `ai/` only; no diagnostics by default.
- **`diagnostics/`** — opt-in statsmodels helpers. Never imported by `jobs/daily`, `web/app`, or paper-trading modules. Researchers and `cli/evaluate_models` may call it explicitly.
- **`research/`** — interactive CLI and exploratory scripts (sector analysis, stock finder, portfolio plots). May import `market_data` and `storage`; not used by the daily job.

## Package layout

| Path | Role |
|------|------|
| `stock_analysis/market_data/` | `MarketDataProvider` protocol and `YFinanceProvider` |
| `stock_analysis/storage/` | SQLite cache and paper-trading portfolio store |
| `stock_analysis/trading/` | Signals, paper broker, daily report |
| `stock_analysis/jobs/` | Scheduled jobs (`daily.py`) |
| `stock_analysis/cli/` | Non-interactive CLIs (`evaluate_models`) |
| `stock_analysis/web/` | Flask app and templates |
| `stock_analysis/research/` | Interactive CLI and exploratory analysis modules |
| `stock_analysis/ai/` | ML models, classifiers, valuation |
| `stock_analysis/diagnostics/` | Opt-in time-series diagnostics (statsmodels) |
| `config/` | Runtime YAML (e.g. `portfolio.yaml`) |
| `scripts/ralph/` | Agent loop tooling (not runtime) |
| `scripts/finviz/` | Finviz scrape script (outputs to `finviz_recs/`) |
| `finviz_recs/` | Local scrape data (gitignored JSON) |

## Naming: `fundamentals` vs `fundamental_analysis`

Two similarly named areas under `ai/` serve different layers:

- **`ai/fundamentals/`** — data layer: `FundamentalsProvider` protocol, yfinance and EDGAR adapters, composite fallback.
- **`ai/fundamental_analysis/`** — analysis layer: valuation models (`StockValuation`), fin statement classifier, examples.

When adding features, put data fetching in `fundamentals/` and valuation logic in `fundamental_analysis/` (or a future `ai/valuation/` rename).

## Development tooling

Agent infrastructure (`.agents/`, `scripts/ralph/`, `skills-lock.json`) lives in the repo for Ralph/Cursor workflows but is not part of the runtime product.

See also [AGENTS.md](../AGENTS.md) and [stack-improvement-plan.md](stack-improvement-plan.md).
