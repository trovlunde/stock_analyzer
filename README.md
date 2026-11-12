# Stock Analysis

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.14+.

```bash
# First-time setup (creates .venv and installs dependencies)
uv sync

# Copy environment config
cp .env.example .env
```

After changing `pyproject.toml` dependencies, run `uv sync` again.

### Project plans

- [Trader container readiness](tasks/trader-container-readiness.md) — coding tasks before the `trader` service runs unattended
- [Architecture](docs/architecture.md) — entry surfaces, layers, import boundaries
- [GitHub issues #1–#13](docs/issues/README.md) — stack tooling epic and child tasks (status per issue)
- [PRD: Stack & tooling improvements](docs/prd-stack-tooling-improvements.md) — problem statement, user stories, implementation & testing decisions
- [Stack & tooling improvements](docs/stack-improvement-plan.md) — phased checklist (dependency hygiene, data sources, ML upgrades)
- [Database cache](docs/database-cache-plan.md) — SQLite cache architecture and phase-2 scope

## Command Line Usage

### Interactive CLI (Recommended for Beginners)

Run the interactive command-line interface:

```bash
uv run python -m stock_analysis.research.main
```

This will start an interactive menu where you can:
1. Single Stock Analysis - Analyze individual stocks
2. Portfolio Analysis - Analyze multiple stocks and optimize portfolio
3. Stock Finder - Find stocks by criteria (high dividend, undervalued, growth)
4. Sector Analysis - Analyze sectors across different markets

### Web API (Flask Application)

Start the Flask web server:

```bash
uv run python -m stock_analysis.web.app
```

The API will be available at `http://localhost:5000` (or the port specified in your `.env` file).

**Available API Endpoints:**
- `GET /` - Landing page
- `GET /movement-analysis` - Movement analysis page
- `GET /health` - Health check
- `GET /api/v1/status` - API status
- `GET /api/v1/models` - List available models
- `DELETE /api/v1/models/cache` - Clear the in-memory model cache
- `POST /api/v1/predict/<model_name>/<model_type>` - Predict with a named model
- `POST /api/v1/train/<ticker>` - Train movement classifier for a ticker
- `GET /api/v1/predict/movement/<ticker>` - Get movement predictions
- `GET /api/v1/models/movement` - List movement classifier models

### Direct Python Script Execution

You can also run individual modules directly:

**Run valuation example:**
```bash
uv run python -m stock_analysis.ai.fundamental_analysis.valuation_example
```

**Run finviz classifier** (requires local scrape data under `finviz_recs/`, not committed to git):
```bash
# From repo root — populates finviz_recs/*.json (Windows: scripts\finviz\finviz_scrape.bat)
uv run python scripts/finviz/finviz_scrape.py

uv run python -m stock_analysis.ai.finviz_classifier.finviz_classifier
```

**Run technical analysis:**
```bash
uv run python -m stock_analysis.ai.technical_analysis.movement_classification
```

### ML Model Evaluation

Compare classifiers with a temporal holdout (train on past, test on recent data):

```bash
# Grid-search all classifiers (sklearn + LightGBM); evaluate best on last 12 months
uv run stock-analysis-evaluate compare --ticker ^GSPC --holdout-months 12 --extra-features

# Run only the LightGBM booster (faster grid-search, leaf-wise trees)
uv run stock-analysis-evaluate compare --ticker ^GSPC --holdout-months 12 --classifier lightgbm

# Run only sklearn ensembles (Random Forest, Voting Classifier, etc.)
uv run stock-analysis-evaluate compare --ticker ^GSPC --holdout-months 12 --classifier "Random Forest" "Voting Classifier"

# Train daily+weekly RandomForest and score them on the holdout
uv run stock-analysis-evaluate train --ticker ^GSPC --holdout-months 6 --extra-features
# Without --holdout-months, evaluation uses the internal chronological 20% test split only
# Add --leveraged-sim for the legacy 10x mini-future profit simulation (off by default)

# Compare dual-RF pipeline variants on the same holdout (features, threshold, RF params)
uv run stock-analysis-evaluate variants --ticker ^GSPC --holdout-months 12
```

**Booster vs ensemble tradeoffs:**

| | LightGBM | sklearn ensembles (RF, Voting) |
|---|---|---|
| Speed | Fast grid-search (leaf-wise growth) | Slower — many tree estimators |
| Tabular data | Often best-in-class on tabular | Competitive; more tuning options |
| Install weight | ~15 MB extra dependency | Bundled with scikit-learn |
| When to prefer | Small dataset, fast iteration, production scoring | Interpretability, no extra deps, voting ensembles |

Use `--classifier lightgbm` for a quick booster benchmark; omit `--classifier` to grid-search all registered classifiers and let CV pick the best.

Interactive CLI (mode 1 with holdout, or mode 5 for classifier comparison):

```bash
uv run python -m stock_analysis.ai.technical_analysis.movement_classification
# Mode 1: enter holdout months when prompted (e.g. 12)
# Mode 5: compare ML classifiers with temporal holdout
# Mode 6: compare dual-RF pipeline variants (baseline, extra features, thresholds, RF depth)
# Mode 2: SP500-trained model — excludes analysis ticker and truncates all tickers at the same holdout cutoff
```

API training with temporal holdout:

```bash
curl -X POST http://localhost:5000/api/v1/train/%5EGSPC \
  -H "Content-Type: application/json" \
  -d '{"period": "10y", "holdout_months": 12, "use_extra_features": true}'
```

### Strategy Backtest

Replays a portfolio config over history with the same rules as the daily job: signals on each session's close, orders filled at the next session's Open, costs and sizing from the config. The report puts the strategy next to buy-and-hold of the same tickers and of SPY.

```bash
# Moving-average crossover on the config's watchlist, last 10 years
uv run stock-analysis-evaluate backtest --strategies ma_crossover

# ML model retrained every 63 sessions on past data only (slow: minutes per ticker-decade)
uv run stock-analysis-evaluate backtest --strategies ml_walk_forward --retrain-every 63

# Other tickers, a fixed window, report saved to a file
uv run stock-analysis-evaluate backtest --strategies ma_crossover \
  --tickers NVDA AMZN JPM --start 2018-01-01 --end 2024-12-31 --output data/backtests/ma.md
```

Reading the report:

- **Strategy vs buy & hold** — CAGR, volatility, Sharpe, Sortino, max drawdown, Calmar; plus trades, hit rate, turnover, cost drag and average exposure for the strategy.
- **Is it better than chance?** — a block-bootstrap interval for the Sharpe ratio and the share of resamples in which the strategy beats buy-and-hold.
- **Configurations tried** — every run is recorded in `data/backtests/runs.jsonl`. Once several configurations have been tried on the same tickers and period, the report adds two checks. The deflated Sharpe ratio discounts the result for the number of attempts. The probability of backtest overfitting splits the period into ten blocks, picks the best configuration on every possible half, and reports how often that pick lands in the bottom half on the remaining blocks: near 0 % means the ranking holds up, 50 % or more means choosing the best backtest is no better than choosing at random. Use `--no-log` for runs that should not count.

`ml_movement` (the saved model) is refused by default because the saved model was trained on the period being tested; use `ml_walk_forward` instead, or `--allow-in-sample-ml` to override.

### Using Python Interactively

You can also import and use functions directly in Python:

```python
# In Python shell or script
from stock_analysis.research.fetch_stock_data import fetch_stock_data
from stock_analysis.research.analyze_stock_data import analyze_stock_data
from stock_analysis.ai.fundamental_analysis.valuation_methods import StockValuation

# Fetch and analyze a stock
df, events = fetch_stock_data('AAPL', period='1y')
analysis = analyze_stock_data(df)

# Get valuations
valuation = StockValuation('AAPL')
all_valuations = valuation.get_all_valuations()
```

### Command Line Examples

**Analyze a single stock:**
```bash
uv run python -c "from stock_analysis.research.fetch_stock_data import fetch_stock_data; from stock_analysis.research.analyze_stock_data import analyze_stock_data; df, _ = fetch_stock_data('AAPL', period='1y'); print(analyze_stock_data(df))"
```

**Find high dividend stocks:**
```bash
uv run python -c "from stock_analysis.research.stock_finder import find_high_dividend_stocks, display_stocks; from stock_analysis.market_data import YFinanceProvider; stocks = YFinanceProvider().get_tickers_obj(['AAPL', 'MSFT', 'GOOGL']); result = find_high_dividend_stocks(stocks=stocks); display_stocks(result)"
```

**Get market index tickers:**
```bash
uv run python -c "from stock_analysis.research.market_indices import MarketIndices; print(MarketIndices.get_sp500_tickers())"
```

### Tests

```bash
uv run pytest
```

### Paper Trading (Daily Job)

Config-driven paper portfolio. Evaluates US equities with the configured signal strategies, queues fills at next open, persists state in SQLite.

**Config:** `config/portfolio.yaml` — watchlist, sizing, risk limits, strategy defaults, per-ticker overrides.

**Signals:** `ma_crossover`, `ml_movement` and `trend_state`, combined per `signal_mode`.
- `ma_crossover` — BUY/SELL on the day the short average crosses the long one.
- `ml_movement` — HOLDs for a ticker until a model trained on the current feature set exists; train one with `POST /api/v1/train/<ticker>`.
- `trend_state` — BUY every day the ticker is in an uptrend (average of its 3-, 6- and 12-month returns above zero and price above its 200-day average), SELL every day that average return is at or below zero. Because it reports the state rather than the day of a change, an entry missed one day is made the next. Parameters: `trend_lookbacks`, `trend_ma`.

**Sizing:** `portfolio.sizing: fixed` (default) puts `position_size_pct` of the portfolio into each new position. `vol_target` gives calmer assets larger positions: weight = `risk_per_position` / annualised volatility over `vol_lookback` sessions, capped at `max_position_pct`.

**Regime filter:** optional top-level `regime_filter: {ticker: SPY, ma: 200}` blocks new BUY orders while that ticker is below its moving average. Existing positions are still sold by their own signals.

**ETF trend portfolio:** `config/portfolio-etf-trend.yaml` runs `trend_state` with volatility sizing and the regime filter on 20 ETFs. It has its own portfolio name, so it keeps separate state from `config/portfolio.yaml`:

```bash
uv run stock-analysis-evaluate backtest --config config/portfolio-etf-trend.yaml
uv run stock-analysis-daily --config config/portfolio-etf-trend.yaml --dry-run
```

**Run once:**
```bash
uv run stock-analysis-daily
uv run stock-analysis-daily --dry-run
uv run stock-analysis-daily --config config/portfolio.yaml
```

**Run on a schedule:** `stock-analysis-trader` is a long-running scheduler that triggers the daily job once per NYSE trading day (16:05 ET with `schedule.run_after_close: true`, otherwise 09:35 ET). A slot missed by more than two hours is skipped and alerted rather than run late. It is what the `trader` compose service runs (see [Docker](#docker)).

```bash
uv run stock-analysis-trader
```

**Broker:** selected by `BROKER_MODE` in `.env`.
- `paper_sim` (default) — local simulator, no external account.
- `alpaca` — Alpaca's paper endpoint; needs `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.

**Kill switch:** trips on a breach of `risk.max_daily_loss_pct` / `risk.max_drawdown_pct`, an unclean shutdown (process killed mid-run), or a broker/local state mismatch. While tripped, new BUY orders are blocked. Clearing it is manual:

```bash
uv run stock-analysis-risk status
uv run stock-analysis-risk reconcile                       # diff local state vs broker
uv run stock-analysis-risk reconcile --accept-broker-truth
uv run stock-analysis-risk clear --reason "why it is safe to resume"
```

**Live trading:** off by default and refused unless every check in `stock_analysis/trading/live_gate.py` passes: `BROKER_MODE=alpaca`, `LIVE_TRADING_ENABLED=true`, `LIVE_TRADING_CONFIRM` set to the exact confirmation phrase, `ALPACA_PAPER=false`, both risk limits configured, kill switch clear, 20 consecutive clean runs, and `LIVE_MAX_NOTIONAL_PER_ORDER` / `LIVE_MAX_PORTFOLIO_NOTIONAL` set. The first live run is read-only (reconcile and report, no orders).

**Email alerts:** optional. Set `SMTP_HOST` and `SMTP_TO` (plus `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_USE_TLS` as needed) to receive the daily summary and failure / kill-switch alerts.

**Outputs:**
- State: `data/stock_analysis.db` (`paper_*` tables)
- Report: `data/reports/YYYY-MM-DD.md` — includes the portfolio's return since its first session next to buy-and-hold of SPY (change the ticker with `defaults.benchmark` in the config; an empty value turns it off)

**Add strategies later:** implement `SignalStrategy` in `stock_analysis/trading/signals/` and register in `registry.py`.

### Docker

```bash
docker compose up --build
```

Starts two services from the same image:

- `stock-analysis` — the Flask API at `http://localhost:5000`. Published on loopback only (the API has no auth); reach it remotely via an SSH tunnel or VPN.
- `trader` — the `stock-analysis-trader` scheduler. Reads `.env`, mounts `./config`, `./data` and `./models`.

### Optional: activate the virtualenv

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```

After activation you can run `python -m stock_analysis.research.main` directly without the `uv run` prefix.

---

Data fetched from https://titlon.uit.no/

Data feched from Yahoo Finance


statsmodels - For more advanced statistical analysis
empyrical - For financial risk metrics
QuantStats - For detailed portfolio analytics
mlfinlab - For machine learning-based financial analysis

scikit-learn - For machine learning

zipline - For backtesting
quantlib - For financial calculations
quantopian - For backtesting
quantstats - For portfolio analytics


1. Certainty - return correlation
2. Certainty based scan 
3. Current portfolio
4. Potential stocks
5. Add volatility/volume features
6. Check different timperiods for backtesting
7. Spread vs hold cost
8. Test models concrete score



Trading bot 4fun, 
IKBR


Reddit 

BnBank
Santander
Odal
DnB
Nordnet
Firi

## Stock Analysis Functions

This section documents the available functions for stock analysis functionality in the codebase.

### Data Fetching

**Module:** `stock_analysis.research.fetch_stock_data`

- `fetch_stock_data(tickers, start_date=None, end_date=None, period="1mo")`
  - Fetches stock price data and events (dividends, earnings) from Yahoo Finance
  - Returns: Tuple of (DataFrame with stock data, DataFrame with events)
  - Supports date ranges or predefined periods (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)

- `fetch_stocks_data(tickers)`
  - Fetches basic information for multiple stocks
  - Returns: DataFrame with company info, market cap, P/E ratios, margins

- `fetch_stocks_data_alt(tickers)`
  - Alternative function with enhanced data cleaning and validation
  - Filters out invalid values and sorts by market cap

### Stock Analysis

**Module:** `stock_analysis.research.analyze_stock_data`

- `analyze_stock_data(df)`
  - Analyzes stock data for risk and return metrics
  - Returns: Dictionary with volatility, annual return, Sharpe ratio, RSI, max drawdown, VaR

- `analyze_portfolio_correlation(stock_data)`
  - Analyzes correlation between stocks in a portfolio
  - Returns: Correlation matrix DataFrame

- `optimize_portfolio(stock_data)`
  - Optimizes portfolio weights using Modern Portfolio Theory
  - Maximizes Sharpe ratio using Efficient Frontier
  - Returns: Dictionary of optimized weights per ticker

### Stock Finding & Screening

**Module:** `stock_analysis.research.stock_finder`

- `find_stocks_by_method(method='high_dividend', stockTickers=[...])`
  - Main function to find stocks based on specified method
  - Methods: 'high_dividend', 'undervalued', 'growth'

- `find_high_dividend_stocks(min_yield=0.03, min_market_cap=1e9, stocks=[])`
  - Finds stocks with high dividend yields
  - Filters by minimum yield and market cap
  - Returns: List of dictionaries with stock information

- `find_undervalued_stocks(min_market_cap=1e9, stocks=[], percentile_threshold=25)`
  - Finds undervalued stocks by comparing metrics within sectors
  - Uses percentile ranking for P/E and P/B ratios
  - Returns: List of potentially undervalued stocks

- `find_growth_stocks(min_market_cap=1e9, marketTicker='^GSPC', growth_threshold=0.2)`
  - Finds growth stocks based on revenue and earnings growth
  - Filters by minimum growth threshold
  - Returns: List of growth stocks

- `display_stocks(stocks, method='high_dividend')`
  - Displays filtered stocks in a formatted table
  - Saves results to CSV file
  - Returns: Formatted DataFrame

### Sector Analysis

**Module:** `stock_analysis.research.sector_analysis`

- `analyze_and_display_sector_metrics(stocks_df, min_market_cap=1e9)`
  - Analyzes and visualizes key metrics for each sector
  - Creates box plots and summary tables
  - Metrics: Forward P/E, Trailing P/E, Price to Book, Profit Margins, Operating Margins
  - Returns: Summary DataFrame and full data DataFrame

- `display_sector_comparison(sector_data, metrics=['forward_pe'])`
  - Creates detailed comparison visualization across sectors
  - Uses violin plots with individual data points
  - Returns: Visualization plots

- `find_undervalued_sectors(marketTicker='^GSPC')`
  - Finds undervalued sectors (placeholder function)

### Market Indices

**Module:** `stock_analysis.research.market_indices`

**Class:** `MarketIndices`
  - Contains methods to scrape ticker lists from Wikipedia for various market indices

- `get_sp500_tickers()` - S&P 500 companies
- `get_nasdaq100_tickers()` - NASDAQ-100 companies
- `get_dow30_tickers()` - Dow Jones Industrial Average
- `get_ftse100_tickers()` - FTSE 100 (UK)
- `get_dax40_tickers()` - DAX 40 (Germany)
- `get_cac40_tickers()` - CAC 40 (France)
- `get_nikkei225_tickers()` - Nikkei 225 (Japan)

- `get_market_tickers(market='sp500')`
  - Convenience function to get tickers for any supported market

### Fundamental Analysis & Valuation

**Module:** `stock_analysis.ai.fundamental_analysis.valuation_methods`

**Class:** `StockValuation`
  - Comprehensive stock valuation class with multiple valuation methods

**Valuation Methods:**

- `asset_based_valuation()` - Asset-based valuation (best for asset-heavy companies)
- `income_based_valuation(discount_rate=0.10, projection_years=5)` - Income-based valuation (best for stable earnings)
- `market_based_valuation()` - Market-based valuation using EBITDA multiples (service and retail companies)
- `dcf_valuation(discount_rate=0.10, growth_rate=0.03, projection_years=5)` - Discounted Cash Flow valuation
- `equity_multiplier_valuation()` - Equity multiplier valuation (financial institutions)
- `book_value_valuation()` - Book value calculation (asset-heavy businesses)
- `liquidation_value()` - Liquidation value (distressed companies)
- `breakup_value()` - Break-up value (conglomerates)
- `epv_valuation(cost_of_capital=0.10)` - Earnings Power Value (Bruce Greenwald method)
- `reverse_dcf(current_price=None)` - Reverse DCF to find implied growth rate
- `graham_valuation()` - Benjamin Graham's intrinsic value (original and revised formulas)
- `calculate_multiples()` - Calculates common valuation multiples (P/E, P/B, P/S, EV/EBITDA, ROE, ROA, ROIC, margins, etc.)
- `get_all_valuations(discount_rate=0.10, growth_rate=0.03, projection_years=5)` - Returns all valuation methods

**Helper Functions:**

- `format_valuation_results(results, currency='USD')` - Formats valuation results into readable DataFrame
- `compare_stocks_valuation(tickers, methods=None, currency='USD')` - Compares valuation metrics across multiple stocks

### Technical Analysis

**Module:** `stock_analysis.ai.technical_analysis.movement_classification`

- `train_classifier_single_stock(stock_data, use_extra_features=False, predict_weekly=False, threshold=0.005, plot=False, overfit_check=False, classifier=...)`
  - Trains a movement classifier for a single stock
  - Returns: Model, scaler, and prepared data

- `train_classifier_tickers(tickers, predict_weekly=False, threshold=0.005, plot=False, overfit_check=False, classifier=..., use_extra_features=False)`
  - Trains classifiers for multiple tickers

- `get_recent_predictions(stock_data, daily_data, daily_model, daily_scaler, weekly_data, weekly_model, weekly_scaler, days=5, threshold=0.005, use_extra_features=False)`
  - Gets recent movement predictions for a stock
  - Returns: DataFrame with predictions

- `full_analysis(stock_data, daily_data, daily_model, daily_scaler, weekly_data, weekly_model, weekly_scaler, threshold=0.01, use_extra_features=False)`
  - Performs comprehensive analysis with predictions and metrics

**Module:** `stock_analysis.ai.technical_analysis.ma_signals`

- `generate_ma_signals(...)` - Generates moving average trading signals
- `generate_multi_ma_signals(...)` - Generates multiple moving average signals

**Module:** `stock_analysis.ai.technical_analysis.rsi_signals`

- `generate_rsi_signals(...)` - Generates RSI-based trading signals

**Module:** `stock_analysis.ai.technical_analysis.nn_predictor`

> **Experimental** — not wired into `stock-analysis-daily` or `stock-analysis-evaluate`. Uses `sklearn.neural_network.MLPRegressor` backend (no TensorFlow or Keras required).

- `train_nn_predictor(data, predict_weekly=False, test_size=0.2, do_tuning=True)` - Trains MLP regressor; `do_tuning=True` runs `RandomizedSearchCV` over hyperparameters. Returns `(model, scaler, history)`.
- `predict_returns(model, scaler, data, predict_weekly=False)` - Makes return predictions using trained model; returns `pd.Series` aligned to prepared feature index.

**Module:** `stock_analysis.ai.technical_analysis.intraday_volatility_analysis`

- `calculate_intraday_metrics(ticker='^GSPC', period='1y', threshold=0.5)` - Calculates intraday volatility metrics
- `analyze_return_distribution(data)` - Analyzes return distribution
- `analyze_trading_strategy(data, target_gain=0.3, stop_loss=0.3, spread_cost=0.5, leverage=1)` - Analyzes trading strategy performance

### Portfolio & Trading Analysis

**Module:** `stock_analysis.research.portfolio_evaluating`

- `analyze_trading_strategies(stock_data, signals, initial_investment=10000, leverage=10, signal_type="")`
  - Analyzes trading strategies with backtesting
  - Returns: Dictionary with performance metrics

- `calculate_max_drawdown(values)` - Calculates maximum drawdown
- `calculate_sharpe_ratio(values, risk_free_rate=0.02)` - Calculates Sharpe ratio
- `plot_trading_analysis(stock_data, results, title="Trading Analysis")` - Plots trading analysis results

**Module:** `stock_analysis.research.trading_cost_analysis`

**Classes:**
- `CalculateCost` - Base class for cost calculations
- `MiniFutureAnalyzer(CalculateCost)` - Analyzes mini futures costs
- `BullBearAnalyzer(CalculateCost)` - Analyzes bull/bear products

- `compare_products(days=31)` - Compares different trading products
- `compare_products_with_volatility(days=31, volatility=0.1)` - Compares products with volatility consideration

### Visualization

**Module:** `stock_analysis.research.efficient_frontier_plot`

- `plot_efficient_frontier(stock_data)` - Plots efficient frontier for portfolio optimization

**Module:** `stock_analysis.research.display_stock_data`

- `display_stock_data(df, events, ticker)` - Displays stock data with events

**Module:** `stock_analysis.visualization.trading_plots`

- `plot_trading_signals(simulation, stock_data, metrics=None, title="Trading Analysis")` - Plots trading signals and performance

### Data Caching

Market and index OHLCV data are cached in SQLite (`data/stock_analysis.db` by default). Configure with `DATABASE_URL` in `.env`. Legacy CSV files under `data/` are imported automatically on first read. See `docs/database-cache-plan.md` for architecture and phase-2 scope.

**Module:** `stock_analysis.storage`

- `get_cache_store()` - Returns the shared `SqlCacheStore` instance
- `CacheStore.get(key, max_age=..., validator=...)` - Read cached DataFrame
- `CacheStore.put(key, df)` - Write cached DataFrame
- `CacheStore.delete(key)` / `clear_prefix(prefix)` - Invalidate entries

**Module:** `stock_analysis.research.cache_data`

- `get_cached_stock_data(market)` - Retrieves cached stock data
- `save_stock_data(stocks_df, market)` - Saves stock data to cache

### Helper Functions

**Module:** `stock_analysis.ai.helpers`

- `get_ticker(ticker)` - Gets yfinance Ticker object
- `get_index_data(ticker, period='10y', start_date=None)` - Gets index/stock data
- `get_indexes_data(indexes, period='10y')` - Gets data for multiple indexes
- `get_ticker_data(ticker)` - Gets ticker price data
- `get_ticker_financials(ticker)` - Gets ticker financial statements
- `get_significant_changes(data, filter_consecutive=False, filter_return=0.03)` - Identifies significant price changes
- `RSI(data, window=14, adjust=False)` - Calculates Relative Strength Index
- `get_features(use_extra_features=False)` - Gets feature list for ML models

### AI/ML Models

**Module:** `stock_analysis.ai.model_manager`

**Class:** `ModelManager`
  - Manages ML model loading, saving, and caching
  - Methods: `load_model()`, `save_model()`, `list_models()`, `clear_cache()`, `get_model_metadata()`

**Module:** `stock_analysis.ai.finviz_classifier.finviz_classifier`

- `prepare_features(signals_df, returns_df, threshold=0.05, use_technical_features=False, use_market_features=False)` - Prepares features for classification
- `train_classifier(df, feature_columns, target='next_day_positive')` - Trains binary classifier
- `train_multiclass_classifier(df, feature_columns, return_type='next_day_return')` - Trains multiclass classifier
- `train_regressor(df, feature_columns, return_type='next_day_return')` - Trains regression model

### Usage Examples

```python
# Fetch stock data
from stock_analysis.research.fetch_stock_data import fetch_stock_data
df, events = fetch_stock_data('AAPL', period='1y')

# Analyze stock
from stock_analysis.research.analyze_stock_data import analyze_stock_data
analysis = analyze_stock_data(df)

# Get valuations
from stock_analysis.ai.fundamental_analysis.valuation_methods import StockValuation
valuation = StockValuation('AAPL')
all_valuations = valuation.get_all_valuations()

# Find high dividend stocks
from stock_analysis.research.stock_finder import find_high_dividend_stocks
stocks = find_high_dividend_stocks(min_yield=0.03, min_market_cap=1e9)

# Optimize portfolio
from stock_analysis.research.analyze_stock_data import optimize_portfolio
weights = optimize_portfolio(stock_data)
```