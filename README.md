How to run:

## Command Line Usage

### Interactive CLI (Recommended for Beginners)

Run the interactive command-line interface:

**Using Poetry:**
```bash
poetry run python -m stock_analysis.main
```

**Or using pip:**
```bash
python -m stock_analysis.main
```

This will start an interactive menu where you can:
1. Single Stock Analysis - Analyze individual stocks
2. Portfolio Analysis - Analyze multiple stocks and optimize portfolio
3. Stock Finder - Find stocks by criteria (high dividend, undervalued, growth)
4. Sector Analysis - Analyze sectors across different markets

### Web API (Flask Application)

Start the Flask web server:

**Using Poetry:**
```bash
poetry run python app.py
```

**Or using pip:**
```bash
python app.py
```

The API will be available at `http://localhost:5000` (or the port specified in your `.env` file).

**Available API Endpoints:**
- `GET /health` - Health check
- `GET /api/v1/status` - API status
- `GET /api/v1/models` - List available models
- `POST /api/v1/train/<ticker>` - Train movement classifier for a ticker
- `GET /api/v1/predict/movement/<ticker>` - Get movement predictions
- `GET /api/v1/models/movement` - List movement classifier models

### Direct Python Script Execution

You can also run individual modules directly:

**Run valuation example:**
```bash
python -m stock_analysis.ai.fundamental_analysis.valuation_example
```

**Run finviz classifier:**
```bash
python -m stock_analysis.ai.finviz_classifier.finviz_classifier
```

**Run technical analysis:**
```bash
python -m stock_analysis.ai.technical_analysis.movement_classification
```

### Using Python Interactively

You can also import and use functions directly in Python:

```python
# In Python shell or script
from stock_analysis.fetch_stock_data import fetch_stock_data
from stock_analysis.analyze_stock_data import analyze_stock_data
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
python -c "from stock_analysis.fetch_stock_data import fetch_stock_data; from stock_analysis.analyze_stock_data import analyze_stock_data; df, _ = fetch_stock_data('AAPL', period='1y'); print(analyze_stock_data(df))"
```

**Find high dividend stocks:**
```bash
python -c "from stock_analysis.stock_finder import find_high_dividend_stocks, display_stocks; import yfinance as yf; stocks = yf.Tickers('AAPL MSFT GOOGL'); result = find_high_dividend_stocks(stocks=stocks); display_stocks(result)"
```

**Get market index tickers:**
```bash
python -c "from stock_analysis.market_indices import MarketIndices; print(MarketIndices.get_sp500_tickers())"
```

### Environment Setup

**Using Poetry (Recommended):**

This project uses Poetry for dependency management. Make sure Poetry is installed, then:

```bash
# Install all dependencies
poetry install

# Run commands using Poetry
poetry run python -m stock_analysis.main
poetry run python app.py

# Or activate the Poetry shell
poetry shell
python -m stock_analysis.main
```

**Using pip:**

Alternatively, you can use pip:

```bash
pip install -r requirements.txt
python -m stock_analysis.main
```

### Windows PowerShell Examples

```powershell
# Run interactive CLI
python -m stock_analysis.main

# Start Flask server
python app.py

# Run with specific ticker
python -c "from stock_analysis.fetch_stock_data import fetch_stock_data; df, _ = fetch_stock_data('MSFT', period='6mo'); print(df.head())"
```

### Linux/Mac Terminal Examples

```bash
# Run interactive CLI
python3 -m stock_analysis.main

# Start Flask server
python3 app.py

# Run with specific ticker
python3 -c "from stock_analysis.fetch_stock_data import fetch_stock_data; df, _ = fetch_stock_data('MSFT', period='6mo'); print(df.head())"
```

---

Data fetched from https://titlon.uit.no/

Data feched from Yahoo Finance


statsmodels - For more advanced statistical analysis
empyrical - For financial risk metrics
QuantStats - For detailed portfolio analytics
mlfinlab - For machine learning-based financial analysis

tensorflow - For machine learning
keras - For machine learning, on top of tensorflow
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

**Module:** `stock_analysis.fetch_stock_data`

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

**Module:** `stock_analysis.analyze_stock_data`

- `analyze_stock_data(df)`
  - Analyzes stock data for risk and return metrics
  - Returns: Dictionary with volatility, annual return, Sharpe ratio, RSI, MACD, max drawdown, VaR
  - Calculates technical indicators using pandas_ta

- `analyze_portfolio_correlation(stock_data)`
  - Analyzes correlation between stocks in a portfolio
  - Returns: Correlation matrix DataFrame

- `optimize_portfolio(stock_data)`
  - Optimizes portfolio weights using Modern Portfolio Theory
  - Maximizes Sharpe ratio using Efficient Frontier
  - Returns: Dictionary of optimized weights per ticker

### Stock Finding & Screening

**Module:** `stock_analysis.stock_finder`

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

**Module:** `stock_analysis.sector_analysis`

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

**Module:** `stock_analysis.market_indices`

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

- `train_nn_predictor(data, predict_weekly=False, test_size=0.2, do_tuning=True)` - Trains neural network predictor
- `predict_returns(model, scaler, data, predict_weekly=False)` - Makes return predictions using trained model

**Module:** `stock_analysis.ai.technical_analysis.intraday_volatility_analysis`

- `calculate_intraday_metrics(ticker='^GSPC', period='1y', threshold=0.5)` - Calculates intraday volatility metrics
- `analyze_return_distribution(data)` - Analyzes return distribution
- `analyze_trading_strategy(data, target_gain=0.3, stop_loss=0.3, spread_cost=0.5, leverage=1)` - Analyzes trading strategy performance

### Portfolio & Trading Analysis

**Module:** `stock_analysis.portfolio_evaluating`

- `analyze_trading_strategies(stock_data, signals, initial_investment=10000, leverage=10, signal_type="")`
  - Analyzes trading strategies with backtesting
  - Returns: Dictionary with performance metrics

- `calculate_max_drawdown(values)` - Calculates maximum drawdown
- `calculate_sharpe_ratio(values, risk_free_rate=0.02)` - Calculates Sharpe ratio
- `plot_trading_analysis(stock_data, results, title="Trading Analysis")` - Plots trading analysis results

**Module:** `stock_analysis.trading_cost_analysis`

**Classes:**
- `CalculateCost` - Base class for cost calculations
- `MiniFutureAnalyzer(CalculateCost)` - Analyzes mini futures costs
- `BullBearAnalyzer(CalculateCost)` - Analyzes bull/bear products

- `compare_products(days=31)` - Compares different trading products
- `compare_products_with_volatility(days=31, volatility=0.1)` - Compares products with volatility consideration

### Visualization

**Module:** `stock_analysis.efficient_frontier_plot`

- `plot_efficient_frontier(stock_data)` - Plots efficient frontier for portfolio optimization

**Module:** `stock_analysis.display_stock_data`

- `display_stock_data(df, events, ticker)` - Displays stock data with events

**Module:** `stock_analysis.visualization.trading_plots`

- `plot_trading_signals(simulation, stock_data, metrics=None, title="Trading Analysis")` - Plots trading signals and performance

### Data Caching

**Module:** `stock_analysis.cache_data`

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
from stock_analysis.fetch_stock_data import fetch_stock_data
df, events = fetch_stock_data('AAPL', period='1y')

# Analyze stock
from stock_analysis.analyze_stock_data import analyze_stock_data
analysis = analyze_stock_data(df)

# Get valuations
from stock_analysis.ai.fundamental_analysis.valuation_methods import StockValuation
valuation = StockValuation('AAPL')
all_valuations = valuation.get_all_valuations()

# Find high dividend stocks
from stock_analysis.stock_finder import find_high_dividend_stocks
stocks = find_high_dividend_stocks(min_yield=0.03, min_market_cap=1e9)

# Optimize portfolio
from stock_analysis.analyze_stock_data import optimize_portfolio
weights = optimize_portfolio(stock_data)
```