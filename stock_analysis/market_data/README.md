# Market data providers

`MarketDataProvider` is the stable interface for prices, fundamentals, and earnings.
Consumers inject a provider (or use the module default `YFinanceProvider`) and never
import `yfinance` directly.

## Protocol methods

| Method | Purpose |
|--------|---------|
| `get_history(ticker, period=..., start=..., end=...)` | OHLCV price history |
| `get_financials(ticker)` | Annual income statement (via composite YFinance + EDGAR) |
| `get_earnings(ticker)` | Earnings calendar DataFrame |

`YFinanceProvider` also exposes non-protocol helpers (`get_raw_ticker`, `get_tickers_obj`,
`get_market_tickers_obj`, `get_quarterly_financials`, etc.) for batch and legacy call sites.

## Adding a paid source

1. Subclass or implement `MarketDataProvider` in a new adapter module.
2. Keep HTTP/API calls inside the adapter only.
3. Wire the adapter at the application boundary (CLI, job, or test fixture).

`PaidMarketDataStub` documents the extension point and raises `NotImplementedError`
until a real adapter exists. Paid APIs (FMP, Polygon, Stooq) are out of scope for
the stock-analysis hobby stack unless explicitly funded in a follow-up issue.
