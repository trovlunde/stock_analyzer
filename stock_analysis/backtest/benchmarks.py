"""Price loading and buy-and-hold benchmarks for backtests."""

import pandas as pd

from stock_analysis.market_data import MarketDataProvider


def load_price_data(
    tickers: list[str], provider: MarketDataProvider, period: str = "10y"
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV per ticker with a tz-naive index; tickers without data are skipped."""
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = provider.get_history(ticker, period=period)
        if df is None or df.empty:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        data[ticker] = df
    return data


def buy_and_hold_equity(
    price_data: dict[str, pd.DataFrame],
    sessions: pd.DatetimeIndex,
    initial_cash: float,
) -> pd.Series:
    """Equal-weight buy-and-hold over `sessions`: bought at the first Close, never
    rebalanced, no costs. Tickers with no price on the first session are left out."""
    closes = pd.DataFrame(
        {ticker: df["Close"] for ticker, df in price_data.items()}
    ).reindex(sessions).ffill()
    closes = closes.loc[:, closes.iloc[0].notna()]
    if closes.empty:
        raise ValueError("No ticker has a price on the first session")
    return initial_cash * (closes / closes.iloc[0]).mean(axis=1)
