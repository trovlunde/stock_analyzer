import yfinance as yf
import pandas as pd

from stock_analysis.ai.fundamentals import CompositeProvider


class YFinanceProvider:
    def __init__(self, fundamentals_provider=None):
        self._fundamentals = fundamentals_provider if fundamentals_provider is not None else CompositeProvider()

    def get_history(
        self,
        ticker: str,
        period: str = "1y",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        t = yf.Ticker(ticker)
        if start is not None or end is not None:
            return t.history(start=start, end=end)
        return t.history(period=period)

    def get_financials(self, ticker: str) -> pd.DataFrame:
        return self._fundamentals.get_annual_financials(ticker)

    def get_earnings(self, ticker: str) -> pd.DataFrame:
        earnings = yf.Ticker(ticker).earnings_dates
        if earnings is None:
            return pd.DataFrame()
        return earnings
