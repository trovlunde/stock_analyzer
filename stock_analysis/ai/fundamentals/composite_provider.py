import pandas as pd

from .edgar_adapter import EdgarAdapter
from .yfinance_adapter import YFinanceAdapter


class CompositeProvider:
    def __init__(self, primary=None, fallback=None):
        self._primary = primary if primary is not None else YFinanceAdapter()
        self._fallback = fallback if fallback is not None else EdgarAdapter()

    def get_annual_financials(self, ticker: str) -> pd.DataFrame:
        df = self._primary.get_annual_financials(ticker)
        if df.empty:
            df = self._fallback.get_annual_financials(ticker)
        return df

    def get_quarterly_financials(self, ticker: str) -> pd.DataFrame:
        df = self._primary.get_quarterly_financials(ticker)
        if df.empty:
            df = self._fallback.get_quarterly_financials(ticker)
        return df
