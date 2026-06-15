from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class FundamentalsProvider(Protocol):
    def get_annual_financials(self, ticker: str) -> pd.DataFrame: ...
    def get_quarterly_financials(self, ticker: str) -> pd.DataFrame: ...
