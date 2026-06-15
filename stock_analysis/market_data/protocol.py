from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class MarketDataProvider(Protocol):
    def get_history(
        self,
        ticker: str,
        period: str = "1y",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame: ...

    def get_financials(self, ticker: str) -> pd.DataFrame: ...

    def get_earnings(self, ticker: str) -> pd.DataFrame: ...
