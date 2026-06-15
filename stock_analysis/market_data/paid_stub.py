"""Placeholder for future paid market-data adapters (FMP, Polygon, Stooq).

Implement ``MarketDataProvider`` and register in application wiring when a paid
API is adopted. HTTP clients stay inside adapter modules — not in consumers.
"""

import pandas as pd

from .protocol import MarketDataProvider


class PaidMarketDataStub:
    """Extension point for paid market-data sources — not implemented."""

    _MSG = (
        "Paid market-data adapters (FMP, Polygon, Stooq) are not implemented. "
        "Use YFinanceProvider or implement MarketDataProvider for your source."
    )

    def get_history(
        self,
        ticker: str,
        period: str = "1y",
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)

    def get_financials(self, ticker: str) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)

    def get_earnings(self, ticker: str) -> pd.DataFrame:
        raise NotImplementedError(self._MSG)


def assert_market_data_provider(provider: object) -> MarketDataProvider:
    """Runtime check that *provider* satisfies MarketDataProvider."""
    if not isinstance(provider, MarketDataProvider):
        raise TypeError(f"Expected MarketDataProvider, got {type(provider)!r}")
    return provider
