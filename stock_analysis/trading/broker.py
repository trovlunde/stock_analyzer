from dataclasses import dataclass, field
from datetime import date
from typing import Protocol, runtime_checkable

import pandas as pd

from stock_analysis.storage.portfolio_store import PortfolioState
from stock_analysis.trading.decision import OrderIntent
from stock_analysis.trading.fills import FillResult


@dataclass(frozen=True)
class AccountSnapshot:
    cash: float
    positions: dict[str, float]  # ticker -> shares, the broker's view of truth
    source: str  # "local" | "alpaca"
    avg_costs: dict[str, float] = field(default_factory=dict)  # ticker -> avg cost, best-effort


@runtime_checkable
class Broker(Protocol):
    def submit_order(
        self,
        state: PortfolioState,
        run_id: int,
        intent: OrderIntent,
        *,
        dry_run: bool,
        fill_date: date | None = None,
    ) -> str | None:
        """`fill_date` is PaperBroker-specific (its next-session-open-fill simulation
        requires it; a real broker fills near-instantly and ignores it)."""
        ...

    def sync_fills(
        self,
        state: PortfolioState,
        price_data: dict[str, pd.DataFrame],
        session_date: date,
        *,
        dry_run: bool,
    ) -> list[FillResult]: ...

    def get_account_snapshot(self, state: PortfolioState) -> AccountSnapshot: ...

    def cancel_order(self, order_id: str) -> bool: ...
