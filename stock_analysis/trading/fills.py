from dataclasses import dataclass
from datetime import date, datetime, timezone

import pandas as pd

from stock_analysis.storage.portfolio_store import PortfolioState, Position
from stock_analysis.trading.cost_models import TradeCost
from stock_analysis.trading.decision import OrderIntent


@dataclass(frozen=True)
class FillResult:
    order_id: int
    ticker: str
    side: str
    shares: float
    fill_price: float
    costs: TradeCost
    notional: float


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def open_price_for_date(df: pd.DataFrame, session_date: date) -> float | None:
    ts = pd.Timestamp(session_date)
    if ts in df.index:
        return float(df.loc[ts, "Open"])
    normalized = df.index.normalize()
    matches = df[normalized == ts]
    if matches.empty:
        return None
    return float(matches.iloc[0]["Open"])


def apply_fill_to_state(
    state: PortfolioState,
    intent: OrderIntent,
    fill_price: float,
    costs: TradeCost,
) -> PortfolioState:
    """Mutate `state.cash`/`state.positions` as if `intent` filled at `fill_price`.

    Broker-agnostic — shared by PaperBroker (simulated fills) and AlpacaBroker
    (real fills reported back by Alpaca)."""
    ticker = intent.ticker
    notional = intent.shares * fill_price

    if intent.side == "BUY":
        total_cost = notional + costs.total
        state.cash -= total_cost
        existing = next((p for p in state.positions if p.ticker == ticker), None)
        if existing is None:
            state.positions.append(
                Position(
                    ticker=ticker,
                    shares=intent.shares,
                    avg_cost=fill_price,
                    opened_at=_utcnow(),
                )
            )
        else:
            total_shares = existing.shares + intent.shares
            existing.avg_cost = (
                (existing.avg_cost * existing.shares) + (fill_price * intent.shares)
            ) / total_shares
            existing.shares = total_shares
        return state

    proceeds = notional - costs.total
    state.cash += proceeds
    state.positions = [p for p in state.positions if p.ticker != ticker]
    return state
