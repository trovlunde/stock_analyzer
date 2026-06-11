from dataclasses import dataclass
from datetime import date, datetime, timezone

import pandas as pd

from stock_analysis.storage.portfolio_store import PortfolioStore, PortfolioState
from stock_analysis.trading.cost_models import CostModel, TradeCost
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
    ticker = intent.ticker
    notional = intent.shares * fill_price

    if intent.side == "BUY":
        total_cost = notional + costs.total
        state.cash -= total_cost
        existing = next((p for p in state.positions if p.ticker == ticker), None)
        if existing is None:
            from stock_analysis.storage.portfolio_store import Position

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


class PaperBroker:
    def __init__(self, store: PortfolioStore, cost_model: CostModel):
        self.store = store
        self.cost_model = cost_model

    def fill_pending_orders(
        self,
        state: PortfolioState,
        price_data: dict[str, pd.DataFrame],
        session_date: date,
        dry_run: bool,
    ) -> list[FillResult]:
        fills: list[FillResult] = []
        pending = self.store.get_pending_orders(state.id)

        for order in pending:
            if order.fill_date is not None and order.fill_date > session_date:
                continue

            df = price_data.get(order.ticker)
            if df is None:
                continue

            fill_price = open_price_for_date(df, session_date)
            if fill_price is None:
                continue

            notional = order.shares * fill_price
            costs = (
                self.cost_model.entry_cost(notional)
                if order.side == "BUY"
                else self.cost_model.exit_cost(notional)
            )

            if dry_run:
                fills.append(
                    FillResult(
                        order_id=order.id,
                        ticker=order.ticker,
                        side=order.side,
                        shares=order.shares,
                        fill_price=fill_price,
                        costs=costs,
                        notional=notional,
                    )
                )
                continue

            intent = OrderIntent(
                ticker=order.ticker,
                side=order.side,
                shares=order.shares,
                reason=order.signal_reason or "",
            )
            apply_fill_to_state(state, intent, fill_price, costs)
            self.store.fill_order(
                order.id, fill_price, costs.commission, costs.slippage
            )

            if order.side == "BUY":
                self.store.upsert_position(
                    state.id, order.ticker, order.shares, fill_price
                )
            else:
                self.store.remove_position(state.id, order.ticker)

            self.store.update_cash(state.id, state.cash)
            fills.append(
                FillResult(
                    order_id=order.id,
                    ticker=order.ticker,
                    side=order.side,
                    shares=order.shares,
                    fill_price=fill_price,
                    costs=costs,
                    notional=notional,
                )
            )

        return fills

    def queue_order(
        self,
        state: PortfolioState,
        run_id: int,
        intent: OrderIntent,
        fill_date: date,
        dry_run: bool,
    ) -> int | None:
        if dry_run:
            return None
        return self.store.create_pending_order(
            portfolio_id=state.id,
            run_id=run_id,
            ticker=intent.ticker,
            side=intent.side,
            shares=intent.shares,
            signal_reason=intent.reason,
            fill_date=fill_date,
        )
