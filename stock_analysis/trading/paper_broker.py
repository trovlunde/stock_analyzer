from datetime import date

import pandas as pd

from stock_analysis.storage.portfolio_store import PortfolioStore, PortfolioState
from stock_analysis.trading.broker import AccountSnapshot
from stock_analysis.trading.cost_models import CostModel
from stock_analysis.trading.decision import OrderIntent
from stock_analysis.trading.fills import FillResult, apply_fill_to_state, open_price_for_date

__all__ = ["PaperBroker", "FillResult", "apply_fill_to_state", "open_price_for_date"]


class PaperBroker:
    """Simulates fills at the next trading session's Open price — a backtest-safe
    shape for daily bars. See trading/broker.py for why a real broker can't share
    this exact shape."""

    def __init__(self, store: PortfolioStore, cost_model: CostModel):
        self.store = store
        self.cost_model = cost_model

    def sync_fills(
        self,
        state: PortfolioState,
        price_data: dict[str, pd.DataFrame],
        session_date: date,
        *,
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
                # Persist the post-fill totals apply_fill_to_state just computed
                # (correctly averaged across any prior fills), not just this
                # fill's shares/price — upsert_position overwrites, it doesn't add.
                position = next(p for p in state.positions if p.ticker == order.ticker)
                self.store.upsert_position(
                    state.id, order.ticker, position.shares, position.avg_cost
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

    def submit_order(
        self,
        state: PortfolioState,
        run_id: int,
        intent: OrderIntent,
        *,
        dry_run: bool,
        fill_date: date | None = None,
    ) -> str | None:
        if fill_date is None:
            raise ValueError("PaperBroker.submit_order requires fill_date")
        if dry_run:
            return None
        order_id = self.store.create_pending_order(
            portfolio_id=state.id,
            run_id=run_id,
            ticker=intent.ticker,
            side=intent.side,
            shares=intent.shares,
            signal_reason=intent.reason,
            fill_date=fill_date,
        )
        return str(order_id)

    def get_account_snapshot(self, state: PortfolioState) -> AccountSnapshot:
        return AccountSnapshot(
            cash=state.cash,
            positions={p.ticker: p.shares for p in state.positions},
            source="local",
        )

    def cancel_order(self, order_id: str) -> bool:
        return self.store.cancel_pending_order(int(order_id))
