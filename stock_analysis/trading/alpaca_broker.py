import logging
from datetime import date

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide as AlpacaOrderSide
from alpaca.trading.enums import QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from stock_analysis.storage.portfolio_store import PortfolioState, PortfolioStore
from stock_analysis.trading.broker import AccountSnapshot
from stock_analysis.trading.cost_models import TradeCost
from stock_analysis.trading.decision import OrderIntent
from stock_analysis.trading.fills import FillResult, apply_fill_to_state

logger = logging.getLogger(__name__)

_NO_COST = TradeCost(commission=0.0, slippage=0.0)  # Alpaca commission-free equities


class BrokerRejected(Exception):
    """Raised when the account is flagged (trading_blocked/account_blocked) before
    we even attempt to submit — distinct from a network/API error so callers can
    tell "the broker said no" from "we couldn't reach the broker"."""


def _client_order_id(portfolio_name: str, run_id: int, ticker: str, side: str) -> str:
    """Deterministic per (portfolio, run, ticker, side) — lets Alpaca's own
    duplicate-client_order_id handling catch a same-run retry. NOTE: this does not
    protect against a retry under a *new* run_id (e.g. a different day's run) —
    that's covered instead by the kill-switch's unclean-shutdown detection forcing
    a human to review before any new run proceeds."""
    return f"{portfolio_name}-{run_id}-{ticker}-{side}"


class AlpacaBroker:
    """Real order execution via Alpaca. Works against the paper endpoint or live
    depending on how `client` was constructed (see trading/broker_config.py) —
    this class has no live/paper branch of its own, by design."""

    def __init__(self, client: TradingClient, store: PortfolioStore):
        self.client = client
        self.store = store

    def submit_order(
        self,
        state: PortfolioState,
        run_id: int,
        intent: OrderIntent,
        *,
        dry_run: bool,
        fill_date: date | None = None,
    ) -> str | None:
        if dry_run:
            return None

        client_order_id = _client_order_id(state.name, run_id, intent.ticker, intent.side)

        existing = self.store.get_order_by_client_id(client_order_id)
        if existing is not None and existing.broker_order_id is not None:
            logger.warning(
                "Order %s already exists for client_order_id=%s — not resubmitting",
                existing.broker_order_id,
                client_order_id,
            )
            return existing.broker_order_id

        self._assert_account_tradeable()

        if existing is not None:
            # A previous attempt persisted the idempotency key but crashed before
            # confirming the submission (broker_order_id is still None). Ask Alpaca
            # directly instead of trusting that silence — don't strand the order
            # forever and don't resubmit if Alpaca already has it.
            order_id = existing.id
            reconciled = self._find_existing_alpaca_order(client_order_id)
            if reconciled is not None:
                broker_order_id, broker_status = reconciled
                self.store.mark_order_submitted(order_id, broker_order_id, broker_status)
                logger.warning(
                    "Reconciled unconfirmed order %s with Alpaca as %s",
                    client_order_id,
                    broker_order_id,
                )
                return broker_order_id
            logger.warning(
                "No record of %s at Alpaca — a previous submission attempt never "
                "reached the broker, submitting now",
                client_order_id,
            )
        else:
            order_id = self.store.create_broker_order(
                portfolio_id=state.id,
                run_id=run_id,
                ticker=intent.ticker,
                side=intent.side,
                shares=intent.shares,
                signal_reason=intent.reason,
                client_order_id=client_order_id,
            )

        request = MarketOrderRequest(
            symbol=intent.ticker,
            qty=intent.shares,
            side=AlpacaOrderSide.BUY if intent.side == "BUY" else AlpacaOrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        alpaca_order = self.client.submit_order(request)

        broker_order_id = str(alpaca_order.id)
        broker_status = str(alpaca_order.status.value)
        self.store.mark_order_submitted(order_id, broker_order_id, broker_status)
        return broker_order_id

    def sync_fills(
        self,
        state: PortfolioState,
        price_data: dict[str, pd.DataFrame],
        session_date: date,
        *,
        dry_run: bool,
    ) -> list[FillResult]:
        if dry_run:
            return []

        open_orders = self.store.get_open_broker_orders(state.id)
        if not open_orders:
            return []

        by_broker_id = {o.broker_order_id: o for o in open_orders if o.broker_order_id}
        fills: list[FillResult] = []

        alpaca_orders = self.client.get_orders(
            GetOrdersRequest(status=QueryOrderStatus.ALL, symbols=list({o.ticker for o in open_orders}))
        )
        for alpaca_order in alpaca_orders:
            local = by_broker_id.get(str(alpaca_order.id))
            if local is None:
                continue

            status = str(alpaca_order.status.value)
            if status == "filled":
                fill_price = float(alpaca_order.filled_avg_price)
                intent = OrderIntent(
                    ticker=local.ticker,
                    side=local.side,
                    shares=local.shares,
                    reason=local.signal_reason or "",
                )
                apply_fill_to_state(state, intent, fill_price, _NO_COST)
                self.store.fill_order(local.id, fill_price, 0.0, 0.0, broker_status=status)

                if local.side == "BUY":
                    # Persist the post-fill totals apply_fill_to_state just computed
                    # (correctly averaged across any prior fills), not just this
                    # fill's shares/price — upsert_position overwrites, it doesn't add.
                    position = next(p for p in state.positions if p.ticker == local.ticker)
                    self.store.upsert_position(
                        state.id, local.ticker, position.shares, position.avg_cost
                    )
                else:
                    self.store.remove_position(state.id, local.ticker)
                self.store.update_cash(state.id, state.cash)

                fills.append(
                    FillResult(
                        order_id=local.id,
                        ticker=local.ticker,
                        side=local.side,
                        shares=local.shares,
                        fill_price=fill_price,
                        costs=_NO_COST,
                        notional=local.shares * fill_price,
                    )
                )
            elif status in ("canceled", "rejected", "expired"):
                self.store.update_broker_order_status(local.id, status.upper(), status)

        return fills

    def get_account_snapshot(self, state: PortfolioState) -> AccountSnapshot:
        account = self.client.get_account()
        positions = self.client.get_all_positions()
        return AccountSnapshot(
            cash=float(account.cash),
            positions={p.symbol: float(p.qty) for p in positions},
            source="alpaca",
        )

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.client.cancel_order_by_id(order_id)
        except Exception:
            logger.exception("Failed to cancel Alpaca order %s", order_id)
            return False
        return True

    def _assert_account_tradeable(self) -> None:
        account = self.client.get_account()
        if bool(account.trading_blocked) or bool(account.account_blocked):
            raise BrokerRejected("Alpaca account is trading_blocked or account_blocked")

    def _find_existing_alpaca_order(self, client_order_id: str) -> tuple[str, str] | None:
        try:
            order = self.client.get_order_by_client_id(client_order_id)
        except Exception:
            return None
        return str(order.id), str(order.status.value)

    # Live-mode notional ceilings (LIVE_MAX_NOTIONAL_PER_ORDER / _PORTFOLIO) are a
    # live_gate.py / Phase 5 concern: the Broker protocol's submit_order() doesn't
    # carry a price, so a per-order ceiling can't be enforced correctly at this
    # layer without an extra quote lookup. Enforce it at the gate that already
    # requires a human to deliberately flip LIVE_TRADING_ENABLED, not here.
