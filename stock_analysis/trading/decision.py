from dataclasses import dataclass
from typing import Literal

from stock_analysis.storage.portfolio_store import PortfolioState, Position
from stock_analysis.trading.config import PortfolioConfig
from stock_analysis.trading.signals.base import SignalAction, SignalResult

OrderSide = Literal["BUY", "SELL"]


@dataclass(frozen=True)
class OrderIntent:
    ticker: str
    side: OrderSide
    shares: float
    reason: str


@dataclass(frozen=True)
class DecisionResult:
    signal: SignalResult
    intent: OrderIntent | None = None
    skipped_reason: str | None = None


def _position_for_ticker(state: PortfolioState, ticker: str) -> Position | None:
    for position in state.positions:
        if position.ticker == ticker:
            return position
    return None


def portfolio_value(state: PortfolioState, prices: dict[str, float]) -> float:
    total = state.cash
    for position in state.positions:
        price = prices.get(position.ticker, position.avg_cost)
        total += position.shares * price
    return total


def _stop_loss_triggered(
    position: Position, current_price: float, stop_loss_pct: float | None
) -> bool:
    if stop_loss_pct is None or position.avg_cost <= 0:
        return False
    drawdown = (position.avg_cost - current_price) / position.avg_cost
    return drawdown >= stop_loss_pct


def decide(
    *,
    ticker: str,
    signal: SignalResult,
    state: PortfolioState,
    config: PortfolioConfig,
    current_price: float,
    prices: dict[str, float],
    stop_loss_pct: float | None,
) -> DecisionResult:
    position = _position_for_ticker(state, ticker)
    open_positions = len(state.positions)

    if position is not None and _stop_loss_triggered(position, current_price, stop_loss_pct):
        return DecisionResult(
            intent=OrderIntent(
                ticker=ticker,
                side="SELL",
                shares=position.shares,
                reason=f"stop loss triggered ({stop_loss_pct:.1%})",
            ),
            signal=signal,
        )

    if signal.action == "BUY":
        if config.long_only and position is not None:
            return DecisionResult(signal=signal, skipped_reason="already long")
        if position is not None:
            return DecisionResult(signal=signal, skipped_reason="already positioned")
        if open_positions >= config.max_positions:
            return DecisionResult(signal=signal, skipped_reason="max positions reached")

        total_value = portfolio_value(state, prices)
        target_notional = total_value * config.position_size_pct
        if target_notional > state.cash:
            target_notional = state.cash

        shares = int(target_notional // current_price)
        if shares <= 0:
            return DecisionResult(signal=signal, skipped_reason="insufficient cash")

        return DecisionResult(
            intent=OrderIntent(
                ticker=ticker,
                side="BUY",
                shares=float(shares),
                reason=signal.reason,
            ),
            signal=signal,
        )

    if signal.action == "SELL":
        if position is None:
            return DecisionResult(signal=signal, skipped_reason="no position to sell")
        return DecisionResult(
            intent=OrderIntent(
                ticker=ticker,
                side="SELL",
                shares=position.shares,
                reason=signal.reason,
            ),
            signal=signal,
        )

    return DecisionResult(signal=signal, skipped_reason=None)
