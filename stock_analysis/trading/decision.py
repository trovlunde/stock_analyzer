from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

from stock_analysis.storage.portfolio_store import PortfolioState, Position
from stock_analysis.trading.config import PortfolioConfig
from stock_analysis.trading.signals.base import SignalAction, SignalResult
from stock_analysis.trading.sizing import position_weight

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


def with_queued_order(
    state: PortfolioState,
    intent: OrderIntent,
    price: float,
) -> PortfolioState:
    """The portfolio as later decisions in the same run should see it once `intent`
    is queued.

    Orders fill at the next session, so `state` itself does not change when one is
    queued. Without this, every BUY decided in one run would see the same cash and
    the same free slots, and together they could exceed both. A queued BUY takes
    its cash and its slot straight away; a queued SELL frees nothing until it fills.
    `state` is not modified.
    """
    if intent.side != "BUY":
        return state
    positions = []
    merged = False
    for position in state.positions:
        if position.ticker == intent.ticker:
            # An addition to an existing holding takes cash but not a new slot
            positions.append(
                Position(
                    ticker=position.ticker,
                    shares=position.shares + intent.shares,
                    avg_cost=position.avg_cost,
                    opened_at=position.opened_at,
                )
            )
            merged = True
        else:
            positions.append(position)
    if not merged:
        positions.append(
            Position(
                ticker=intent.ticker,
                shares=intent.shares,
                avg_cost=price,
                opened_at=datetime.now(timezone.utc),
            )
        )
    return PortfolioState(
        id=state.id,
        name=state.name,
        cash=state.cash - intent.shares * price,
        positions=positions,
    )


def _stop_loss_triggered(
    position: Position, current_price: float, stop_loss_pct: float | None
) -> bool:
    if stop_loss_pct is None or position.avg_cost <= 0:
        return False
    drawdown = (position.avg_cost - current_price) / position.avg_cost
    return drawdown >= stop_loss_pct


def _sector_room(
    ticker: str,
    state: PortfolioState,
    prices: dict[str, float],
    sectors: dict[str, str] | None,
    max_sector_value: float | None,
) -> float | None:
    """Value that can still be bought in `ticker`'s sector, or None when uncapped."""
    if max_sector_value is None or not sectors or ticker not in sectors:
        return None
    sector = sectors[ticker]
    held = sum(
        position.shares * prices.get(position.ticker, position.avg_cost)
        for position in state.positions
        if sectors.get(position.ticker) == sector and position.ticker != ticker
    )
    return max_sector_value - held


def _rebalance(
    *,
    ticker: str,
    signal: SignalResult,
    position: Position,
    state: PortfolioState,
    config: PortfolioConfig,
    current_price: float,
    total_value: float,
    target_weight: float,
) -> DecisionResult | None:
    """An order that moves a holding back to its target weight, or None when it is
    inside the band (or the trade would be too small to be worth its costs)."""
    band = config.rebalance_band
    if band is None or total_value <= 0 or target_weight <= 0:
        return None
    current_weight = position.shares * current_price / total_value
    drift = current_weight - target_weight
    if abs(drift) <= band * target_weight:
        return None
    if abs(drift) < config.rebalance_min_trade_pct:
        return None

    if drift > 0:
        shares = int(drift * total_value // current_price)
        side: OrderSide = "SELL"
        shares = min(shares, int(position.shares))
    else:
        notional = min(-drift * total_value, state.cash - config.cash_buffer_pct * total_value)
        shares = int(notional // current_price)
        side = "BUY"
    if shares <= 0:
        return None
    return DecisionResult(
        intent=OrderIntent(
            ticker=ticker,
            side=side,
            shares=float(shares),
            reason=(
                f"rebalance: weight {current_weight:.1%} outside the band around "
                f"target {target_weight:.1%}"
            ),
        ),
        signal=signal,
    )


def decide(
    *,
    ticker: str,
    signal: SignalResult,
    state: PortfolioState,
    config: PortfolioConfig,
    current_price: float,
    prices: dict[str, float],
    stop_loss_pct: float | None,
    volatility: float | None = None,
    universe_weight: float | None = None,
    exposure_scale: float = 1.0,
    stop_price: float | None = None,
    stop_reason: str | None = None,
    sectors: dict[str, str] | None = None,
    correlations: dict[str, float] | None = None,
    rebalance_due: bool = False,
) -> DecisionResult:
    """Turn a signal into an order, or a reason for not placing one.

    Everything after `stop_loss_pct` is optional and computed by the caller from
    price history (see `trading.portfolio_rules`):

    - `volatility`: the ticker's annualised volatility, for `vol_target` sizing.
    - `universe_weight`: the ticker's weight from the allocation over the whole
      universe, for `min_variance` / `hrp` sizing.
    - `exposure_scale`: multiplier on new position sizes from the portfolio-level
      volatility target (1 = no scaling).
    - `stop_price`: a trailing stop level. A holding at or below it is sold, and a
      BUY is not placed while the price is at or below it — otherwise a state-based
      signal would buy straight back the day after being stopped out.
    - `sectors`: {ticker: sector}, for `config.max_sector_pct`.
    - `correlations`: this ticker's return correlation with every other ticker,
      for `config.max_correlation`.
    - `rebalance_due`: whether holdings may be moved back to their target weight
      in this run (`config.rebalance_band`).

    A signal may carry `metadata["size_factor"]` (0–1) to take a smaller position
    than the sizing rule gives (used by the score-based signal mode).
    """
    position = _position_for_ticker(state, ticker)
    open_positions = len(state.positions)
    below_trailing_stop = stop_price is not None and current_price <= stop_price

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
    if position is not None and below_trailing_stop:
        return DecisionResult(
            intent=OrderIntent(
                ticker=ticker,
                side="SELL",
                shares=position.shares,
                reason=stop_reason or f"trailing stop at {stop_price:.2f}",
            ),
            signal=signal,
        )

    size_factor = signal.metadata.get("size_factor") if signal.metadata else None
    weight = position_weight(config, volatility, universe_weight) * exposure_scale
    if size_factor is not None:
        weight *= min(1.0, max(0.0, float(size_factor)))

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

    if position is not None:
        if rebalance_due:
            rebalance = _rebalance(
                ticker=ticker,
                signal=signal,
                position=position,
                state=state,
                config=config,
                current_price=current_price,
                total_value=portfolio_value(state, prices),
                target_weight=weight,
            )
            if rebalance is not None:
                return rebalance
        if signal.action == "BUY":
            reason = "already long" if config.long_only else "already positioned"
            return DecisionResult(signal=signal, skipped_reason=reason)
        return DecisionResult(signal=signal, skipped_reason=None)

    if signal.action == "BUY":
        if open_positions >= config.max_positions:
            return DecisionResult(signal=signal, skipped_reason="max positions reached")
        if below_trailing_stop:
            return DecisionResult(
                signal=signal,
                skipped_reason=f"price is at or below the trailing stop ({stop_price:.2f}) — no entry",
            )
        if config.max_correlation is not None and correlations:
            correlated = [
                held.ticker
                for held in state.positions
                if correlations.get(held.ticker, 0.0) >= config.max_correlation
            ]
            if len(correlated) >= config.max_correlated_positions:
                return DecisionResult(
                    signal=signal,
                    skipped_reason=(
                        f"correlation cap: already holding {len(correlated)} position(s) "
                        f"correlated {config.max_correlation:.2f} or more with {ticker}"
                    ),
                )

        total_value = portfolio_value(state, prices)
        target_notional = total_value * weight
        sector_room = _sector_room(
            ticker,
            state,
            prices,
            sectors,
            total_value * config.max_sector_pct if config.max_sector_pct is not None else None,
        )
        if sector_room is not None:
            if sector_room < current_price:
                return DecisionResult(
                    signal=signal,
                    skipped_reason=f"sector cap reached ({config.max_sector_pct:.0%} in {sectors[ticker]})",
                )
            target_notional = min(target_notional, sector_room)
        # Keep a little of the portfolio in cash: orders are sized at today's Close
        # but fill at the next Open plus commission and slippage, so spending the
        # last of the cash would leave it negative.
        target_notional = min(target_notional, state.cash - config.cash_buffer_pct * total_value)

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

    return DecisionResult(signal=signal, skipped_reason=None)
