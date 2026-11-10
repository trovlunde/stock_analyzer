"""Replay history through the same signal → decision → fill path as the daily job.

Each session mirrors `jobs/daily.py`: fill yesterday's orders at today's Open,
mark the portfolio at today's Close, check the kill switch, then evaluate
signals on data up to and including today and queue orders for the next session.
Strategies only ever receive bars up to the session being decided, so a strategy
cannot see the future however it is written.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Callable

import pandas as pd

from stock_analysis.storage.portfolio_store import PortfolioState
from stock_analysis.trading.config import (
    PortfolioSettings,
    stop_loss_for_position,
    strategy_names_for_position,
    strategy_params_for_position,
)
from stock_analysis.trading.cost_models import build_cost_model
from stock_analysis.trading.decision import (
    DecisionResult,
    OrderIntent,
    decide,
    portfolio_value,
    with_queued_order,
)
from stock_analysis.trading.fills import apply_fill_to_state
from stock_analysis.trading.regime import risk_off_reason
from stock_analysis.trading.risk import check_kill_switch
from stock_analysis.trading.sizing import annualised_volatility
from stock_analysis.trading.signals import (
    SignalResult,
    SignalStrategy,
    combine_signals,
    get_signal_strategy,
)

StrategyResolver = Callable[[str], SignalStrategy]


@dataclass(frozen=True)
class Trade:
    date: date
    ticker: str
    side: str
    shares: float
    price: float
    costs: float
    reason: str
    # SELL only: proceeds minus cost basis and this fill's costs
    realized_pnl: float | None = None

    @property
    def notional(self) -> float:
        return self.shares * self.price


@dataclass
class BacktestResult:
    initial_cash: float
    equity: pd.Series  # portfolio value at each session's Close
    exposure: pd.Series  # share of portfolio value held in positions
    trades: list[Trade] = field(default_factory=list)
    kill_switch_tripped_on: date | None = None
    kill_switch_reason: str | None = None


def _sessions(
    price_data: dict[str, pd.DataFrame], start: date | None, end: date | None
) -> pd.DatetimeIndex:
    calendar = pd.DatetimeIndex([])
    for df in price_data.values():
        calendar = calendar.union(df.index)
    if start is not None:
        calendar = calendar[calendar >= pd.Timestamp(start)]
    if end is not None:
        calendar = calendar[calendar <= pd.Timestamp(end)]
    return calendar


def _combined_signal(
    settings: PortfolioSettings,
    strategy_names: list[str],
    window: pd.DataFrame,
    ticker: str,
    params: dict,
    resolve_strategy: StrategyResolver,
) -> SignalResult:
    results = {
        name: resolve_strategy(name).evaluate(window, ticker, params)
        for name in strategy_names
    }
    if len(results) == 1:
        return next(iter(results.values()))
    return combine_signals(results, mode=settings.defaults.get("signal_mode", "consensus"))


def run_backtest(
    settings: PortfolioSettings,
    price_data: dict[str, pd.DataFrame],
    *,
    start: date | None = None,
    end: date | None = None,
    lookback_bars: int | None = None,
    resolve_strategy: StrategyResolver = get_signal_strategy,
    apply_kill_switch: bool = True,
    market_data: pd.DataFrame | None = None,
) -> BacktestResult:
    """Simulate `settings` over `price_data` ({ticker: OHLCV frame, tz-naive daily index}).

    Bars before `start` are still passed to strategies as history, so fetch more
    data than the period being tested to avoid a warm-up gap. `lookback_bars`
    caps how much history each strategy call receives (None = everything so far).
    `market_data` is the price history of `settings.regime_filter.ticker` and is
    required when the config has a regime filter.
    """
    portfolio_cfg = settings.portfolio
    regime_filter = settings.regime_filter
    if regime_filter is not None and market_data is None:
        raise ValueError(
            f"The config has a regime filter on {regime_filter.ticker}; pass its "
            f"price history as market_data"
        )
    market_close = market_data["Close"] if regime_filter is not None else None
    vol_sized = portfolio_cfg.sizing == "vol_target"
    cost_model = build_cost_model(settings.costs)
    positions = [
        p for p in settings.positions if p.enabled and p.ticker in price_data
    ]
    sessions = _sessions({p.ticker: price_data[p.ticker] for p in positions}, start, end)
    if sessions.empty:
        raise ValueError("No sessions to backtest for the given tickers and dates")

    state = PortfolioState(
        id=0, name=portfolio_cfg.name, cash=portfolio_cfg.initial_cash, positions=[]
    )
    pending: list[OrderIntent] = []
    trades: list[Trade] = []
    last_close: dict[str, float] = {}
    equity_history: list[tuple[str, float]] = []
    exposure_history: list[float] = []
    tripped_on: date | None = None
    tripped_reason: str | None = None

    for session in sessions:
        # 1. Orders queued at the previous close fill at this session's Open.
        unfilled: list[OrderIntent] = []
        for intent in pending:
            df = price_data[intent.ticker]
            if session not in df.index or pd.isna(df.at[session, "Open"]):
                unfilled.append(intent)  # ticker did not trade today; try next session
                continue
            fill_price = float(df.at[session, "Open"])
            notional = intent.shares * fill_price
            realized_pnl = None
            if intent.side == "BUY":
                costs = cost_model.entry_cost(notional)
            else:
                costs = cost_model.exit_cost(notional)
                held = next(p for p in state.positions if p.ticker == intent.ticker)
                realized_pnl = (fill_price - held.avg_cost) * intent.shares - costs.total
            apply_fill_to_state(state, intent, fill_price, costs)
            trades.append(
                Trade(
                    date=session.date(),
                    ticker=intent.ticker,
                    side=intent.side,
                    shares=intent.shares,
                    price=fill_price,
                    costs=costs.total,
                    reason=intent.reason,
                    realized_pnl=realized_pnl,
                )
            )
        pending = unfilled

        # 2. Mark to market at the Close.
        for position_entry in positions:
            df = price_data[position_entry.ticker]
            if session in df.index and not pd.isna(df.at[session, "Close"]):
                last_close[position_entry.ticker] = float(df.at[session, "Close"])
        prices = dict(last_close)
        value = portfolio_value(state, prices)

        # 3. Kill switch — once tripped it blocks new BUYs for the rest of the run,
        # as in production where it stays tripped until reset by hand.
        if apply_kill_switch and tripped_on is None:
            result = check_kill_switch(
                config=settings.risk,
                session_start_value=(
                    equity_history[-1][1] if equity_history else portfolio_cfg.initial_cash
                ),
                current_value=value,
                equity_curve=equity_history,
            )
            if result.tripped:
                tripped_on = session.date()
                tripped_reason = result.reason

        # 4. Decide on data up to and including this session; orders fill next session.
        awaiting_fill = {intent.ticker for intent in pending}
        # What decide() sees: the portfolio plus BUY orders already queued, so the
        # BUYs of one session cannot together spend more cash or slots than exist.
        decision_state = state
        for intent in pending:
            decision_state = with_queued_order(
                decision_state, intent, prices.get(intent.ticker, 0.0)
            )
        risk_off = (
            risk_off_reason(regime_filter, market_close.loc[:session])
            if regime_filter is not None
            else None
        )
        for position_entry in positions:
            ticker = position_entry.ticker
            df = price_data[ticker]
            if session not in df.index or ticker not in prices or ticker in awaiting_fill:
                continue
            window = df.loc[:session]
            if lookback_bars:
                window = window.iloc[-lookback_bars:]

            signal = _combined_signal(
                settings,
                strategy_names_for_position(settings, position_entry),
                window,
                ticker,
                strategy_params_for_position(settings, position_entry),
                resolve_strategy,
            )
            decision: DecisionResult = decide(
                ticker=ticker,
                signal=signal,
                state=decision_state,
                config=portfolio_cfg,
                current_price=prices[ticker],
                prices=prices,
                stop_loss_pct=stop_loss_for_position(settings, position_entry),
                volatility=(
                    annualised_volatility(window["Close"], portfolio_cfg.vol_lookback)
                    if vol_sized and signal.action == "BUY"
                    else None
                ),
            )
            if decision.intent is None:
                continue
            if decision.intent.side == "BUY" and (tripped_on is not None or risk_off):
                continue
            pending.append(decision.intent)
            decision_state = with_queued_order(decision_state, decision.intent, prices[ticker])

        equity_history.append((session.date().isoformat(), value))
        exposure_history.append((value - state.cash) / value if value else 0.0)

    return BacktestResult(
        initial_cash=portfolio_cfg.initial_cash,
        equity=pd.Series([v for _, v in equity_history], index=sessions, dtype=float),
        exposure=pd.Series(exposure_history, index=sessions, dtype=float),
        trades=trades,
        kill_switch_tripped_on=tripped_on,
        kill_switch_reason=tripped_reason,
    )
