from datetime import date, datetime, timezone

import pandas as pd

from stock_analysis.storage.portfolio_store import PortfolioState, Position
from stock_analysis.trading.cost_models import TradeCost
from stock_analysis.trading.decision import OrderIntent
from stock_analysis.trading.fills import apply_fill_to_state, open_price_for_date


def test_open_price_for_date():
    index = pd.to_datetime(["2024-06-03", "2024-06-04"])
    df = pd.DataFrame({"Open": [100.0, 101.0], "Close": [100.5, 101.5]}, index=index)
    assert open_price_for_date(df, date(2024, 6, 4)) == 101.0


def test_open_price_for_date_returns_none_when_missing():
    index = pd.to_datetime(["2024-06-03"])
    df = pd.DataFrame({"Open": [100.0], "Close": [100.5]}, index=index)
    assert open_price_for_date(df, date(2024, 6, 10)) is None


def test_apply_fill_to_state_buy_opens_new_position():
    state = PortfolioState(id=1, name="test", cash=10_000, positions=[])
    intent = OrderIntent(ticker="AAPL", side="BUY", shares=10, reason="test")
    costs = TradeCost(commission=1.0, slippage=0.5)

    apply_fill_to_state(state, intent, fill_price=100.0, costs=costs)

    assert state.cash == 10_000 - (1000.0 + 1.5)
    assert len(state.positions) == 1
    assert state.positions[0].ticker == "AAPL"
    assert state.positions[0].shares == 10
    assert state.positions[0].avg_cost == 100.0


def test_apply_fill_to_state_buy_averages_into_existing_position():
    state = PortfolioState(
        id=1,
        name="test",
        cash=10_000,
        positions=[
            Position(ticker="AAPL", shares=10, avg_cost=100.0, opened_at=datetime.now(timezone.utc))
        ],
    )
    intent = OrderIntent(ticker="AAPL", side="BUY", shares=10, reason="test")
    costs = TradeCost(commission=0.0, slippage=0.0)

    apply_fill_to_state(state, intent, fill_price=110.0, costs=costs)

    assert len(state.positions) == 1
    assert state.positions[0].shares == 20
    assert state.positions[0].avg_cost == 105.0


def test_apply_fill_to_state_sell_closes_position_and_credits_cash():
    state = PortfolioState(
        id=1,
        name="test",
        cash=5_000,
        positions=[
            Position(ticker="AAPL", shares=10, avg_cost=100.0, opened_at=datetime.now(timezone.utc))
        ],
    )
    intent = OrderIntent(ticker="AAPL", side="SELL", shares=10, reason="test")
    costs = TradeCost(commission=1.0, slippage=0.5)

    apply_fill_to_state(state, intent, fill_price=120.0, costs=costs)

    assert state.cash == 5_000 + (1200.0 - 1.5)
    assert state.positions == []
