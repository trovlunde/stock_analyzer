from datetime import datetime, timezone

from stock_analysis.storage.portfolio_store import PortfolioState, Position
from stock_analysis.trading.config import PortfolioConfig
from stock_analysis.trading.decision import decide
from stock_analysis.trading.signals.base import SignalResult


def _config(**kwargs) -> PortfolioConfig:
    defaults = dict(
        name="test",
        enabled=True,
        initial_cash=100_000,
        max_positions=3,
        position_size_pct=0.10,
        long_only=True,
        stop_loss_pct=None,
    )
    defaults.update(kwargs)
    return PortfolioConfig(**defaults)


def test_buy_when_signal_and_cash_available():
    state = PortfolioState(id=1, name="test", cash=100_000, positions=[])
    signal = SignalResult(action="BUY", reason="golden cross")
    decision = decide(
        ticker="AAPL",
        signal=signal,
        state=state,
        config=_config(),
        current_price=100.0,
        prices={"AAPL": 100.0},
        stop_loss_pct=None,
    )
    assert decision.intent is not None
    assert decision.intent.side == "BUY"
    assert decision.intent.shares == 100


def test_sell_when_holding_and_signal_sell():
    state = PortfolioState(
        id=1,
        name="test",
        cash=50_000,
        positions=[
            Position(
                ticker="AAPL",
                shares=50,
                avg_cost=100.0,
                opened_at=datetime.now(timezone.utc),
            ),
        ],
    )
    signal = SignalResult(action="SELL", reason="death cross")
    decision = decide(
        ticker="AAPL",
        signal=signal,
        state=state,
        config=_config(),
        current_price=95.0,
        prices={"AAPL": 95.0},
        stop_loss_pct=None,
    )
    assert decision.intent is not None
    assert decision.intent.side == "SELL"
    assert decision.intent.shares == 50


def test_stop_loss_triggers_sell():
    from datetime import datetime, timezone

    state = PortfolioState(
        id=1,
        name="test",
        cash=50_000,
        positions=[
            Position(
                ticker="AAPL",
                shares=50,
                avg_cost=100.0,
                opened_at=datetime.now(timezone.utc),
            ),
        ],
    )
    signal = SignalResult(action="HOLD", reason="no crossover")
    decision = decide(
        ticker="AAPL",
        signal=signal,
        state=state,
        config=_config(),
        current_price=90.0,
        prices={"AAPL": 90.0},
        stop_loss_pct=0.08,
    )
    assert decision.intent is not None
    assert decision.intent.side == "SELL"
    assert "stop loss" in decision.intent.reason
