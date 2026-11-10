"""decide(): trailing stops, caps, rebalancing, score sizing and the cash buffer."""

from datetime import datetime, timezone

import pytest

from stock_analysis.storage.portfolio_store import PortfolioState, Position
from stock_analysis.trading.config import PortfolioConfig
from stock_analysis.trading.decision import OrderIntent, decide, with_queued_order
from stock_analysis.trading.fills import apply_fill_to_state
from stock_analysis.trading.cost_models import TradeCost
from stock_analysis.trading.signals.base import SignalResult

BUY = SignalResult(action="BUY", reason="buy")
HOLD = SignalResult(action="HOLD", reason="hold")
SELL = SignalResult(action="SELL", reason="sell")


def _config(**kwargs) -> PortfolioConfig:
    values = dict(
        name="test",
        enabled=True,
        initial_cash=100_000,
        max_positions=10,
        position_size_pct=0.10,
        long_only=True,
        stop_loss_pct=None,
    )
    values.update(kwargs)
    return PortfolioConfig(**values)


def _position(ticker, shares, avg_cost=100.0) -> Position:
    return Position(
        ticker=ticker, shares=shares, avg_cost=avg_cost, opened_at=datetime.now(timezone.utc)
    )


def _state(cash=100_000, positions=()) -> PortfolioState:
    return PortfolioState(id=1, name="test", cash=cash, positions=list(positions))


def _decide(signal, state, config=None, ticker="AAA", price=100.0, prices=None, **kwargs):
    return decide(
        ticker=ticker,
        signal=signal,
        state=state,
        config=config or _config(),
        current_price=price,
        prices=prices or {ticker: price},
        stop_loss_pct=kwargs.pop("stop_loss_pct", None),
        **kwargs,
    )


# --- trailing stop -----------------------------------------------------------------


def test_holding_at_or_below_the_trailing_stop_is_sold():
    state = _state(cash=90_000, positions=[_position("AAA", 100)])

    decision = _decide(BUY, state, price=95.0, stop_price=95.0, stop_reason="trailing stop: test")

    assert decision.intent == OrderIntent("AAA", "SELL", 100, "trailing stop: test")


def test_no_entry_while_the_price_is_below_the_trailing_stop():
    decision = _decide(BUY, _state(), price=95.0, stop_price=96.0)

    assert decision.intent is None
    assert "trailing stop" in decision.skipped_reason


def test_trailing_stop_above_the_price_changes_nothing():
    decision = _decide(BUY, _state(), price=100.0, stop_price=95.0)

    assert decision.intent.side == "BUY"


# --- cash buffer -------------------------------------------------------------------


def test_last_order_leaves_the_cash_buffer():
    # 5,000 cash in a 100,000 portfolio: the 10% order is cut to cash less 1% of the portfolio
    state = _state(cash=5_000, positions=[_position("BBB", 950)])

    decision = _decide(BUY, state, prices={"AAA": 100.0, "BBB": 100.0})

    assert decision.intent.shares == 40


def test_buffer_does_not_touch_an_order_that_fits():
    decision = _decide(BUY, _state())

    assert decision.intent.shares == 100


def test_queued_addition_takes_cash_but_no_new_slot():
    state = _state(cash=50_000, positions=[_position("AAA", 100)])

    queued = with_queued_order(state, OrderIntent("AAA", "BUY", 50, "rebalance"), 100.0)

    assert [(p.ticker, p.shares) for p in queued.positions] == [("AAA", 150)]
    assert queued.cash == 45_000
    assert state.positions[0].shares == 100  # the real state is untouched


# --- score sizing ------------------------------------------------------------------


def test_size_factor_scales_the_position():
    signal = SignalResult(action="BUY", reason="blend", metadata={"size_factor": 0.5})

    assert _decide(signal, _state()).intent.shares == 50


# --- sector cap --------------------------------------------------------------------


def test_sector_cap_cuts_the_order_to_the_room_left():
    config = _config(max_sector_pct=0.25)
    state = _state(cash=80_000, positions=[_position("BBB", 200)])  # 20% in tech
    sectors = {"AAA": "tech", "BBB": "tech"}

    decision = _decide(BUY, state, config, prices={"AAA": 100.0, "BBB": 100.0}, sectors=sectors)

    assert decision.intent.shares == 50  # 5% of 100,000 left under the 25% cap


def test_sector_cap_blocks_when_the_sector_is_full():
    config = _config(max_sector_pct=0.20)
    state = _state(cash=80_000, positions=[_position("BBB", 200)])
    sectors = {"AAA": "tech", "BBB": "tech"}

    decision = _decide(BUY, state, config, prices={"AAA": 100.0, "BBB": 100.0}, sectors=sectors)

    assert decision.intent is None
    assert "sector cap" in decision.skipped_reason


def test_sector_cap_ignores_other_sectors_and_unlabelled_tickers():
    config = _config(max_sector_pct=0.20)
    state = _state(cash=80_000, positions=[_position("BBB", 200)])

    other = _decide(BUY, state, config, prices={"AAA": 100.0, "BBB": 100.0},
                    sectors={"AAA": "energy", "BBB": "tech"})
    unlabelled = _decide(BUY, state, config, prices={"AAA": 100.0, "BBB": 100.0},
                         sectors={"BBB": "tech"})

    assert other.intent.shares == 100
    assert unlabelled.intent.shares == 100


# --- correlation cap ---------------------------------------------------------------


def test_correlation_cap_blocks_another_lookalike():
    config = _config(max_correlation=0.8, max_correlated_positions=2)
    state = _state(cash=70_000, positions=[_position(t, 100) for t in ("BBB", "CCC", "DDD")])
    prices = {t: 100.0 for t in ("AAA", "BBB", "CCC", "DDD")}

    blocked = _decide(BUY, state, config, prices=prices,
                      correlations={"BBB": 0.9, "CCC": 0.85, "DDD": 0.1})
    allowed = _decide(BUY, state, config, prices=prices,
                      correlations={"BBB": 0.9, "CCC": 0.5, "DDD": 0.1})

    assert blocked.intent is None
    assert "correlation cap" in blocked.skipped_reason
    assert allowed.intent.side == "BUY"


# --- rebalancing -------------------------------------------------------------------


def test_rebalance_trims_a_holding_that_has_grown_past_the_band():
    config = _config(rebalance_band=0.25)
    # 10% target; the holding has doubled to about 18% of the portfolio
    state = _state(cash=90_000, positions=[_position("AAA", 100)])

    decision = _decide(HOLD, state, config, price=200.0, rebalance_due=True)

    assert decision.intent.side == "SELL"
    assert decision.intent.shares == 45  # back to 10% of 110,000
    assert "rebalance" in decision.intent.reason


def test_rebalance_tops_up_a_holding_that_has_shrunk():
    config = _config(rebalance_band=0.25)
    state = _state(cash=90_000, positions=[_position("AAA", 100)])

    decision = _decide(BUY, state, config, price=50.0, rebalance_due=True)

    assert decision.intent.side == "BUY"
    assert decision.intent.shares == 90  # from 5,000 to 10% of 95,000


def test_no_rebalance_inside_the_band_or_when_not_due():
    config = _config(rebalance_band=0.25)
    state = _state(cash=90_000, positions=[_position("AAA", 100)])

    inside = _decide(BUY, state, config, price=110.0, rebalance_due=True)
    not_due = _decide(BUY, state, config, price=200.0, rebalance_due=False)

    assert inside.intent is None and inside.skipped_reason == "already long"
    assert not_due.intent is None and not_due.skipped_reason == "already long"


def test_rebalance_is_off_without_a_band():
    state = _state(cash=90_000, positions=[_position("AAA", 100)])

    assert _decide(HOLD, state, price=200.0, rebalance_due=True).intent is None


def test_sell_signal_wins_over_rebalancing():
    config = _config(rebalance_band=0.25)
    state = _state(cash=90_000, positions=[_position("AAA", 100)])

    decision = _decide(SELL, state, config, price=200.0, rebalance_due=True)

    assert decision.intent == OrderIntent("AAA", "SELL", 100, "sell")


def test_partial_sale_keeps_the_rest_at_the_same_cost():
    state = _state(cash=0, positions=[_position("AAA", 100, avg_cost=80.0)])

    apply_fill_to_state(state, OrderIntent("AAA", "SELL", 40, "rebalance"), 100.0, TradeCost(1.0, 0.0))

    assert [(p.ticker, p.shares, p.avg_cost) for p in state.positions] == [("AAA", 60, 80.0)]
    assert state.cash == pytest.approx(3_999.0)


def test_full_sale_removes_the_position():
    state = _state(cash=0, positions=[_position("AAA", 100)])

    apply_fill_to_state(state, OrderIntent("AAA", "SELL", 100, "sell"), 100.0, TradeCost(0.0, 0.0))

    assert state.positions == []
