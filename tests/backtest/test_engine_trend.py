"""Backtest engine with the state-based trend signal, volatility sizing and regime filter."""

import dataclasses

import numpy as np
import pandas as pd
import pytest

from stock_analysis.backtest.engine import run_backtest
from stock_analysis.trading.config import PortfolioConfig, PortfolioSettings, PositionEntry
from stock_analysis.trading.regime import RegimeFilter
from stock_analysis.trading.signals.base import SignalResult
from stock_analysis.trading.signals.trend_state import TrendStateStrategy

DATES = pd.date_range("2022-01-03", periods=200, freq="B")
TREND_PARAMS = {"trend_lookbacks": [20, 40, 60], "trend_ma": 50}


def _frame(closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame({"Open": closes, "Close": closes}, index=DATES[: len(closes)])


def _settings(tickers=("AAA",), **portfolio) -> PortfolioSettings:
    values = dict(
        name="bt",
        enabled=True,
        initial_cash=100_000,
        max_positions=10,
        position_size_pct=0.10,
        long_only=True,
        stop_loss_pct=None,
    )
    values.update(portfolio)
    return PortfolioSettings(
        portfolio=PortfolioConfig(**values),
        costs={"model": "equity", "commission_per_trade": 0.0, "slippage_pct": 0.0},
        defaults={"strategies": ["trend_state"], **TREND_PARAMS},
        positions=[PositionEntry(ticker=ticker) for ticker in tickers],
    )


class AlwaysBuy:
    """BUY on every session once it has been shown `after` bars."""

    def __init__(self, after: int = 0):
        self.after = after

    def evaluate(self, df, ticker, params):
        action = "BUY" if len(df) > self.after else "HOLD"
        return SignalResult(action=action, reason="scripted")


def _run(settings, price_data, strategy=None, **kwargs):
    strategy = strategy or TrendStateStrategy()
    return run_backtest(settings, price_data, resolve_strategy=lambda name: strategy, **kwargs)


def test_trend_state_enters_mid_trend_and_exits_when_the_trend_breaks():
    # Up for 120 sessions, then down for 80
    up = 100 * 1.003 ** np.arange(120)
    down = up[-1] * 0.99 ** np.arange(1, 81)
    result = _run(_settings(), {"AAA": _frame(np.concatenate([up, down]))})

    sides = [trade.side for trade in result.trades]
    assert sides == ["BUY", "SELL"]
    buy, sell = result.trades
    # First session with enough history is index 60; the order fills the next session.
    # No cross happened that day — the strategy bought because the trend was already up.
    assert buy.date == DATES[61].date()
    assert DATES[120].date() < sell.date < DATES[160].date()


def test_trend_state_buys_again_after_a_missed_entry():
    """A BUY that could not be placed (portfolio full) is retried once there is room."""
    rising = 100 * 1.003 ** np.arange(200)
    falls_late = np.concatenate([rising[:120], rising[119] * 0.99 ** np.arange(1, 81)])
    price_data = {"AAA": _frame(falls_late), "BBB": _frame(rising)}
    settings = _settings(("AAA", "BBB"), max_positions=1)

    result = _run(settings, price_data)

    trades = [(trade.ticker, trade.side) for trade in result.trades]
    assert trades[:2] == [("AAA", "BUY"), ("AAA", "SELL")]
    # BBB was in an uptrend all along but had no slot until AAA was sold
    assert trades[2] == ("BBB", "BUY")
    assert result.trades[2].date > result.trades[1].date


def test_vol_target_sizes_the_jumpy_ticker_smaller():
    rng = np.random.default_rng(1)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.005, 200)))
    jumpy = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 200)))
    settings = _settings(
        ("CALM", "JUMPY"), sizing="vol_target", risk_per_position=0.01,
        max_position_pct=0.50, vol_lookback=60,
    )

    result = _run(
        settings, {"CALM": _frame(calm), "JUMPY": _frame(jumpy)}, AlwaysBuy(after=100)
    )

    notional = {trade.ticker: trade.notional for trade in result.trades}
    # About four times the volatility, so about a quarter of the position
    assert notional["CALM"] / notional["JUMPY"] == pytest.approx(4, rel=0.35)


def test_vol_target_uses_the_fixed_size_until_volatility_is_known():
    settings = _settings(sizing="vol_target", vol_lookback=60)

    result = _run(settings, {"AAA": _frame(np.linspace(100, 120, 200))}, AlwaysBuy())

    # Bought on the first session, long before 60 returns exist: 10% of 100,000
    assert result.trades[0].notional == pytest.approx(10_000, rel=0.01)


def test_regime_filter_blocks_buys_while_the_market_is_below_its_average():
    # Flat, then sliding for 50 sessions, then a jump back above the average
    market = _frame(
        np.concatenate([np.full(100, 100.0), np.linspace(99, 60, 50), np.full(50, 120.0)])
    )
    settings = dataclasses.replace(
        _settings(), regime_filter=RegimeFilter(ticker="MKT", ma=20)
    )
    price_data = {"AAA": _frame(np.full(200, 50.0))}

    # The ticker's own signal says BUY from session 110 on, in the middle of the slide
    result = _run(settings, price_data, AlwaysBuy(after=110), market_data=market)

    assert len(result.trades) == 1
    # The market recovers at index 150; the order fills the next session
    assert result.trades[0].date == DATES[151].date()


def test_buys_on_the_same_session_share_the_free_slots_and_cash():
    tickers = ("AAA", "BBB", "CCC")
    price_data = {ticker: _frame(np.full(200, 50.0)) for ticker in tickers}

    limited_slots = _run(_settings(tickers, max_positions=2), price_data, AlwaysBuy())
    limited_cash = _run(_settings(tickers, position_size_pct=0.40), price_data, AlwaysBuy())

    assert [trade.ticker for trade in limited_slots.trades] == ["AAA", "BBB"]
    # 40% + 40% + what is left after the 1% cash buffer, never more than the cash there is
    assert [trade.notional for trade in limited_cash.trades] == [40_000, 40_000, 19_000]
    assert limited_cash.exposure.max() == pytest.approx(0.99)


def test_regime_filter_does_not_block_sells():
    market = _frame(np.concatenate([np.full(100, 100.0), np.full(100, 80.0)]))
    settings = dataclasses.replace(
        _settings(), regime_filter=RegimeFilter(ticker="MKT", ma=20)
    )
    strategy = type(
        "Scripted",
        (),
        {
            "evaluate": lambda self, df, ticker, params: SignalResult(
                action="BUY" if len(df) < 50 else "SELL", reason="scripted"
            )
        },
    )()

    result = _run(settings, {"AAA": _frame(np.full(200, 50.0))}, strategy, market_data=market)

    assert [trade.side for trade in result.trades] == ["BUY", "SELL"]


def test_regime_filter_without_market_data_is_an_error():
    settings = dataclasses.replace(
        _settings(), regime_filter=RegimeFilter(ticker="MKT", ma=20)
    )

    with pytest.raises(ValueError, match="regime filter on MKT"):
        _run(settings, {"AAA": _frame(np.full(200, 50.0))})
