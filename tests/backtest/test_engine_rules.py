"""Backtest engine with trailing stops, allocation sizing, caps and rebalancing."""

import numpy as np
import pandas as pd
import pytest

from stock_analysis.backtest.engine import run_backtest
from stock_analysis.trading.config import PortfolioConfig, PortfolioSettings, PositionEntry
from stock_analysis.trading.signals.base import SignalResult

DATES = pd.date_range("2022-01-03", periods=300, freq="B")


def _frame(closes, spread=0.5) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"Open": closes, "High": closes + spread, "Low": closes - spread, "Close": closes},
        index=DATES[: len(closes)],
    )


def _settings(tickers=("AAA",), sectors=None, **portfolio) -> PortfolioSettings:
    values = dict(
        name="bt", enabled=True, initial_cash=100_000, max_positions=10,
        position_size_pct=0.10, long_only=True, stop_loss_pct=None,
    )
    values.update(portfolio)
    return PortfolioSettings(
        portfolio=PortfolioConfig(**values),
        costs={"model": "equity", "commission_per_trade": 0.0, "slippage_pct": 0.0},
        defaults={"strategies": ["scripted"]},
        positions=[PositionEntry(ticker=t, sector=(sectors or {}).get(t)) for t in tickers],
    )


class AlwaysBuy:
    def __init__(self, after: int = 0):
        self.after = after

    def evaluate(self, df, ticker, params):
        return SignalResult(action="BUY" if len(df) > self.after else "HOLD", reason="scripted")


def _run(settings, price_data, strategy=None):
    strategy = strategy or AlwaysBuy()
    return run_backtest(settings, price_data, resolve_strategy=lambda name: strategy)


def test_trailing_stop_exits_and_keeps_the_signal_from_buying_straight_back():
    # Up to 150, a sharp fall to 120, flat there, then a new climb
    closes = np.concatenate(
        [np.linspace(100, 150, 100), np.linspace(148, 120, 15), np.full(60, 120.0),
         np.linspace(121, 160, 40)]
    )
    settings = _settings(stop_type="atr_trailing", stop_atr_multiple=3.0, stop_atr_period=10,
                         stop_lookback=40)

    result = _run(settings, {"AAA": _frame(closes)}, AlwaysBuy(after=45))

    sides = [(trade.side, trade.date) for trade in result.trades]
    assert [side for side, _ in sides] == ["BUY", "SELL", "BUY"]
    assert "trailing stop" in result.trades[1].reason
    # The signal says BUY every day, but the re-entry waits until the old high has
    # left the lookback window — well after the stop-out, not the next session
    assert (sides[2][1] - sides[1][1]).days > 30


def test_without_a_trailing_stop_the_position_is_simply_held():
    closes = np.concatenate([np.linspace(100, 150, 100), np.linspace(148, 120, 15)])

    result = _run(_settings(), {"AAA": _frame(closes)}, AlwaysBuy(after=45))

    assert [trade.side for trade in result.trades] == ["BUY"]


def test_rebalancing_trims_a_winner_back_towards_its_target():
    flat = np.full(200, 100.0)
    winner = np.concatenate([np.full(60, 100.0), np.linspace(100, 300, 140)])
    price_data = {"WIN": _frame(winner), "FLAT": _frame(flat)}
    drifting = _run(_settings(("WIN", "FLAT")), price_data)
    rebalanced = _run(_settings(("WIN", "FLAT"), rebalance_band=0.25), price_data)

    assert [trade.side for trade in drifting.trades] == ["BUY", "BUY"]
    trims = [t for t in rebalanced.trades if t.side == "SELL"]
    assert trims and all(t.ticker == "WIN" and "rebalance" in t.reason for t in trims)
    # Partial sales: the position is still open at the end, and smaller than at entry
    assert sum(t.shares for t in trims) < rebalanced.trades[0].shares
    # Rebalancing happens on the first session of a month only
    months = [(t.date.year, t.date.month) for t in trims]
    assert len(months) == len(set(months))
    assert all(t.realized_pnl > 0 for t in trims)


def test_hrp_sizing_gives_the_calm_asset_the_larger_position():
    rng = np.random.default_rng(3)
    calm = 100 * np.exp(np.cumsum(rng.normal(0, 0.004, 300)))
    wild = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))
    settings = _settings(("CALM", "WILD"), sizing="hrp", max_position_pct=1.0,
                         allocation_lookback=100)

    result = _run(settings, {"CALM": _frame(calm), "WILD": _frame(wild)}, AlwaysBuy(after=150))

    notional = {trade.ticker: trade.notional for trade in result.trades}
    assert notional["CALM"] > 3 * notional["WILD"]
    assert sum(notional.values()) <= 100_000


def test_sector_and_correlation_caps_limit_lookalikes():
    rng = np.random.default_rng(5)
    market = rng.normal(0, 0.01, 300)

    def lookalike():
        return _frame(100 * np.exp(np.cumsum(market + rng.normal(0, 0.001, 300))))

    price_data = {"E1": lookalike(), "E2": lookalike(), "E3": lookalike(),
                  "OTHER": _frame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))}
    tickers = tuple(price_data)

    by_correlation = _run(
        _settings(tickers, max_correlation=0.8, max_correlated_positions=1,
                  correlation_lookback=60),
        price_data, AlwaysBuy(after=100),
    )
    by_sector = _run(
        _settings(tickers, sectors={"E1": "eq", "E2": "eq", "E3": "eq"}, max_sector_pct=0.15),
        price_data, AlwaysBuy(after=100),
    )

    assert {t.ticker for t in by_correlation.trades} == {"E1", "OTHER"}
    sector_notional = sum(t.notional for t in by_sector.trades if t.ticker.startswith("E"))
    assert sector_notional == pytest.approx(15_000, rel=0.02)
    assert any(t.ticker == "OTHER" for t in by_sector.trades)
