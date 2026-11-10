from datetime import date

import pandas as pd
import pytest

from stock_analysis.backtest.benchmarks import buy_and_hold_equity
from stock_analysis.backtest.engine import run_backtest
from stock_analysis.trading.config import PortfolioConfig, PortfolioSettings, PositionEntry
from stock_analysis.trading.risk import RiskConfig
from stock_analysis.trading.signals.base import SignalResult

DATES = pd.date_range("2024-01-01", periods=6, freq="B")


def _prices(opens, closes, dates=DATES):
    return pd.DataFrame({"Open": opens, "Close": closes}, index=dates[: len(closes)], dtype=float)


def _settings(tickers=("AAA",), position_size_pct=0.10, risk=None):
    return PortfolioSettings(
        portfolio=PortfolioConfig(
            name="bt",
            enabled=True,
            initial_cash=100_000,
            max_positions=10,
            position_size_pct=position_size_pct,
            long_only=True,
            stop_loss_pct=None,
        ),
        costs={"model": "equity", "commission_per_trade": 1.0, "slippage_pct": 0.0005},
        defaults={"strategies": ["scripted"]},
        positions=[PositionEntry(ticker=ticker) for ticker in tickers],
        risk=risk or RiskConfig(),
    )


class ScriptedStrategy:
    """Emits the action scripted for the latest bar it is shown; HOLD otherwise."""

    def __init__(self, actions: dict):
        self.actions = {(ticker, pd.Timestamp(day)): action for (ticker, day), action in actions.items()}
        self.seen: list[tuple[str, pd.Timestamp, int]] = []

    def evaluate(self, df, ticker, params):
        latest = df.index[-1]
        self.seen.append((ticker, latest, len(df)))
        action = self.actions.get((ticker, latest), "HOLD")
        return SignalResult(action=action, reason=f"scripted {action}")


def _run(settings, price_data, strategy, **kwargs):
    return run_backtest(settings, price_data, resolve_strategy=lambda name: strategy, **kwargs)


def test_buy_signal_fills_at_next_session_open_sized_at_signal_close():
    price_data = {"AAA": _prices([100, 100, 105, 110, 121, 121], [100, 100, 110, 121, 121, 121])}
    strategy = ScriptedStrategy({("AAA", DATES[1]): "BUY"})

    result = _run(_settings(), price_data, strategy)

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade.date == DATES[2].date()
    assert trade.side == "BUY"
    assert trade.price == 105  # next session's Open, not the signal day's Close
    assert trade.shares == 100  # 10% of 100,000 at the signal day's Close of 100
    assert trade.costs == pytest.approx(1.0 + 0.0005 * 10_500)

    cash = 100_000 - 10_500 - trade.costs
    assert result.equity.loc[DATES[1]] == pytest.approx(100_000)
    assert result.equity.loc[DATES[2]] == pytest.approx(cash + 100 * 110)
    assert result.equity.loc[DATES[3]] == pytest.approx(cash + 100 * 121)
    assert result.exposure.loc[DATES[1]] == 0
    assert result.exposure.loc[DATES[3]] == pytest.approx(12_100 / (cash + 12_100))


def test_sell_records_realized_pnl_net_of_exit_costs():
    price_data = {"AAA": _prices([100, 100, 105, 110, 121, 121], [100, 100, 110, 121, 121, 121])}
    strategy = ScriptedStrategy({("AAA", DATES[1]): "BUY", ("AAA", DATES[3]): "SELL"})

    result = _run(_settings(), price_data, strategy)

    buy, sell = result.trades
    assert sell.date == DATES[4].date()
    assert sell.price == 121
    assert buy.realized_pnl is None
    assert sell.realized_pnl == pytest.approx((121 - 105) * 100 - (1.0 + 0.0005 * 12_100))
    assert result.exposure.iloc[-1] == 0


def test_strategies_never_see_bars_after_the_session_being_decided():
    price_data = {"AAA": _prices([100] * 6, [100, 101, 102, 103, 104, 105])}
    strategy = ScriptedStrategy({})

    _run(_settings(), price_data, strategy)

    assert [latest for _, latest, _ in strategy.seen] == list(DATES)
    assert [length for _, _, length in strategy.seen] == [1, 2, 3, 4, 5, 6]


def test_hold_only_strategy_keeps_equity_flat():
    price_data = {"AAA": _prices([100] * 6, [100, 90, 110, 80, 120, 100])}

    result = _run(_settings(), price_data, ScriptedStrategy({}))

    assert result.trades == []
    assert (result.equity == 100_000).all()


def test_start_limits_trading_but_earlier_bars_remain_available_as_history():
    price_data = {"AAA": _prices([100] * 6, [100] * 6)}
    strategy = ScriptedStrategy({})

    result = _run(_settings(), price_data, strategy, start=DATES[3].date())

    assert list(result.equity.index) == list(DATES[3:])
    assert strategy.seen[0] == ("AAA", DATES[3], 4)


def test_lookback_bars_caps_history_passed_to_strategies():
    price_data = {"AAA": _prices([100] * 6, [100] * 6)}
    strategy = ScriptedStrategy({})

    _run(_settings(), price_data, strategy, lookback_bars=2)

    assert [length for _, _, length in strategy.seen] == [1, 2, 2, 2, 2, 2]


def test_order_waits_for_next_session_the_ticker_actually_trades():
    aaa = _prices([100] * 6, [100] * 6).drop(DATES[2])  # AAA has no bar on DATES[2]
    bbb = _prices([50] * 6, [50] * 6)
    strategy = ScriptedStrategy({("AAA", DATES[1]): "BUY"})

    result = _run(_settings(tickers=("AAA", "BBB")), {"AAA": aaa, "BBB": bbb}, strategy)

    assert [(t.ticker, t.date) for t in result.trades] == [("AAA", DATES[3].date())]


def test_kill_switch_blocks_buys_after_daily_loss_limit():
    # Fully invested, then a 20% one-day fall trips the 5% daily loss limit
    price_data = {"AAA": _prices([100, 100, 100, 80, 80, 80], [100, 100, 80, 80, 80, 80])}
    strategy = ScriptedStrategy(
        {
            ("AAA", DATES[0]): "BUY",
            ("AAA", DATES[2]): "SELL",
            ("AAA", DATES[4]): "BUY",
        }
    )
    settings = _settings(position_size_pct=1.0, risk=RiskConfig(max_daily_loss_pct=0.05))

    result = _run(settings, price_data, strategy)

    assert result.kill_switch_tripped_on == DATES[2].date()
    assert "daily loss" in result.kill_switch_reason
    # The exit still goes through; the later BUY does not
    assert [t.side for t in result.trades] == ["BUY", "SELL"]

    unrestricted = _run(settings, price_data, ScriptedStrategy(strategy_actions(strategy)),
                        apply_kill_switch=False)
    assert [t.side for t in unrestricted.trades] == ["BUY", "SELL", "BUY"]
    assert unrestricted.kill_switch_tripped_on is None


def strategy_actions(strategy: ScriptedStrategy) -> dict:
    return {key: action for key, action in strategy.actions.items()}


def test_no_sessions_raises():
    price_data = {"AAA": _prices([100] * 6, [100] * 6)}

    with pytest.raises(ValueError, match="No sessions"):
        _run(_settings(), price_data, ScriptedStrategy({}), start=date(2030, 1, 1))


def test_buy_and_hold_equity_is_equal_weight_from_first_close():
    price_data = {
        "AAA": _prices([100] * 3, [100, 110, 120]),
        "BBB": _prices([50] * 3, [50, 50, 25]),
    }

    equity = buy_and_hold_equity(price_data, DATES[:3], 100_000)

    assert equity.tolist() == pytest.approx([100_000, 105_000, 85_000])


def test_buy_and_hold_equity_leaves_out_tickers_without_a_first_price():
    price_data = {
        "AAA": _prices([100] * 3, [100, 110, 120]),
        "LATE": _prices([10] * 2, [10, 20], dates=DATES[1:]),
    }

    equity = buy_and_hold_equity(price_data, DATES[:3], 100_000)

    assert equity.tolist() == pytest.approx([100_000, 110_000, 120_000])
