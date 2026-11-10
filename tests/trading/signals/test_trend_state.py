import numpy as np
import pandas as pd

from stock_analysis.trading.signals import get_signal_strategy
from stock_analysis.trading.signals.trend_state import TrendStateStrategy

PARAMS = {"trend_lookbacks": [20, 40, 60], "trend_ma": 50}


def _frame(closes) -> pd.DataFrame:
    index = pd.date_range("2022-01-03", periods=len(closes), freq="B")
    return pd.DataFrame({"Close": np.asarray(closes, dtype=float)}, index=index)


def _rising(n=120):
    return _frame(100 * 1.002 ** np.arange(n))


def _falling(n=120):
    return _frame(100 * 0.998 ** np.arange(n))


def test_registered_under_trend_state():
    assert isinstance(get_signal_strategy("trend_state"), TrendStateStrategy)


def test_buy_in_an_uptrend():
    result = TrendStateStrategy().evaluate(_rising(), "AAA", PARAMS)

    assert result.action == "BUY"
    assert result.metadata["momentum"] > 0
    assert result.metadata["close"] > result.metadata["moving_average"]


def test_sell_in_a_downtrend():
    result = TrendStateStrategy().evaluate(_falling(), "AAA", PARAMS)

    assert result.action == "SELL"
    assert result.metadata["momentum"] < 0


def test_reports_the_state_every_day_not_only_on_the_day_it_changes():
    data = _rising()
    strategy = TrendStateStrategy()

    actions = [strategy.evaluate(data.iloc[:end], "AAA", PARAMS).action for end in (100, 101, 120)]

    assert actions == ["BUY", "BUY", "BUY"]


def test_holds_when_momentum_is_positive_but_price_is_below_its_average():
    # A long climb, then a sharp recent drop: still up over 20/40/60 sessions on
    # average, but below the 50-day average
    closes = list(100 * 1.01 ** np.arange(110)) + [0] * 10
    peak = closes[109]
    closes[110:] = [peak * 0.80] * 10
    result = TrendStateStrategy().evaluate(_frame(closes), "AAA", PARAMS)

    assert result.metadata["momentum"] > 0
    assert result.metadata["close"] < result.metadata["moving_average"]
    assert result.action == "HOLD"
    assert "no new entry" in result.reason


def test_momentum_is_the_average_return_over_the_lookbacks():
    data = _rising()
    close = data["Close"]
    expected = np.mean([close.iloc[-1] / close.iloc[-1 - n] - 1 for n in (20, 40, 60)])

    result = TrendStateStrategy().evaluate(data, "AAA", PARAMS)

    assert result.metadata["momentum"] == np.float64(expected)


def test_holds_without_enough_history():
    result = TrendStateStrategy().evaluate(_rising(60), "AAA", PARAMS)

    assert result.action == "HOLD"
    assert "insufficient data (60 bars, need 61)" in result.reason


def test_moving_average_check_can_be_switched_off():
    closes = list(100 * 1.01 ** np.arange(110))
    closes += [closes[-1] * 0.80] * 10
    result = TrendStateStrategy().evaluate(
        _frame(closes), "AAA", {"trend_lookbacks": [20, 40, 60], "trend_ma": 0}
    )

    assert result.action == "BUY"


def test_defaults_need_about_a_year_of_history():
    result = TrendStateStrategy().evaluate(_rising(252), "AAA", {})
    assert "need 253" in result.reason
    assert TrendStateStrategy().evaluate(_rising(253), "AAA", {}).action == "BUY"
