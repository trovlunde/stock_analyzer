import numpy as np
import pandas as pd
import pytest

from stock_analysis.ai.helpers import RSI
from stock_analysis.ai.technical_analysis.rsi_signals import generate_rsi_signals
from stock_analysis.trading.signals import RsiStrategy, get_signal_strategy

NO_FILTER = {"rsi_trend_ma": 0}


def _frame(closes) -> pd.DataFrame:
    index = pd.date_range("2022-01-03", periods=len(closes), freq="B")
    return pd.DataFrame({"Close": np.asarray(closes, dtype=float)}, index=index)


def _path(*legs, start=100.0):
    """Closes from (sessions, daily change) legs, e.g. (60, +0.5) then (15, -2.0)."""
    steps = np.concatenate([np.full(sessions, change) for sessions, change in legs])
    return start + np.concatenate([[0.0], np.cumsum(steps)])


def _actions(closes, params, first=20):
    """The strategy's action on each session, given only the bars up to it."""
    strategy = RsiStrategy()
    return [
        strategy.evaluate(_frame(closes[:end]), "AAA", params).action
        for end in range(first, len(closes) + 1)
    ]


# Uptrend, a sharp dip that drives RSI deep below 30, then a recovery
DIP_AND_RECOVERY = _path((60, 0.5), (15, -2.0), (30, 1.0))


def test_registered_under_rsi():
    assert isinstance(get_signal_strategy("rsi"), RsiStrategy)


def test_one_rsi_definition_wilder_smoothing():
    rng = np.random.default_rng(7)
    data = _frame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    period = 14

    # Wilder's RSI written out independently: smoothing factor 1 / period
    delta = data["Close"].diff().dropna()
    average_gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    average_loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    expected = 100 - 100 / (1 + average_gain / average_loss)

    from_helpers = RSI(data, window=period)
    from_signals = generate_rsi_signals(data, period=period)["RSI"]
    from_strategy = RsiStrategy().evaluate(data, "AAA", NO_FILTER).metadata["rsi"]

    pd.testing.assert_series_equal(from_helpers, expected, check_names=False)
    pd.testing.assert_series_equal(
        from_signals.dropna(), from_helpers, check_names=False, check_freq=False
    )
    assert from_strategy == pytest.approx(expected.iloc[-1])

    # The old rolling-mean variant gives different numbers; it must not come back
    rolling_gain = delta.clip(lower=0).rolling(period).mean()
    rolling_loss = (-delta.clip(upper=0)).rolling(period).mean()
    rolling_rsi = 100 - 100 / (1 + rolling_gain / rolling_loss)
    assert (from_signals - rolling_rsi).abs().max() > 1


def test_research_signals_keep_their_level_based_columns():
    signals = generate_rsi_signals(_frame(DIP_AND_RECOVERY))

    assert list(signals.columns) == ["Close", "RSI", "Signal"]
    assert set(signals["Signal"].unique()) == {-1, 0, 1}
    assert (signals.loc[signals["Signal"] == 1, "RSI"] < 30).all()
    assert (signals.loc[signals["Signal"] == -1, "RSI"] > 70).all()
    (single,) = generate_rsi_signals([_frame(DIP_AND_RECOVERY)])
    pd.testing.assert_frame_equal(single, signals)


def test_buys_once_on_the_cross_back_up_not_while_oversold():
    closes = DIP_AND_RECOVERY
    actions = _actions(closes, NO_FILTER)
    rsi = RSI(_frame(closes)).to_numpy()[19 - 1:]  # RSI has no value for the first bar

    oversold_sessions = int((rsi < 30).sum())
    assert oversold_sessions > 5  # the level-based rule would have fired on each of these
    assert actions.count("BUY") == 1
    buy_at = actions.index("BUY")
    assert rsi[buy_at - 1] <= 30 < rsi[buy_at]
    # No BUY on any session where RSI is still below the threshold
    assert all(action != "BUY" for action, value in zip(actions, rsi) if value < 30)


def test_buy_metadata_reports_the_cross():
    closes = DIP_AND_RECOVERY
    end = 20 + _actions(closes, NO_FILTER).index("BUY")

    result = RsiStrategy().evaluate(_frame(closes[:end]), "AAA", NO_FILTER)

    assert result.action == "BUY"
    assert "crossed back above 30" in result.reason
    assert result.metadata["previous_rsi"] <= 30 < result.metadata["rsi"]
    assert result.metadata["close"] == closes[end - 1]
    assert result.metadata["moving_average"] is None
    assert result.metadata["rsi_period"] == 14


def test_sells_once_on_the_cross_into_overbought():
    # Flat-to-down drift, then a rally that pushes RSI above 70 and keeps it there
    closes = _path((20, 0.5), (20, -0.5), (30, 1.0))
    actions = _actions(closes, NO_FILTER, first=30)
    rsi = RSI(_frame(closes)).to_numpy()[29 - 1:]

    assert int((rsi > 70).sum()) > 5
    assert actions.count("SELL") == 1
    sell_at = actions.index("SELL")
    assert rsi[sell_at - 1] <= 70 < rsi[sell_at]


def test_trend_filter_blocks_a_buy_below_the_average():
    closes = DIP_AND_RECOVERY
    end = 20 + _actions(closes, NO_FILTER).index("BUY")
    window = _frame(closes[:end])

    # After a 30-point fall the close is well below its 50-day average
    blocked = RsiStrategy().evaluate(window, "AAA", {"rsi_trend_ma": 50})

    assert blocked.action == "HOLD"
    assert "below its 50-day average" in blocked.reason
    assert blocked.metadata["close"] < blocked.metadata["moving_average"]
    assert "BUY" not in _actions(closes, {"rsi_trend_ma": 50}, first=50)


def test_trend_filter_allows_a_buy_above_the_average():
    # A long climb, then a short sharp dip that leaves the close above the average
    closes = _path((150, 1.0), (6, -6.0), (10, 2.0))
    params = {"rsi_trend_ma": 100}
    actions = _actions(closes, params, first=140)

    assert actions.count("BUY") == 1
    end = 140 + actions.index("BUY")
    result = RsiStrategy().evaluate(_frame(closes[:end]), "AAA", params)
    assert result.metadata["close"] > result.metadata["moving_average"]


def test_thresholds_and_period_come_from_params():
    closes = DIP_AND_RECOVERY
    params = {**NO_FILTER, "rsi_period": 5, "rsi_oversold": 20, "rsi_overbought": 80}
    actions = _actions(closes, params)
    end = 20 + actions.index("BUY")

    result = RsiStrategy().evaluate(_frame(closes[:end]), "AAA", params)

    assert result.metadata["previous_rsi"] <= 20 < result.metadata["rsi"]
    assert result.metadata["rsi"] == pytest.approx(RSI(_frame(closes[:end]), window=5).iloc[-1])


def test_insufficient_data_holds():
    short = RsiStrategy().evaluate(_frame(DIP_AND_RECOVERY[:15]), "AAA", NO_FILTER)
    default_filter = RsiStrategy().evaluate(_frame(DIP_AND_RECOVERY), "AAA", {})

    assert short.action == "HOLD"
    assert short.reason == "insufficient data (15 bars, need 16)"
    # The default 200-day trend filter needs 200 bars
    assert default_filter.action == "HOLD"
    assert default_filter.reason == f"insufficient data ({len(DIP_AND_RECOVERY)} bars, need 200)"
    assert default_filter.metadata["rsi_trend_ma"] == 200


def test_flat_prices_hold():
    result = RsiStrategy().evaluate(_frame(np.full(40, 100.0)), "AAA", NO_FILTER)

    assert result.action == "HOLD"
    assert result.reason == "RSI undefined"
