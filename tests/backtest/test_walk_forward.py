import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import stock_analysis.backtest.walk_forward as walk_forward_mod
from stock_analysis.backtest.walk_forward import WalkForwardMlStrategy


def _synthetic_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({"Open": close, "Close": close, "Volume": volume}, index=dates)


def _strategy(**kwargs):
    return WalkForwardMlStrategy(
        classifier_factory=lambda: RandomForestClassifier(n_estimators=5, random_state=0),
        **kwargs,
    )


def _record_training(monkeypatch):
    """Record the labelled rows each training call receives."""
    calls = []
    real_train = walk_forward_mod.train_classifier_single_stock

    def recording_train(prepared, **kwargs):
        calls.append(prepared[prepared["Target"].notna()].index)
        return real_train(prepared, **kwargs)

    monkeypatch.setattr(walk_forward_mod, "train_classifier_single_stock", recording_train)
    return calls


def test_trains_only_on_labels_known_at_the_session_being_decided(monkeypatch):
    data = _synthetic_ohlcv()
    calls = _record_training(monkeypatch)
    strategy = _strategy(retrain_every=10, min_train_rows=100)

    window = data.iloc[:300]
    result = strategy.evaluate(window, "AAA", {})

    assert result.action in {"BUY", "SELL", "HOLD"}
    assert len(calls) == 1
    # The newest label needs the next bar's close, so it is one bar behind the data
    assert calls[0].max() == window.index[-2]
    assert result.metadata["trained_through"] == str(window.index[-1].date())


def test_retrains_on_schedule_not_every_session(monkeypatch):
    data = _synthetic_ohlcv()
    calls = _record_training(monkeypatch)
    strategy = _strategy(retrain_every=10, min_train_rows=100)

    for end in range(300, 321):
        strategy.evaluate(data.iloc[:end], "AAA", {})

    # Sessions 300, 310 and 320
    assert len(calls) == 3
    assert [index.max() for index in calls] == [data.index[298], data.index[308], data.index[318]]


def test_models_are_kept_per_ticker(monkeypatch):
    data = _synthetic_ohlcv()
    calls = _record_training(monkeypatch)
    strategy = _strategy(retrain_every=10, min_train_rows=100)

    strategy.evaluate(data.iloc[:300], "AAA", {})
    strategy.evaluate(data.iloc[:300], "BBB", {})
    strategy.evaluate(data.iloc[:301], "AAA", {})

    assert len(calls) == 2


def test_holds_until_there_is_enough_history(monkeypatch):
    data = _synthetic_ohlcv()
    calls = _record_training(monkeypatch)
    strategy = _strategy(min_train_rows=252)

    result = strategy.evaluate(data.iloc[:100], "AAA", {})

    assert result.action == "HOLD"
    assert "not enough history" in result.reason
    assert calls == []


def test_low_confidence_predictions_hold():
    data = _synthetic_ohlcv()
    strategy = _strategy(min_train_rows=100)

    result = strategy.evaluate(data.iloc[:300], "AAA", {"ml_confidence_threshold": 1.01})

    assert result.action == "HOLD"
    assert "below threshold" in result.reason
