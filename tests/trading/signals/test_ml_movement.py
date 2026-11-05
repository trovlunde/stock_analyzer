from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from stock_analysis.ai.helpers import get_features
from stock_analysis.trading.signals.ml_movement import MlMovementStrategy


def _model_manager_with_models(metadata=None) -> MagicMock:
    model_manager = MagicMock()
    model_manager.load_model.side_effect = [MagicMock(), MagicMock()]
    model_manager.get_model_metadata.return_value = (
        {"feature_columns": list(get_features(False))} if metadata is None else metadata
    )
    return model_manager


def _synthetic_ohlcv(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


def test_hold_when_no_trained_model():
    model_manager = MagicMock()
    model_manager.load_model.return_value = None
    strategy = MlMovementStrategy(model_manager=model_manager)

    result = strategy.evaluate(_synthetic_ohlcv(), "AAPL", {})

    assert result.action == "HOLD"
    assert "no trained ML model" in result.reason


def test_hold_when_scaler_missing_but_model_present():
    model_manager = MagicMock()
    model_manager.load_model.side_effect = [MagicMock(), None]  # model ok, scaler missing
    strategy = MlMovementStrategy(model_manager=model_manager)

    result = strategy.evaluate(_synthetic_ohlcv(), "AAPL", {})

    assert result.action == "HOLD"
    assert "scaler missing" in result.reason


def test_hold_when_model_missing_but_scaler_present():
    model_manager = MagicMock()
    model_manager.load_model.side_effect = [None, MagicMock()]  # model missing, scaler ok
    strategy = MlMovementStrategy(model_manager=model_manager)

    result = strategy.evaluate(_synthetic_ohlcv(), "AAPL", {})

    assert result.action == "HOLD"
    assert "model missing" in result.reason


def test_hold_when_model_predates_feature_tracking():
    """Legacy models saved without feature_columns can't be proven to match today's
    feature pipeline — a scaler fitted on a bare array with the same column count
    would otherwise mis-predict silently."""
    strategy = MlMovementStrategy(model_manager=_model_manager_with_models(metadata={}))

    with patch("stock_analysis.trading.signals.ml_movement.predict_latest") as predict:
        result = strategy.evaluate(_synthetic_ohlcv(), "AAPL", {})

    assert result.action == "HOLD"
    assert "retrain" in result.reason
    predict.assert_not_called()


def test_hold_when_model_trained_on_different_features():
    strategy = MlMovementStrategy(
        model_manager=_model_manager_with_models(metadata={"feature_columns": ["old_feature"]})
    )

    with patch("stock_analysis.trading.signals.ml_movement.predict_latest") as predict:
        result = strategy.evaluate(_synthetic_ohlcv(), "AAPL", {})

    assert result.action == "HOLD"
    assert "different feature set" in result.reason
    predict.assert_not_called()


def _strategy_with_prediction(label: str, confidence: float) -> MlMovementStrategy:
    strategy = MlMovementStrategy(model_manager=_model_manager_with_models())
    patcher = patch(
        "stock_analysis.trading.signals.ml_movement.predict_latest",
        return_value=(label, confidence),
    )
    patcher.start()
    strategy._patcher = patcher  # keep alive for the caller to stop
    return strategy


def test_buy_on_positive_label_above_threshold():
    strategy = _strategy_with_prediction("positive", 0.8)
    try:
        result = strategy.evaluate(_synthetic_ohlcv(), "AAPL", {})
        assert result.action == "BUY"
        assert result.metadata["confidence"] == 0.8
    finally:
        strategy._patcher.stop()


def test_sell_on_negative_label_above_threshold():
    strategy = _strategy_with_prediction("negative", 0.9)
    try:
        result = strategy.evaluate(_synthetic_ohlcv(), "AAPL", {})
        assert result.action == "SELL"
    finally:
        strategy._patcher.stop()


def test_hold_on_neutral_label():
    strategy = _strategy_with_prediction("neutral", 0.9)
    try:
        result = strategy.evaluate(_synthetic_ohlcv(), "AAPL", {})
        assert result.action == "HOLD"
    finally:
        strategy._patcher.stop()


def test_hold_when_confidence_below_threshold():
    strategy = _strategy_with_prediction("positive", 0.4)
    try:
        result = strategy.evaluate(
            _synthetic_ohlcv(), "AAPL", {"ml_confidence_threshold": 0.55}
        )
        assert result.action == "HOLD"
        assert "below threshold" in result.reason
    finally:
        strategy._patcher.stop()


def test_hold_when_prediction_raises_value_error():
    strategy = MlMovementStrategy(model_manager=_model_manager_with_models())

    with patch(
        "stock_analysis.trading.signals.ml_movement.predict_latest",
        side_effect=ValueError("Insufficient history"),
    ):
        result = strategy.evaluate(_synthetic_ohlcv(), "AAPL", {})

    assert result.action == "HOLD"
    assert "ML prediction unavailable" in result.reason


def test_model_lookup_uses_ticker_movement_naming_convention():
    model_manager = MagicMock()
    model_manager.load_model.return_value = None
    strategy = MlMovementStrategy(model_manager=model_manager)

    strategy.evaluate(_synthetic_ohlcv(), "MSFT", {})

    model_manager.load_model.assert_any_call("MSFT_movement", "daily")
    model_manager.load_model.assert_any_call("MSFT_movement_scaler", "daily")
