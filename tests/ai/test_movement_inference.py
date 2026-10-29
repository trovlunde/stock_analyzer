from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from stock_analysis.ai.technical_analysis.movement_inference import predict_latest


def _synthetic_ohlcv(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


def _mock_model_and_scaler(predicted_label="positive", probabilities=(0.1, 0.2, 0.7)):
    scaler = MagicMock()
    scaler.transform.side_effect = lambda df: df.to_numpy()

    model = MagicMock()
    model.classes_ = ["negative", "neutral", "positive"]
    model.predict.return_value = [predicted_label]
    model.predict_proba.return_value = [list(probabilities)]
    return model, scaler


def test_predict_latest_returns_label_and_matching_confidence():
    stock_data = _synthetic_ohlcv()
    model, scaler = _mock_model_and_scaler("positive", (0.1, 0.2, 0.7))

    label, confidence = predict_latest(stock_data, model, scaler)

    assert label == "positive"
    assert confidence == pytest.approx(0.7)


def test_predict_latest_confidence_matches_predicted_class_not_just_max():
    stock_data = _synthetic_ohlcv()
    # Predicted label is "negative" even though "positive" has higher probability —
    # confidence must reflect the predicted class, not argmax.
    model, scaler = _mock_model_and_scaler("negative", (0.3, 0.2, 0.5))

    label, confidence = predict_latest(stock_data, model, scaler)

    assert label == "negative"
    assert confidence == pytest.approx(0.3)


def test_predict_latest_calls_scaler_with_single_row():
    stock_data = _synthetic_ohlcv()
    model, scaler = _mock_model_and_scaler()

    predict_latest(stock_data, model, scaler)

    called_with = scaler.transform.call_args[0][0]
    assert len(called_with) == 1


def test_predict_latest_uses_extra_features_when_requested():
    stock_data = _synthetic_ohlcv()
    model, scaler = _mock_model_and_scaler()

    predict_latest(stock_data, model, scaler, use_extra_features=True)

    called_with = scaler.transform.call_args[0][0]
    assert called_with.shape[1] == 11  # 3 base + 8 extra features


def test_predict_latest_raises_on_insufficient_history():
    # Too few bars for the 21-day rolling extra features to be defined at the end.
    stock_data = _synthetic_ohlcv(n=10)
    model, scaler = _mock_model_and_scaler()

    with pytest.raises(ValueError, match="Insufficient history"):
        predict_latest(stock_data, model, scaler, use_extra_features=True)


def test_predict_latest_no_plotting_or_print_side_effects(monkeypatch, capsys):
    stock_data = _synthetic_ohlcv()
    model, scaler = _mock_model_and_scaler()

    import plotly.graph_objects as go

    show_mock = MagicMock()
    monkeypatch.setattr(go.Figure, "show", show_mock)

    predict_latest(stock_data, model, scaler)

    show_mock.assert_not_called()
