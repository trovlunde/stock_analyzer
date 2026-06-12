import numpy as np
import pandas as pd

from stock_analysis.ai.technical_analysis.nn_predictor import (
    prepare_data,
    predict_returns,
    train_nn_predictor,
)


def _synthetic_ohlcv(n: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


def test_train_nn_predictor_returns_model_scaler_history():
    data = _synthetic_ohlcv()
    model, scaler, history = train_nn_predictor(data, do_tuning=False)

    assert model is not None
    assert scaler is not None
    assert isinstance(history, dict)
    assert "loss" in history
    assert len(history["loss"]) > 0


def test_predict_returns_length_matches_prepared_features():
    data = _synthetic_ohlcv()
    model, scaler, _ = train_nn_predictor(data, do_tuning=False)

    predictions = predict_returns(model, scaler, data)

    prepared = prepare_data(data)
    expected_len = len(prepared.drop("target", axis=1))
    assert isinstance(predictions, pd.Series)
    assert len(predictions) == expected_len
