import numpy as np
import pandas as pd

from stock_analysis.ai.helpers import get_features
from stock_analysis.ai.technical_analysis.prepare_classification_data import (
    prepare_classification_data,
    prepare_classification_data_enhanced,
)


def _synthetic_ohlcv(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


def test_prepare_classification_data_extra_features_are_finite():
    stock_data = _synthetic_ohlcv()
    data = prepare_classification_data(stock_data, use_extra_features=True)
    feature_columns = get_features(use_extra_features=True)

    valid = data[feature_columns].dropna()
    assert len(valid) > 0
    assert np.isfinite(valid.to_numpy()).all()


def test_prepare_classification_data_enhanced_macd_histogram_is_finite():
    stock_data = _synthetic_ohlcv()
    data = prepare_classification_data_enhanced(stock_data, use_extra_features=True)

    macd = data["macd_diff"].dropna()
    assert len(macd) > 0
    assert np.isfinite(macd.to_numpy()).all()
