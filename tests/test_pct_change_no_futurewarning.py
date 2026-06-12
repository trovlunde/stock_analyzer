import warnings
import numpy as np
import pandas as pd

from stock_analysis.ai.technical_analysis.prepare_classification_data import (
    prepare_classification_data,
)
from stock_analysis.ai.metrics import calculate_sharpe_ratio


def _synthetic_ohlcv(n: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


def _pct_change_warnings(caught):
    return [
        w for w in caught
        if issubclass(w.category, FutureWarning)
        and "pct_change" in str(w.message)
        and "fill_method" in str(w.message)
    ]


def test_prepare_classification_data_no_pct_change_futurewarning():
    stock_data = _synthetic_ohlcv()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        prepare_classification_data(stock_data, use_extra_features=True)
    bad = _pct_change_warnings(caught)
    assert not bad, f"pct_change FutureWarning(s) raised: {[str(w.message) for w in bad]}"


def test_calculate_sharpe_ratio_no_pct_change_futurewarning():
    values = list(range(1, 121))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        calculate_sharpe_ratio(values)
    bad = _pct_change_warnings(caught)
    assert not bad, f"pct_change FutureWarning(s) raised: {[str(w.message) for w in bad]}"
