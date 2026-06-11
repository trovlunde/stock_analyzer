import numpy as np
import pandas as pd

from stock_analysis.trading.signals.ma_crossover import MaCrossoverStrategy


def _price_series(values: list[float]) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=len(values), freq="B")
    return pd.DataFrame({"Close": values, "Open": values}, index=index)


def test_insufficient_data_returns_hold():
    strategy = MaCrossoverStrategy()
    df = _price_series([100.0] * 10)
    result = strategy.evaluate(df, {"ma_short": 20, "ma_long": 50})
    assert result.action == "HOLD"
    assert "insufficient data" in result.reason


def test_evaluate_returns_valid_action():
    rng = np.random.default_rng(0)
    prices = 100 + np.cumsum(rng.normal(0, 1, 120))
    strategy = MaCrossoverStrategy()
    result = strategy.evaluate(
        _price_series(prices.tolist()), {"ma_short": 5, "ma_long": 20}
    )
    assert result.action in {"BUY", "SELL", "HOLD"}
