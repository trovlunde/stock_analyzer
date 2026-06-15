"""P2 regression tests: synthetic stationarity and regression diagnostics, CLI smoke test."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from stock_analysis.diagnostics.stationarity import check_stationarity
from stock_analysis.diagnostics.regression import regression_diagnostics
from stock_analysis.cli.evaluate_models import main


def _white_noise(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n))


def _random_walk(n: int = 300, seed: int = 42) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n).cumsum())


def _synthetic_ohlcv(n: int = 120, seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


class TestSyntheticStationarityAtAlpha005:
    def test_white_noise_flagged_stationary(self):
        result = check_stationarity(_white_noise(), alpha=0.05)
        assert result["is_stationary"] is True
        assert result["insufficient_data"] is False

    def test_random_walk_flagged_non_stationary(self):
        result = check_stationarity(_random_walk(), alpha=0.05)
        assert result["is_stationary"] is False
        assert result["insufficient_data"] is False


class TestSyntheticRegressionDiagnostics:
    def test_near_perfect_r_squared(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(200)
        y = pd.Series(2.5 * x + rng.standard_normal(200) * 0.01)
        X = pd.DataFrame({"x": x})
        result = regression_diagnostics(X, y)
        assert result["r_squared"] > 0.99

    def test_positive_slope_sign(self):
        rng = np.random.default_rng(1)
        x = rng.standard_normal(100)
        y = pd.Series(3.0 * x + rng.standard_normal(100) * 0.05)
        X = pd.DataFrame({"x": x})
        result = regression_diagnostics(X, y)
        assert result["coefficients"]["x"] > 0

    def test_negative_slope_sign(self):
        rng = np.random.default_rng(2)
        x = rng.standard_normal(100)
        y = pd.Series(-2.0 * x + rng.standard_normal(100) * 0.05)
        X = pd.DataFrame({"x": x})
        result = regression_diagnostics(X, y)
        assert result["coefficients"]["x"] < 0


class TestCliDiagnoseSmoke:
    def test_exit_code_0_with_mocked_index_data(self):
        ohlcv = _synthetic_ohlcv()
        with patch("stock_analysis.cli.evaluate_models.get_index_data", return_value=ohlcv):
            # main() returns None on success; raises SystemExit(non-zero) on error
            try:
                result = main(["diagnose", "--ticker", "TEST", "--period", "1y"])
                assert result is None
            except SystemExit as e:
                pytest.fail(f"CLI diagnose exited with code {e.code}")
