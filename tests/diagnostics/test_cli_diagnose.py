import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from stock_analysis.cli.evaluate_models import main


def _synthetic_ohlcv(n: int = 120, seed: int = 42) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


class TestDiagnoseCommand:
    def test_diagnose_prints_stationarity_header(self, capsys):
        ohlcv = _synthetic_ohlcv()
        with patch("stock_analysis.cli.evaluate_models.get_index_data", return_value=ohlcv):
            main(["diagnose", "--ticker", "TEST", "--period", "1y"])
        out = capsys.readouterr().out
        assert "Stationarity Report" in out

    def test_diagnose_prints_regression_header(self, capsys):
        ohlcv = _synthetic_ohlcv()
        with patch("stock_analysis.cli.evaluate_models.get_index_data", return_value=ohlcv):
            main(["diagnose", "--ticker", "TEST", "--period", "1y"])
        out = capsys.readouterr().out
        assert "Regression Diagnostics" in out

    def test_diagnose_includes_base_feature_columns(self, capsys):
        ohlcv = _synthetic_ohlcv()
        with patch("stock_analysis.cli.evaluate_models.get_index_data", return_value=ohlcv):
            main(["diagnose", "--ticker", "TEST"])
        out = capsys.readouterr().out
        assert "prev_day_return" in out

    def test_diagnose_extra_features(self, capsys):
        ohlcv = _synthetic_ohlcv(n=200)
        with patch("stock_analysis.cli.evaluate_models.get_index_data", return_value=ohlcv):
            main(["diagnose", "--ticker", "TEST", "--extra-features"])
        out = capsys.readouterr().out
        assert "Stationarity Report" in out
        assert "Regression Diagnostics" in out

    def test_diagnose_default_ticker_is_gspc(self, capsys):
        ohlcv = _synthetic_ohlcv()
        with patch("stock_analysis.cli.evaluate_models.get_index_data", return_value=ohlcv) as mock_fn:
            main(["diagnose"])
        mock_fn.assert_called_once_with("^GSPC", "20y")
