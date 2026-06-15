import numpy as np
import pandas as pd
import pytest

from stock_analysis.diagnostics.regression import regression_diagnostics


def _linear_data(n: int = 100, slope: float = 3.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    y = slope * x + rng.standard_normal(n) * 0.05
    X = pd.DataFrame({"x": x})
    return X, pd.Series(y, name="target")


class TestRegressionDiagnosticsReturnShape:
    def test_returns_dict_with_required_keys(self):
        X, y = _linear_data()
        result = regression_diagnostics(X, y)
        assert set(result.keys()) == {
            "r_squared",
            "coefficients",
            "durbin_watson",
            "condition_number",
            "n_obs",
            "insufficient_data",
        }

    def test_insufficient_data_false_on_valid_input(self):
        X, y = _linear_data()
        result = regression_diagnostics(X, y)
        assert result["insufficient_data"] is False

    def test_n_obs_matches_input(self):
        X, y = _linear_data(n=50)
        result = regression_diagnostics(X, y)
        assert result["n_obs"] == 50


class TestRegressionDiagnosticsValues:
    def test_near_perfect_r_squared(self):
        X, y = _linear_data(n=200)
        result = regression_diagnostics(X, y)
        assert result["r_squared"] > 0.99

    def test_coefficient_sign_correct(self):
        X, y = _linear_data(slope=3.0)
        result = regression_diagnostics(X, y)
        assert result["coefficients"]["x"] > 0

    def test_negative_slope_sign_correct(self):
        X, y = _linear_data(slope=-2.5)
        result = regression_diagnostics(X, y)
        assert result["coefficients"]["x"] < 0

    def test_durbin_watson_in_range(self):
        X, y = _linear_data()
        result = regression_diagnostics(X, y)
        assert 0.0 <= result["durbin_watson"] <= 4.0

    def test_condition_number_positive(self):
        X, y = _linear_data()
        result = regression_diagnostics(X, y)
        assert result["condition_number"] > 0


class TestRegressionDiagnosticsEdgeCases:
    def test_non_numeric_columns_dropped(self):
        X, y = _linear_data()
        X["label"] = "foo"
        result = regression_diagnostics(X, y)
        assert result["insufficient_data"] is False
        assert "label" not in result["coefficients"]
        assert "x" in result["coefficients"]

    def test_all_non_numeric_returns_insufficient(self):
        X = pd.DataFrame({"a": ["foo", "bar", "baz"]})
        y = pd.Series([1.0, 2.0, 3.0])
        result = regression_diagnostics(X, y)
        assert result["insufficient_data"] is True
        assert result["r_squared"] is None

    def test_empty_X_returns_insufficient(self):
        X = pd.DataFrame()
        y = pd.Series([1.0, 2.0, 3.0])
        result = regression_diagnostics(X, y)
        assert result["insufficient_data"] is True

    def test_fewer_than_3_obs_returns_insufficient(self):
        X = pd.DataFrame({"x": [1.0, 2.0]})
        y = pd.Series([1.0, 2.0])
        result = regression_diagnostics(X, y)
        assert result["insufficient_data"] is True
        assert result["n_obs"] == 2

    def test_index_alignment(self):
        X = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=[0, 1, 2, 3, 4])
        y = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
        result = regression_diagnostics(X, y)
        assert result["n_obs"] == 3
        assert result["insufficient_data"] is False

    def test_nan_rows_excluded(self):
        X, y = _linear_data(n=50)
        X.iloc[0, 0] = np.nan
        result = regression_diagnostics(X, y)
        assert result["n_obs"] == 49
        assert result["insufficient_data"] is False

    def test_multi_feature(self):
        rng = np.random.default_rng(7)
        n = 100
        x1 = rng.standard_normal(n)
        x2 = rng.standard_normal(n)
        y = 2.0 * x1 - 1.5 * x2 + rng.standard_normal(n) * 0.01
        X = pd.DataFrame({"x1": x1, "x2": x2})
        result = regression_diagnostics(X, pd.Series(y))
        assert result["r_squared"] > 0.99
        assert result["coefficients"]["x1"] > 0
        assert result["coefficients"]["x2"] < 0
