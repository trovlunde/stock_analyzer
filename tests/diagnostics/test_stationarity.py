import numpy as np
import pandas as pd
import pytest

from stock_analysis.diagnostics.stationarity import (
    check_stationarity,
    check_stationarity_frame,
)


def _white_noise(n: int = 200, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n))


def _random_walk(n: int = 200, seed: int = 1) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(n).cumsum())


class TestCheckStationarity:
    def test_white_noise_is_stationary(self):
        result = check_stationarity(_white_noise())
        assert result["is_stationary"] is True
        assert result["insufficient_data"] is False
        assert isinstance(result["statistic"], float)
        assert isinstance(result["p_value"], float)

    def test_random_walk_is_not_stationary(self):
        result = check_stationarity(_random_walk())
        assert result["is_stationary"] is False
        assert result["insufficient_data"] is False

    def test_empty_series_returns_insufficient(self):
        result = check_stationarity(pd.Series([], dtype=float))
        assert result["insufficient_data"] is True
        assert result["is_stationary"] is None

    def test_too_short_series_returns_insufficient(self):
        result = check_stationarity(pd.Series([1.0, 2.0, 3.0]))
        assert result["insufficient_data"] is True

    def test_series_with_nans_handled(self):
        s = _white_noise(200)
        s.iloc[::10] = np.nan
        result = check_stationarity(s)
        assert result["insufficient_data"] is False

    def test_custom_alpha(self):
        result = check_stationarity(_white_noise(), alpha=0.0)
        assert result["is_stationary"] is False

    def test_returns_dict_keys(self):
        result = check_stationarity(_white_noise())
        assert set(result.keys()) == {"statistic", "p_value", "is_stationary", "insufficient_data"}


class TestCheckStationarityFrame:
    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame({"wn": _white_noise(), "rw": _random_walk()})

    def test_returns_dataframe(self):
        report = check_stationarity_frame(self._frame())
        assert isinstance(report, pd.DataFrame)
        assert set(report.columns) == {"statistic", "p_value", "is_stationary", "insufficient_data"}
        assert list(report.index) == ["wn", "rw"]

    def test_column_subset(self):
        report = check_stationarity_frame(self._frame(), columns=["wn"])
        assert list(report.index) == ["wn"]

    def test_skips_non_numeric(self):
        df = self._frame()
        df["label"] = "x"
        report = check_stationarity_frame(df)
        assert "label" not in report.index

    def test_white_noise_col_stationary(self):
        report = check_stationarity_frame(self._frame())
        assert report.loc["wn", "is_stationary"] == True

    def test_random_walk_col_not_stationary(self):
        report = check_stationarity_frame(self._frame())
        assert report.loc["rw", "is_stationary"] == False
