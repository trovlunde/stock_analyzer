import pandas as pd
import pytest
from unittest.mock import MagicMock

import stock_analysis.ai.fundamental_analysis.fin_statement_classifier as fin_mod
from stock_analysis.ai.fundamental_analysis.fin_statement_classifier import (
    prepare_classification_data_cache,
)

_FEATURE_COLS = [
    "gross_margin", "profit_margin", "current_ratio", "debt_to_equity",
    "revenue_growth", "eps_estimate", "eps_actual", "eps_difference", "surprise_percent",
]


def _make_features():
    idx = pd.date_range("2024-01-01", periods=5, freq="QE")
    return pd.DataFrame({col: [0.1] * 5 for col in _FEATURE_COLS}, index=idx)


def _make_targets():
    idx = pd.date_range("2024-01-01", periods=5, freq="QE")
    return pd.DataFrame(
        {"daily_target": ["positive"] * 5, "weekly_target": ["positive"] * 5},
        index=idx,
    )


def _make_returns():
    idx = pd.date_range("2024-01-01", periods=5, freq="QE")
    return pd.DataFrame(
        {"daily_return": [0.01] * 5, "weekly_return": [0.02] * 5},
        index=idx,
    )


@pytest.fixture
def mock_ticker():
    t = MagicMock()
    t.ticker = "TESTCACHE"
    return t


def test_cache_miss_returns_two_dataframes(tmp_path, mock_ticker, monkeypatch):
    features, targets, returns = _make_features(), _make_targets(), _make_returns()
    monkeypatch.setattr(fin_mod, "prepare_classification_data", lambda t: (features, targets, returns))

    result = prepare_classification_data_cache(mock_ticker, _cache_dir=str(tmp_path))

    assert isinstance(result, tuple), "cache miss must return tuple"
    assert len(result) == 2, "cache miss must return exactly 2 values"
    X, y = result
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    assert len(X) == 5
    assert list(X.columns) == _FEATURE_COLS

    assert (tmp_path / "TESTCACHE_data.csv").exists(), "features CSV not written"
    assert (tmp_path / "TESTCACHE_targets.csv").exists(), "targets CSV not written"


def test_cache_hit_returns_two_dataframes_without_recompute(tmp_path, mock_ticker, monkeypatch):
    features, targets, returns = _make_features(), _make_targets(), _make_returns()
    call_count = {"n": 0}

    def fake_prepare(t):
        call_count["n"] += 1
        return features, targets, returns

    monkeypatch.setattr(fin_mod, "prepare_classification_data", fake_prepare)

    # First call — cache miss, writes files
    prepare_classification_data_cache(mock_ticker, _cache_dir=str(tmp_path))
    assert call_count["n"] == 1

    # Second call — cache hit, must not recompute
    result = prepare_classification_data_cache(mock_ticker, _cache_dir=str(tmp_path))
    assert call_count["n"] == 1, "cache hit must not call prepare_classification_data"

    assert isinstance(result, tuple)
    assert len(result) == 2
    X, y = result
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.DataFrame)
    assert list(X.columns) == _FEATURE_COLS
    assert len(X) == len(features)
    assert list(y.columns) == list(targets.columns)
    assert len(y) == len(targets)
