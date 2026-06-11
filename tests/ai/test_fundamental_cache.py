import pandas as pd
import pytest
from unittest.mock import MagicMock

import stock_analysis.ai.fundamental_analysis.fin_statement_classifier as fin_mod
from stock_analysis.ai.fundamental_analysis.fin_statement_classifier import (
    prepare_classification_data_cache,
    train_classifier,
)

_FEATURE_COLS = [
    "gross_margin", "profit_margin", "current_ratio", "debt_to_equity",
    "revenue_growth", "eps_estimate", "eps_actual", "eps_difference", "surprise_percent",
]


def _make_features(n=5):
    idx = pd.date_range("2024-01-01", periods=n, freq="QE")
    return pd.DataFrame({col: [0.1] * n for col in _FEATURE_COLS}, index=idx)


def _make_targets(n=5):
    idx = pd.date_range("2024-01-01", periods=n, freq="QE")
    labels = (["positive", "neutral", "negative", "positive"] * ((n // 4) + 1))[:n]
    return pd.DataFrame(
        {"daily_target": labels, "weekly_target": labels},
        index=idx,
    )


def _make_returns(n=5):
    idx = pd.date_range("2024-01-01", periods=n, freq="QE")
    return pd.DataFrame(
        {"daily_return": [0.01] * n, "weekly_return": [0.02] * n},
        index=idx,
    )


@pytest.fixture
def mock_ticker():
    t = MagicMock()
    t.ticker = "TESTCACHE"
    return t


def test_cache_miss_returns_two_dataframes(tmp_path, mock_ticker, monkeypatch):
    features, targets, returns = _make_features(5), _make_targets(5), _make_returns(5)
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
    features, targets, returns = _make_features(5), _make_targets(5), _make_returns(5)
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


def test_train_classifier_batch_no_unpack_error(monkeypatch):
    # 4 samples per ticker = 8 total — below 10 threshold so no stratification
    features_a = _make_features(4)
    targets_a = _make_targets(4)
    features_b = _make_features(4)
    targets_b = _make_targets(4)

    call_count = {"n": 0}

    def fake_cache(ticker, _cache_dir=None):
        call_count["n"] += 1
        if ticker.ticker == "TICKER_A":
            return features_a.copy(), targets_a.copy()
        return features_b.copy(), targets_b.copy()

    monkeypatch.setattr(fin_mod, "prepare_classification_data_cache", fake_cache)

    ticker_a = MagicMock(); ticker_a.ticker = "TICKER_A"
    ticker_b = MagicMock(); ticker_b.ticker = "TICKER_B"

    daily_clf, weekly_clf = train_classifier([ticker_a, ticker_b])

    assert call_count["n"] == 2, "expected one cache call per ticker"
    assert hasattr(daily_clf, "predict"), "daily_clf must be a fitted classifier"
    assert hasattr(weekly_clf, "predict"), "weekly_clf must be a fitted classifier"
