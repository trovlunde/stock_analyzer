import pandas as pd

from stock_analysis.ai.technical_analysis.movement_classification import (
    resolve_backtest_period,
    temporal_holdout_split,
)
from stock_analysis.portfolio_evaluating import normalize_trading_signals


def _sample_prepared(n=100):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "prev_day_return": [0.01] * n,
            "prev_2day_return": [0.01] * n,
            "prev_week_return": [0.01] * n,
            "return": [0.01] * n,
            "Target": ["positive"] * n,
        },
        index=dates,
    )


def _sample_stock(n=100):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Close": range(n)}, index=dates)


def test_normalize_trading_signals_uses_date_column():
    signals = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=3, freq="B"),
            "prediction": ["positive", "neutral", "negative"],
        }
    )
    normalized = normalize_trading_signals(signals)
    assert isinstance(normalized.index, pd.DatetimeIndex)
    assert list(normalized["prediction"]) == ["positive", "neutral", "negative"]


def test_resolve_backtest_period_holdout():
    prepared = _sample_prepared(100)
    stock = _sample_stock(100)
    cutoff = prepared.index[80]
    eval_daily, _, eval_stock, label = resolve_backtest_period(
        prepared,
        prepared,
        stock,
        holdout_months=12,
        cutoff_date=cutoff,
    )
    assert len(eval_daily) == 20
    assert eval_daily.index.min() == cutoff
    assert "holdout" in label


def test_multi_ticker_training_cutoff_excludes_recent_rows():
    first = _sample_prepared(100)
    second = _sample_prepared(100)
    first["ticker"] = "AAA"
    second["ticker"] = "BBB"
    combined = pd.concat([first, second])
    cutoff = combined.index[80]
    filtered = combined[combined.index < cutoff]
    assert filtered.index.max() < cutoff
    assert len(filtered) < len(combined)


def test_resolve_backtest_period_internal_test_set():
    prepared = _sample_prepared(100)
    stock = _sample_stock(100)
    train_meta = prepared.copy()
    train_meta["test_set"] = False
    train_meta.iloc[-20:, train_meta.columns.get_loc("test_set")] = True

    eval_daily, _, eval_stock, label = resolve_backtest_period(
        prepared,
        prepared,
        stock,
        train_meta=train_meta,
    )
    assert len(eval_daily) == 20
    assert "internal chronological test set" in label
