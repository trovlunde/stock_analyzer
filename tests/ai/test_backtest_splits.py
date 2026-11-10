import pandas as pd

from stock_analysis.ai.technical_analysis.movement_classification import (
    resolve_backtest_period,
    temporal_holdout_split,
)
from stock_analysis.research.portfolio_evaluating import normalize_trading_signals


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


def _run_backtests_with_stubs(monkeypatch, **kwargs):
    import stock_analysis.ai.technical_analysis.movement_classification as mc

    leveraged_calls = []
    monkeypatch.setattr(mc, "simulate_trading", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(mc, "full_analysis", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(
        mc, "analyze_trading_strategies", lambda *a, **k: leveraged_calls.append(k)
    )
    stock = _sample_stock(20)
    mc.run_movement_backtests(
        stock, None, None, None, None, None, None, 0.005, False,
        None, None, stock, "test", **kwargs,
    )
    return leveraged_calls


def test_leveraged_simulation_is_off_by_default(monkeypatch, capsys):
    assert _run_backtests_with_stubs(monkeypatch) == []
    assert "stock-analysis-evaluate backtest" in capsys.readouterr().out


def test_leveraged_simulation_runs_when_asked_for(monkeypatch):
    calls = _run_backtests_with_stubs(monkeypatch, leveraged_simulation=True)
    assert [call["leverage"] for call in calls] == [10, 10, 10]
