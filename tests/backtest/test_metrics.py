from datetime import date

import numpy as np
import pandas as pd
import pytest

from stock_analysis.backtest.engine import BacktestResult, Trade
from stock_analysis.backtest.metrics import (
    deflated_sharpe_ratio,
    paired_block_bootstrap,
    performance_metrics,
    trade_statistics,
)
from stock_analysis.backtest.report import append_run, format_comparison, logged_runs


def _equity(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="B"), dtype=float)


def _noisy_returns(n=750, mean=0.0005, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(mean, 0.01, n))


def test_performance_metrics_for_steady_growth_over_one_year():
    daily_growth = 2 ** (1 / 252)
    equity = _equity([100 * daily_growth**i for i in range(253)])

    metrics = performance_metrics(equity)

    assert metrics["total_return"] == pytest.approx(1.0)
    assert metrics["cagr"] == pytest.approx(1.0)
    assert metrics["max_drawdown"] == 0
    assert metrics["calmar"] is None  # no drawdown to divide by
    assert metrics["sortino"] is None  # no losing days


def test_max_drawdown_is_measured_from_the_running_peak():
    metrics = performance_metrics(_equity([100, 120, 90, 110]))

    assert metrics["max_drawdown"] == pytest.approx(-0.25)
    assert metrics["total_return"] == pytest.approx(0.10)


def test_flat_equity_has_no_sharpe_rather_than_a_division_error():
    metrics = performance_metrics(_equity([100] * 10))

    assert metrics["sharpe"] is None
    assert metrics["total_return"] == 0
    assert metrics["max_drawdown"] == 0


def test_sharpe_matches_definition():
    returns = _noisy_returns(300)
    equity = _equity((1 + returns).cumprod().tolist())
    realised = equity.pct_change().dropna()

    metrics = performance_metrics(equity)

    assert metrics["sharpe"] == pytest.approx(realised.mean() / realised.std() * np.sqrt(252))
    assert metrics["volatility"] == pytest.approx(realised.std() * np.sqrt(252))


def test_trade_statistics_counts_costs_turnover_and_hit_rate():
    equity = _equity([100_000] * 253)
    trades = [
        Trade(date(2024, 1, 2), "AAA", "BUY", 100, 100.0, 6.0, "buy"),
        Trade(date(2024, 2, 2), "AAA", "SELL", 100, 110.0, 6.5, "sell", realized_pnl=993.5),
        Trade(date(2024, 3, 2), "BBB", "BUY", 200, 50.0, 6.0, "buy"),
        Trade(date(2024, 4, 2), "BBB", "SELL", 200, 45.0, 5.5, "sell", realized_pnl=-1005.5),
    ]
    result = BacktestResult(
        initial_cash=100_000,
        equity=equity,
        exposure=pd.Series(0.25, index=equity.index),
        trades=trades,
    )

    stats = trade_statistics(result)

    assert stats["trades"] == 4
    assert stats["round_trips"] == 2
    assert stats["hit_rate"] == 0.5
    assert stats["total_costs"] == pytest.approx(24.0)
    assert stats["cost_drag"] == pytest.approx(0.00024)
    # 10,000 + 11,000 + 10,000 + 9,000 traded over one year on 100,000
    assert stats["annual_turnover"] == pytest.approx(0.40)
    assert stats["avg_exposure"] == pytest.approx(0.25)


def test_trade_statistics_without_trades():
    equity = _equity([100_000] * 10)
    result = BacktestResult(
        initial_cash=100_000, equity=equity, exposure=pd.Series(0.0, index=equity.index)
    )

    stats = trade_statistics(result)

    assert stats["trades"] == 0
    assert stats["hit_rate"] is None
    assert stats["cost_drag"] == 0


def test_bootstrap_is_reproducible_and_brackets_the_observed_sharpe():
    strategy = _noisy_returns(seed=1)
    benchmark = _noisy_returns(seed=2)
    observed = strategy.mean() / strategy.std() * np.sqrt(252)

    first = paired_block_bootstrap(strategy, benchmark, n_resamples=500, seed=7)
    second = paired_block_bootstrap(strategy, benchmark, n_resamples=500, seed=7)

    assert first == second
    assert first["sharpe_ci_low"] < observed < first["sharpe_ci_high"]
    assert 0 <= first["prob_beats_benchmark"] <= 1


def test_bootstrap_favours_a_strategy_that_adds_steady_return_to_the_benchmark():
    benchmark = _noisy_returns(mean=0.0, seed=3)
    better = benchmark + 0.002
    worse = benchmark - 0.002

    assert paired_block_bootstrap(better, benchmark, n_resamples=300)["prob_beats_benchmark"] > 0.95
    assert paired_block_bootstrap(worse, benchmark, n_resamples=300)["prob_beats_benchmark"] < 0.05


def test_bootstrap_needs_enough_data():
    short = _noisy_returns(n=20)

    assert paired_block_bootstrap(short, short)["sharpe_ci_low"] is None


def test_deflated_sharpe_falls_as_more_configurations_are_tried():
    returns = _noisy_returns(mean=0.0008)
    sharpe = returns.mean() / returns.std()
    few_trials = [sharpe, sharpe - 0.03]
    many_trials = [sharpe] + [sharpe - 0.03 * (i % 5) for i in range(1, 50)]

    few = deflated_sharpe_ratio(returns, few_trials)
    many = deflated_sharpe_ratio(returns, many_trials)

    assert 0 <= many < few <= 1


def test_deflated_sharpe_needs_at_least_two_distinct_trials():
    returns = _noisy_returns()

    assert deflated_sharpe_ratio(returns, [0.05]) is None
    assert deflated_sharpe_ratio(returns, [0.05, 0.05]) is None


def test_run_log_round_trips_and_filters_by_experiment(tmp_path):
    log = tmp_path / "nested" / "runs.jsonl"

    append_run({"experiment": "A", "daily_sharpe": 0.1}, log)
    append_run({"experiment": "B", "daily_sharpe": 0.2}, log)
    append_run({"experiment": "A", "daily_sharpe": 0.3}, log)

    assert [run["daily_sharpe"] for run in logged_runs("A", log)] == [0.1, 0.3]
    assert logged_runs("A", tmp_path / "missing.jsonl") == []


def test_format_comparison_marks_metrics_a_column_lacks():
    table = format_comparison(
        {
            "Strategy": {"total_return": 0.1234, "sharpe": 1.5, "trades": 4},
            "Buy & hold": {"total_return": 0.2, "sharpe": None},
        }
    )

    assert "| Total return | 12.34% | 20.00% |" in table
    assert "| Sharpe | 1.50 | — |" in table
    assert "| Trades | 4 | — |" in table
    assert "Hit rate" not in table
