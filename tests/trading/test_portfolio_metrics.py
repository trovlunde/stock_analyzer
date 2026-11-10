"""Regression tests for compute_portfolio_metrics against empyrical reference."""

import numpy as np
import pandas as pd
import pytest
import empyrical

from stock_analysis.trading.portfolio_metrics import compute_portfolio_metrics


@pytest.fixture()
def synthetic_returns() -> pd.Series:
    """Fixed 252-day return series via seeded RNG — no network."""
    rng = np.random.default_rng(42)
    return pd.Series(rng.normal(0.001, 0.02, 252), dtype=float)


def test_calmar_matches_empyrical(synthetic_returns):
    result = compute_portfolio_metrics(synthetic_returns)
    expected = empyrical.calmar_ratio(synthetic_returns)
    assert result["calmar"] is not None
    assert abs(result["calmar"] - float(expected)) < 1e-9


def test_sortino_matches_empyrical(synthetic_returns):
    result = compute_portfolio_metrics(synthetic_returns)
    expected = empyrical.sortino_ratio(synthetic_returns)
    assert result["sortino"] is not None
    assert abs(result["sortino"] - float(expected)) < 1e-9


def test_rolling_sharpe_matches_empyrical(synthetic_returns):
    result = compute_portfolio_metrics(synthetic_returns, rolling_window=63)
    expected = empyrical.roll_sharpe_ratio(synthetic_returns, window=63).dropna()
    rs = result["rolling_sharpe"]
    assert len(rs) > 0
    assert abs(rs.iloc[-1] - expected.iloc[-1]) < 1e-9


def test_short_series_returns_none_metrics():
    short = pd.Series([0.01], dtype=float)
    result = compute_portfolio_metrics(short)
    assert result["calmar"] is None
    assert result["sortino"] is None
    assert len(result["rolling_sharpe"]) == 0


def test_insufficient_rolling_window_returns_empty_rolling():
    returns = pd.Series([0.01, 0.02, -0.01, 0.005], dtype=float)
    result = compute_portfolio_metrics(returns, rolling_window=63)
    assert len(result["rolling_sharpe"]) == 0


def test_empty_series_returns_none_metrics():
    result = compute_portfolio_metrics(pd.Series([], dtype=float))
    assert result["calmar"] is None
    assert result["sortino"] is None
    assert len(result["rolling_sharpe"]) == 0


def _benchmark_close():
    index = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    return pd.Series([100.0, 102.0, 104.0, 110.0], index=index)


def test_benchmark_relative_return_compares_over_the_equity_curve_span():
    from stock_analysis.trading.portfolio_metrics import benchmark_relative_return

    curve = [("2024-01-02", 10_000.0), ("2024-01-03", 10_100.0), ("2024-01-05", 10_500.0)]

    result = benchmark_relative_return(curve, _benchmark_close(), "SPY")

    assert result["benchmark"] == "SPY"
    assert result["since"] == "2024-01-02"
    assert result["portfolio_return"] == pytest.approx(0.05)
    assert result["benchmark_return"] == pytest.approx(0.10)
    assert result["excess_return"] == pytest.approx(-0.05)


def test_benchmark_relative_return_uses_last_close_on_or_before_each_date():
    from stock_analysis.trading.portfolio_metrics import benchmark_relative_return

    # 2024-01-06/07 is a weekend: the benchmark is priced at Friday's close
    curve = [("2024-01-03", 10_000.0), ("2024-01-07", 10_000.0)]

    result = benchmark_relative_return(curve, _benchmark_close(), "SPY")

    assert result["benchmark_return"] == pytest.approx(110 / 102 - 1)


def test_benchmark_relative_return_needs_two_points_and_a_starting_price():
    from stock_analysis.trading.portfolio_metrics import benchmark_relative_return

    close = _benchmark_close()
    assert benchmark_relative_return([("2024-01-02", 10_000.0)], close, "SPY") is None
    # Portfolio history starts before the benchmark has any price
    early = [("2023-12-01", 10_000.0), ("2024-01-05", 10_500.0)]
    assert benchmark_relative_return(early, close, "SPY") is None
    assert benchmark_relative_return(early, pd.Series(dtype=float), "SPY") is None
