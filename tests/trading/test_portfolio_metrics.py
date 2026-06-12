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
