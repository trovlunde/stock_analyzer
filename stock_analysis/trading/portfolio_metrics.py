import numpy as np
import pandas as pd
import empyrical


def benchmark_relative_return(
    equity_curve: list[tuple[str, float]],
    benchmark_close: pd.Series,
    benchmark: str,
) -> dict | None:
    """Portfolio return against buy-and-hold of `benchmark` over the equity curve's span.

    `equity_curve` is [(ISO date, end-of-day value)], oldest first; `benchmark_close`
    is a daily close series with a tz-naive index. Each end is priced at the
    benchmark's last close on or before that date. Returns None when the curve has
    fewer than two points or the benchmark has no price for the first date.
    """
    if len(equity_curve) < 2 or benchmark_close is None or benchmark_close.empty:
        return None

    (start_date, start_value), (end_date, end_value) = equity_curve[0], equity_curve[-1]
    closes = benchmark_close.dropna().sort_index()
    start_close = closes.asof(pd.Timestamp(start_date))
    end_close = closes.asof(pd.Timestamp(end_date))
    if pd.isna(start_close) or pd.isna(end_close) or start_close <= 0 or start_value <= 0:
        return None

    portfolio_return = end_value / start_value - 1
    benchmark_return = float(end_close / start_close) - 1
    return {
        "benchmark": benchmark,
        "since": start_date,
        "portfolio_return": portfolio_return,
        "benchmark_return": benchmark_return,
        "excess_return": portfolio_return - benchmark_return,
    }


def compute_portfolio_metrics(
    returns: pd.Series,
    *,
    rolling_window: int = 63,
) -> dict:
    """Compute Calmar, Sortino, and rolling Sharpe from a daily returns series.

    Returns a dict with keys: calmar, sortino, rolling_sharpe.
    Metrics are None (or empty Series) when there is insufficient data.
    """
    result: dict = {
        "calmar": None,
        "sortino": None,
        "rolling_sharpe": pd.Series(dtype=float),
    }

    if len(returns) < 2:
        return result

    try:
        calmar = empyrical.calmar_ratio(returns)
        if np.isfinite(calmar):
            result["calmar"] = float(calmar)
    except Exception:
        pass

    try:
        sortino = empyrical.sortino_ratio(returns)
        if np.isfinite(sortino):
            result["sortino"] = float(sortino)
    except Exception:
        pass

    if len(returns) >= rolling_window:
        try:
            rolling = empyrical.roll_sharpe_ratio(returns, window=rolling_window)
            result["rolling_sharpe"] = rolling.dropna()
        except Exception:
            pass

    return result
