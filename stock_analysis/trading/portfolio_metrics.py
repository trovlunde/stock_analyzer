import numpy as np
import pandas as pd
import empyrical


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
