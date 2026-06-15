"""Stationarity diagnostics using ADF test from statsmodels."""

from __future__ import annotations

import pandas as pd


def check_stationarity(series: pd.Series, *, alpha: float = 0.05) -> dict:
    """Run ADF test on *series* and return a result dict.

    Returns
    -------
    dict with keys:
        statistic      float   ADF test statistic
        p_value        float   MacKinnon p-value
        is_stationary  bool    True when p_value < alpha
        insufficient_data bool  True when series too short for ADF
    """
    import numpy as np
    from statsmodels.tsa.stattools import adfuller

    clean = series.dropna()
    # ADF requires at least ~20 obs to be meaningful; statsmodels needs >2
    if len(clean) < 5:
        return {
            "statistic": None,
            "p_value": None,
            "is_stationary": None,
            "insufficient_data": True,
        }

    result = adfuller(clean, autolag="AIC")
    return {
        "statistic": float(result[0]),
        "p_value": float(result[1]),
        "is_stationary": float(result[1]) < alpha,
        "insufficient_data": False,
    }


def check_stationarity_frame(
    df: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    """Run ADF stationarity check on numeric columns of *df*.

    Parameters
    ----------
    df      DataFrame with feature columns.
    columns Subset of columns to check; defaults to all numeric columns.

    Returns
    -------
    DataFrame indexed by column name with columns:
        statistic, p_value, is_stationary, insufficient_data
    """
    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    rows = []
    for col in columns:
        result = check_stationarity(df[col])
        rows.append({"column": col, **result})

    return pd.DataFrame(rows).set_index("column")
