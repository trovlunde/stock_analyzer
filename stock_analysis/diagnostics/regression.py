"""Regression diagnostics using OLS from statsmodels."""

from __future__ import annotations

import pandas as pd


def regression_diagnostics(X: pd.DataFrame, y: pd.Series) -> dict:
    """Fit OLS on *X* → *y* and return a diagnostic summary dict.

    Non-numeric columns in X are dropped silently before fitting.
    Pass only numeric features if exact column coverage matters.

    Parameters
    ----------
    X   Feature DataFrame (rows = observations, cols = features).
        Non-numeric columns are dropped before fitting.
    y   Target Series. Aligned to X by index; rows with NaN in either
        X or y are excluded.

    Returns
    -------
    dict with keys:
        r_squared        float  In-sample R²
        coefficients     dict   {feature_name: float} (excludes intercept)
        durbin_watson    float  DW statistic (2 ≈ no autocorrelation in residuals)
        condition_number float  Condition number of numeric X matrix
        n_obs            int    Number of observations used after alignment/dropna
        insufficient_data bool  True when fewer than 3 usable observations or no
                                numeric features remain after dropping non-numeric cols
    """
    import numpy as np
    import statsmodels.api as sm
    from statsmodels.stats.stattools import durbin_watson

    X_num = X.select_dtypes(include="number")

    common = X_num.index.intersection(y.index)
    X_aligned = X_num.loc[common].dropna()
    y_aligned = y.loc[X_aligned.index].dropna()
    X_aligned = X_aligned.loc[y_aligned.index]

    n_obs = len(X_aligned)
    n_features = X_aligned.shape[1] if not X_aligned.empty else 0

    if n_obs < 3 or n_features < 1:
        return {
            "r_squared": None,
            "coefficients": {},
            "durbin_watson": None,
            "condition_number": None,
            "n_obs": n_obs,
            "insufficient_data": True,
        }

    X_const = sm.add_constant(X_aligned, has_constant="add")
    model = sm.OLS(y_aligned, X_const).fit()

    coefs = {col: float(model.params[col]) for col in X_aligned.columns}
    cond = float(np.linalg.cond(X_aligned.values.astype(float)))
    dw = float(durbin_watson(model.resid))

    return {
        "r_squared": float(model.rsquared),
        "coefficients": coefs,
        "durbin_watson": dw,
        "condition_number": cond,
        "n_obs": n_obs,
        "insufficient_data": False,
    }
