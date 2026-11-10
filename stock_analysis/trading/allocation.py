"""Weights over a whole universe from its return covariance.

Mean-variance "max Sharpe" needs expected returns, and estimates of those from
history are mostly noise, so the weights swing wildly and pile into a few names.
Both methods here use the covariance only, and shrink it (Ledoit-Wolf) towards a
simple structure so that it is stable with few observations per asset.
"""

import pandas as pd

ALLOCATION_METHODS = ("min_variance", "hrp")


def universe_weights(
    returns: pd.DataFrame, method: str, *, max_weight: float = 1.0
) -> dict[str, float] | None:
    """Long-only weights summing to 1 for the columns of `returns` (daily returns).

    `min_variance`: the lowest-volatility mix, no asset above `max_weight`.
    `hrp`: hierarchical risk parity — clusters correlated assets and splits risk
    between clusters, so a group of near-duplicates shares one allocation.
    Columns with a missing value in the window are left out. None when fewer than
    two assets remain or there are fewer rows than assets to estimate from.
    """
    if method not in ALLOCATION_METHODS:
        raise ValueError(f"Unknown allocation method {method!r}; known: {ALLOCATION_METHODS}")
    clean = returns.dropna(axis=1, how="any")
    clean = clean.loc[:, clean.std(ddof=1) > 0]
    if clean.shape[1] < 2 or len(clean) <= clean.shape[1]:
        return None

    # Imported here: pypfopt pulls in cvxpy, which the daily job does not need
    # unless one of these sizing modes is configured.
    from pypfopt import EfficientFrontier, HRPOpt, risk_models

    covariance = risk_models.CovarianceShrinkage(clean, returns_data=True).ledoit_wolf()
    if method == "hrp":
        weights = HRPOpt(returns=clean, cov_matrix=covariance).optimize()
    else:
        # A cap below 1/n cannot sum to 1
        upper = max(max_weight, 1.0 / clean.shape[1])
        frontier = EfficientFrontier(None, covariance, weight_bounds=(0, upper))
        frontier.min_volatility()
        weights = frontier.clean_weights()
    return {ticker: float(weight) for ticker, weight in weights.items()}
