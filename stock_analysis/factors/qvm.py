"""Quality / Value / Momentum factor definitions and sector-relative scoring.

Shared by the snapshot screen (`research/qvm.py`, from `yfinance.info`) and the
backtestable strategy (`factors/qvm_strategy.py`, from point-in-time
fundamentals), so both rank on the same definitions.

Every value metric is a *yield* (something / price), never a multiple. A
multiple breaks down for a loss-making company: its P/E is negative, and
"lowest P/E is cheapest" then ranks it above every profitable one. As a yield
the same company is simply at the bottom.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from .valuation import margin_of_safety

# Higher is better
VALUE_METRICS = (
    "earnings_yield",  # net income / market value
    "forward_earnings_yield",  # analyst estimate; snapshot screen only
    "fcf_yield",  # (operating cash flow - capex) / market value
    "ebitda_ev",  # EBITDA / enterprise value
    "book_to_price",  # equity / market value
)
# Optional extra value metric from the valuation models; off by default
MARGIN_OF_SAFETY_METRIC = "epv_to_price"
QUALITY_HIGHER = (
    "roe",
    "profit_margin",
    "operating_margin",
    "gross_profitability",  # gross profit / total assets
)
# Lower is better
QUALITY_LOWER = (
    "debt_to_equity",  # snapshot screen
    "leverage",  # total debt / total assets; point-in-time
    "accruals",  # (net income - operating cash flow) / total assets
    "earnings_variability",
)
MOMENTUM_METRICS = ("momentum_12_1",)

SCORE_COLUMNS = ("quality_score", "value_score", "momentum_score", "qvm_score")

DEFAULT_MIN_SECTOR_SIZE = 4
DEFAULT_WINSOR_LIMITS = (0.05, 0.95)
MIN_STABILITY_CHANGES = 4

_NAN = float("nan")


def winsorise(
    series: pd.Series, lower: float = DEFAULT_WINSOR_LIMITS[0], upper: float = DEFAULT_WINSOR_LIMITS[1]
) -> pd.Series:
    """Pull the lowest `lower` and highest `1 - upper` share of values in to the nearest value kept.

    Stops one absurd figure (an earnings yield of 400 % on a near-zero market
    value, say) from counting for more than "very high". The bounds are actual
    observations, not interpolated quantiles, which an outlier would drag along
    with it; with fewer than `1 / lower` values nothing is clipped. NaNs stay NaN.
    """
    values = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    ordered = np.sort(values.dropna().to_numpy())
    count = len(ordered)
    if count == 0:
        return values
    # round() guards against 20 * (1 - 0.95) coming out as 0.99999...
    below = int(round(count * lower, 9))
    above = int(round(count * (1 - upper), 9))
    return values.clip(ordered[below], ordered[count - 1 - above])


def percentile_rank(series: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    """Rank within the group as 0–100 (100 = best). NaNs stay NaN."""
    values = series if higher_is_better else -series
    return values.rank(pct=True, na_option="keep") * 100


def _safe_ratio(numerator: float, denominator: float) -> float:
    """numerator / denominator, NaN unless both are known and the denominator is positive."""
    if pd.isna(numerator) or pd.isna(denominator) or denominator <= 0:
        return _NAN
    return float(numerator) / float(denominator)


def earnings_variability(net_income_quarters: pd.Series | None, total_assets: float) -> float:
    """Standard deviation of the year-over-year change in quarterly net income, over total assets.

    Lower means steadier earnings. Each quarter is compared with the same
    quarter a year earlier so that an ordinary seasonal pattern does not count
    as instability. NaN with fewer than `MIN_STABILITY_CHANGES` such changes
    (eight quarters of history).
    """
    if net_income_quarters is None or pd.isna(total_assets) or total_assets <= 0:
        return _NAN
    changes = net_income_quarters.diff(4).dropna()
    if len(changes) < MIN_STABILITY_CHANGES:
        return _NAN
    return float(changes.std(ddof=1)) / float(total_assets)


def fundamental_metrics(
    values: Mapping[str, float],
    market_cap: float,
    *,
    net_income_quarters: pd.Series | None = None,
    sector: str | None = None,
) -> dict[str, float]:
    """Value and quality metrics from point-in-time fundamentals and a market value.

    `values` uses the names of `FundamentalSnapshot.values`: trailing-twelve-month
    flows and the latest balance sheet. Anything whose inputs are missing is NaN
    and is left out of the factor average rather than guessed.
    """

    def get(name: str) -> float:
        value = values.get(name, _NAN)
        return _NAN if value is None else float(value)

    net_income = get("net_income")
    revenue = get("revenue")
    assets = get("total_assets")
    equity = get("total_equity")
    # No debt (or cash) line on a balance sheet that exists means there is none
    debt = get("total_debt") if not pd.isna(get("total_debt")) else 0.0
    cash = get("cash") if not pd.isna(get("cash")) else 0.0
    enterprise_value = market_cap + debt - cash

    metrics = {
        "market_cap": market_cap,
        "earnings_yield": _safe_ratio(net_income, market_cap),
        "fcf_yield": _safe_ratio(get("free_cash_flow"), market_cap),
        "ebitda_ev": _safe_ratio(get("ebitda"), enterprise_value),
        "book_to_price": _safe_ratio(equity, market_cap),
        # Negative equity makes ROE meaningless (a loss would give a positive ratio)
        "roe": _safe_ratio(net_income, equity),
        "profit_margin": _safe_ratio(net_income, revenue),
        "operating_margin": _safe_ratio(get("operating_income"), revenue),
        "gross_profitability": _safe_ratio(get("gross_profit"), assets),
        "leverage": _safe_ratio(debt, assets),
        "accruals": _safe_ratio(net_income - get("operating_cash_flow"), assets),
        "earnings_variability": earnings_variability(net_income_quarters, assets),
    }
    metrics.update(margin_of_safety({**values, "total_debt": debt, "cash": cash}, market_cap, sector))
    return metrics


def _factor_score(df: pd.DataFrame, columns: Sequence[tuple[str, bool]]) -> pd.Series:
    """Mean of the percentile ranks of the metrics present; all-NaN → NaN."""
    parts = [
        percentile_rank(df[column], higher_is_better=higher_is_better)
        for column, higher_is_better in columns
        if column in df.columns
    ]
    if not parts:
        return pd.Series(_NAN, index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def score_factors(
    df: pd.DataFrame,
    *,
    sector_col: str | None = "sector",
    min_sector_size: int = DEFAULT_MIN_SECTOR_SIZE,
    winsor_limits: tuple[float, float] | None = DEFAULT_WINSOR_LIMITS,
    value_metrics: Sequence[str] = VALUE_METRICS,
    quality_higher: Sequence[str] = QUALITY_HIGHER,
    quality_lower: Sequence[str] = QUALITY_LOWER,
    momentum_metrics: Sequence[str] = MOMENTUM_METRICS,
    require_all_factors: bool = False,
) -> pd.DataFrame:
    """Add quality_score, value_score, momentum_score and qvm_score (0–100).

    Each metric is winsorised across the whole frame, then ranked *within its
    sector*, so a bank is compared with banks and the result does not simply
    favour whichever sector is cheap or in fashion. A factor score is the mean
    of its metrics' ranks; the QVM score is the mean of the three factors.
    Sectors with fewer than `min_sector_size` rows get NaN scores: a rank among
    two or three names says little. `sector_col=None` ranks the frame as one group.

    With rank-based scores winsorising only changes ties at the extremes; it is
    done first so that the same cleaned metrics are what any later use sees.

    `require_all_factors` leaves the QVM score NaN unless all three factors
    could be scored, so a stock with prices but no fundamentals is not ranked
    on momentum alone.
    """
    out = df.copy()
    for column in SCORE_COLUMNS:
        out[column] = pd.Series(_NAN, index=out.index, dtype=float)
    if out.empty:
        return out

    quality = [(c, True) for c in quality_higher] + [(c, False) for c in quality_lower]
    value = [(c, True) for c in value_metrics]
    momentum = [(c, True) for c in momentum_metrics]

    cleaned = out.copy()
    for column, _ in quality + value + momentum:
        if column in cleaned.columns:
            cleaned[column] = (
                winsorise(cleaned[column], *winsor_limits)
                if winsor_limits is not None
                else pd.to_numeric(cleaned[column], errors="coerce")
            )

    if sector_col is None:
        groups = [cleaned]
    else:
        groups = [group for _, group in cleaned.groupby(sector_col, dropna=False)]
    for group in groups:
        if len(group) < min_sector_size:
            continue
        factors = pd.concat(
            [_factor_score(group, quality), _factor_score(group, value), _factor_score(group, momentum)],
            axis=1,
        )
        out.loc[group.index, "quality_score"] = factors[0]
        out.loc[group.index, "value_score"] = factors[1]
        out.loc[group.index, "momentum_score"] = factors[2]
        combined = factors.mean(axis=1, skipna=True)
        if require_all_factors:
            combined = combined.where(factors.notna().all(axis=1))
        out.loc[group.index, "qvm_score"] = combined
    return out
