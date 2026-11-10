"""Factor definitions shared by the research screens, the backtester and trading.

Kept outside `research/` so `backtest/` and `trading/` can use the same maths
without importing the interactive research code.
"""

from .prices import momentum_12_1, split_factor, unadjusted_close
from .qvm import (
    SCORE_COLUMNS,
    earnings_variability,
    fundamental_metrics,
    percentile_rank,
    score_factors,
    winsorise,
)
from .valuation import (
    SECTOR_DISCOUNT_RATES,
    discount_rate_for,
    epv_to_price,
    implied_growth,
    margin_of_safety,
)

__all__ = [
    "SCORE_COLUMNS",
    "SECTOR_DISCOUNT_RATES",
    "discount_rate_for",
    "earnings_variability",
    "epv_to_price",
    "fundamental_metrics",
    "implied_growth",
    "margin_of_safety",
    "momentum_12_1",
    "percentile_rank",
    "score_factors",
    "split_factor",
    "unadjusted_close",
    "winsorise",
]
