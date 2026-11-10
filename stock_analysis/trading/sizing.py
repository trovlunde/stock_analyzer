"""Position sizing: how much of the portfolio a new position gets."""

import math
from typing import Protocol

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
# `min_variance` and `hrp` take the weight from an allocation over the whole
# universe — see trading.allocation.universe_weights.
ALLOCATION_MODES = ("min_variance", "hrp")
SIZING_MODES = ("fixed", "vol_target", *ALLOCATION_MODES)


class SizingConfig(Protocol):
    sizing: str
    position_size_pct: float
    risk_per_position: float
    max_position_pct: float
    vol_lookback: int


def annualised_volatility(close: pd.Series, lookback: int) -> float | None:
    """Annualised standard deviation of the last `lookback` daily returns.

    None when there is not enough history or the price never moved.
    """
    returns = close.dropna().iloc[-(lookback + 1):].pct_change().dropna()
    if len(returns) < lookback:
        return None
    volatility = float(returns.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    return volatility if volatility > 0 else None


def position_weight(
    config: SizingConfig,
    volatility: float | None,
    universe_weight: float | None = None,
) -> float:
    """Share of portfolio value to put into a new position.

    `fixed`: always `position_size_pct`. `vol_target`: `risk_per_position /
    volatility`, so a calm asset gets a larger position and a jumpy one a smaller
    one and each contributes about the same risk. `min_variance` / `hrp`: the
    ticker's weight in an allocation over the whole universe. All but `fixed` are
    capped at `max_position_pct` and fall back to the fixed size when their input
    is unknown.
    """
    if config.sizing == "vol_target" and volatility is not None:
        return min(config.max_position_pct, config.risk_per_position / volatility)
    if config.sizing in ALLOCATION_MODES and universe_weight is not None:
        return min(config.max_position_pct, universe_weight)
    return config.position_size_pct


def exposure_scale(
    equity: list[float], target_volatility: float | None, lookback: int
) -> float:
    """Multiplier (at most 1) on new position sizes so the portfolio as a whole
    aims at `target_volatility`.

    Compares the target with the annualised volatility of the portfolio's own last
    `lookback` daily returns: when the portfolio has been moving more than the
    target, new positions are sized down in proportion. Never sizes up, and is 1
    while there is no target or too little history.
    """
    if target_volatility is None or len(equity) < lookback + 1:
        return 1.0
    realised = annualised_volatility(pd.Series(equity, dtype=float), lookback)
    if realised is None or realised <= target_volatility:
        return 1.0
    return target_volatility / realised
