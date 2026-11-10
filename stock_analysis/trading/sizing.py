"""Position sizing: how much of the portfolio a new position gets."""

import math
from typing import Protocol

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
SIZING_MODES = ("fixed", "vol_target")


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


def position_weight(config: SizingConfig, volatility: float | None) -> float:
    """Share of portfolio value to put into a new position.

    `fixed`: always `position_size_pct`. `vol_target`: `risk_per_position /
    volatility`, so a calm asset gets a larger position and a jumpy one a smaller
    one and each contributes about the same risk; capped at `max_position_pct`.
    Falls back to the fixed size when the volatility is unknown.
    """
    if config.sizing != "vol_target" or volatility is None:
        return config.position_size_pct
    return min(config.max_position_pct, config.risk_per_position / volatility)
