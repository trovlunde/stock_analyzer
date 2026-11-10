"""Market-regime filter: no new long positions while the market itself is in a downtrend."""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class RegimeFilter:
    ticker: str
    ma: int = 200


def risk_off_reason(regime_filter: RegimeFilter, market_close: pd.Series) -> str | None:
    """Why new BUY orders should be blocked right now, or None if they are allowed.

    The market is "risk-off" when the filter's ticker closed below its own
    `ma`-day average. `market_close` must end at the session being decided.
    With too little history to tell, nothing is blocked.
    """
    close = market_close.dropna()
    if len(close) < regime_filter.ma:
        return None
    average = float(close.iloc[-regime_filter.ma:].mean())
    if float(close.iloc[-1]) >= average:
        return None
    return (
        f"market regime: {regime_filter.ticker} is below its {regime_filter.ma}-day "
        f"average — new BUY orders blocked"
    )
