"""Trailing stops computed from price history alone.

The fixed stop (`stop_loss_pct` below average cost) depends on the entry price.
The stops here do not: the level trails the highest close of the last
`stop_lookback` sessions, so it is the same for a holding and for a candidate.
That lets `decide()` use one level both to exit and to hold off re-entry, and
makes a backtest and the daily job agree without tracking entry dates.
"""

import pandas as pd

STOP_TYPES = ("fixed", "pct_trailing", "atr_trailing")


def average_true_range(df: pd.DataFrame, period: int) -> float | None:
    """Mean true range of the last `period` sessions; None with too little data.

    True range is the day's High − Low widened by any gap from the previous Close.
    """
    if len(df) < period + 1:
        return None
    previous_close = df["Close"].shift(1)
    true_range = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - previous_close).abs(),
            (df["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    value = float(true_range.iloc[-period:].mean())
    return value if value > 0 else None


def trailing_stop(
    df: pd.DataFrame,
    *,
    stop_type: str,
    stop_loss_pct: float | None,
    atr_multiple: float,
    atr_period: int,
    lookback: int,
) -> tuple[float, str] | None:
    """(stop level, description) for the latest bar of `df`, or None when the stop
    type has no trailing level (`fixed`) or there is too little history.

    `pct_trailing`: `stop_loss_pct` below the highest close of the last `lookback`
    sessions. `atr_trailing`: `atr_multiple` average true ranges below it, so the
    distance adapts to how much the asset normally moves.
    """
    if stop_type == "fixed":
        return None
    if stop_type not in STOP_TYPES:
        raise ValueError(f"Unknown stop type {stop_type!r}; known: {STOP_TYPES}")
    # Only the tail matters; this is called for every ticker on every session
    df = df.iloc[-(max(lookback, atr_period) + 1):]
    close = df["Close"].dropna()
    if len(close) < lookback:
        return None
    high = float(close.iloc[-lookback:].max())

    if stop_type == "pct_trailing":
        if stop_loss_pct is None:
            return None
        level = high * (1 - stop_loss_pct)
        return level, f"trailing stop: {stop_loss_pct:.1%} below the {lookback}-day high"

    atr = average_true_range(df, atr_period)
    if atr is None:
        return None
    level = high - atr_multiple * atr
    return level, f"trailing stop: {atr_multiple:g} ATR below the {lookback}-day high"
