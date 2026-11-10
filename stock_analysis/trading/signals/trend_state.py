import pandas as pd

from .base import SignalResult

DEFAULT_LOOKBACKS = (63, 126, 252)  # about 3, 6 and 12 months of sessions
DEFAULT_MA = 200


class TrendStateStrategy:
    """Long while the asset is in an uptrend, flat otherwise.

    Unlike `ma_crossover`, which signals only on the day of a cross, this reports
    the current state every day: BUY for as long as the uptrend holds and SELL for
    as long as it is broken. `decide()` skips a BUY when already long and a SELL
    when flat, so a missed run, a full portfolio or a ticker added mid-trend is
    corrected on the next run instead of waiting for the next cross.

    Momentum is the average return over `trend_lookbacks` sessions.
    - BUY: momentum above zero and the price above its `trend_ma`-day average.
    - SELL: momentum at or below zero.
    - HOLD: momentum above zero but the price below the average — no new entry,
      and an existing position is kept. This gap stops the signal flipping daily
      when the price hovers around its average.
    """

    def evaluate(self, df: pd.DataFrame, ticker: str, params: dict) -> SignalResult:
        lookbacks = [int(n) for n in params.get("trend_lookbacks", DEFAULT_LOOKBACKS)]
        ma_period = int(params.get("trend_ma", DEFAULT_MA))
        close = df["Close"].dropna()

        needed = max(max(lookbacks) + 1, ma_period)
        if len(close) < needed:
            return SignalResult(
                action="HOLD",
                reason=f"insufficient data ({len(close)} bars, need {needed})",
                metadata={"trend_lookbacks": lookbacks, "trend_ma": ma_period},
            )

        latest = float(close.iloc[-1])
        returns = {n: latest / float(close.iloc[-1 - n]) - 1 for n in lookbacks}
        momentum = sum(returns.values()) / len(returns)
        average = float(close.iloc[-ma_period:].mean()) if ma_period > 0 else None
        above_average = average is None or latest > average
        metadata = {
            "trend_lookbacks": lookbacks,
            "trend_ma": ma_period,
            "momentum": momentum,
            "returns": {str(n): value for n, value in returns.items()},
            "moving_average": average,
            "close": latest,
        }

        if momentum <= 0:
            return SignalResult(
                action="SELL",
                reason=f"downtrend (momentum {momentum:+.1%})",
                metadata=metadata,
            )
        if not above_average:
            return SignalResult(
                action="HOLD",
                reason=(
                    f"momentum {momentum:+.1%} but price below its "
                    f"{ma_period}-day average — no new entry"
                ),
                metadata=metadata,
            )
        return SignalResult(
            action="BUY",
            reason=f"uptrend (momentum {momentum:+.1%}, price above its average)",
            metadata=metadata,
        )
