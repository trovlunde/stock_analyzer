import math

import pandas as pd

from stock_analysis.ai.helpers import RSI

from .base import SignalResult

DEFAULT_PERIOD = 14
DEFAULT_OVERSOLD = 30
DEFAULT_OVERBOUGHT = 70
DEFAULT_TREND_MA = 200


class RsiStrategy:
    """Buy the recovery from an oversold dip in an uptrend; sell into overbought.

    Signals are crosses, not levels, so each one fires on a single day however
    long the RSI stays beyond the threshold.
    - BUY: RSI crosses back up through `rsi_oversold`, and the close is above its
      `rsi_trend_ma`-day average (0 disables this filter). Waiting for the cross
      back up avoids buying while the price is still falling; the filter avoids
      buying dips in a downtrend.
    - SELL: RSI crosses up through `rsi_overbought`.
    - HOLD: otherwise, including an oversold recovery blocked by the trend filter.
    """

    def evaluate(self, df: pd.DataFrame, ticker: str, params: dict) -> SignalResult:
        period = int(params.get("rsi_period", DEFAULT_PERIOD))
        oversold = float(params.get("rsi_oversold", DEFAULT_OVERSOLD))
        overbought = float(params.get("rsi_overbought", DEFAULT_OVERBOUGHT))
        trend_ma = int(params.get("rsi_trend_ma", DEFAULT_TREND_MA))
        close = df["Close"].dropna()
        metadata = {
            "rsi_period": period,
            "rsi_oversold": oversold,
            "rsi_overbought": overbought,
            "rsi_trend_ma": trend_ma,
        }

        # A cross needs two RSI values, each smoothed over at least `period` changes
        needed = max(period + 2, trend_ma)
        if len(close) < needed:
            return SignalResult(
                action="HOLD",
                reason=f"insufficient data ({len(close)} bars, need {needed})",
                metadata=metadata,
            )

        rsi = RSI(pd.DataFrame({"Close": close}), window=period)
        current, previous = float(rsi.iloc[-1]), float(rsi.iloc[-2])
        latest = float(close.iloc[-1])
        average = float(close.iloc[-trend_ma:].mean()) if trend_ma > 0 else None
        metadata.update(
            {
                "rsi": current,
                "previous_rsi": previous,
                "close": latest,
                "moving_average": average,
            }
        )

        # RSI is undefined while the price has not moved at all
        if math.isnan(current) or math.isnan(previous):
            return SignalResult(action="HOLD", reason="RSI undefined", metadata=metadata)

        if previous <= oversold < current:
            if average is not None and latest <= average:
                return SignalResult(
                    action="HOLD",
                    reason=(
                        f"RSI recovered above {oversold:g} but price below its "
                        f"{trend_ma}-day average — no new entry"
                    ),
                    metadata=metadata,
                )
            return SignalResult(
                action="BUY",
                reason=f"RSI crossed back above {oversold:g} ({previous:.1f} to {current:.1f})",
                metadata=metadata,
            )
        if previous <= overbought < current:
            return SignalResult(
                action="SELL",
                reason=f"RSI crossed above {overbought:g} ({previous:.1f} to {current:.1f})",
                metadata=metadata,
            )
        return SignalResult(
            action="HOLD", reason=f"no RSI cross (RSI {current:.1f})", metadata=metadata
        )
