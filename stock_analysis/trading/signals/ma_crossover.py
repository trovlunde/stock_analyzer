import pandas as pd

from stock_analysis.ai.technical_analysis.ma_signals import generate_ma_signals

from .base import SignalResult


class MaCrossoverStrategy:
    def evaluate(self, df: pd.DataFrame, params: dict) -> SignalResult:
        short_period = int(params.get("ma_short", 20))
        long_period = int(params.get("ma_long", 50))
        ma_type = params.get("ma_type", "sma")

        if len(df) < long_period + 1:
            return SignalResult(
                action="HOLD",
                reason=f"insufficient data ({len(df)} bars, need {long_period + 1})",
                metadata={"ma_short": short_period, "ma_long": long_period},
            )

        signals_df = generate_ma_signals(
            df, short_period=short_period, long_period=long_period, ma_type=ma_type
        )
        latest = signals_df.iloc[-1]
        signal_value = int(latest["Signal"])
        short_ma = float(latest[f"MA_{short_period}"])
        long_ma = float(latest[f"MA_{long_period}"])
        metadata = {
            "ma_short": short_period,
            "ma_long": long_period,
            "short_ma": short_ma,
            "long_ma": long_ma,
            "signal_value": signal_value,
        }

        if signal_value == 1:
            return SignalResult(
                action="BUY",
                reason=f"golden cross (MA{short_period} crossed above MA{long_period})",
                metadata=metadata,
            )
        if signal_value == -1:
            return SignalResult(
                action="SELL",
                reason=f"death cross (MA{short_period} crossed below MA{long_period})",
                metadata=metadata,
            )

        trend = "bullish" if short_ma > long_ma else "bearish"
        return SignalResult(
            action="HOLD",
            reason=f"no crossover ({trend} regime)",
            metadata=metadata,
        )
