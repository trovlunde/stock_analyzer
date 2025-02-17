import pandas as pd
import numpy as np
from typing import List, Union


def generate_rsi_signals(
    stock_data: Union[pd.DataFrame, List[pd.DataFrame]],
    period: int = 14,
    overbought_threshold: float = 70,
    oversold_threshold: float = 30
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Generate buy/sell signals based on RSI indicator.

    Args:
        stock_data: Single DataFrame or list of DataFrames with stock price data
        period: RSI calculation period (default: 14)
        overbought_threshold: RSI threshold for sell signal (default: 70)
        oversold_threshold: RSI threshold for buy signal (default: 30)

    Returns:
        DataFrame or list of DataFrames with added RSI and signal columns
    """

    def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
        # Make a copy to avoid modifying original data
        df = df.copy()

        # Calculate price changes
        delta = df['Close'].diff()

        # Calculate gains (positive) and losses (negative)
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Calculate relative strength
        rs = gain / loss

        # Calculate RSI
        df['RSI'] = 100 - (100 / (1 + rs))

        # Generate signals
        df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell

        # Buy signal when RSI crosses below oversold threshold
        df.loc[df['RSI'] < oversold_threshold, 'Signal'] = 1

        # Sell signal when RSI crosses above overbought threshold
        df.loc[df['RSI'] > overbought_threshold, 'Signal'] = -1

        return df

    # Handle both single DataFrame and list of DataFrames
    if isinstance(stock_data, pd.DataFrame):
        return calculate_signals(stock_data)
    else:
        return [calculate_signals(df) for df in stock_data]
