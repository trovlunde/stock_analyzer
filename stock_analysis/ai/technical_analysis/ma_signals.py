import pandas as pd
import numpy as np
from typing import Union, List, Tuple


def generate_ma_signals(
    stock_data: Union[pd.DataFrame, List[pd.DataFrame]],
    short_period: int = 20,
    long_period: int = 50,
    ma_type: str = 'sma'  # 'sma' or 'ema'
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Generate buy/sell signals based on moving average crossovers.

    Args:
        stock_data: Single DataFrame or list of DataFrames with stock price data
        short_period: Period for shorter moving average (default: 20)
        long_period: Period for longer moving average (default: 50)
        ma_type: Type of moving average - 'sma' or 'ema' (default: 'sma')

    Returns:
        DataFrame or list of DataFrames with added MA and signal columns
    """

    def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Calculate moving averages
        if ma_type.lower() == 'ema':
            short_ma = df['Close'].ewm(span=short_period, adjust=False).mean()
            long_ma = df['Close'].ewm(span=long_period, adjust=False).mean()
        else:  # default to SMA
            short_ma = df['Close'].rolling(window=short_period).mean()
            long_ma = df['Close'].rolling(window=long_period).mean()

        # Add MA columns
        df[f'MA_{short_period}'] = short_ma
        df[f'MA_{long_period}'] = long_ma

        # Calculate crossover signals
        df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell

        # Golden Cross (short MA crosses above long MA)
        golden_cross = (short_ma > long_ma) & (
            short_ma.shift(1) <= long_ma.shift(1))
        df.loc[golden_cross, 'Signal'] = 1

        # Death Cross (short MA crosses below long MA)
        death_cross = (short_ma < long_ma) & (
            short_ma.shift(1) >= long_ma.shift(1))
        df.loc[death_cross, 'Signal'] = -1

        return df

    # Handle both single DataFrame and list of DataFrames
    if isinstance(stock_data, pd.DataFrame):
        return calculate_signals(stock_data)
    else:
        return [calculate_signals(df) for df in stock_data]


def generate_multi_ma_signals(
    stock_data: Union[pd.DataFrame, List[pd.DataFrame]],
    ma_pairs: List[Tuple[int, int]] = [(5, 20), (20, 50), (50, 200)],
    ma_type: str = 'sma'
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Generate signals based on multiple MA crossover combinations.

    Args:
        stock_data: Single DataFrame or list of DataFrames with stock price data
        ma_pairs: List of tuples containing (short_period, long_period) combinations
        ma_type: Type of moving average - 'sma' or 'ema' (default: 'sma')

    Returns:
        DataFrame or list of DataFrames with added MA and signal columns
    """

    def calculate_multi_signals(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Initialize combined signal column
        df['Combined_Signal'] = 0

        # Calculate signals for each MA pair
        for short_period, long_period in ma_pairs:
            # Get signals for this MA pair
            signals_df = generate_ma_signals(
                df, short_period, long_period, ma_type)

            # Add individual signal column
            signal_col = f'Signal_{short_period}_{long_period}'
            df[signal_col] = signals_df['Signal']

            # Add to combined signal (weighted by MA period difference)
            weight = 1 / (long_period - short_period)
            df['Combined_Signal'] += df[signal_col] * weight

        # Normalize combined signals
        df['Combined_Signal'] = np.sign(df['Combined_Signal'])

        return df

    # Handle both single DataFrame and list of DataFrames
    if isinstance(stock_data, pd.DataFrame):
        return calculate_multi_signals(stock_data)
    else:
        return [calculate_multi_signals(df) for df in stock_data]
