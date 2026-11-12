from typing import List, Union

import pandas as pd

from ..helpers import RSI


def generate_rsi_signals(
    stock_data: Union[pd.DataFrame, List[pd.DataFrame]],
    period: int = 14,
    overbought_threshold: float = 70,
    oversold_threshold: float = 30
) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Add RSI and level-based signal columns, for research and plotting.

    The RSI is the Wilder-smoothed one from `helpers.RSI`, the only RSI definition
    in the package. `Signal` marks every day the RSI is beyond a threshold (1 below
    oversold, -1 above overbought), so it repeats for as long as the level holds.
    For trading use the `rsi` signal strategy, which acts on crosses and has a
    trend filter.

    Args:
        stock_data: Single DataFrame or list of DataFrames with stock price data
        period: RSI calculation period (default: 14)
        overbought_threshold: RSI level above which Signal is -1 (default: 70)
        oversold_threshold: RSI level below which Signal is 1 (default: 30)

    Returns:
        DataFrame or list of DataFrames with added RSI and signal columns
    """

    def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
        # Make a copy to avoid modifying original data
        df = df.copy()

        df['RSI'] = RSI(df, window=period)

        df['Signal'] = 0  # 0: no signal, 1: buy, -1: sell
        df.loc[df['RSI'] < oversold_threshold, 'Signal'] = 1
        df.loc[df['RSI'] > overbought_threshold, 'Signal'] = -1

        return df

    # Handle both single DataFrame and list of DataFrames
    if isinstance(stock_data, pd.DataFrame):
        return calculate_signals(stock_data)
    else:
        return [calculate_signals(df) for df in stock_data]
