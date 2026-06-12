import os
import time
import pandas as pd
import numpy as np
from ..helpers import RSI, get_features
import pandas_ta as ta


def prepare_classification_data(stock_data, predict_weekly=False, threshold=0.01, use_extra_features=False):
    """Prepare data for three-class classification."""
    # Initialize with full date range
    data = pd.DataFrame(index=stock_data.index)

    # Target: today's or next week's return (only needed for training)
    if predict_weekly:
        data['return'] = stock_data['Close'].pct_change(
            periods=5, fill_method=None).shift(-4)  # Next week's return
    else:
        data['return'] = stock_data['Close'].pct_change(
            fill_method=None).shift(-1)  # Tomorrow's return

    # Base features (all must be from past data)
    data['prev_day_return'] = stock_data['Close'].pct_change(fill_method=None)  # Today's return
    data['prev_2day_return'] = stock_data['Close'].pct_change(fill_method=None).shift(
        1)  # Yesterday's return
    data['prev_week_return'] = stock_data['Close'].pct_change(
        periods=5, fill_method=None)  # Past week's return

    # Extra features
    if use_extra_features:
        # Raw volume values normalized by rolling mean
        data['volume_1d'] = stock_data['Volume'] / \
            stock_data['Volume'].rolling(21).mean()
        data['volume_5d'] = stock_data['Volume'].rolling(
            5).mean() / stock_data['Volume'].rolling(21).mean()

        # Calculate volatility using annualized standard deviation of returns
        data['volatility_5d'] = stock_data['Close'].pct_change(fill_method=None).rolling(
            5).std() * np.sqrt(252/5)
        data['volatility_21d'] = stock_data['Close'].pct_change(fill_method=None).rolling(
            21).std() * np.sqrt(252/21)

        # Moving averages relative to current price
        data['ma_5d'] = stock_data['Close'].rolling(
            5).mean() / stock_data['Close']
        data['ma_21d'] = stock_data['Close'].rolling(
            21).mean() / stock_data['Close']

        # RSI
        data['rsi_5d'] = RSI(stock_data, window=5)
        data['rsi_21d'] = RSI(stock_data, window=21)

    # Create categorical target where we have the data
    data['Target'] = pd.cut(data['return'],
                            bins=[-np.inf, -threshold, threshold, np.inf],
                            labels=['negative', 'neutral', 'positive'])

    # Get the appropriate feature list
    feature_columns = get_features(use_extra_features)

    # Fill any missing features with forward fill followed by backward fill
    data[feature_columns] = data[feature_columns].fillna(
        method='ffill').fillna(method='bfill')

    print(f"Number of samples with complete features: {len(data)}")
    print(
        f"Number of samples with complete targets: {data['Target'].notna().sum()}")

    return data


def prepare_classification_data_enhanced(stock_data, predict_weekly=False, threshold=0.01, use_extra_features=False):
    """Prepare data for three-class classification."""
    # Initialize with full date range
    data = pd.DataFrame(index=stock_data.index)

    # Target: today's or next week's return (only needed for training)
    if predict_weekly:
        data['return'] = stock_data['Close'].pct_change(
            periods=5, fill_method=None).shift(-4)  # Next week's return
    else:
        data['return'] = stock_data['Close'].pct_change(
            fill_method=None).shift(-1)  # Tomorrow's return

    # Base features (all must be from past data)
    data['prev_day_return'] = stock_data['Close'].pct_change(fill_method=None)  # Today's return
    data['prev_2day_return'] = stock_data['Close'].pct_change(fill_method=None).shift(
        1)  # Yesterday's return
    data['prev_week_return'] = stock_data['Close'].pct_change(
        periods=5, fill_method=None)  # Past week's return

    # Extra features
    if use_extra_features:
        # Raw volume values normalized by rolling mean
        data['volume_1d'] = stock_data['Volume'] / \
            stock_data['Volume'].rolling(21).mean()
        data['volume_5d'] = stock_data['Volume'].rolling(
            5).mean() / stock_data['Volume'].rolling(21).mean()

        # Calculate volatility using annualized standard deviation of returns
        data['volatility_5d'] = stock_data['Close'].pct_change(fill_method=None).rolling(
            5).std() * np.sqrt(252/5)
        data['volatility_21d'] = stock_data['Close'].pct_change(fill_method=None).rolling(
            21).std() * np.sqrt(252/21)

        # Moving averages relative to current price
        data['ma_5d'] = stock_data['Close'].rolling(
            5).mean() / stock_data['Close']
        data['ma_21d'] = stock_data['Close'].rolling(
            21).mean() / stock_data['Close']

        # RSI
        data['rsi_5d'] = RSI(stock_data, window=5)
        data['rsi_21d'] = RSI(stock_data, window=21)

        data['momentum'] = stock_data['Close'].pct_change(20, fill_method=None)
        data['bollinger_band_position'] = (stock_data['Close'] - stock_data['Close'].rolling(20).mean()) / \
            (stock_data['Close'].rolling(20).std() * 2)

        macd_result = ta.macd(stock_data['Close'])
        if isinstance(macd_result, pd.DataFrame) and len(macd_result.columns) > 0:
            histogram_col = next(
                (col for col in macd_result.columns if col.startswith('MACDh')),
                macd_result.columns[-1],
            )
            data['macd_diff'] = macd_result[histogram_col]
        else:
            data['macd_diff'] = np.nan

    # Create categorical target where we have the data
    data['Target'] = pd.cut(data['return'],
                            bins=[-np.inf, -threshold, threshold, np.inf],
                            labels=['negative', 'neutral', 'positive'])

    # Get the appropriate feature list
    feature_columns = get_features(use_extra_features)

    # Fill any missing features with forward fill followed by backward fill
    data[feature_columns] = data[feature_columns].fillna(
        method='ffill').fillna(method='bfill')

    print(f"Number of samples with complete features: {len(data)}")
    print(
        f"Number of samples with complete targets: {data['Target'].notna().sum()}")

    return data


def prepare_classification_data_iterative(stock_data, predict_weekly=False, threshold=0.01, use_extra_features=False, last_n_months=6):
    """Prepare data for three-class classification using a sliding window approach."""
    # Get the appropriate feature list first
    feature_columns = get_features(use_extra_features)

    # Get the last 6 months of data
    last_date = stock_data.index[-1]
    start_date = last_date - pd.DateOffset(months=last_n_months)
    recent_data = stock_data[start_date:]

    # Initialize final dataframe with all expected columns
    final_data = pd.DataFrame(columns=feature_columns)

    # Sliding window size (30 days)
    window_size = 30

    # Iterate through each day in the last 6 months
    for end_idx in range(len(recent_data)):
        if end_idx < window_size:  # Skip until we have enough historical data
            continue

        # Get the window of data
        window_end = recent_data.index[end_idx]
        window_start = recent_data.index[end_idx - window_size]
        window_data = stock_data[window_start:window_end]

        # Create temporary dataframe for this window with all expected columns
        temp_data = pd.DataFrame(index=[window_end], columns=feature_columns)

        # Calculate base features (always included)
        temp_data['prev_day_return'] = window_data['Close'].pct_change(fill_method=None).iloc[-1]
        temp_data['prev_2day_return'] = window_data['Close'].pct_change(fill_method=None).shift(
            1).iloc[-1]
        temp_data['prev_week_return'] = window_data['Close'].pct_change(
            periods=5, fill_method=None).iloc[-1]

        if use_extra_features:
            # Volume features
            temp_data['volume_1d'] = (window_data['Volume'].iloc[-1] /
                                      window_data['Volume'].rolling(21).mean().iloc[-1])
            temp_data['volume_5d'] = (window_data['Volume'].rolling(5).mean().iloc[-1] /
                                      window_data['Volume'].rolling(21).mean().iloc[-1])

            # Volatility features
            temp_data['volatility_5d'] = (window_data['Close'].pct_change(fill_method=None).rolling(5).std().iloc[-1] *
                                          np.sqrt(252/5))
            temp_data['volatility_21d'] = (window_data['Close'].pct_change(fill_method=None).rolling(21).std().iloc[-1] *
                                           np.sqrt(252/21))

            # Moving averages
            temp_data['ma_5d'] = (window_data['Close'].rolling(5).mean().iloc[-1] /
                                  window_data['Close'].iloc[-1])
            temp_data['ma_21d'] = (window_data['Close'].rolling(21).mean().iloc[-1] /
                                   window_data['Close'].iloc[-1])

            # RSI (using the erroneous version)
            temp_data['rsi_5d'] = RSI_erroneous(window_data, window=5).iloc[-1]
            temp_data['rsi_21d'] = RSI_erroneous(
                window_data, window=21).iloc[-1]

        # Add to final dataset
        final_data = pd.concat([final_data, temp_data])

    # Calculate returns for target (using full dataset to get actual returns)
    if predict_weekly:
        final_data['return'] = stock_data['Close'].pct_change(
            periods=5, fill_method=None).shift(-4)
    else:
        final_data['return'] = stock_data['Close'].pct_change(fill_method=None).shift(-1)

    # Create categorical target
    final_data['Target'] = pd.cut(final_data['return'],
                                  bins=[-np.inf, -threshold,
                                        threshold, np.inf],
                                  labels=['negative', 'neutral', 'positive'])

    # Get the appropriate feature list
    feature_columns = get_features(use_extra_features)

    # Fill any missing features
    final_data[feature_columns] = final_data[feature_columns].fillna(
        method='ffill').fillna(method='bfill')

    print(f"Number of samples with complete features: {len(final_data)}")
    print(
        f"Number of samples with complete targets: {final_data['Target'].notna().sum()}")

    return final_data


def prepare_classification_data_cache(stock_data, predict_weekly=False, threshold=0.01, use_extra_features=False):
    """Prepare classification data with file-based caching."""

    # Create cache directory in the project's data folder
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(__file__)))), 'data', 'technical_classifier_cache')
    os.makedirs(cache_dir, exist_ok=True)

    # Create cache filename based on stock symbol and parameters
    symbol = stock_data.index.name or 'unknown'
    cache_file = os.path.join(
        cache_dir, f"{symbol}_{predict_weekly}_{threshold}_data.csv")

    # Check if cache exists and is recent (less than 24 hours old)
    if os.path.exists(cache_file):
        if (time.time() - os.path.getmtime(cache_file)) < 24*60*60:  # 24 hours
            data = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            print(f"Using cached data for {symbol}")
            return data

    try:
        # If no cache or cache is old, calculate the data
        data = prepare_classification_data(
            stock_data, predict_weekly, threshold, use_extra_features)

        # Save to cache
        data.to_csv(cache_file)
        print(f"Cached data for {symbol}")

        return data
    except Exception as e:
        print(f"Error processing {symbol}: {str(e)}")
        return None
