import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import time
import os
import json


def get_cache_path():
    """Returns the path to the cache file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    return os.path.join(data_dir, 'stock_returns_cache.parquet')


def load_cache():
    """Load cached stock returns data if it exists"""
    cache_path = get_cache_path()
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        df['timestamp'] = pd.to_datetime(
            df['timestamp'])  # Convert to datetime
        return df
    return pd.DataFrame()


def save_cache(df):
    """Save stock returns data to cache"""
    cache_path = get_cache_path()
    df.to_parquet(cache_path)


def fetch_daily_returns(ticker, signal_date, current_date):
    """
    Fetch detailed daily returns for visualization.
    Returns both the standard format and detailed daily data.
    """
    time.sleep(0.1)
    print(f"\nProcessing {ticker} (signal date: {signal_date})", end='')

    # Adjust signal date to previous business day if it's a weekend
    # If signal is on Saturday or Sunday, use Friday's data
    original_signal_date = signal_date

    if signal_date.weekday() == 0:  # 5 = Saturday, 6 = Sunday
        days_to_subtract = signal_date.weekday() + 3  # 4 = Friday
        signal_date = signal_date - timedelta(days=days_to_subtract)
        print(f" (adjusted to {signal_date})", end='')

    # Fetch from day before signal to capture both the signal day and future returns
    start_date = signal_date - timedelta(days=45)
    # Use original date for end range
    end_date = original_signal_date + timedelta(days=10)

    stock = yf.Ticker(ticker)
    hist = stock.history(
        start=start_date,
        end=min(end_date, current_date.date() + timedelta(days=1)),
        interval='1d'
    )

    if hist.empty:
        print(" ✗ (No data available)")
        return None, None

    print(signal_date)

    # Find signal day data to get the open price
    # For Monday signals after weekend, look for Friday's data
    # Convert index to date for comparison
    signal_day_data = hist[hist.index.date == signal_date]

    if len(signal_day_data) == 0:
        print(" ✗ (No signal day data)")
        return None, None

    # Use signal day's open as reference price
    reference_price = signal_day_data['Open'].iloc[0]

    # Calculate standard returns (for compatibility)
    next_day_return = None
    next_week_return = None
    next_day_high_return = None
    next_week_high_return = None

    # Calculate returns starting from signal day
    signal_day_returns = hist[hist.index.date >= signal_date].copy()

    if len(signal_day_returns) > 0:
        # Day 1 returns (same day)
        next_day_return = (
            signal_day_data['Close'].iloc[0] - reference_price) / reference_price
        next_day_high_return = (
            signal_day_data['High'].iloc[0] - reference_price) / reference_price

        if len(signal_day_returns) >= 5:
            next_week_return = (
                signal_day_returns['Close'].iloc[4] - reference_price) / reference_price
            next_week_high_return = (
                signal_day_returns['High'].iloc[4] - reference_price) / reference_price

    standard_returns = {
        'Ticker': ticker,
        'timestamp': original_signal_date,
        'next_day_return': next_day_return,
        'next_week_return': next_week_return,
        'next_day_high_return': next_day_high_return,
        'next_week_high_return': next_week_high_return,
        'data_start_date': hist.index[0].date(),
        'data_end_date': hist.index[-1].date()
    }

    # Calculate detailed daily returns
    daily_returns = {}

    for idx, row in signal_day_returns.iterrows():
        days_after_signal = (idx.date() - signal_date).days
        if 0 <= days_after_signal <= 5:  # Only include up to 5 days after signal
            daily_returns[days_after_signal] = {
                'close_return': (row['Close'] - reference_price) / reference_price,
                'high_return': (row['High'] - reference_price) / reference_price,
                'date': idx.date()
            }

    detailed_returns = {
        'Ticker': ticker,
        'signal_date': original_signal_date,
        'reference_price': reference_price,
        'daily_returns': daily_returns
    }

    print(" ✓")
    return standard_returns, detailed_returns


def fetch_stock_returns(df):
    """Fetch future returns for each stock signal"""
    current_date = datetime.now()
    print("\nFetching stock returns...")
    print(f"Current date: {current_date}")
    print(
        f"Date range in data: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Load cached data
    cached_data = load_cache()
    returns_data = []
    detailed_returns_data = []
    total_stocks = len(df)
    successful_fetches = 0

    for idx, row in df.iterrows():
        ticker = row['Ticker']
        signal_date = pd.to_datetime(row['timestamp']).tz_localize(
            None)  # Convert to timezone-naive datetime

        # Check if data exists in cache first
        cached_entry = None
        if not cached_data.empty:
            cached_entry = cached_data[
                (cached_data['Ticker'] == ticker) &
                (cached_data['timestamp'] == signal_date)
            ]

        if cached_entry is not None and not cached_entry.empty:
            returns_data.append(cached_entry.iloc[0].to_dict())
            successful_fetches += 1
            print(f"\n{ticker} found in cache ✓")
            continue

        try:
            standard_returns, detailed_returns = fetch_daily_returns(
                ticker, signal_date.date(), current_date)
            if standard_returns:
                # Ensure consistent datetime types
                standard_returns['timestamp'] = pd.to_datetime(
                    standard_returns['timestamp'])
                standard_returns['data_start_date'] = pd.to_datetime(
                    standard_returns['data_start_date'])
                standard_returns['data_end_date'] = pd.to_datetime(
                    standard_returns['data_end_date'])
                returns_data.append(standard_returns)
                if detailed_returns:
                    detailed_returns_data.append(detailed_returns)
                successful_fetches += 1
        except Exception as e:
            print(f" ✗ (Error: {str(e)})")
            continue

    # Create DataFrames
    returns_df = pd.DataFrame(returns_data)
    detailed_returns_df = pd.DataFrame(detailed_returns_data)

    # Update cache with new data
    if not returns_df.empty:
        # Ensure all datetime columns are timezone-naive
        for col in ['timestamp', 'data_start_date', 'data_end_date']:
            if col in returns_df.columns:
                returns_df[col] = pd.to_datetime(
                    returns_df[col]).dt.tz_localize(None)

        if cached_data.empty:
            save_cache(returns_df)
        else:
            # Ensure cached data has consistent types
            for col in ['timestamp', 'data_start_date', 'data_end_date']:
                if col in cached_data.columns:
                    cached_data[col] = pd.to_datetime(
                        cached_data[col]).dt.tz_localize(None)

            combined_cache = pd.concat([cached_data, returns_df]).drop_duplicates(
                subset=['Ticker', 'timestamp'])
            save_cache(combined_cache)

    return returns_df, detailed_returns_df


def load_historical_signals():
    """Load and parse the historical signals from JSON file"""
    try:
        with open('finviz_recs/finviz_recs.json', 'r') as f:
            data = json.load(f)

        # Convert signals to DataFrame
        df = pd.DataFrame(data['signals'])

        # Convert timestamp to datetime and ensure timezone-naive
        df['timestamp'] = pd.to_datetime(
            df['timestamp'], format='ISO8601').dt.tz_localize(None)

        # Clean numeric columns with error handling
        try:
            # Convert Price - handle any non-numeric values
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

            # Convert Change - remove % and convert to float
            df['Change'] = df['Change'].astype(str)  # Ensure string type
            df['Change'] = df['Change'].str.rstrip('%').astype(float) / 100

            # Convert Volume - handle any non-numeric values
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

            # Convert Market Cap - extract numbers and convert
            df['Market Cap'] = df['Market Cap'].astype(
                str)  # Ensure string type
            df['Market Cap'] = df['Market Cap'].str.extract(
                r'(\d+\.?\d*)')[0].astype(float)

        except Exception as e:
            print(f"Error cleaning numeric columns: {str(e)}")
            print("DataFrame head:")
            print(df.head())
            print("\nDataFrame info:")
            print(df.info())
            return None

        return df
    except Exception as e:
        print(f"Error loading historical signals: {str(e)}")
        return None
