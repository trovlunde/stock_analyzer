import json
import os
import time
from datetime import date, datetime, timedelta

import pandas as pd
import yfinance as yf


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
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        if 'daily_returns_json' not in df.columns:
            df['daily_returns_json'] = None
        if 'reference_price' not in df.columns:
            df['reference_price'] = None
        return df
    return pd.DataFrame()


def save_cache(df):
    """Save stock returns data to cache"""
    cache_path = get_cache_path()
    df.to_parquet(cache_path)


def is_weekend(day):
    """Check if a date falls on a weekend"""
    return day.weekday() >= 5


def get_last_business_day(day):
    """Get the last business day on or before the given date"""
    while is_weekend(day):
        day = day - timedelta(days=1)
    return day


def serialize_daily_returns(daily_returns):
    """Serialize daily_returns dict for parquet storage."""
    if not daily_returns:
        return None
    payload = {}
    for day_offset, values in daily_returns.items():
        payload[str(day_offset)] = {
            'close_return': values['close_return'],
            'high_return': values['high_return'],
            'date': values['date'].isoformat()
            if isinstance(values['date'], (date, datetime))
            else values['date'],
        }
    return json.dumps(payload)


def deserialize_daily_returns(json_str):
    """Deserialize daily_returns dict from parquet storage."""
    if json_str is None or (isinstance(json_str, float) and pd.isna(json_str)):
        return None
    if not json_str:
        return None
    payload = json.loads(json_str)
    daily_returns = {}
    for day_offset, values in payload.items():
        daily_returns[int(day_offset)] = {
            'close_return': values['close_return'],
            'high_return': values['high_return'],
            'date': pd.to_datetime(values['date']).date(),
        }
    return daily_returns


def _cache_entry_usable(cached_entry, signal_date, current_date):
    """Return True when a cached row can be used without refetching."""
    if cached_entry.empty:
        return False
    row = cached_entry.iloc[0]
    if pd.isna(row.get('daily_returns_json')):
        return False
    if row['is_complete']:
        return True
    cache_age = (current_date - signal_date).days
    return cache_age <= 1


def _append_cache_hit(cached_entry, returns_data, detailed_returns_data):
    """Append standard and detailed returns from a cache row."""
    row = cached_entry.iloc[0]
    returns_data.append(row.to_dict())
    detailed_returns_data.append({
        'Ticker': row['Ticker'],
        'signal_date': row['timestamp'],
        'reference_price': row['reference_price'],
        'daily_returns': deserialize_daily_returns(row['daily_returns_json']),
    })


def fetch_ticker_history(ticker, start_date, end_date):
    """Fetch OHLCV history for a ticker over a date range."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(
            start=start_date,
            end=end_date + timedelta(days=1),
            interval='1d',
        )
    except Exception as e:
        print(f" ✗ (Error fetching {ticker}: {e})")
        return None
    if hist.empty:
        print(f" ✗ (No data available for {ticker})")
        return None
    return hist


def compute_returns_from_history(ticker, signal_date, hist):
    """
    Compute standard and detailed returns for a signal from pre-fetched history.
    Returns (standard_returns, detailed_returns, has_complete_data) or (None, None, False).
    """
    original_signal_date = signal_date
    expected_end_date = get_last_business_day(signal_date + timedelta(days=10))

    actual_end_date = hist.index[-1].date()
    has_complete_data = actual_end_date >= expected_end_date

    signal_day_data = hist[hist.index.date == signal_date]

    if len(signal_day_data) == 0:
        prev_days_data = hist[hist.index.date < signal_date].tail(1)
        if len(prev_days_data) == 0:
            return None, None, False

        reference_price = prev_days_data['Close'].iloc[0]
        signal_day_data = pd.DataFrame({
            'Open': reference_price,
            'High': reference_price,
            'Low': reference_price,
            'Close': reference_price,
        }, index=[pd.Timestamp(signal_date)])

        today_partial = hist[hist.index.date == signal_date]
        if len(today_partial) > 0:
            signal_day_data['High'] = today_partial['High'].iloc[0]
            signal_day_data['Low'] = today_partial['Low'].iloc[0]
            signal_day_data['Close'] = today_partial['Close'].iloc[0]
    else:
        reference_price = signal_day_data['Open'].iloc[0]

    next_day_return = None
    next_week_return = None
    next_day_high_return = None
    next_week_high_return = None
    next_day_low_return = None
    next_week_low_return = None

    signal_day_returns = hist[hist.index.date >= signal_date].copy()

    if len(signal_day_returns) > 0:
        next_day_return = (
            signal_day_data['Close'].iloc[0] - reference_price) / reference_price
        next_day_high_return = (
            signal_day_data['High'].iloc[0] - reference_price) / reference_price
        next_day_low_return = (
            signal_day_data['Low'].iloc[0] - reference_price) / reference_price

        if len(signal_day_returns) >= 5:
            next_week_return = (
                signal_day_returns['Close'].iloc[4] - reference_price) / reference_price
            next_week_high_return = (
                signal_day_returns['High'].iloc[4] - reference_price) / reference_price
            next_week_low_return = (
                signal_day_returns['Low'].iloc[4] - reference_price) / reference_price

    standard_returns = {
        'Ticker': ticker,
        'timestamp': original_signal_date,
        'next_day_return': next_day_return,
        'next_week_return': next_week_return,
        'next_day_high_return': next_day_high_return,
        'next_week_high_return': next_week_high_return,
        'next_day_low_return': next_day_low_return,
        'next_week_low_return': next_week_low_return,
        'data_start_date': hist.index[0].date(),
        'data_end_date': hist.index[-1].date(),
        'reference_price': reference_price,
    }

    daily_returns = {}
    for idx, row in signal_day_returns.iterrows():
        days_after_signal = (idx.date() - signal_date).days
        if 0 <= days_after_signal <= 5:
            daily_returns[days_after_signal] = {
                'close_return': (row['Close'] - reference_price) / reference_price,
                'high_return': (row['High'] - reference_price) / reference_price,
                'date': idx.date(),
            }

    detailed_returns = {
        'Ticker': ticker,
        'signal_date': original_signal_date,
        'reference_price': reference_price,
        'daily_returns': daily_returns,
    }

    return standard_returns, detailed_returns, has_complete_data


def _normalize_returns_for_cache(standard_returns, detailed_returns, has_complete_data):
    """Normalize return dicts and add cache columns."""
    standard_returns['is_complete'] = has_complete_data
    standard_returns['timestamp'] = pd.to_datetime(standard_returns['timestamp'])
    standard_returns['data_start_date'] = pd.to_datetime(
        standard_returns['data_start_date'])
    standard_returns['data_end_date'] = pd.to_datetime(
        standard_returns['data_end_date'])
    standard_returns['daily_returns_json'] = serialize_daily_returns(
        detailed_returns['daily_returns'])
    return standard_returns


def fetch_stock_returns(df):
    """Fetch future returns for each stock signal, batched by ticker."""
    current_date = datetime.now()
    print("\nFetching stock returns...")
    print(f"Current date: {current_date}")
    print(
        f"Date range in data: {df['timestamp'].min()} to {df['timestamp'].max()}")

    cached_data = load_cache()
    returns_data = []
    detailed_returns_data = []
    pending_by_ticker = {}

    for _, row in df.iterrows():
        ticker = row['Ticker']
        signal_date = pd.to_datetime(row['timestamp']).tz_localize(None)
        signal_day = signal_date.date()

        cached_entry = pd.DataFrame()
        if not cached_data.empty:
            cached_entry = cached_data[
                (cached_data['Ticker'] == ticker) &
                (cached_data['timestamp'] == signal_date)
            ]

        if _cache_entry_usable(cached_entry, signal_date, current_date):
            _append_cache_hit(cached_entry, returns_data, detailed_returns_data)
            print(f"\n{ticker} ({signal_day}) found in cache ✓")
            continue

        pending_by_ticker.setdefault(ticker, []).append(signal_day)

    if pending_by_ticker:
        print(
            f"\nFetching {sum(len(v) for v in pending_by_ticker.values())} signals "
            f"across {len(pending_by_ticker)} tickers...")

    newly_fetched = []

    for ticker, signal_dates in pending_by_ticker.items():
        time.sleep(0.1)
        min_signal = min(signal_dates)
        max_signal = max(signal_dates)
        start_date = min_signal - timedelta(days=45)
        end_date = max_signal + timedelta(days=10)

        print(f"\nFetching {ticker}: {len(signal_dates)} signal(s), "
              f"{start_date} to {end_date}", end='')

        hist = fetch_ticker_history(ticker, start_date, end_date)
        if hist is None:
            continue

        print(" ✓")

        for signal_day in signal_dates:
            print(f"  Processing {ticker} (signal date: {signal_day})", end='')
            try:
                standard_returns, detailed_returns, has_complete_data = (
                    compute_returns_from_history(ticker, signal_day, hist))
            except Exception as e:
                print(f" ✗ (Error: {e})")
                continue

            if not standard_returns:
                print(" ✗ (No signal day data)")
                continue

            standard_returns = _normalize_returns_for_cache(
                standard_returns, detailed_returns, has_complete_data)
            returns_data.append(standard_returns)
            detailed_returns_data.append(detailed_returns)
            newly_fetched.append(standard_returns)
            print(" ✓")

    returns_df = pd.DataFrame(returns_data)
    detailed_returns_df = pd.DataFrame(detailed_returns_data)

    if newly_fetched:
        new_cache_df = pd.DataFrame(newly_fetched)
        for col in ['timestamp', 'data_start_date', 'data_end_date']:
            if col in new_cache_df.columns:
                new_cache_df[col] = pd.to_datetime(
                    new_cache_df[col]).dt.tz_localize(None)

        if cached_data.empty:
            save_cache(new_cache_df)
        else:
            for col in ['timestamp', 'data_start_date', 'data_end_date']:
                if col in cached_data.columns:
                    cached_data[col] = pd.to_datetime(
                        cached_data[col]).dt.tz_localize(None)
            combined_cache = pd.concat([cached_data, new_cache_df])
            combined_cache = combined_cache.sort_values(
                'is_complete', ascending=False).drop_duplicates(
                subset=['Ticker', 'timestamp'], keep='first')
            save_cache(combined_cache)

    return returns_df, detailed_returns_df


def load_historical_signals(file_name='finviz_recs_up.json'):
    """Load and parse the historical signals from JSON file"""
    try:
        with open(f'finviz_recs/{file_name}', 'r') as f:
            data = json.load(f)

        df = pd.DataFrame(data['signals'])

        df['timestamp'] = pd.to_datetime(
            df['timestamp'], format='ISO8601').dt.tz_localize(None)

        try:
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
            df['Change'] = df['Change'].astype(str)
            df['Change'] = df['Change'].str.rstrip('%').astype(float) / 100
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
            df['Market Cap'] = df['Market Cap'].astype(str)
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
