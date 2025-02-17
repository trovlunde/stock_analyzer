from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd


def get_cached_stock_data(market):
    """
    Check for and return cached stock data if it exists and is recent

    Args:
        market (str): Market identifier (e.g., 'sp500')
    Returns:
        DataFrame or None: Cached data if valid, None if needs refresh
    """
    # Create data directory if it doesn't exist
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    # Check for cached file
    cache_file = data_dir / f"{market}_stock_data.csv"
    if not cache_file.exists():
        return None

    # Check file age
    file_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
    if datetime.now() - file_time > timedelta(hours=12):
        return None

    try:
        # Load cached data
        return pd.read_csv(cache_file)
    except Exception:
        return None


def save_stock_data(stocks_df, market):
    """
    Save stock data to cache file

    Args:
        stocks_df (DataFrame): Stock data to cache
        market (str): Market identifier
    """
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)

    cache_file = data_dir / f"{market}_stock_data.csv"
    stocks_df.to_csv(cache_file, index=False)
