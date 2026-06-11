from datetime import timedelta

from .storage import get_cache_store


def get_cached_stock_data(market):
    """
    Check for and return cached stock data if it exists and is recent

    Args:
        market (str): Market identifier (e.g., 'sp500')
    Returns:
        DataFrame or None: Cached data if valid, None if needs refresh
    """
    return get_cache_store().get(
        f"market:{market}",
        max_age=timedelta(hours=12),
    )


def save_stock_data(stocks_df, market):
    """
    Save stock data to cache

    Args:
        stocks_df (DataFrame): Stock data to cache
        market (str): Market identifier
    """
    get_cache_store().put(f"market:{market}", stocks_df)
