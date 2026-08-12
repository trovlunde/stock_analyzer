import pandas as pd
from tqdm import tqdm
import numpy as np
from requests_ratelimiter import LimiterSession
import time

from stock_analysis.market_data import MarketDataProvider, YFinanceProvider

_default_provider: MarketDataProvider = YFinanceProvider()


def fetch_stock_data(tickers, start_date=None, end_date=None, period="1mo", retry_count=3, provider=None):
    """
    Fetch stock data and events from Yahoo Finance with retry logic

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        period (str, optional): Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
        retry_count (int): Number of retry attempts if data fetch fails

    Returns:
        tuple: (DataFrame with stock data, DataFrame with events)
    """
    _provider = provider or _default_provider
    for attempt in range(retry_count):
        try:
            stock = _provider.get_raw_ticker(tickers)

            # Fetch price data with retry logic
            if start_date and end_date:
                df = stock.history(start=start_date, end=end_date)
            else:
                df = stock.history(period=period)

            # If empty, try alternative methods
            if df.empty:
                if attempt < retry_count - 1:
                    # Try with a longer period as fallback
                    if period and period != "max":
                        print(f"Retrying {tickers} with longer period...")
                        time.sleep(1)  # Brief delay to avoid rate limiting
                        df = stock.history(period="max")
                        if not df.empty:
                            # Filter to requested period if possible
                            if period in ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y"]:
                                days_map = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365,
                                            "2y": 730, "5y": 1825, "10y": 3650}
                                days = days_map.get(period, 365)
                                df = df.tail(days)

                    # If still empty, wait and retry
                    if df.empty:
                        wait_time = 2 ** attempt
                        print(
                            f"No data on attempt {attempt + 1}, retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                else:
                    # Last attempt: try to get info to verify ticker is valid
                    try:
                        info = stock.info
                        if info and 'symbol' in info:
                            print(
                                f"Warning: No price data available for {tickers} but ticker appears valid.")
                            print(
                                "This may be a temporary API issue. Try again later or use a different period.")
                        else:
                            print(
                                f"No price data found for {tickers}. Ticker may be invalid or delisted.")
                    except:
                        print(f"No price data found for {tickers}")
                    return None, None

            # Fetch events
            events = pd.DataFrame()

            # Get dividends
            try:
                dividends = stock.dividends
                if dividends is not None and not dividends.empty:
                    div_events = pd.DataFrame({
                        'Event': 'Dividend Payment',
                        'Details': dividends.apply(lambda x: f'${x:.2f}')
                    }, index=dividends.index)
                    events = pd.concat([events, div_events])
            except Exception as e:
                pass  # Silently skip if dividends can't be fetched

            # Get earnings dates
            try:
                earnings = stock.earnings_dates
                if earnings is not None and not earnings.empty:
                    earnings_events = pd.DataFrame({
                        'Event': 'Earnings',
                        'Details': earnings.apply(lambda row: f"EPS Est: ${row['EPS Estimate']:.2f}, Actual: ${row['Reported EPS']:.2f}"
                                                  if pd.notnull(row['Reported EPS'])
                                                  else f"EPS Est: ${row['EPS Estimate']:.2f}", axis=1)
                    }, index=earnings.index)
                    events = pd.concat([events, earnings_events])
            except Exception as e:
                pass  # Silently skip if earnings can't be fetched

            # Sort events by date
            if not events.empty:
                events = events.sort_index()

            return df, events

        except Exception as e:
            if attempt < retry_count - 1:
                wait_time = 2 ** attempt
                print(
                    f"Error fetching data for {tickers} (attempt {attempt + 1}/{retry_count}): {str(e)}")
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                print(
                    f"Error fetching data for {tickers} after {retry_count} attempts: {str(e)}")
                return None, None

    return None, None


def fetch_stocks_data(tickers, provider=None):
    """
    Fetch stock data for a list of tickers

    Args:
        tickers (list): List of stock tickers
    Returns:
        DataFrame: Stock data
    """
    _provider = provider or _default_provider
    stocks = []
    print("Fetching stock data...")

    # Create Tickers object
    tickers_obj = _provider.get_tickers_obj(tickers)

    for symbol, ticker in tqdm(tickers_obj.tickers.items()):
        try:
            info = ticker.info

            # Skip if we can't get basic info
            if not info or 'marketCap' not in info:
                continue

            stock_data = {
                'ticker': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'forward_pe': info.get('forwardPE'),
                'trailing_pe': info.get('trailingPE'),
                'price_to_book': info.get('priceToBook'),
                'profit_margins': info.get('profitMargins'),
                'operating_margins': info.get('operatingMargins')
            }

            stocks.append(stock_data)

        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            continue

    return pd.DataFrame(stocks).replace([np.inf, -np.inf], np.nan)


def fetch_stocks_data_alt(tickers, provider=None):
    """
    Fetch and clean stock data for a list of tickers

    Args:
        tickers (list): List of stock tickers
    Returns:
        DataFrame: Clean stock data with numeric values and no infinities
    """
    _provider = provider or _default_provider
    stocks = []
    print("Fetching stock data...")

    # Create Tickers object
    tickers_obj = _provider.get_tickers_obj(tickers)

    # Define numeric fields and their valid ranges
    numeric_fields = {
        'market_cap': (0, None),  # Must be positive, no upper limit
        'forward_pe': (0, 1000),  # Reasonable P/E range
        'trailing_pe': (0, 1000),
        'price_to_book': (0, 100),
        'price_to_sales': (0, 100),
        'profit_margins': (-1, 1),  # -100% to 100%
        'operating_margins': (-1, 1)
    }

    for symbol, ticker in tqdm(tickers_obj.tickers.items()):
        try:
            info = ticker.info

            # Skip if we can't get basic info
            if not info or 'marketCap' not in info:
                continue

            stock_data = {
                'ticker': symbol,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A')
            }

            # Clean and validate numeric fields
            for field, (min_val, max_val) in numeric_fields.items():
                value = info.get(field)

                # Convert to float and validate
                try:
                    value = float(value)

                    # Check if value is infinite
                    if np.isinf(value):
                        value = None
                    # Check if value is within valid range
                    elif min_val is not None and value < min_val:
                        value = None
                    elif max_val is not None and value > max_val:
                        value = None
                except (TypeError, ValueError):
                    value = None

                stock_data[field] = value

            stocks.append(stock_data)

        except Exception as e:
            print(f"Error fetching {symbol}: {str(e)}")
            continue

    df = pd.DataFrame(stocks)

    # Remove rows where all numeric fields are None
    numeric_columns = list(numeric_fields.keys())
    df = df.dropna(subset=numeric_columns, how='all')

    # Replace inf values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Sort by market cap (descending)
    if not df.empty:
        df = df.sort_values('market_cap', ascending=False)

    return df
