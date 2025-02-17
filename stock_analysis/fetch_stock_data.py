import yfinance as yf
import pandas as pd
from tqdm import tqdm
import numpy as np
from requests_ratelimiter import LimiterSession


def fetch_stock_data(tickers, start_date=None, end_date=None, period="1mo"):
    """
    Fetch stock data and events from Yahoo Finance

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
        start_date (str, optional): Start date in 'YYYY-MM-DD' format
        end_date (str, optional): End date in 'YYYY-MM-DD' format
        period (str, optional): Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max

    Returns:
        tuple: (DataFrame with stock data, DataFrame with events)
    """
    try:
        stock = yf.Ticker(tickers)

        # Fetch price data
        if start_date and end_date:
            df = stock.history(start=start_date, end=end_date)
        else:
            df = stock.history(period=period)

        if df.empty:
            print(f"No price data found for {tickers}")
            return None, None

        # Fetch events
        events = pd.DataFrame()

        # Get dividends
        dividends = stock.dividends
        if dividends is not None and not dividends.empty:
            div_events = pd.DataFrame({
                'Event': 'Dividend Payment',
                'Details': dividends.apply(lambda x: f'${x:.2f}')
            }, index=dividends.index)
            events = pd.concat([events, div_events])

        # # Add ex-dividend date
        # try:
        #     ex_div_date = stock.info.get('exDividendDate')
        #     if ex_div_date:
        #         ex_div_timestamp = pd.to_datetime(ex_div_date, unit='s')
        #         # Localize the timestamp to UTC
        #         ex_div_timestamp = ex_div_timestamp.tz_localize('UTC')
        #         next_div = stock.info.get('dividendRate', 'N/A')
        #         ex_div_event = pd.DataFrame({
        #             'Event': 'Ex-Dividend Date',
        #             'Details': f'Next dividend: ${next_div:.2f}' if isinstance(next_div, (int, float)) else 'Amount TBA'
        #         }, index=[ex_div_timestamp])
        #         events = pd.concat([events, ex_div_event])
        # except Exception as e:
        #     print(f"Error processing ex-dividend date: {e}")

        try:
            print(stock)
            earnings = stock.earnings_dates
            if not earnings.empty:
                earnings_events = pd.DataFrame({
                    'Event': 'Earnings',
                    'Details': earnings.apply(lambda row: f"EPS Est: ${row['EPS Estimate']:.2f}, Actual: ${row['Reported EPS']:.2f}"
                                              if pd.notnull(row['Reported EPS'])
                                              else f"EPS Est: ${row['EPS Estimate']:.2f}", axis=1)
                }, index=earnings.index)
                events = pd.concat([events, earnings_events])
        except Exception as e:
            print(f"Error processing earnings dates: {e}")

        # Sort events by date
        if not events.empty:
            events = events.sort_index()

        return df, events

    except Exception as e:
        print(f"Error fetching data: {e}")
        return None, None


def fetch_stocks_data(tickers):
    """
    Fetch stock data for a list of tickers

    Args:
        tickers (list): List of stock tickers
    Returns:
        DataFrame: Stock data
    """
    stocks = []
    print("Fetching stock data...")

    # Create Tickers object
    tickers_obj = yf.Tickers(' '.join(tickers))

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


def fetch_stocks_data_alt(tickers):
    """
    Fetch and clean stock data for a list of tickers

    Args:
        tickers (list): List of stock tickers
    Returns:
        DataFrame: Clean stock data with numeric values and no infinities
    """
    stocks = []
    print("Fetching stock data...")

    # Create Tickers object
    tickers_obj = yf.Tickers(' '.join(tickers))

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
