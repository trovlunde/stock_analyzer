import yfinance as yf
import pandas as pd
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import seaborn as sns


def find_stocks_by_method(method='high_dividend', stockTickers=["AAPL", "MSFT", "GOOGL", "AMZN", "FB", "TSLA", "NVDA", "AMD", "INTC", "QCOM"]):
    """
    Find stocks based on specified method

    Args:
        method (str): Method to use ('high_dividend', 'undervalued', 'growth')
        min_yield (float): Minimum dividend yield for dividend method
        min_market_cap (float): Minimum market cap in dollars
        stockTickers (list): List of stock tickers to search

    Returns:
        list: List of dictionaries containing stock info
    """
    print(f"Fetching {stockTickers} stocks...")
    stocks = yf.tickers(stockTickers)
    filtered_stocks = []
    errors = []

    print(f"Analyzing stocks using {method} method...")

    # Method-specific filters and data
    if method == 'high_dividend':
        dividendStocks = find_high_dividend_stocks(stocks=stocks)

    elif method == 'undervalued':
        undervaluedStocks = find_undervalued_stocks(stocks=stocks)

    elif method == 'growth':
        growthStocks = find_growth_stocks(stocks=stocks)

    # Print summary
    print(f"\nFound {len(filtered_stocks)} matching stocks")
    if errors:
        print(f"Encountered {len(errors)} errors")

    return filtered_stocks


def display_stocks(stocks, method='high_dividend'):
    """
    Display the filtered stocks in a formatted table
    """
    if not stocks:
        print("No matching stocks found.")
        return

    # Create DataFrame
    df = pd.DataFrame(stocks)

    # Format columns based on method
    df['market_cap'] = df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B")
    df['price'] = df['price'].apply(lambda x: f"${x:.2f}")

    if method == 'high_dividend':
        df['dividend_yield'] = df['dividend_yield'].apply(lambda x: f"{x:.2%}")
        df['dividend_rate'] = df['dividend_rate'].apply(lambda x: f"${x:.2f}")
        columns = ['ticker', 'company_name', 'sector', 'dividend_yield',
                   'dividend_rate', 'price', 'market_cap', 'payout_ratio',
                   'beta', 'forward_pe', 'trailing_pe']
        column_names = ['Ticker', 'Company', 'Sector', 'Yield', 'Div Rate',
                        'Price', 'Market Cap', 'Payout Ratio', 'Beta',
                        'Forward P/E', 'Trailing P/E']

    elif method == 'undervalued':
        columns = ['ticker', 'company_name', 'sector', 'price', 'market_cap',
                   'forward_pe', 'price_to_book', 'debt_to_equity', 'beta']
        column_names = ['Ticker', 'Company', 'Sector', 'Price', 'Market Cap',
                        'Forward P/E', 'P/B', 'Debt/Equity', 'Beta']

    elif method == 'growth':
        df['revenue_growth'] = df['revenue_growth'].apply(
            lambda x: f"{x:.2%}" if isinstance(x, (int, float)) else x)
        columns = ['ticker', 'company_name', 'sector', 'price', 'market_cap',
                   'revenue_growth', 'peg_ratio', 'analyst_target_price']
        column_names = ['Ticker', 'Company', 'Sector', 'Price', 'Market Cap',
                        'Rev Growth', 'PEG', 'Target Price']

    # Reorder and rename columns
    df = df[columns]
    df.columns = column_names

    # Display the table
    print(f"\n{method.title()} Stocks:")
    print(df.to_string(index=False))

    # Save to CSV
    filename = f"{method}_stocks_{time.strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

    return df

# Keep the existing functions for backward compatibility


def find_high_dividend_stocks(min_yield=0.03, min_market_cap=1e9, stocks=[]):
    """
    Find stocks with high dividend yields
    """
    filtered_stocks = []
    errors = []

    print("Analyzing stocks for high dividends...")
    for ticker, stock in tqdm(stocks.tickers.items()):
        try:
            info = stock.info

            # Basic filters
            current_price = info.get('currentPrice')
            market_cap = info.get('marketCap')
            dividend_rate = info.get('dividendRate', 0)

            if not all([current_price, market_cap]):
                continue

            if market_cap < min_market_cap:
                continue

            if dividend_rate == 0:
                continue

            dividend_yield = dividend_rate / current_price
            if dividend_yield < min_yield:
                continue

            stock_data = {
                'ticker': ticker,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'price': current_price,
                'market_cap': market_cap,
                'dividend_yield': dividend_yield,
                'dividend_rate': dividend_rate,
                'payout_ratio': info.get('payoutRatio', 'N/A'),
                'beta': info.get('beta', 'N/A'),
                'forward_pe': info.get('forwardPE', 'N/A'),
                'trailing_pe': info.get('trailingPE', 'N/A')
            }

            filtered_stocks.append(stock_data)

        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")
            continue

    # Sort by dividend yield
    filtered_stocks.sort(key=lambda x: x['dividend_yield'], reverse=True)

    print(f"\nFound {len(filtered_stocks)} high dividend stocks")
    if errors:
        print(f"Encountered {len(errors)} errors")

    return filtered_stocks


def find_growth_stocks(min_market_cap=1e9, marketTicker='^GSPC', growth_threshold=0.2):
    """
    Find growth stocks based on revenue and earnings growth
    """
    print(f"Fetching {marketTicker.upper()} stocks...")
    stocks = yf.tickers(market=marketTicker.upper())
    filtered_stocks = []
    errors = []

    print("Analyzing stocks for growth potential...")
    for ticker, stock in tqdm(stocks.tickers.items()):
        try:
            info = stock.info

            # Basic filters
            current_price = info.get('currentPrice')
            market_cap = info.get('marketCap')
            revenue_growth = info.get('revenueGrowth', 0)
            earnings_growth = info.get('earningsGrowth', 0)

            if not all([current_price, market_cap]):
                continue

            if market_cap < min_market_cap:
                continue

            if revenue_growth < growth_threshold or earnings_growth < growth_threshold:
                continue

            stock_data = {
                'ticker': ticker,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'price': current_price,
                'market_cap': market_cap,
                'revenue_growth': revenue_growth,
                'earnings_growth': earnings_growth,
                'peg_ratio': info.get('pegRatio', 'N/A'),
                'forward_pe': info.get('forwardPE', 'N/A'),
                'analyst_target_price': info.get('targetMeanPrice', 'N/A')
            }

            filtered_stocks.append(stock_data)

        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")
            continue

    # Sort by revenue growth
    filtered_stocks.sort(key=lambda x: x['revenue_growth'], reverse=True)

    print(f"\nFound {len(filtered_stocks)} growth stocks")
    if errors:
        print(f"Encountered {len(errors)} errors")

    return filtered_stocks


def find_undervalued_stocks(min_market_cap=1e9, stocks=[], percentile_threshold=25):
    """
    Find undervalued stocks by comparing their metrics within their sectors

    Args:
        min_market_cap (float): Minimum market cap in dollars
        market (str): Market to search ('sp500', 'nasdaq', etc.)
        percentile_threshold (int): Percentile threshold for considering a stock undervalued (lower = stricter)

    Returns:
        list: List of dictionaries containing undervalued stock info
    """
    print(f"Fetching {stocks} stocks...")
    all_stocks = []
    errors = []

    # First pass: collect all valid stock data
    for ticker, stock in tqdm(stocks.tickers.items(), desc="Collecting stock data"):
        try:
            info = stock.info

            # Basic filters
            current_price = info.get('currentPrice')
            market_cap = info.get('marketCap')
            sector = info.get('sector')

            if not all([current_price, market_cap, sector]):
                continue

            if market_cap < min_market_cap:
                continue

            stock_data = {
                'ticker': ticker,
                'company_name': info.get('longName', 'N/A'),
                'sector': sector,
                'price': current_price,
                'market_cap': market_cap,
                'forward_pe': info.get('forwardPE'),
                'trailing_pe': info.get('trailingPE'),
                'price_to_book': info.get('priceToBook'),
                'enterprise_value': info.get('enterpriseValue'),
                'debt_to_equity': info.get('debtToEquity'),
                'profit_margins': info.get('profitMargins'),
                'operating_margins': info.get('operatingMargins'),
                'revenue_growth': info.get('revenueGrowth')
            }

            all_stocks.append(stock_data)

        except Exception as e:
            errors.append(f"{ticker}: {str(e)}")
            continue

    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_stocks)

    # Group by sector and calculate percentiles
    undervalued_stocks = []

    for sector in df['sector'].unique():
        sector_df = df[df['sector'] == sector].copy()

        if len(sector_df) < 4:  # Skip sectors with too few companies
            continue

        # Calculate percentiles for key metrics within sector
        sector_df['forward_pe_percentile'] = sector_df['forward_pe'].rank(
            pct=True) * 100
        sector_df['trailing_pe_percentile'] = sector_df['trailing_pe'].rank(
            pct=True) * 100
        sector_df['price_to_book_percentile'] = sector_df['price_to_book'].rank(
            pct=True) * 100

        # Calculate sector averages for comparison
        sector_averages = {
            'avg_forward_pe': sector_df['forward_pe'].mean(),
            'avg_trailing_pe': sector_df['trailing_pe'].mean(),
            'avg_price_to_book': sector_df['price_to_book'].mean()
        }

        # Find potentially undervalued stocks
        undervalued = sector_df[
            (sector_df['forward_pe_percentile'] <= percentile_threshold) &
            (sector_df['trailing_pe_percentile'] <= percentile_threshold) &
            (sector_df['price_to_book_percentile'] <= percentile_threshold)
        ]

        # Add sector averages and percentile ranks to each stock
        for _, stock in undervalued.iterrows():
            stock_data = stock.to_dict()
            stock_data.update({
                'sector_avg_forward_pe': sector_averages['avg_forward_pe'],
                'sector_avg_trailing_pe': sector_averages['avg_trailing_pe'],
                'sector_avg_price_to_book': sector_averages['avg_price_to_book'],
                'forward_pe_percentile': stock['forward_pe_percentile'],
                'trailing_pe_percentile': stock['trailing_pe_percentile'],
                'price_to_book_percentile': stock['price_to_book_percentile']
            })
            undervalued_stocks.append(stock_data)

    # Sort by average percentile rank
    undervalued_stocks.sort(key=lambda x: (
        x['forward_pe_percentile'] +
        x['trailing_pe_percentile'] +
        x['price_to_book_percentile']
    ) / 3)

    # Print summary
    print(f"\nFound {len(undervalued_stocks)} potentially undervalued stocks")
    if errors:
        print(f"Encountered {len(errors)} errors")

    return undervalued_stocks


def display_high_dividend_stocks(stocks):
    """
    Display high dividend stocks in a formatted table
    """
    if not stocks:
        print("No high dividend stocks found.")
        return

    df = pd.DataFrame(stocks)

    # Format columns
    df['market_cap'] = df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B")
    df['price'] = df['price'].apply(lambda x: f"${x:.2f}")
    df['dividend_yield'] = df['dividend_yield'].apply(lambda x: f"{x:.2%}")
    df['dividend_rate'] = df['dividend_rate'].apply(lambda x: f"${x:.2f}")

    columns = ['ticker', 'company_name', 'sector', 'dividend_yield',
               'dividend_rate', 'price', 'market_cap', 'payout_ratio',
               'beta', 'forward_pe', 'trailing_pe']

    column_names = ['Ticker', 'Company', 'Sector', 'Yield', 'Div Rate',
                    'Price', 'Market Cap', 'Payout Ratio', 'Beta',
                    'Forward P/E', 'Trailing P/E']

    df = df[columns]
    df.columns = column_names

    print("\nHigh Dividend Stocks:")
    print(df.to_string(index=False))

    filename = f"high_dividend_stocks_{time.strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

    return df


def display_growth_stocks(stocks):
    """
    Display growth stocks in a formatted table
    """
    if not stocks:
        print("No growth stocks found.")
        return

    df = pd.DataFrame(stocks)

    # Format columns
    df['market_cap'] = df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B")
    df['price'] = df['price'].apply(lambda x: f"${x:.2f}")
    df['revenue_growth'] = df['revenue_growth'].apply(lambda x: f"{x:.2%}")
    df['earnings_growth'] = df['earnings_growth'].apply(lambda x: f"{x:.2%}")

    columns = ['ticker', 'company_name', 'sector', 'price', 'market_cap',
               'revenue_growth', 'earnings_growth', 'peg_ratio',
               'forward_pe', 'analyst_target_price']

    column_names = ['Ticker', 'Company', 'Sector', 'Price', 'Market Cap',
                    'Rev Growth', 'Earn Growth', 'PEG', 'Forward P/E',
                    'Target Price']

    df = df[columns]
    df.columns = column_names

    print("\nGrowth Stocks:")
    print(df.to_string(index=False))

    filename = f"growth_stocks_{time.strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

    return df


def display_undervalued_stocks(stocks):
    """
    Display the undervalued stocks in a formatted table
    """
    if not stocks:
        print("No undervalued stocks found.")
        return

    # Create DataFrame
    df = pd.DataFrame(stocks)

    # Format columns
    df['market_cap'] = df['market_cap'].apply(lambda x: f"${x/1e9:.1f}B")
    df['price'] = df['price'].apply(lambda x: f"${x:.2f}")
    df['forward_pe'] = df['forward_pe'].apply(lambda x: f"{x:.1f}")
    df['trailing_pe'] = df['trailing_pe'].apply(lambda x: f"{x:.1f}")
    df['price_to_book'] = df['price_to_book'].apply(lambda x: f"{x:.2f}")

    # Add comparison columns
    df['PE_vs_Sector'] = df.apply(
        lambda x: f"{float(x['forward_pe']):.1f} vs {x['sector_avg_forward_pe']:.1f}", axis=1)
    df['PB_vs_Sector'] = df.apply(
        lambda x: f"{float(x['price_to_book']):.2f} vs {x['sector_avg_price_to_book']:.2f}", axis=1)

    # Select and rename columns for display
    columns = ['ticker', 'company_name', 'sector', 'price', 'market_cap',
               'PE_vs_Sector', 'PB_vs_Sector', 'forward_pe_percentile',
               'price_to_book_percentile']

    column_names = ['Ticker', 'Company', 'Sector', 'Price', 'Market Cap',
                    'PE (Stock vs Sector)', 'P/B (Stock vs Sector)',
                    'PE %ile', 'P/B %ile']

    df = df[columns]
    df.columns = column_names

    # Display the table
    print("\nUndervalued Stocks:")
    print(df.to_string(index=False))

    # Save to CSV
    filename = f"undervalued_stocks_{time.strftime('%Y%m%d')}.csv"
    df.to_csv(filename, index=False)
    print(f"\nResults saved to {filename}")

    return df


def find_undervalued_sectors(marketTicker='^GSPC'):
    stocks = yf.Tickers(market=marketTicker)
    info = stocks.info
    print(info)


# Example usage:
# summary_df, full_data = analyzeSectorMetrics()
# displaySectorComparison(full_data, 'forward_pe')



