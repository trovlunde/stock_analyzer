from .fetch_stock_data import fetch_stock_data, fetch_stocks_data
from .display_stock_data import display_stock_data
from .analyze_stock_data import analyze_stock_data, analyze_portfolio_correlation, optimize_portfolio
from .efficient_frontier_plot import plot_efficient_frontier
from .stock_finder import find_stocks_by_method, display_stocks
from .market_indices import MarketIndices
from .cache_data import get_cached_stock_data, save_stock_data
from .sector_analysis import analyze_and_display_sector_metrics, display_sector_comparison


def get_tickers_from_finder():
    """Get tickers using one of the stock finder functions"""
    print("\nChoose stock finding method:")
    print("1. High Dividend Stocks")
    print("2. Undervalued Stocks")
    print("3. Growth Stocks")

    choice = input("> ")

    if choice == "1":
        min_yield = float(
            input("\nEnter minimum dividend yield (e.g., 0.03 for 3%): ") or "0.03")
        min_market_cap = float(
            input("Enter minimum market cap in billions (e.g., 1 for $1B): ") or "1") * 1e9
        market = input("Enter market (default: sp500): ") or "sp500"

        stocks = find_stocks_by_method(
            'high_dividend', min_yield, min_market_cap, market)
        display_stocks(stocks, method='high_dividend')

        if stocks:
            print("\nSelect tickers from the list above (comma-separated):")
            selected = input("> ").upper()
            return [ticker.strip() for ticker in selected.split(",")]

    elif choice == "2":
        min_market_cap = float(
            input("\nEnter minimum market cap in billions (e.g., 1 for $1B): ") or "1") * 1e9
        market = input("Enter market (default: sp500): ") or "sp500"

        stocks = find_stocks_by_method(
            'undervalued', min_yield=0.0, min_market_cap=min_market_cap, market=market)
        display_stocks(stocks, method='undervalued')

        if stocks:
            print("\nSelect tickers from the list above (comma-separated):")
            selected = input("> ").upper()
            return [ticker.strip() for ticker in selected.split(",")]

    elif choice == "3":
        min_market_cap = float(
            input("\nEnter minimum market cap in billions (e.g., 1 for $1B): ") or "1") * 1e9
        market = input("Enter market (default: sp500): ") or "sp500"

        stocks = find_stocks_by_method(
            'growth', min_yield=0.0, min_market_cap=min_market_cap, market=market)
        display_stocks(stocks, method='growth')

        if stocks:
            print("\nSelect tickers from the list above (comma-separated):")
            selected = input("> ").upper()
            return [ticker.strip() for ticker in selected.split(",")]

    return []


def main():
    """
    Main function to interact with stock data fetching
    """
    print("Stock Data Analysis")
    print("-" * 20)

    while True:
        # Get operation mode

        # Look at specific metrics in detail
        mode = input(
            "\nChoose mode:\n1. Single Stock Analysis\n2. Portfolio Analysis\n3. Stock Finder\n4. Sector Analysis\n5. Quit\n> ")

        if mode == '5':
            print("Exiting program...")
            break

        elif mode == '1':
            # Single stock analysis
            ticker = input("\nEnter stock ticker (e.g., AAPL): ").upper()
            use_dates = input(
                "Do you want to specify a date range? (y/n): ").lower()

            if use_dates == 'y':
                start_date = input("Enter start date (YYYY-MM-DD): ")
                end_date = input("Enter end date (YYYY-MM-DD): ")
                df, events = fetch_stock_data(ticker, start_date, end_date)
            else:
                period = input(
                    "Enter period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max) [default: 1mo]: ") or "1mo"
                df, events = fetch_stock_data(ticker, period=period)

            if df is None:
                print(f"No data found for {ticker}")
                continue

            display_stock_data(df, events, ticker)
            try:
                analysis = analyze_stock_data(df)
                print("\nStock Analysis:")
                for metric, value in analysis.items():
                    if value is not None:
                        if isinstance(value, (int, float)):
                            print(f"{metric}: {value:.4f}")
                        else:
                            print(f"{metric}: {value}")
                    else:
                        print(f"{metric}: N/A")
            except Exception as e:
                print(f"\nError analyzing stock data: {e}")

        elif mode == '2':
            # Portfolio analysis
            print("\nHow would you like to select stocks?")
            print("1. Enter tickers manually")
            print("2. Use stock finder")
            select_mode = input("> ")

            tickers = []
            if select_mode == "1":
                print("\nEnter stock tickers (one per line, empty line to finish):")
                while True:
                    ticker = input("> ").upper()
                    if not ticker:
                        break
                    tickers.append(ticker)
            elif select_mode == "2":
                tickers = get_tickers_from_finder()

            if not tickers:
                print("No tickers selected.")
                continue

            print(f"\nSelected tickers: {', '.join(tickers)}")

            # Get time period for all stocks
            use_dates = input(
                "Do you want to specify a date range? (y/n): ").lower()

            if use_dates == 'y':
                start_date = input("Enter start date (YYYY-MM-DD): ")
                end_date = input("Enter end date (YYYY-MM-DD): ")
                period = None
            else:
                period = input(
                    "Enter period (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max) [default: 1y]: ") or "1y"
                start_date = end_date = None

            # Fetch data for all stocks
            print("\nFetching data...")
            stock_data = {}
            for ticker in tickers:
                df, _ = fetch_stock_data(ticker, start_date, end_date, period)
                if df is not None:
                    stock_data[ticker] = df
                    print(f"Data fetched for {ticker}")
                else:
                    print(f"Could not fetch data for {ticker}")

            if not stock_data:
                print("No data could be fetched for any ticker.")
                continue

            # Calculate and display correlation matrix
            if len(stock_data) > 1:
                print("\nPortfolio Correlation Matrix:")
                try:
                    correlation_matrix = analyze_portfolio_correlation(
                        stock_data)
                    print(correlation_matrix)
                except Exception as e:
                    print(f"Error calculating correlation matrix: {e}")
            else:
                print("\nSkipping correlation matrix (requires at least 2 stocks)")

            # Calculate and display optimal portfolio weights
            if len(stock_data) > 1:
                print("\nOptimal Portfolio Weights:")
                try:
                    weights = optimize_portfolio(stock_data)
                    for ticker, weight in weights.items():
                        print(f"{ticker}: {weight:.2%}")
                except Exception as e:
                    print(f"Error calculating optimal weights: {e}")
                    print(
                        "Note: Portfolio optimization requires at least 2 stocks with sufficient data overlap.")
            else:
                print("\nSkipping portfolio optimization (requires at least 2 stocks)")

            # Display individual stock metrics
            print("\nIndividual Stock Metrics:")
            for ticker, df in stock_data.items():
                try:
                    analysis = analyze_stock_data(df)
                    print(f"\n{ticker}:")
                    for metric, value in analysis.items():
                        if value is not None:
                            if isinstance(value, (int, float)):
                                print(f"  {metric}: {value:.4f}")
                            else:
                                print(f"  {metric}: {value}")
                        else:
                            print(f"  {metric}: N/A")
                except Exception as e:
                    print(f"\n{ticker}: Error analyzing - {e}")

            if len(stock_data) > 1:
                try:
                    plot_efficient_frontier(stock_data)
                except Exception as e:
                    print(f"Error plotting efficient frontier: {e}")

        elif mode == '3':
            # Stock finder
            print("\nWhich stock finding method would you like to use?")
            print("1. High Dividend Stocks")
            print("2. Undervalued Stocks")
            print("3. Growth Stocks")
            select_mode = input("> ")

            if select_mode == "1":
                min_yield = float(
                    input("\nEnter minimum dividend yield (e.g., 0.03 for 3%): ") or "0.03")
                min_market_cap = float(
                    input("Enter minimum market cap in billions (e.g., 1 for $1B): ") or "1") * 1e9
                market = input("Enter market (default: sp500): ") or "sp500"

                tickers = find_stocks_by_method(
                    'high_dividend', min_yield, min_market_cap, market)
                display_stocks(tickers, method='high_dividend')

            elif select_mode == "2":
                min_market_cap = float(
                    input("\nEnter minimum market cap in billions (e.g., 1 for $1B): ") or "1") * 1e9
                market = input("Enter market (default: sp500): ") or "sp500"

                tickers = find_stocks_by_method(
                    'undervalued', min_yield=0.0, min_market_cap=min_market_cap, market=market)
                display_stocks(tickers, method='undervalued')

            elif select_mode == "3":
                min_market_cap = float(
                    input("\nEnter minimum market cap in billions (e.g., 1 for $1B): ") or "1") * 1e9
                market = input("Enter market (default: sp500): ") or "sp500"

                tickers = find_stocks_by_method(
                    'growth', min_yield=0.0, min_market_cap=min_market_cap, market=market)
                display_stocks(tickers, method='growth')

            else:
                print("Invalid choice")

        elif mode == '4':
            # Sector Analysis
            print("\nChoose market index:")
            print("1. SP500")
            print("2. FTSE100")
            print("3. DAX40")
            print("4. CAC40")
            print("5. Nikkei225")

            market_choice = input("> ")

            # Map numeric choice to market name
            market_map = {
                '1': 'sp500',
                '2': 'ftse100',
                '3': 'dax40',
                '4': 'cac40',
                '5': 'nikkei225'
            }

            market = market_map.get(market_choice)
            if not market:
                print("Invalid market choice")
                continue

            try:
                # Check for cached data
                cached_data = get_cached_stock_data(market)

                if cached_data is not None:
                    print("Using cached data...")
                    stocks_df = cached_data
                else:
                    print("Fetching fresh data...")
                    # Get tickers for selected market
                    tickers = MarketIndices.get_market_tickers(market)

                    # Fetch stock data
                    stocks_df = fetch_stocks_data(tickers)

                    # Cache the data
                    save_stock_data(stocks_df, market)

                if stocks_df.empty:
                    print("No stock data found")
                    continue

                # Analyze sectors
                summary_df, full_data = analyze_and_display_sector_metrics(
                    stocks_df)

                # Display sector comparisons
                display_sector_comparison(full_data, [
                                          'forward_pe', 'trailing_pe', 'price_to_book', 'profit_margins', 'operating_margins'])

            except Exception as e:
                print(f"Error analyzing data: {str(e)}")


if __name__ == "__main__":
    main()
