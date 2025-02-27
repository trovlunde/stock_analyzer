import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import pytz
import os
import logging
import pandas_market_calendars as mcal

# Set up logging with absolute path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(base_dir, 'finviz_recs', 'finviz_scrape.log')

# Create log directory if it doesn't exist
os.makedirs(os.path.dirname(log_file), exist_ok=True)

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log the paths being used
logging.info(f"Log file location: {log_file}")

# Map countries to their primary exchanges
COUNTRY_EXCHANGE_MAP = {
    'USA': 'NYSE',      # New York Stock Exchange
    'China': 'SSE',     # Shanghai Stock Exchange
    'Canada': 'TSX',    # Toronto Stock Exchange
    'Germany': 'XETR',  # Deutsche Börse
    'Australia': 'ASX',  # Australian Securities Exchange
    'Israel': 'TASE',   # Tel Aviv Stock Exchange
    'Sweden': 'OMXS',   # Stockholm Stock Exchange
    'Singapore': 'SGX',  # Singapore Exchange
    'Greece': 'ASEX',   # Athens Stock Exchange
    'UK': 'LSE',        # London Stock Exchange
    'France': 'EPAX',   # Euronext Paris
    'Italy': 'BME',     # Borsa Italiana
    'Spain': 'BME',     # Bolsa de Madrid
    'Netherlands': 'EPAX',  # Euronext Amsterdam
    'Belgium': 'EPAX',   # Euronext Brussels
    'Japan': 'TSE',      # Tokyo Stock Exchange
    'Hong Kong': 'HKEX',  # Hong Kong Stock Exchange
    'South Korea': 'KRX',  # Korea Exchange
    'Taiwan': 'TWSE',    # Taiwan Stock Exchange
    'India': 'BSE',      # Bombay Stock Exchange
    'Brazil': 'B3',      # B3
    'Argentina': 'MERV',  # Buenos Aires Stock Exchange
}


def is_market_open(country='USA'):
    """
    Check if the market is open for a given country
    Args:
        country (str): Country name from the stock data
    Returns:
        bool: True if market is open, False otherwise
    """
    try:
        # Get the exchange calendar for the country
        # Default to NYSE if country not found
        exchange_name = COUNTRY_EXCHANGE_MAP.get(country, '')

        if not exchange_name:
            logging.warning(f"No exchange found for country: {country}")
            return True

        calendar = mcal.get_calendar(exchange_name)

        # Get current time in UTC
        now = datetime.now(pytz.UTC)

        # Get schedule for today
        schedule = calendar.schedule(
            start_date=now.date(), end_date=now.date())

        if schedule.empty:
            logging.info(f"Market closed - No trading day for {exchange_name}")
            return False

        return True

    except Exception as e:
        logging.warning(
            f"Error checking market hours for {country}: {str(e)}. Defaulting to NYSE hours.")
        # Fall back to NYSE if there's an error
        return True


def scrape_finviz(url):
    logging.info("Starting Finviz scraping...")
    try:
        # Use a desktop browser User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        logging.info(f"Requesting URL: {url}")
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Find the main data table using the updated class selector
        table = soup.find('table', {'class': 'styled-table-new'})

        if not table:
            logging.error(
                "No data table found. The website structure might have changed.")
            return []

        # Extract headers
        headers = []
        for th in table.find_all('th'):
            headers.append(th.text.strip())

        # Extract rows
        stocks = []
        for tr in table.find_all('tr')[1:]:  # Skip the header row
            row = {}
            cells = tr.find_all('td')
            for i, cell in enumerate(cells):
                if i < len(headers):
                    value = cell.find('a').text.strip() if cell.find(
                        'a') else cell.text.strip()

                    # Convert Market Cap to numeric millions
                    if headers[i] == "Market Cap":
                        try:
                            value = value.replace(',', '').strip()
                            if value.endswith('B'):
                                value = float(value[:-1]) * 1000
                            elif value.endswith('M'):
                                value = float(value[:-1])
                            elif value.endswith('K'):
                                value = float(value[:-1]) / 1000
                            else:
                                value = float(value) / 1000000
                        except (ValueError, AttributeError):
                            value = None

                    # Convert other numeric fields
                    elif headers[i] in ["Price", "Change"]:
                        try:
                            value = float(value.rstrip(
                                '%') if headers[i] == "Change" else value)
                        except ValueError:
                            value = None
                    elif headers[i] == "Volume":
                        try:
                            value = int(value.replace(',', ''))
                        except ValueError:
                            value = None

                    row[headers[i]] = value
            if row:
                stocks.append(row)

        logging.info(f"Successfully scraped {len(stocks)} stocks")
        return stocks

    except Exception as e:
        logging.error(
            f"Error occurred during scraping: {str(e)}", exc_info=True)
        return []


def update_json_file(new_stocks, filename):
    """Update the JSON file with new stock data, checking market hours for each stock"""
    if not new_stocks:
        logging.info("No stocks to update")
        return

    # Group stocks by country to minimize API calls
    stocks_by_country = {}
    for stock in new_stocks:
        country = stock.get('Country', 'USA')
        if country not in stocks_by_country:
            stocks_by_country[country] = []
        stocks_by_country[country].append(stock)

    # Check market status for each country and only include stocks from open markets
    valid_stocks = []
    for country, stocks in stocks_by_country.items():
        if is_market_open(country):
            valid_stocks.extend(stocks)
        else:
            logging.info(
                f"Skipping {len(stocks)} stocks from {country} - Market closed")

    if not valid_stocks:
        logging.info("No stocks from open markets to update")
        return

    # Continue with the existing update logic using valid_stocks instead of new_stocks
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_file = os.path.join(base_dir, 'finviz_recs', filename)

    logging.info(f"Working directory: {os.getcwd()}")
    logging.info(f"Attempting to write to: {json_file}")

    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(json_file), exist_ok=True)

        # Read existing data if file exists
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Check for recent data
            if data.get('signals'):
                latest_timestamp = max(signal['timestamp']
                                       for signal in data['signals'])
                latest_time = datetime.fromisoformat(
                    latest_timestamp.replace('Z', '+00:00'))
                if datetime.now(pytz.UTC) - latest_time < timedelta(hours=12):
                    logging.info(
                        "Recent data exists (less than 12 hours old). Skipping update.")
                    return
        else:
            # Create new data structure
            data = {
                "timestamp": datetime.now(pytz.UTC).isoformat(),
                "headers": ["No.", "Ticker", "Company", "Sector", "Industry", "Country",
                            "Market Cap", "P/E", "Price", "Change", "Volume"],
                "signals": [],
                "filters": "ind:stocksonly sh_curvol:o5000 sh_price:u1 sh_relvol:o1.5 ta_change:u",
                "export": 1
            }

        # Add new stocks
        for stock in valid_stocks:
            stock['timestamp'] = datetime.now(pytz.UTC).isoformat()
            data['signals'].append(stock)
            logging.info(f"Added stock: {stock['Ticker']}")

        # Save updated data
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(
            f"Successfully wrote {len(valid_stocks)} stocks to JSON file")

    except Exception as e:
        logging.error(f"Error updating JSON file: {str(e)}", exc_info=True)
        raise


def main():
    logging.info("=== Starting Finviz scraping script ===")
    url_up = "https://finviz.com/screener.ashx?v=111&f=ind_stocksonly,sh_curvol_o5000,sh_price_u1,sh_relvol_o1.5,ta_change_u&ft=4"
    url = "https://finviz.com/screener.ashx?v=111&f=ind_stocksonly,sh_curvol_o5000,sh_price_u1,sh_relvol_o1.5&ft=4"
    url_alt_up = "https://finviz.com/screener.ashx?v=111&f=ind_stocksonly,sh_curvol_o2000,sh_price_u3,sh_relvol_o1.5,sh_short_o15,ta_change_u&ft=4&o=-volume"
    url_alt = "https://finviz.com/screener.ashx?v=111&f=ind_stocksonly,sh_curvol_o2000,sh_price_u3,sh_relvol_o1.5,sh_short_o15&ft=4&o=-volume"

    stocks = scrape_finviz(url)
    stocks_alt = scrape_finviz(url_alt)
    stocks_alt_up = scrape_finviz(url_alt_up)
    stocks_up = scrape_finviz(url_up)

    if stocks:
        logging.info(f"Found {len(stocks)} stocks matching criteria.")
        update_json_file(stocks, 'finviz_recs.json')
    else:
        logging.error("No stocks found or error occurred during scraping.")

    if stocks_alt:
        logging.info(f"Found {len(stocks_alt)} stocks matching criteria.")
        update_json_file(stocks_alt, 'finviz_recs_alt.json')
    else:
        logging.error("No stocks found or error occurred during scraping.")

    if stocks_alt_up:
        logging.info(f"Found {len(stocks_alt_up)} stocks matching criteria.")
        update_json_file(stocks_alt_up, 'finviz_recs_alt_up.json')
    else:
        logging.error("No stocks found or error occurred during scraping.")

    if stocks_up:
        logging.info(f"Found {len(stocks_up)} stocks matching criteria.")
        update_json_file(stocks_up, 'finviz_recs_up.json')
    else:
        logging.error("No stocks found or error occurred during scraping.")


if __name__ == "__main__":
    main()
