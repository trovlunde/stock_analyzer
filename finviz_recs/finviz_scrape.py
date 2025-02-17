import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime, timedelta
import pytz
import os
import logging

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


def scrape_finviz():
    logging.info("Starting Finviz scraping...")
    try:
        # Use a desktop browser User-Agent
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Make request to Finviz screener
        url = "https://finviz.com/screener.ashx?v=111&f=ind_stocksonly,sh_curvol_o5000,sh_price_u1,sh_relvol_o1.5,ta_change_u&ft=4"
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


def update_json_file(new_stocks):
    # Use absolute path for JSON file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_file = os.path.join(base_dir, 'finviz_recs', 'finviz_recs.json')

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
        for stock in new_stocks:
            stock['timestamp'] = datetime.now(pytz.UTC).isoformat()
            data['signals'].append(stock)
            logging.info(f"Added stock: {stock['Ticker']}")

        # Save updated data
        with open(json_file, 'w') as f:
            json.dump(data, f, indent=2)
        logging.info(
            f"Successfully wrote {len(new_stocks)} stocks to JSON file")

    except Exception as e:
        logging.error(f"Error updating JSON file: {str(e)}", exc_info=True)
        raise


def main():
    logging.info("=== Starting Finviz scraping script ===")
    stocks = scrape_finviz()

    if stocks:
        logging.info(f"Found {len(stocks)} stocks matching criteria.")
        update_json_file(stocks)
    else:
        logging.error("No stocks found or error occurred during scraping.")


if __name__ == "__main__":
    main()
