import pandas as pd
import requests
from bs4 import BeautifulSoup


class MarketIndices:
    """
    Contains Wikipedia URLs and methods to scrape tickers for different market indices
    """

    WIKI_URLS = {
        'sp500': 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'nasdaq100': 'https://en.wikipedia.org/wiki/Nasdaq-100',
        'dow30': 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average',
        'ftse100': 'https://en.wikipedia.org/wiki/FTSE_100_Index',
        'dax40': 'https://en.wikipedia.org/wiki/DAX',
        'cac40': 'https://en.wikipedia.org/wiki/CAC_40',
        'nikkei225': 'https://en.wikipedia.org/wiki/Nikkei_225'
    }

    @staticmethod
    def get_sp500_tickers():
        """Scrape S&P 500 tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['sp500']
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        # Try multiple ways to find the table
        table = soup.find('table', {'class': 'wikitable'}) or soup.find(
            'table', {'id': 'constituents'})
        if table is None:
            # Try finding any table with S&P 500 data
            tables = soup.find_all('table')
            for t in tables:
                if t.find('th') and ('Symbol' in t.find('th').text or 'Ticker' in t.find('th').text):
                    table = t
                    break

        if table is None:
            raise ValueError(
                f"Could not find ticker table on {url}. The page structure may have changed.")

        tickers = []
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                # Remove any footnote references
                ticker = ticker.split()[0] if ticker else ''
                if ticker:
                    tickers.append(ticker)

        return tickers

    @staticmethod
    def get_nasdaq100_tickers():
        """Scrape NASDAQ-100 tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['nasdaq100']
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        if table is None:
            tables = soup.find_all('table')
            for t in tables:
                if t.find('th') and ('Symbol' in t.find('th').text or 'Ticker' in t.find('th').text):
                    table = t
                    break

        if table is None:
            raise ValueError(
                f"Could not find ticker table on {url}. The page structure may have changed.")

        tickers = []
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if len(cells) >= 2:  # Ensure there are enough cells
                ticker = cells[1].text.strip()
                ticker = ticker.split()[0] if ticker else ''
                if ticker:
                    tickers.append(ticker)

        return tickers

    @staticmethod
    def get_dow30_tickers():
        """Scrape Dow Jones Industrial Average tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['dow30']
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        if table is None:
            tables = soup.find_all('table')
            for t in tables:
                if t.find('th') and ('Symbol' in t.find('th').text or 'Ticker' in t.find('th').text):
                    table = t
                    break

        if table is None:
            raise ValueError(
                f"Could not find ticker table on {url}. The page structure may have changed.")

        tickers = []
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if len(cells) >= 2:
                ticker = cells[1].text.strip()
                ticker = ticker.split()[0] if ticker else ''
                if ticker:
                    tickers.append(ticker)

        return tickers

    @staticmethod
    def get_ftse100_tickers():
        """Scrape FTSE 100 tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['ftse100']
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        if table is None:
            tables = soup.find_all('table')
            for t in tables:
                if t.find('th') and ('Symbol' in t.find('th').text or 'Ticker' in t.find('th').text):
                    table = t
                    break

        if table is None:
            raise ValueError(
                f"Could not find ticker table on {url}. The page structure may have changed.")

        tickers = []
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                ticker = ticker.split()[0] if ticker else ''
                if ticker:
                    tickers.append(ticker)

        return tickers

    @staticmethod
    def get_dax40_tickers():
        """Scrape DAX 40 tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['dax40']
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        if table is None:
            tables = soup.find_all('table')
            for t in tables:
                if t.find('th') and ('Symbol' in t.find('th').text or 'Ticker' in t.find('th').text):
                    table = t
                    break

        if table is None:
            raise ValueError(
                f"Could not find ticker table on {url}. The page structure may have changed.")

        tickers = []
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                ticker = ticker.split()[0] if ticker else ''
                if ticker:
                    tickers.append(ticker)

        return tickers

    @staticmethod
    def get_cac40_tickers():
        """Scrape CAC 40 tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['cac40']
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        if table is None:
            tables = soup.find_all('table')
            for t in tables:
                if t.find('th') and ('Symbol' in t.find('th').text or 'Ticker' in t.find('th').text):
                    table = t
                    break

        if table is None:
            raise ValueError(
                f"Could not find ticker table on {url}. The page structure may have changed.")

        tickers = []
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                ticker = ticker.split()[0] if ticker else ''
                if ticker:
                    tickers.append(ticker)

        return tickers

    @staticmethod
    def get_nikkei225_tickers():
        """Scrape Nikkei 225 tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['nikkei225']
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        if table is None:
            tables = soup.find_all('table')
            for t in tables:
                if t.find('th') and ('Symbol' in t.find('th').text or 'Ticker' in t.find('th').text):
                    table = t
                    break

        if table is None:
            raise ValueError(
                f"Could not find ticker table on {url}. The page structure may have changed.")

        tickers = []
        for row in table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if cells:
                ticker = cells[0].text.strip()
                ticker = ticker.split()[0] if ticker else ''
                if ticker:
                    tickers.append(ticker)

        return tickers

    @staticmethod
    def get_market_tickers(market='sp500'):
        """
        Get tickers for specified market index

        Args:
            market (str): Market index to fetch ('sp500', 'nasdaq100', 'dow30', etc.)

        Returns:
            list: List of tickers
        """
        market = market.lower()

        if market == 'sp500':
            return MarketIndices.get_sp500_tickers()
        elif market == 'nasdaq100':
            return MarketIndices.get_nasdaq100_tickers()
        elif market == 'dow30':
            return MarketIndices.get_dow30_tickers()
        elif market == 'ftse100':
            return MarketIndices.get_ftse100_tickers()
        elif market == 'dax40':
            return MarketIndices.get_dax40_tickers()
        elif market == 'cac40':
            return MarketIndices.get_cac40_tickers()
        elif market == 'nikkei225':
            return MarketIndices.get_nikkei225_tickers()
        else:
            raise ValueError(
                f"Market '{market}' not supported. Supported markets: {', '.join(MarketIndices.WIKI_URLS.keys())}")
