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
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        tickers = []

        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip()
            tickers.append(ticker)

        return tickers

    @staticmethod
    def get_nasdaq100_tickers():
        """Scrape NASDAQ-100 tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['nasdaq100']
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        tickers = []

        for row in table.findAll('tr')[1:]:
            cells = row.findAll('td')
            if len(cells) >= 2:  # Ensure there are enough cells
                ticker = cells[1].text.strip()
                tickers.append(ticker)

        return tickers

    @staticmethod
    def get_dow30_tickers():
        """Scrape Dow Jones Industrial Average tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['dow30']
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        tickers = []

        for row in table.findAll('tr')[1:]:
            cells = row.findAll('td')
            if len(cells) >= 2:
                ticker = cells[1].text.strip()
                tickers.append(ticker)

        return tickers

    @staticmethod
    def get_ftse100_tickers():
        """Scrape FTSE 100 tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['ftse100']
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        tickers = []

        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip()
            tickers.append(ticker)

        return tickers

    @staticmethod
    def get_dax40_tickers():
        """Scrape DAX 40 tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['dax40']
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        tickers = []

        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip()
            tickers.append(ticker)

        return tickers

    @staticmethod
    def get_cac40_tickers():
        """Scrape CAC 40 tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['cac40']
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        tickers = []

        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip()
            tickers.append(ticker)

        return tickers

    @staticmethod
    def get_nikkei225_tickers():
        """Scrape Nikkei 225 tickers from Wikipedia"""
        url = MarketIndices.WIKI_URLS['nikkei225']
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        table = soup.find('table', {'class': 'wikitable'})
        tickers = []

        for row in table.findAll('tr')[1:]:
            ticker = row.findAll('td')[0].text.strip()
            tickers.append(ticker)

        return tickers

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
