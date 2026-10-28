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
        'nikkei225': 'https://en.wikipedia.org/wiki/Nikkei_225',
        # Oslo-listed equities (Yahoo .OL). Broader than the official OSEBX basket;
        # useful as the Norwegian screening universe.
        'osebx': 'https://en.wikipedia.org/wiki/List_of_companies_listed_on_the_Oslo_Stock_Exchange',
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
    def _to_yahoo_ol_ticker(raw: str) -> str | None:
        """Normalize Wikipedia OSE ticker text to yfinance ``SYMBOL.OL``."""
        from stock_analysis.research.oslo_tickers import to_yahoo_ol_ticker

        return to_yahoo_ol_ticker(raw)

    @staticmethod
    def get_osebx_tickers(validate: bool = True, use_search_fallback: bool = True):
        """Scrape Oslo listings and resolve to working Yahoo tickers.

        Source is Wikipedia's list of companies listed on Oslo Bors (practical
        Norwegian screening universe; broader than the licensed OSEBX PDF).
        When ``validate`` is True (default), drops delisted symbols, applies
        known remaps (e.g. ARCHER->ARCH.OL), tries dual-listed US tickers, and
        optionally searches Yahoo by company name for remaining misses.
        Aliases: ``osebx``, ``oslo``, ``oslo_bors``.
        """
        from stock_analysis.research.oslo_tickers import resolve_oslo_yahoo_tickers

        url = MarketIndices.WIKI_URLS["osebx"]
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")

        table = None
        for candidate in soup.find_all("table", {"class": "wikitable"}):
            header_cells = candidate.find("tr")
            if header_cells is None:
                continue
            headers_text = " ".join(
                th.get_text(strip=True).lower()
                for th in header_cells.find_all(["th", "td"])
            )
            if "ticker" in headers_text:
                table = candidate
                break

        if table is None:
            raise ValueError(
                f"Could not find ticker table on {url}. The page structure may have changed."
            )

        entries: list[tuple[str, str]] = []
        for row in table.find_all("tr")[1:]:
            cells = row.find_all("td")
            if len(cells) < 2:
                continue
            company = cells[0].get_text(strip=True)
            raw = cells[1].get_text(strip=True)
            if raw:
                entries.append((company, raw))

        tickers, remaps = resolve_oslo_yahoo_tickers(
            entries,
            validate=validate,
            use_search_fallback=use_search_fallback,
        )
        if not tickers:
            raise ValueError(f"No Oslo tickers resolved from {url}")

        if remaps:
            sample = ", ".join(f"{k}->{v}" for k, v in list(remaps.items())[:5])
            more = f" (+{len(remaps) - 5} more)" if len(remaps) > 5 else ""
            print(f"Resolved {len(remaps)} Oslo ticker remaps: {sample}{more}")
        print(
            f"Oslo universe: {len(tickers)} Yahoo-valid tickers "
            f"(from {len(entries)} Wikipedia rows)"
        )
        return tickers

    @staticmethod
    def get_market_tickers(market='sp500'):
        """
        Get tickers for specified market index

        Args:
            market (str): Market index to fetch ('sp500', 'nasdaq100', 'dow30',
                'osebx' / 'oslo', etc.)

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
        elif market in ('osebx', 'oslo', 'oslo_bors'):
            return MarketIndices.get_osebx_tickers()
        else:
            raise ValueError(
                f"Market '{market}' not supported. Supported markets: {', '.join(MarketIndices.WIKI_URLS.keys())}")
