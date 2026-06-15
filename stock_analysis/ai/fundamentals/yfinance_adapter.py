import yfinance as yf
import pandas as pd


class YFinanceAdapter:
    def get_annual_financials(self, ticker: str) -> pd.DataFrame:
        return yf.Ticker(ticker).financials

    def get_quarterly_financials(self, ticker: str) -> pd.DataFrame:
        return yf.Ticker(ticker).quarterly_financials
