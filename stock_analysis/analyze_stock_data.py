import pandas as pd
import numpy as np
from scipy import stats
import pandas_ta as ta
from pypfopt import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


def analyze_stock_data(df):
    """
    Analyze stock data for risk and returns

    Args:
        df (pandas.DataFrame): DataFrame with stock price data

    Returns:
        dict: Dictionary containing analysis results
    """
    # Calculate daily returns
    df['Returns'] = df['Close'].pct_change()

    # Calculate volatility (annualized)
    volatility = df['Returns'].std() * np.sqrt(252)

    # Calculate annualized return
    annual_return = df['Returns'].mean() * 252

    # Calculate Sharpe Ratio (assuming risk-free rate of 0.01)
    risk_free_rate = 0.01
    sharpe_ratio = (annual_return - risk_free_rate) / volatility

    # Calculate technical indicators
    # RSI (Relative Strength Index)
    rsi = ta.rsi(df['Close'])

    # MACD (Moving Average Convergence Divergence)
    macd = ta.macd(df['Close'])

    # Beta calculation (requires market index data)
    # This is just a placeholder for now

    results = {
        'Volatility (Annual)': volatility,
        'Return (Annual)': annual_return,
        'Sharpe Ratio': sharpe_ratio,
        'Latest RSI': rsi.iloc[-1] if not rsi.empty else None,
        'Current Price': df['Close'].iloc[-1],
        'Max Drawdown': (df['Close'] / df['Close'].expanding(min_periods=1).max() - 1).min(),
        'Value at Risk (95%)': np.percentile(df['Returns'].dropna(), 5),
    }

    return results


def analyze_portfolio_correlation(stock_data):
    """
    Analyze correlation between stocks in portfolio
    """
    # Create returns DataFrame
    returns_df = pd.DataFrame()
    for ticker, df in stock_data.items():
        returns_df[ticker] = df['Close'].pct_change()

    # Calculate correlation matrix
    correlation_matrix = returns_df.corr()
    return correlation_matrix


def optimize_portfolio(stock_data):
    """
    Optimize portfolio weights using Modern Portfolio Theory
    """
    # Combine price data
    prices_df = pd.DataFrame()
    for ticker, df in stock_data.items():
        prices_df[ticker] = df['Close']

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(prices_df)
    S = risk_models.sample_cov(prices_df)

    # Optimize for maximal Sharpe ratio
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()

    return ef.clean_weights()
