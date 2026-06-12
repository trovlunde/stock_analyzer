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
    df['Returns'] = df['Close'].pct_change(fill_method=None)

    # Calculate volatility (annualized)
    volatility = df['Returns'].std() * np.sqrt(252)

    # Calculate annualized return
    annual_return = df['Returns'].mean() * 252

    # Calculate Sharpe Ratio (assuming risk-free rate of 0.01)
    risk_free_rate = 0.01
    sharpe_ratio = (annual_return - risk_free_rate) / volatility

    # Calculate technical indicators
    # RSI (Relative Strength Index)
    try:
        rsi = ta.rsi(df['Close'])
        # Handle both Series and DataFrame returns
        if isinstance(rsi, pd.DataFrame):
            rsi = rsi.iloc[:, 0] if len(rsi.columns) > 0 else pd.Series(
                dtype=float, index=df.index)
        elif not isinstance(rsi, pd.Series):
            rsi = pd.Series(dtype=float, index=df.index)
    except Exception as e:
        rsi = pd.Series(dtype=float, index=df.index)

    # MACD (Moving Average Convergence Divergence)
    try:
        macd_result = ta.macd(df['Close'])
        # MACD returns a DataFrame with MACD, MACD_signal, MACD_diff columns
        if isinstance(macd_result, pd.DataFrame):
            macd = macd_result.iloc[:, 0] if len(
                macd_result.columns) > 0 else pd.Series(dtype=float, index=df.index)
        elif isinstance(macd_result, pd.Series):
            macd = macd_result
        else:
            macd = pd.Series(dtype=float, index=df.index)
    except Exception as e:
        macd = pd.Series(dtype=float, index=df.index)

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


def analyze_stock_vs_sp500(stock_df, sp500_ticker='^GSPC', period=None):
    """
    Compare stock performance against S&P 500 index

    Args:
        stock_df (pd.DataFrame): Stock price data
        sp500_ticker (str): S&P 500 ticker symbol (default: ^GSPC)
        period (str, optional): Period string if using period-based fetching (e.g., '1y', '6mo')

    Returns:
        dict: Comparison metrics including Beta, Alpha, relative performance
    """
    import yfinance as yf

    try:
        # Get S&P 500 data for the same period
        sp500 = yf.Ticker(sp500_ticker)

        # Use period if provided, otherwise use date range
        if period:
            sp500_df = sp500.history(period=period)
        else:
            # Add a small buffer to ensure we get all dates
            start_date = stock_df.index[0] - pd.Timedelta(days=5)
            end_date = stock_df.index[-1] + pd.Timedelta(days=1)
            sp500_df = sp500.history(start=start_date, end=end_date)

        if sp500_df.empty:
            return None

        # Align dates
        common_dates = stock_df.index.intersection(sp500_df.index)
        if len(common_dates) < 10:
            return None

        stock_aligned = stock_df.loc[common_dates]
        sp500_aligned = sp500_df.loc[common_dates]

        # Calculate returns
        stock_returns = stock_aligned['Close'].pct_change(fill_method=None).dropna()
        sp500_returns = sp500_aligned['Close'].pct_change(fill_method=None).dropna()

        # Align returns
        common_returns = stock_returns.index.intersection(sp500_returns.index)
        stock_returns = stock_returns.loc[common_returns]
        sp500_returns = sp500_returns.loc[common_returns]

        if len(stock_returns) < 10:
            return None

        # Calculate Beta (slope of regression)
        covariance = np.cov(stock_returns, sp500_returns)[0][1]
        sp500_variance = np.var(sp500_returns)
        beta = covariance / sp500_variance if sp500_variance > 0 else None

        # Calculate Alpha (excess return)
        stock_annual_return = stock_returns.mean() * 252
        sp500_annual_return = sp500_returns.mean() * 252
        risk_free_rate = 0.01  # 1% risk-free rate
        alpha = stock_annual_return - \
            (risk_free_rate + beta *
             (sp500_annual_return - risk_free_rate)) if beta else None

        # Calculate correlation
        correlation = np.corrcoef(stock_returns, sp500_returns)[0][1]

        # Calculate relative performance
        stock_cumulative = (1 + stock_returns).cumprod().iloc[-1] - 1
        sp500_cumulative = (1 + sp500_returns).cumprod().iloc[-1] - 1
        relative_performance = stock_cumulative - sp500_cumulative

        # Calculate tracking error
        tracking_error = np.std(stock_returns - sp500_returns) * np.sqrt(252)

        return {
            'Beta': beta,
            'Alpha (Annual)': alpha,
            'Correlation with S&P 500': correlation,
            'Stock Return': stock_cumulative,
            'S&P 500 Return': sp500_cumulative,
            'Relative Performance': relative_performance,
            'Tracking Error (Annual)': tracking_error,
            'Stock Volatility': stock_returns.std() * np.sqrt(252),
            'S&P 500 Volatility': sp500_returns.std() * np.sqrt(252),
        }
    except Exception as e:
        print(f"Error comparing with S&P 500: {e}")
        return None


def analyze_portfolio_correlation(stock_data):
    """
    Analyze correlation between stocks in portfolio
    """
    # Create returns DataFrame
    returns_df = pd.DataFrame()
    for ticker, df in stock_data.items():
        returns_df[ticker] = df['Close'].pct_change(fill_method=None)

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
