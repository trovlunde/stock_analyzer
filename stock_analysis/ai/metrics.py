import numpy as np
import pandas as pd


def calculate_max_drawdown(values):
    """Calculate the maximum drawdown from a series of values"""
    peak = values[0]
    max_drawdown = 0

    for value in values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown


def calculate_sharpe_ratio(values, risk_free_rate=0.02):
    """Calculate the Sharpe ratio for a series of values"""
    returns = pd.Series(values).pct_change(fill_method=None).dropna()
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    if len(excess_returns) > 0:
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    return 0


def calculate_performance_metrics(portfolio_values, investment_amount):
    """Calculate key performance metrics for a portfolio"""
    total_return = (portfolio_values[-1] -
                    investment_amount) / investment_amount
    max_drawdown = calculate_max_drawdown(portfolio_values)
    sharpe_ratio = calculate_sharpe_ratio(portfolio_values)

    return {
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }
