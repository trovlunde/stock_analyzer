import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pypfopt import expected_returns, risk_models, EfficientFrontier


def plot_efficient_frontier(stock_data):
    """
    Plot the efficient frontier and mark the optimal portfolio

    Args:
        stock_data (dict): Dictionary of DataFrames containing stock data

    Returns:
        tuple: (optimal_weights, expected_return, expected_volatility)
    """
    # Combine price data
    prices_df = pd.DataFrame()
    for ticker, df in stock_data.items():
        prices_df[ticker] = df['Close']

    # Calculate expected returns and sample covariance
    mu = expected_returns.mean_historical_return(prices_df)
    S = risk_models.sample_cov(prices_df)

    # Set risk-free rate
    risk_free_rate = 0.0425  # Current 1-year Treasury rate

    # Generate random portfolios for plotting
    n_samples = 1000
    weights = np.random.dirichlet(np.ones(len(prices_df.columns)), n_samples)
    rets = weights @ mu
    vols = np.sqrt(np.sum((weights @ S) * weights, axis=1))

    # Calculate Sharpe ratios for random portfolios
    sharpe_ratios = (rets - risk_free_rate) / vols

    # Create hover text for random portfolios
    random_hover_text = []
    for i in range(n_samples):
        text = (
            f"Sharpe Ratio: {sharpe_ratios[i]:.2f}<br>" +
            "<br>".join([f"{ticker}: {weight:.1%}" for ticker,
                        weight in zip(prices_df.columns, weights[i])])
        )
        random_hover_text.append(text)

    # Find minimum volatility portfolio
    ef_min = EfficientFrontier(mu, S)
    ef_min.min_volatility()
    min_vol = ef_min.portfolio_performance()[1]
    min_ret = ef_min.portfolio_performance()[0]

    # Find maximum Sharpe ratio portfolio (this will be our optimal portfolio)
    ef = EfficientFrontier(mu, S)
    weights = ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    opt_ret, opt_vol, _ = ef.portfolio_performance()

    # Find maximum return portfolio by optimizing for a single asset
    # This will give us the highest possible return (and highest risk)
    max_ret_idx = mu.argmax()
    max_ret = mu.iloc[max_ret_idx]
    single_asset_weights = np.zeros(len(mu))
    single_asset_weights[max_ret_idx] = 1
    max_vol = np.sqrt(single_asset_weights @ S @ single_asset_weights)

    # Generate efficient frontier line
    # Increased number of points
    target_returns = np.linspace(min_ret, max_ret, 100)
    ef_line_volatilities = []
    ef_line_returns = []

    # Create new EfficientFrontier instance for each point
    for target_return in target_returns:
        try:
            ef_new = EfficientFrontier(mu, S)
            ef_new.efficient_return(target_return)
            ret, vol, _ = ef_new.portfolio_performance()
            ef_line_volatilities.append(vol)
            ef_line_returns.append(ret)
        except Exception as e:
            continue

    # Sort points by volatility to ensure proper line plotting
    ef_points = sorted(zip(ef_line_volatilities, ef_line_returns))
    ef_line_volatilities, ef_line_returns = zip(*ef_points)

    # Create the plot
    fig = go.Figure()

    # Add random portfolios with color based on Sharpe ratio
    fig.add_trace(go.Scatter(
        x=vols,
        y=rets,
        mode='markers',
        name='Random Portfolios',
        marker=dict(
            size=4,
            color=sharpe_ratios,  # Color based on Sharpe ratio
            colorscale='Viridis',  # Use a color scale
            colorbar=dict(
                title='Sharpe<br>Ratio',
                thickness=20,
                len=0.75,
                x=0.95
            ),
            showscale=True,
        ),
        text=random_hover_text,
        hovertemplate=(
            "<b>Portfolio Metrics:</b><br>" +
            "Annual Return: %{y:.01%}<br>" +
            "Annual Volatility: %{x:.1%}<br>" +
            "<br><b>Portfolio Details:</b><br>" +
            "%{text}<extra></extra>"
        )
    ))

    # Add efficient frontier line
    fig.add_trace(go.Scatter(
        x=ef_line_volatilities,
        y=ef_line_returns,
        mode='lines',
        name='Efficient Frontier',
        line=dict(color='blue'),
        hovertemplate=(
            "<b>Efficient Portfolio:</b><br>" +
            "Annual Return: %{y:.01%}<br>" +
            "Annual Volatility: %{x:.1%}<extra></extra>"
        )
    ))

    # Create hover text for optimal portfolio
    opt_hover = "<br>".join(
        [f"{ticker}: {weight:.1%}" for ticker, weight in cleaned_weights.items()])

    # Add optimal portfolio point
    fig.add_trace(go.Scatter(
        x=[opt_vol],
        y=[opt_ret],
        mode='markers',
        name='Optimal Portfolio (Maximum Sharpe Ratio)',
        marker=dict(
            size=15,
            color='red',
            symbol='star'
        ),
        text=[opt_hover],
        hovertemplate=(
            "<b>Optimal Portfolio:</b><br>" +
            "Annual Return: %{y:.01%}<br>" +
            "Annual Volatility: %{x:.1%}<br>" +
            "<br><b>Portfolio Weights:</b><br>" +
            "%{text}<extra></extra>"
        )
    ))

    # Calculate Sharpe Ratio for optimal portfolio
    risk_free_rate = 0.0425  # Current 1-year Treasury rate
    sharpe_ratio = (opt_ret - risk_free_rate) / opt_vol

    # Update layout with text on the right side
    fig.update_layout(
        title={
            'text': 'Portfolio Efficient Frontier',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24)
        },
        xaxis_title={
            'text': 'Expected Annual Volatility (Risk)',
            'font': dict(size=16)
        },
        yaxis_title={
            'text': 'Expected Annual Return',
            'font': dict(size=16)
        },
        showlegend=True,
        height=800,
        width=1200,  # Increased width to accommodate text
    )

    # Update hover template for optimal portfolio
    fig.data[-1].hovertemplate = (
        "<b>Optimal Portfolio:</b><br>" +
        "Annual Return: %{y:.1%}<br>" +
        "Annual Volatility: %{x:.1%}<br>" +
        f"Sharpe Ratio: {sharpe_ratio:.2f}<br>" +
        "<br><b>Portfolio Weights:</b><br>" +
        "%{text}<extra></extra>"
    )

    # Show the plot
    fig.show()

    return cleaned_weights, opt_ret, opt_vol
