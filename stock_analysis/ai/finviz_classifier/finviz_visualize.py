import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stock_analysis.ai.finviz_classifier.data_fetching import fetch_stock_returns, load_historical_signals
import numpy as np


def feature_engineering(df):
    """
    Perform feature engineering on the input DataFrame.
    Calculates returns paths and other features needed for visualization.

    Args:
        df (pd.DataFrame): Merged DataFrame containing signals and price data

    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    # Map the existing columns to our day-based format
    df['day_0_return'] = 0  # Day 0 is always 0 (signal day)
    df['day_1_return'] = df['next_day_return']
    df['day_5_return'] = df['next_week_return']

    df['day_0_high_return'] = 0  # Day 0 is always 0 (signal day)
    df['day_1_high_return'] = df['next_day_high_return']
    df['day_5_high_return'] = df['next_week_high_return']

    # Fill intermediate days (2-4) with NaN for now
    for day in range(2, 5):
        df[f'day_{day}_return'] = np.nan
        df[f'day_{day}_high_return'] = np.nan

    # Convert all returns to percentages
    for day in range(6):
        df[f'day_{day}_return'] = df[f'day_{day}_return'].fillna(0) * 100
        df[f'day_{day}_high_return'] = df[f'day_{day}_high_return'].fillna(
            0) * 100

    # Calculate basic statistics
    df['max_return'] = df[[
        col for col in df.columns if 'day_' in col and 'return' in col]].max(axis=1)
    df['min_return'] = df[[
        col for col in df.columns if 'day_' in col and 'return' in col]].min(axis=1)
    df['return_range'] = df['max_return'] - df['min_return']

    # Add signal metadata features
    if 'Change' in df.columns:
        df['signal_day_change'] = df['Change'] * 100  # Convert to percentage

    if 'Volume' in df.columns and 'Market Cap' in df.columns:
        df['volume_to_mcap'] = df['Volume'] / df['Market Cap']

    return df


def visualize_signal_performance():
    """
    Visualize the performance of signals over time using high and close prices.
    Each line represents a single signal's return path.
    """
    signals_df = load_historical_signals()
    if signals_df is None:
        print("Failed to load historical signals")
        return

    returns_df, detailed_returns_df = fetch_stock_returns(signals_df)
    if returns_df.empty:
        print("No return data available")
        return

    # Normalize timestamps and ensure datetime type
    signals_df['timestamp'] = pd.to_datetime(
        signals_df['timestamp']).dt.normalize()
    detailed_returns_df['signal_date'] = pd.to_datetime(
        detailed_returns_df['signal_date']).dt.normalize()

    # Debug print
    print("\nBefore merge:")
    print("Signals timestamp dtype:", signals_df['timestamp'].dtype)
    print("Detailed returns signal_date dtype:",
          detailed_returns_df['signal_date'].dtype)

    df = pd.merge(
        signals_df,
        detailed_returns_df,
        left_on=['Ticker', 'timestamp'],
        right_on=['Ticker', 'signal_date'],
        how='inner'
    )

    if df.empty:
        print("No data after merging signals and returns")
        return

    # Create figure with subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=('High Price Paths', 'Close Price Paths'),
        vertical_spacing=0.1
    )

    days = list(range(6))  # 0 to 5 days

    # Track min and max returns for y-axis scaling
    all_returns = []

    # Plot individual paths for each signal
    for idx, row in df.iterrows():
        # Create return paths starting at 0 for day 0
        high_returns = [0]
        close_returns = [0]

        # Add returns for days 1-5 where available
        for day in range(1, 6):
            if day in row['daily_returns']:
                high_returns.append(
                    row['daily_returns'][day]['high_return'] * 100)
                close_returns.append(
                    row['daily_returns'][day]['close_return'] * 100)
            else:
                high_returns.append(None)
                close_returns.append(None)

        # Track all non-None returns for y-axis scaling
        all_returns.extend(
            [r for r in high_returns + close_returns if r is not None])

        # Create hover text
        hover_text = [
            f"Ticker: {row['Ticker']}<br>" +
            f"Signal Date: {row['signal_date'].strftime('%Y-%m-%d')}<br>" +
            f"Price: ${row['Price']:.2f}<br>" +
            f"Change: {row['Change']*100:.1f}%<br>" +
            f"Volume: {row['Volume']:,.0f}<br>" +
            f"Market Cap: {row['Market Cap']:,.0f}<br>" +
            f"Day {i} Return: {ret:.1f}%" if ret is not None else
            f"Ticker: {row['Ticker']}<br>Day {i}: No data"
            for i, ret in enumerate(high_returns)
        ]

        # Plot high returns path
        fig.add_trace(
            go.Scatter(
                x=days,
                y=high_returns,
                mode='lines',
                line=dict(width=1, color='rgba(0,100,80,0.1)'),
                showlegend=False,
                hovertext=hover_text,
                hoverinfo='text'
            ),
            row=1,
            col=1
        )

        # Update hover text for close returns
        hover_text = [
            f"Ticker: {row['Ticker']}<br>" +
            f"Signal Date: {row['signal_date'].strftime('%Y-%m-%d')}<br>" +
            f"Price: ${row['Price']:.2f}<br>" +
            f"Change: {row['Change']*100:.1f}%<br>" +
            f"Volume: {row['Volume']:,.0f}<br>" +
            f"Market Cap: {row['Market Cap']:,.0f}<br>" +
            f"Day {i} Return: {ret:.1f}%" if ret is not None else
            f"Ticker: {row['Ticker']}<br>Day {i}: No data"
            for i, ret in enumerate(close_returns)
        ]

        # Plot close returns path
        fig.add_trace(
            go.Scatter(
                x=days,
                y=close_returns,
                mode='lines',
                line=dict(width=1, color='rgba(0,100,80,0.1)'),
                showlegend=False,
                hovertext=hover_text,
                hoverinfo='text'
            ),
            row=2,
            col=1
        )

    # Calculate y-axis range with 10% padding
    y_min = min(all_returns)
    y_max = max(all_returns)
    y_padding = (y_max - y_min) * 0.1
    y_range = [y_min - y_padding, y_max + y_padding]

    # Update layout
    fig.update_layout(
        title='Individual Signal Return Paths',
        height=1200,
        width=1600,
        template='plotly_white',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Roboto"
        )
    )

    # Update y-axes with calculated range
    fig.update_yaxes(
        title_text="Return (%)",
        ticksuffix="%",
        range=y_range
    )

    # Update x-axes
    fig.update_xaxes(
        title_text="Days After Signal",
        tickmode='array',
        ticktext=['Day 0', 'Day 1', 'Day 2', 'Day 3', 'Day 4', 'Day 5'],
        tickvals=days
    )

    # Show the figure
    fig.show(renderer="browser")


if __name__ == "__main__":
    visualize_signal_performance()
