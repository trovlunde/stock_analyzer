import warnings
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def sp500_analysis():
    print("Fetching SP500 data...")
    try:
        sp500 = yf.Ticker("^GSPC").history(period="20y")
    except Exception as e:
        print(f"Error fetching SP500 data: {e}")
        return None
    sp500 = sp500.drop(columns=['Dividends', 'Stock Splits'])
    sp500.info()

    # Calculate moving averages and returns
    sp500['50ma'] = sp500['Close'].rolling(window=50).mean()
    sp500['200ma'] = sp500['Close'].rolling(window=200).mean()
    sp500['return'] = sp500['Close'].pct_change(fill_method=None)

    # Create figure
    fig = go.Figure()

    # Add main price lines with return info in hover
    fig.add_trace(go.Scatter(
        x=sp500.index,
        y=sp500['Close'],
        name='Close',
        line=dict(color='blue'),
        hovertemplate="<br>".join([
            "Date: %{x}",
            "Close: $%{y:.2f}",
            "Return: %{customdata:.2%}",
            "<extra></extra>"
        ]),
        customdata=sp500['return']
    ))

    fig.add_trace(go.Scatter(
        x=sp500.index,
        y=sp500['50ma'],
        name='50MA',
        line=dict(color='orange'),
        hovertemplate="<br>".join([
            "50MA: $%{y:.2f}",
            "<extra></extra>"
        ])
    ))

    fig.add_trace(go.Scatter(
        x=sp500.index,
        y=sp500['200ma'],
        name='200MA',
        line=dict(color='red'),
        hovertemplate="<br>".join([
            "200MA: $%{y:.2f}",
            "<extra></extra>"
        ])
    ))

    # Add vertical lines for significant changes
    significant_changes = sp500[abs(sp500['return']) > 0.04]
    for date in significant_changes.index:
        color = 'green' if significant_changes.loc[date,
                                                   'return'] > 0 else 'red'
        fig.add_vline(
            x=date,
            line_width=1,
            line_dash="dash",
            line_color=color,
            opacity=0.3
        )

    # Update layout
    fig.update_layout(
        title='S&P 500 Close Price with 200 and 50-Day Moving Average',
        xaxis_title='Date',
        yaxis_title='Price',
        yaxis_type='log',
        height=600,
        showlegend=True,
        hovermode='x unified'
    )

    fig.show()

    # Create analysis table for significant changes
    significant_changes = sp500[abs(sp500['return']) > 0.03].copy()

    # Calculate prior returns
    significant_changes['prior_day_return'] = sp500['return'].shift(1)
    significant_changes['prior_week_return'] = sp500['Close'].pct_change(
        periods=5, fill_method=None)  # 5 trading days
    significant_changes['prior_month_return'] = sp500['Close'].pct_change(
        periods=21, fill_method=None)  # ~21 trading days

    # Format the table
    table_df = pd.DataFrame({
        'Date': significant_changes.index.strftime('%Y-%m-%d'),
        'Return': significant_changes['return'].map('{:.2%}'.format),
        'Prior Day': significant_changes['prior_day_return'].map('{:.2%}'.format),
        'Prior Week': significant_changes['prior_week_return'].map('{:.2%}'.format),
        'Prior Month': significant_changes['prior_month_return'].map('{:.2%}'.format)
    })

    # Create color arrays for each column
    def get_color(value_str):
        try:
            value = float(value_str.strip('%'))/100
            if value > 0.01:  # More than 1%
                return '#c8e6c9'  # green
            elif value < -0.01:  # Less than -1%
                return '#ffcdd2'  # red
            else:  # Between -1% and 1%
                return 'white'
        except:
            return 'white'

    color_arrays = {
        'Return': [get_color(x) for x in table_df['Return']],
        'Prior Day': [get_color(x) for x in table_df['Prior Day']],
        'Prior Week': [get_color(x) for x in table_df['Prior Week']],
        'Prior Month': [get_color(x) for x in table_df['Prior Month']]
    }

    # Display table using plotly
    fig2 = go.Figure(data=[go.Table(
        header=dict(
            values=list(table_df.columns),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[table_df[col] for col in table_df.columns],
            fill_color=[['white'] * len(table_df) if col == 'Date' else color_arrays[col]
                        for col in table_df.columns],
            align='left'
        )
    )])

    fig2.update_layout(
        title='Significant Price Changes (>5%) Analysis',
        # Adjust height based on number of rows
        height=400 + len(significant_changes) * 25
    )

    fig2.show()

    # ... rest of the code ...

    sp500['return'] = sp500['Close'].pct_change(fill_method=None)

    plt.figure(figsize=(15, 5))
    plt.title('S&P 500 Daily Return')
    plt.plot(sp500['return'])
    plt.show()

    # Calculate the volatility of the close price
    sp500['volatility'] = sp500['return'].rolling(
        window=252).std() * (252 ** 0.5)

    # Calculate the rolling correlation between the return and volatility
    sp500['corr252'] = sp500['return'].rolling(
        window=252).corr(sp500['volatility'])

    # Plot the rolling correlation
    plt.figure(figsize=(15, 5))
    plt.title(
        'Rolling Correlation between S&P 500 Return and Volatility on a 252 days timeframe')
    plt.plot(sp500['corr252'])
    plt.show()

    return sp500


sp500_analysis()
