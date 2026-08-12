import plotly.graph_objects as go
from plotly.subplots import make_subplots


def display_stock_data(df, events, ticker):
    """
    Display stock data with interactive plots and events

    Args:
        df (pandas.DataFrame): DataFrame containing stock data
        events (pandas.DataFrame): DataFrame containing events data
        ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple)
    """
    # Create figure with secondary y-axis
    fig = make_subplots(rows=2, cols=1,
                        vertical_spacing=0.2,
                        subplot_titles=(f'{ticker} Stock Price', 'Volume'),
                        row_heights=[0.7, 0.3])

    # Add candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                                 open=df['Open'],
                                 high=df['High'],
                                 low=df['Low'],
                                 close=df['Close'],
                                 name='OHLC'),
                  row=1, col=1)

    # Add volume bar chart
    fig.add_trace(go.Bar(x=df.index,
                         y=df['Volume'],
                         name='Volume'),
                  row=2, col=1)

    # Add events as annotations
    if events is not None and not events.empty:
        max_price = df['High'].max()
        min_price = df['Low'].min()
        price_range = max_price - min_price

        for idx, event in events.iterrows():
            # Add vertical lines for events
            fig.add_vline(x=idx, line_dash="dash",
                          line_color="gray", opacity=0.5)

            # Add annotations
            fig.add_annotation(
                x=idx,
                y=max_price + price_range * 0.05,  # Slightly above the chart
                text=f"{event['Event']}<br>{event['Details']}",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="gray",
                row=1, col=1
            )

    # Update layout
    fig.update_layout(
        title=f'{ticker} Stock Price and Volume',
        yaxis_title='Stock Price (USD)',
        yaxis2_title='Volume',
        xaxis_rangeslider_visible=False,
        height=800,  # Increase height of the figure
        showlegend=True
    )

    # Show the plot in browser
    fig.show()

    # Print summary statistics
    print(f"\nSummary Statistics for {ticker}:")
    print(f"Current Price: ${df['Close'].iloc[-1]:.2f}")
    print(f"Period High: ${df['High'].max():.2f}")
    print(f"Period Low: ${df['Low'].min():.2f}")
    print(f"Average Volume: {df['Volume'].mean():.0f}")

    if events is not None and not events.empty:
        print("\nMajor Events:")
        for idx, event in events.iterrows():
            # Convert index to string if it's a datetime, otherwise use as is
            date_str = idx.strftime(
                '%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            print(f"{date_str}: {event['Event']} - {event['Details']}")
