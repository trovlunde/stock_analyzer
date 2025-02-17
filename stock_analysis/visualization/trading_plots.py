import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_trading_signals(simulation, stock_data, metrics=None, title="Trading Analysis"):
    """Create a visualization of trading signals and portfolio performance"""
    ticker = stock_data.index.name or 'Unknown'

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            f'Trading Signals and Price - {ticker}',
            f'Portfolio Value - {ticker}'
        ),
        vertical_spacing=0.1,
        shared_xaxes=True
    )

    # Add price series to first subplot
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data['Close'],
            mode='lines',
            name='Price',
            line=dict(color='rgba(0, 0, 255, 0.5)', width=1)
        ),
        row=1, col=1
    )

    # Add correct positive predictions
    correct_long = simulation[
        (simulation['prediction'] == 'positive') &
        (simulation['actual_return'] > 0)
    ].index

    if len(correct_long) > 0:
        fig.add_trace(
            go.Scatter(
                x=correct_long,
                y=simulation.loc[correct_long, 'price'],
                mode='markers',
                name='Correct Long',
                marker=dict(symbol='triangle-up', size=10,
                            color='green', opacity=0.7),
                hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<br>Prediction: Correct Long<extra></extra>"
            ),
            row=1, col=1
        )

    # Add wrong positive predictions
    for actual, color, symbol in [
        ('neutral', 'white', 'triangle-up'),
        ('negative', 'red', 'triangle-up')
    ]:
        wrong_long = simulation[
            (simulation['prediction'] == 'positive') &
            (simulation['actual_return'] == actual)
        ].index

        if len(wrong_long) > 0:
            fig.add_trace(
                go.Scatter(
                    x=wrong_long,
                    y=simulation.loc[wrong_long, 'price'],
                    mode='markers',
                    name=f'Wrong Long ({actual})',
                    marker=dict(
                        symbol=symbol,
                        size=10,
                        color=color,
                        line=dict(color='black',
                                  width=1) if color == 'white' else None,
                        opacity=0.7
                    ),
                    hovertemplate=f"Date: %{{x}}<br>Price: %{{y:.2f}}<br>Prediction: Wrong Long ({actual})<extra></extra>"
                ),
                row=1, col=1
            )

    # Add portfolio value to second subplot
    fig.add_trace(
        go.Scatter(
            x=simulation.index,
            y=simulation['portfolio_value'],
            name='Portfolio Value',
            line=dict(color='green')
        ),
        row=2, col=1
    )

    # Add metrics annotation if provided
    if metrics:
        metrics_text = (
            f"Total Return: {metrics['total_return']:.2%}<br>"
            f"Max Drawdown: {metrics['max_drawdown']:.2%}<br>"
            f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}"
        )

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.01,
            y=0.25,
            text=metrics_text,
            showarrow=False,
            font=dict(size=12),
            align="left"
        )

    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified'
    )

    # Update axes labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig
