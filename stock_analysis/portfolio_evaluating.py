import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stock_analysis.trading_cost_analysis import MiniFutureAnalyzer


def normalize_trading_signals(signals: pd.DataFrame) -> pd.DataFrame:
    """Ensure signals use a DatetimeIndex and a single prediction column."""
    signals = signals.copy()
    if 'prediction' not in signals.columns:
        if 'daily_pred' in signals.columns:
            signals = signals.rename(columns={'daily_pred': 'prediction'})
        elif 'weekly_pred' in signals.columns:
            signals = signals.rename(columns={'weekly_pred': 'prediction'})
        else:
            raise ValueError("signals must include a 'prediction' column")

    if 'date' in signals.columns:
        signals = signals.set_index('date')
    elif signals.index.name == 'date':
        signals.index.name = None

    signals.index = pd.to_datetime(signals.index)
    return signals[['prediction']]


def analyze_trading_strategies(stock_data, signals, initial_investment=10000, leverage=10, signal_type=""):
    """
    Analyze two different trading strategies based on signals:
    1. Day trading: Enter position at signal and exit at end of day
    2. Position trading: Hold position until opposite signal

    Args:
        stock_data (pd.DataFrame): DataFrame with 'Open', 'High', 'Low', 'Close' prices
        signals (pd.DataFrame): DataFrame with trading signals ('prediction' column)
        initial_investment (float): Initial investment amount
        leverage (int): Leverage multiplier for positions
        signal_type (str): Type of signals being analyzed (e.g., "Daily Model", "Weekly Model", "Combined")

    Returns:
        dict: Results of both strategies including daily values and metrics
    """
    signals = normalize_trading_signals(signals)

    # Check and align data ranges
    first_signal_date = signals.index.min()
    last_signal_date = signals.index.max()

    # Filter stock_data to match signals date range
    stock_data = stock_data[
        (stock_data.index >= first_signal_date) &
        (stock_data.index <= last_signal_date)
    ]

    if len(stock_data) == 0:
        raise ValueError("No overlapping dates between stock data and signals")

    # Initialize TradingCostAnalyzer for cost calculations
    cost_analyzer = MiniFutureAnalyzer(
        initial_investment=initial_investment,
        leverage=leverage,
        spread_pct=0.5,  # 0.5% spread
        variable_rate=0.04,  # 4% variable rate
        fixed_rate=0.03  # 3% fixed rate
    )

    # Strategy 1: Day Trading
    day_trading_values = [initial_investment] * len(stock_data)
    day_trading_positions = ['none'] * len(stock_data)
    current_value = float(initial_investment)
    is_bankrupt = False

    # Add cost tracking for day trading (only spread costs since positions are closed daily)
    day_trading_spread_costs = [0] * len(stock_data)
    day_trading_holding_costs = [0] * len(stock_data)  # Will remain zeros
    day_trading_total_costs = [0] * len(stock_data)
    accumulated_spread_cost_day = 0

    for i, date in enumerate(stock_data.index):
        if is_bankrupt:
            day_trading_values[i] = 0
            day_trading_positions[i] = 'none'
            continue

        # First, apply any returns from previous day's position
        if i > 0 and day_trading_positions[i-1] in ['long', 'short']:
            # Calculate return based on day-to-day price movement
            price_change = float(
                (stock_data.loc[date, 'Close'] -
                 stock_data.loc[stock_data.index[i-1], 'Close']
                 ) / stock_data.loc[stock_data.index[i-1], 'Close']
            )

            # Calculate position return
            position_multiplier = 1 if day_trading_positions[i -
                                                             1] == 'long' else -1
            position_return = float(
                current_value * price_change * position_multiplier * leverage)

            # Update current value (no holding costs for day trading)
            current_value = float(current_value + position_return)

        # Then, set position for today based on signal
        position = 'none'
        if date in signals.index:
            signal = signals.loc[date, 'prediction']
            if signal == 'positive':
                position = 'long'
            elif signal == 'negative':
                position = 'short'

        day_trading_values[i] = current_value
        day_trading_positions[i] = position

        # When opening a new position, add spread cost
        if position != 'none' and (i == 0 or day_trading_positions[i-1] == 'none'):
            spread_cost = current_value * cost_analyzer.spread_pct
            current_value -= spread_cost
            accumulated_spread_cost_day += spread_cost

        # When closing a position (end of day), add spread cost
        if position != 'none':
            spread_cost = current_value * cost_analyzer.spread_pct
            current_value -= spread_cost
            accumulated_spread_cost_day += spread_cost

        day_trading_spread_costs[i] = accumulated_spread_cost_day
        # Total costs are just spread costs
        day_trading_total_costs[i] = accumulated_spread_cost_day

    # Strategy 2: Position Trading
    position_trading_values = [initial_investment] * len(stock_data)
    position_trading_positions = ['none'] * len(stock_data)
    current_value = float(initial_investment)
    current_position = 'none'
    entry_price = None  # Track entry price for position
    position_size = 0   # Track the size of position in currency
    is_bankrupt = False

    # Add cost tracking for position trading
    position_trading_spread_costs = [0] * len(stock_data)
    position_trading_holding_costs = [0] * len(stock_data)
    position_trading_total_costs = [0] * len(stock_data)
    accumulated_spread_cost_pos = 0
    accumulated_holding_cost_pos = 0

    for i, date in enumerate(stock_data.index):
        if is_bankrupt:
            position_trading_values[i] = 0
            position_trading_positions[i] = 'none'
            continue

        # Calculate returns if we have a position
        if current_position in ['long', 'short'] and entry_price is not None:
            current_price = stock_data.loc[date, 'Close']
            previous_price = stock_data.loc[stock_data.index[i-1], 'Close']

            # Calculate daily return based on previous day's close
            price_change = float(
                (current_price - previous_price) / previous_price)
            position_multiplier = 1 if current_position == 'long' else -1

            # Calculate position return based on current value
            position_return = float(
                current_value * price_change * leverage * position_multiplier)

            # Update current value with the return
            current_value = float(current_value + position_return)

            # Calculate daily holding cost based on the borrowed amount (not full leveraged position)
            # This is the amount we're actually borrowing

            daily_holding_cost = float(cost_analyzer.simple_calculate_holding_cost(
                portfolio_value=current_value
            ))
            accumulated_holding_cost_pos += daily_holding_cost
            current_value -= daily_holding_cost

        # Process signals and position changes
        if date in signals.index:
            signal = signals.loc[date, 'prediction']
            next_days = stock_data.index[stock_data.index > date]

            if signal in ['positive', 'negative'] and current_value > 0 and len(next_days) > 0:
                next_day = next_days[0]

                # Close existing position if opposite signal
                if ((current_position == 'long' and signal == 'negative') or
                        (current_position == 'short' and signal == 'positive')):
                    # Apply spread cost for closing position
                    spread_cost = current_value * cost_analyzer.spread_pct
                    current_value -= spread_cost
                    accumulated_spread_cost_pos += spread_cost
                    current_position = 'none'
                    entry_price = None
                    position_size = 0

                # Open new position starting next day
                if current_position == 'none':
                    # Apply spread cost for opening position
                    spread_cost = current_value * cost_analyzer.spread_pct
                    current_value -= spread_cost
                    accumulated_spread_cost_pos += spread_cost
                    current_position = 'long' if signal == 'positive' else 'short'
                    # Store entry price
                    entry_price = stock_data.loc[date, 'Close']
                    position_size = current_value * leverage  # Store leveraged position size

        # Check for bankruptcy
        if current_value <= 0:
            is_bankrupt = True
            current_value = 0
            current_position = 'none'
            entry_price = None
            position_size = 0

        position_trading_values[i] = current_value
        position_trading_positions[i] = current_position
        position_trading_holding_costs[i] = accumulated_holding_cost_pos
        position_trading_spread_costs[i] = accumulated_spread_cost_pos
        position_trading_total_costs[i] = accumulated_holding_cost_pos + \
            accumulated_spread_cost_pos

    # Create results DataFrame
    results = pd.DataFrame({
        'Date': stock_data.index,
        'Price': stock_data['Close'],
        'Day_Trading_Value': day_trading_values,
        'Day_Trading_Position': day_trading_positions,
        'Day_Trading_Spread_Costs': day_trading_spread_costs,
        'Day_Trading_Holding_Costs': day_trading_holding_costs,
        'Day_Trading_Total_Costs': day_trading_total_costs,
        'Position_Trading_Value': position_trading_values,
        'Position_Trading_Position': position_trading_positions,
        'Position_Trading_Spread_Costs': position_trading_spread_costs,
        'Position_Trading_Holding_Costs': position_trading_holding_costs,
        'Position_Trading_Total_Costs': position_trading_total_costs
    })
    print(results)

    # Calculate metrics
    metrics = {
        'Day_Trading': {
            'Total_Return': (day_trading_values[-1] - initial_investment) / initial_investment,
            'Max_Drawdown': calculate_max_drawdown(day_trading_values),
            'Sharpe_Ratio': calculate_sharpe_ratio(day_trading_values)
        },
        'Position_Trading': {
            'Total_Return': (position_trading_values[-1] - initial_investment) / initial_investment,
            'Max_Drawdown': calculate_max_drawdown(position_trading_values),
            'Sharpe_Ratio': calculate_sharpe_ratio(position_trading_values)
        }
    }

    # Create results dictionary
    results_dict = {
        'results': results,
        'metrics': metrics
    }

    # Update the plot title to include signal type
    title = f"Trading Analysis - {signal_type} Signals (Leverage: {leverage}x)"
    plot_trading_analysis(stock_data, results, title=title)

    return results_dict


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
    returns = pd.Series(values).pct_change().dropna()
    excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
    if len(excess_returns) > 0:
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    return 0


def plot_trading_analysis(stock_data, results, title="Trading Analysis"):
    """
    Plot trading signals and portfolio performance.

    Args:
        stock_data (pd.DataFrame): DataFrame with price data
        results (pd.DataFrame): DataFrame with trading signals and portfolio values
        title (str): Plot title
    """
    # Calculate next day returns (shift -1 to get next day's return)
    next_day_returns = stock_data['Close'].pct_change().shift(-1) * 100

    # Create figure with three subplots instead of two
    fig = make_subplots(rows=3, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=('Price and Signals',
                                        'Portfolio Value (Log Scale)',
                                        'Accumulated Costs'))

    # Plot 1: Price and Signals
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=stock_data['Close'],
            name='Price',
            line=dict(color='grey', width=1),
            hovertemplate="<br>".join([
                "Date: %{x}",
                "Price: $%{y:.2f}",
                "Next Day Return: %{customdata:.2f}%",
                "<extra></extra>"
            ]),
            customdata=next_day_returns
        ),
        row=1, col=1
    )

    # Add long signals — use Price from results (same row as Date), not integer index
    long_mask_day = results['Day_Trading_Position'] == 'long'
    if long_mask_day.any():
        fig.add_trace(
            go.Scatter(
                x=results.loc[long_mask_day, 'Date'],
                y=results.loc[long_mask_day, 'Price'],
                mode='markers',
                name='Day Trade Long',
                marker=dict(
                    symbol='triangle-up',
                    size=10,
                    color='green',
                    opacity=0.7
                )
            ),
            row=1, col=1
        )

    # Add short signals
    short_mask_day = results['Day_Trading_Position'] == 'short'
    if short_mask_day.any():
        fig.add_trace(
            go.Scatter(
                x=results.loc[short_mask_day, 'Date'],
                y=results.loc[short_mask_day, 'Price'],
                mode='markers',
                name='Day Trade Short',
                marker=dict(
                    symbol='triangle-down',
                    size=10,
                    color='red',
                    opacity=0.7
                )
            ),
            row=1, col=1
        )

    # Add position trading signals
    long_mask_pos = results['Position_Trading_Position'] == 'long'
    if long_mask_pos.any():
        fig.add_trace(
            go.Scatter(
                x=results.loc[long_mask_pos, 'Date'],
                y=results.loc[long_mask_pos, 'Price'],
                mode='markers',
                name='Position Long',
                marker=dict(
                    symbol='circle',
                    size=8,
                    color='lightgreen',
                    opacity=0.7
                )
            ),
            row=1, col=1
        )

    short_mask_pos = results['Position_Trading_Position'] == 'short'
    if short_mask_pos.any():
        fig.add_trace(
            go.Scatter(
                x=results.loc[short_mask_pos, 'Date'],
                y=results.loc[short_mask_pos, 'Price'],
                mode='markers',
                name='Position Short',
                marker=dict(
                    symbol='circle',
                    size=8,
                    color='lightcoral',
                    opacity=0.7
                )
            ),
            row=1, col=1
        )

    # Plot 2: Portfolio Values
    fig.add_trace(
        go.Scatter(
            x=results['Date'],
            y=results['Day_Trading_Value'],
            name='Day Trading Portfolio',
            line=dict(color='blue')
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=results['Date'],
            y=results['Position_Trading_Value'],
            name='Position Trading Portfolio',
            line=dict(color='orange')
        ),
        row=2, col=1
    )

    # Add third plot for costs
    # Day Trading Costs
    fig.add_trace(
        go.Scatter(
            x=results['Date'],
            y=results['Day_Trading_Spread_Costs'],
            name='Day Trading Spread Costs',
            line=dict(color='blue', dash='dot')
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=results['Date'],
            y=results['Day_Trading_Holding_Costs'],
            name='Day Trading Holding Costs',
            line=dict(color='blue', dash='dash')
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=results['Date'],
            y=results['Day_Trading_Total_Costs'],
            name='Day Trading Total Costs',
            line=dict(color='blue')
        ),
        row=3, col=1
    )

    # Position Trading Costs
    fig.add_trace(
        go.Scatter(
            x=results['Date'],
            y=results['Position_Trading_Spread_Costs'],
            name='Position Trading Spread Costs',
            line=dict(color='orange', dash='dot')
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=results['Date'],
            y=results['Position_Trading_Holding_Costs'],
            name='Position Trading Holding Costs',
            line=dict(color='orange', dash='dash')
        ),
        row=3, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=results['Date'],
            y=results['Position_Trading_Total_Costs'],
            name='Position Trading Total Costs',
            line=dict(color='orange')
        ),
        row=3, col=1
    )

    # Update layout for three plots
    fig.update_layout(
        height=1000,  # Increased height for three plots
        title_text=title,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        hovermode='x unified'
    )

    # Update y-axes labels for all three plots
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value", type="log", row=2, col=1)
    fig.update_yaxes(title_text="Accumulated Costs", row=3, col=1)

    fig.show()
