import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class CalculateCost:
    def __init__(self, initial_investment=100, leverage=15, spread_pct=0.5,
                 variable_rate=0.04, fixed_rate=0.03):
        self.initial_investment = initial_investment
        self.current_value = initial_investment
        self.leverage = leverage
        self.spread_pct = spread_pct / 100  # Convert to decimal
        self.variable_rate = max(variable_rate, 0)  # NIBOR, minimum 0
        self.fixed_rate = fixed_rate  # Fixed 3% for Mini Futures
        # Daily rate calculation adjusted per documentation
        self.daily_rate = (self.variable_rate + self.fixed_rate) / 365
        self.cost_multiplier = self.daily_rate * (self.leverage - 1)

    def simple_calculate_holding_cost(self, portfolio_value):
        """Simple method for calculating holding cost"""
        print(self.cost_multiplier, portfolio_value)
        return self.cost_multiplier * portfolio_value

    def calculate_holding_costs(self, days, price_changes=None, current_value=None):
        """Base method for calculating costs and returns for holding a position"""
        if price_changes is None:
            price_changes = [0] * days
        elif isinstance(price_changes, (int, float)):
            price_changes = [price_changes] * days

        if current_value:
            self.current_value = current_value
        else:
            self.current_value = self.initial_investment

        # Initial costs (spread only applies when opening new positions)
        spread_cost = self.current_value * self.spread_pct

        # Daily financing cost based on leverage
        daily_costs = []
        cumulative_costs = []
        position_values = []

        for day, daily_change in enumerate(price_changes):
            # Calculate daily financing cost based on current position value
            daily_finance_cost = self._calculate_daily_finance_cost()
            daily_costs.append(daily_finance_cost)

            # Update cumulative costs
            total_cost = spread_cost + sum(daily_costs)
            cumulative_costs.append(total_cost)

            # Update current value based on price changes
            self.current_value = self._update_position_value(
                daily_change, day)

            # Track position value
            position_values.append(self.current_value - total_cost)

        return {
            'daily_costs': daily_costs,
            'cumulative_costs': cumulative_costs,
            'position_values': position_values
        }

    def _calculate_daily_finance_cost(self, value=None):
        """Calculate daily financing cost based on leverage"""
        # Use provided value or fall back to current_value
        position_value = value if value is not None else self.current_value
        leveraged_amount = position_value * (self.leverage - 1)
        return leveraged_amount * self.daily_rate

    def _update_position_value(self, daily_change, day, value=None):
        """Update position value - to be overridden by child classes"""
        position_value = value if value is not None else self.current_value
        return position_value * (1 + daily_change)


class MiniFutureAnalyzer(CalculateCost):
    def _calculate_daily_finance_cost(self, value=None):
        """Mini Future specific daily cost calculation"""
        position_value = value if value is not None else self.current_value
        borrowed_amount = position_value * (self.leverage - 1)
        return borrowed_amount * self.daily_rate

    def _update_position_value(self, daily_change, day, value=None):
        """Update position value with leverage effect"""
        position_value = value if value is not None else self.current_value
        leveraged_change = daily_change * self.leverage
        return position_value * (1 + leveraged_change)

    def compare_strategies(self, total_days, trade_interval, price_change_pct=0):
        """Compare holding vs periodic trading strategies"""
        # Strategy 1: Hold position
        hold_results = self.calculate_holding_costs(
            total_days, price_change_pct, self.initial_investment)

        # Strategy 2: Periodic trading
        daily_values = []
        total_cost = 0

        for day in range(total_days):
            if trade_interval == 1:
                # For daily trading: only spread costs, no financing
                daily_spread_cost = self.initial_investment * \
                    self.spread_pct * 2  # Buy and sell each day
                total_cost += daily_spread_cost
            else:
                # For other intervals: spread costs on trade days, no costs on other days
                if day % trade_interval == 0:
                    # Add spread costs for entry and exit based on initial investment
                    total_cost += self.initial_investment * self.spread_pct * 2

            # Update value based on initial investment minus accumulated costs
            current_value = self.initial_investment - total_cost
            daily_values.append(current_value)

        return {
            'hold_strategy': hold_results['position_values'],
            'trade_strategy': daily_values
        }

    def plot_comparison(self, total_days, trade_intervals=[1, 2, 3, 5, 10], price_change_pct=0):
        """Plot comparison of different trading strategies"""
        fig = make_subplots(rows=1, cols=1,
                            subplot_titles=['Strategy Comparison - Position Value Over Time'])

        # Plot hold strategy
        hold_results = self.calculate_holding_costs(
            total_days, price_change_pct)
        fig.add_trace(
            go.Scatter(
                x=list(range(total_days)),
                y=hold_results['position_values'],
                name='Hold Strategy',
                line=dict(color='blue')
            )
        )

        # Plot different trading intervals
        colors = ['red', 'purple', 'orange', 'green', 'brown']
        for interval, color in zip(trade_intervals, colors):
            results = self.compare_strategies(
                total_days, interval, price_change_pct)

            # Plot daily values
            fig.add_trace(
                go.Scatter(
                    x=list(range(total_days)),
                    y=results['trade_strategy'],
                    name=f'Trade Every {interval} Days',
                    line=dict(color=color)
                )
            )

            # Add markers for trade points
            trade_days = list(range(0, total_days, interval))
            if trade_days[-1] != total_days - 1:
                trade_days.append(total_days - 1)

            fig.add_trace(
                go.Scatter(
                    x=trade_days,
                    y=[results['trade_strategy'][i] for i in trade_days],
                    name=f'Trade Points ({interval} Days)',
                    mode='markers',
                    marker=dict(color=color, size=10),
                    showlegend=False
                )
            )

        # Update layout
        fig.update_layout(
            title='Trading Strategy Comparison',
            xaxis_title='Days',
            yaxis_title='Position Value (NOK)',
            height=600,
            showlegend=True,
            hovermode='x unified'
        )

        fig.show()


class BullBearAnalyzer(CalculateCost):
    def __init__(self, initial_investment=100, leverage=15, spread_pct=0.5):
        super().__init__(initial_investment, leverage, spread_pct)
        # Add tracking of previous day's change for rebalancing calculation
        self.previous_day_change = 0
        # Override rates based on leverage per documentation
        if leverage > 5:
            self.fixed_rate = 0.03  # 3% for leverage > 5
        elif leverage == 5:
            self.fixed_rate = 0.015  # 1.5% for leverage = 5
        else:
            self.fixed_rate = 0.01  # 1% for leverage < 5

        self.admin_fee = 0.0059  # 0.59% administration fee
        self.daily_rate = (self.fixed_rate + self.admin_fee) / 365

    def _calculate_daily_finance_cost(self, value=None):
        """Calculate daily financing cost based on leverage"""
        position_value = value if value is not None else self.current_value
        leveraged_amount = position_value * self.leverage
        return leveraged_amount * self.daily_rate

    def _update_position_value(self, daily_change, day, value=None):
        """Update position value with leverage and rebalancing effect"""
        position_value = value if value is not None else self.current_value

        # Calculate rebalancing decay from previous day's movement
        rebalancing_decay = 0
        if self.previous_day_change != 0:
            rebalancing_decay = position_value * \
                (self.leverage * (self.previous_day_change ** 2)) / 2

        # Apply leveraged price change
        leveraged_change = daily_change * self.leverage
        new_value = position_value * (1 + leveraged_change)

        # Subtract rebalancing decay after price movement
        new_value -= rebalancing_decay

        # Store current day's change for next day's rebalancing calculation
        self.previous_day_change = daily_change

        return new_value


def compare_products(days=31):
    """Compare Mini Futures vs Bull/Bear certificates for different leverages"""
    leverages = [2, 5, 15]

    # Create figure
    fig = make_subplots(rows=1, cols=1,
                        subplot_titles=['Mini Futures vs Bull/Bear Certificates Cost Comparison'])

    colors = ['blue', 'red']  # Mini Futures, Bull/Bear

    for leverage in leverages:
        # Initialize analyzers
        mini_future = MiniFutureAnalyzer(leverage=leverage)
        bull_bear = BullBearAnalyzer(leverage=leverage)

        # Calculate costs
        mini_results = mini_future.calculate_holding_costs(days)
        bull_results = bull_bear.calculate_holding_costs(days)

        # Plot Mini Future
        fig.add_trace(
            go.Scatter(
                x=list(range(days)),
                y=mini_results['position_values'],
                name=f'Mini Future {leverage}x',
                line=dict(color='blue', dash='solid' if leverage ==
                          15 else 'dash')
            )
        )

        # Plot Bull/Bear
        fig.add_trace(
            go.Scatter(
                x=list(range(days)),
                y=bull_results['position_values'],
                name=f'Bull/Bear {leverage}x',
                line=dict(color='red', dash='solid' if leverage ==
                          15 else 'dash')
            )
        )

    # Update layout
    fig.update_layout(
        title='Mini Futures vs Bull/Bear Certificates Cost Comparison',
        xaxis_title='Days',
        yaxis_title='Position Value (NOK)',
        height=600,
        showlegend=True,
        hovermode='x unified'
    )

    fig.show()


def compare_products_with_volatility(days=31, volatility=0.1):
    """
    Compare Mini Futures vs Bull/Bear certificates with volatile price movements
    First plot shows total portfolio values, second plot shows isolated costs
    """
    leverages = [2, 5, 15]

    # Generate a volatile but mean-reverting price series
    np.random.seed(42)  # For reproducibility
    price_changes = [0]  # Start with no change on day 0
    current_price = 100
    target_price = 100

    for _ in range(days-1):
        price_delta = (target_price - current_price) * \
            0.1 + np.random.normal(0, volatility)
        price_change = price_delta / current_price
        price_changes.append(price_change)
        current_price = current_price * (1 + price_change)

    # Create figure with 3 subplots
    fig = make_subplots(rows=3, cols=1,
                        subplot_titles=['Portfolio Values',
                                        'Product Costs', 'Underlying Asset Price'],
                        row_heights=[0.4, 0.4, 0.2])

    # Plot underlying asset price
    underlying_prices = [100]
    for change in price_changes:
        underlying_prices.append(underlying_prices[-1] * (1 + change))

    fig.add_trace(
        go.Scatter(x=list(range(days + 1)), y=underlying_prices,
                   name='Underlying Asset', line=dict(color='green')),
        row=3, col=1
    )

    for leverage in leverages:
        # Initialize analyzers
        mini_future = MiniFutureAnalyzer(leverage=leverage)
        bull_bear = BullBearAnalyzer(leverage=leverage)

        initial_investment = mini_future.initial_investment

        # Initialize arrays
        mini_portfolio_values = [initial_investment]
        bull_portfolio_values = [initial_investment, initial_investment]
        mini_costs = [0, 0]
        bull_total_costs = [0, 0]

        mini_cumulative_cost = 0
        bull_cumulative_cost = 0
        current_bull_value = initial_investment

        # Apply initial spread cost
        mini_cumulative_cost = - \
            abs(initial_investment * mini_future.spread_pct)
        bull_cumulative_cost = -abs(initial_investment * bull_bear.spread_pct)

        # Calculate cumulative price change from initial price
        cumulative_price_change = 0

        # Start from index 1
        for day, price_change in enumerate(price_changes[1:], 1):
            # Update cumulative price change for Mini Future
            cumulative_price_change = (
                underlying_prices[day] / underlying_prices[0]) - 1

            # Calculate Mini Future position value based on initial investment
            mini_position_value = initial_investment * \
                (1 + (cumulative_price_change * leverage))

            # Calculate Bull/Bear position value based on current value
            bull_value_change = current_bull_value * leverage * price_change
            current_bull_value += bull_value_change

            # Add daily financing costs
            if mini_position_value > 0:
                daily_mini_cost = - \
                    abs(mini_future._calculate_daily_finance_cost(
                        mini_position_value))
                mini_cumulative_cost += daily_mini_cost

            if current_bull_value > 0:
                daily_bull_financing = - \
                    abs(bull_bear._calculate_daily_finance_cost(current_bull_value))
                bull_cumulative_cost += daily_bull_financing

                # Add rebalancing cost for bull/bear
                if price_change != 0:
                    rebalancing_cost = - \
                        abs(current_bull_value *
                            (leverage * (price_change ** 2)) / 2)
                    bull_cumulative_cost += rebalancing_cost

            # Store values and costs
            mini_portfolio_values.append(
                max(0, mini_position_value + mini_cumulative_cost))
            bull_portfolio_values.append(
                max(0, current_bull_value + bull_cumulative_cost))
            mini_costs.append(mini_cumulative_cost)
            bull_total_costs.append(bull_cumulative_cost)

        # Plot portfolio values (first subplot)
        fig.add_trace(
            go.Scatter(
                x=list(range(days+1)),
                y=mini_portfolio_values,
                name=f'Mini Future {leverage}x',
                line=dict(color='blue', dash='solid' if leverage ==
                          15 else 'dash')
            ),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(days+1)),
                y=bull_portfolio_values,
                name=f'Bull/Bear {leverage}x',
                line=dict(color='red', dash='solid' if leverage ==
                          15 else 'dash')
            ),
            row=1, col=1
        )

        # Plot isolated costs (second subplot)
        fig.add_trace(
            go.Scatter(
                x=list(range(days+1)),
                y=mini_costs,
                name=f'Mini Future Costs {leverage}x',
                line=dict(color='blue', dash='solid' if leverage ==
                          15 else 'dash'),
                showlegend=False
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(days+1)),
                y=bull_total_costs,
                name=f'Bull/Bear Total Costs {leverage}x',
                line=dict(color='red', dash='solid' if leverage ==
                          15 else 'dash'),
                showlegend=False
            ),
            row=2, col=1
        )

    # Update layout
    fig.update_layout(
        title='Mini Futures vs Bull/Bear Certificates Comparison',
        height=1000,
        showlegend=True,
        hovermode='x unified'
    )
    fig.update_xaxes(title_text='Days', row=3, col=1)
    fig.update_yaxes(title_text='Portfolio Value (NOK)', row=1, col=1)
    fig.update_yaxes(title_text='Costs (NOK)', row=2, col=1)
    fig.update_yaxes(title_text='Underlying Price', row=3, col=1)

    fig.show()


"""
# Create analyzer instance
analyzer = MiniFutureAnalyzer(
    initial_investment=100,  # 10,000 NOK
    leverage=15,              # 15x leverage
    spread_pct=0.5,          # 0.5% spread
    variable_rate=0.04,      # 4% variable rate (NIBOR)
    fixed_rate=0.03          # 3% fixed rate
)

# Compare strategies over 30 days with different trading intervals
analyzer.plot_comparison(
    total_days=31,
    # Compare daily, 2-day, 3-day, weekly, and 10-day trading
    trade_intervals=[1, 2, 3, 5, 10],
    price_change_pct=0           # Assume stationary price
)

compare_products()

# Run comparison with volatility
compare_products_with_volatility(
    days=31, volatility=5)  # 2% daily volatility
"""
