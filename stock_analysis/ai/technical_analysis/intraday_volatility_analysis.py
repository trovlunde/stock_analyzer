import pandas as pd
import matplotlib.pyplot as plt
from ..helpers import get_index_data
import seaborn as sns


def calculate_intraday_metrics(ticker='^GSPC', period='1y', threshold=0.5):
    """
    Calculate intraday volatility and volume metrics for a given ticker.

    Args:
        ticker (str): Stock ticker symbol (default: '^GSPC' for S&P 500)
        period (str): Time period to analyze (default: '1y')

    Returns:
        pandas.DataFrame: DataFrame with intraday volatility and volume metrics
    """
    # Get the data
    data = get_index_data(ticker, period=period)

    # Calculate intraday volatility (High-Low)/Open
    data['intraday_volatility'] = (
        data['High'] - data['Low']) / data['Open'] * 100

    # Calculate returns from open
    data['return_from_open_high'] = (
        data['High'] - data['Open']) / data['Open'] * 100
    data['return_from_open_low'] = (
        data['Low'] - data['Open']) / data['Open'] * 100

    # Classify days based on 0.5% threshold from open
    data['up_move_day'] = data['return_from_open_high'] > threshold
    data['down_move_day'] = data['return_from_open_low'] < -threshold
    data['significant_move_day'] = data['up_move_day'] | data['down_move_day']
    data['return_percentile'] = data['return_from_open_high'].abs().rank(
        pct=True) * 100

    # Calculate normalized volume (volume relative to 20-day moving average)
    data['volume_ma20'] = data['Volume'].rolling(window=20).mean()
    data['normalized_volume'] = data['Volume'] / data['volume_ma20']

    # Calculate additional metrics
    data['volatility_ma5'] = data['intraday_volatility'].rolling(
        window=5).mean()
    data['volatility_ma20'] = data['intraday_volatility'].rolling(
        window=20).mean()

    return data


def analyze_return_distribution(data):
    """
    Analyze the distribution of intraday returns and provide statistics.

    Args:
        data (pandas.DataFrame): DataFrame containing the return metrics

    Returns:
        dict: Dictionary containing return statistics
    """
    total_days = len(data)
    up_move_days = data['up_move_day'].sum()
    down_move_days = data['down_move_day'].sum()
    significant_move_days = data['significant_move_day'].sum()

    # Calculate exclusive categories
    up_only_days = data[data['up_move_day'] & ~data['down_move_day']].shape[0]
    down_only_days = data[~data['up_move_day']
                          & data['down_move_day']].shape[0]
    both_direction_days = data[data['up_move_day']
                               & data['down_move_day']].shape[0]
    no_move_days = data[~data['up_move_day'] & ~data['down_move_day']].shape[0]

    stats = {
        'total_days': total_days,
        'up_move_days': up_move_days,
        'down_move_days': down_move_days,
        'significant_move_days': significant_move_days,
        'up_move_percentage': (up_move_days / total_days) * 100,
        'down_move_percentage': (down_move_days / total_days) * 100,
        'significant_move_percentage': (significant_move_days / total_days) * 100,

        # Exclusive categories
        'up_only_days': up_only_days,
        'down_only_days': down_only_days,
        'both_direction_days': both_direction_days,
        'no_move_days': no_move_days,
        'up_only_percentage': (up_only_days / total_days) * 100,
        'down_only_percentage': (down_only_days / total_days) * 100,
        'both_direction_percentage': (both_direction_days / total_days) * 100,
        'no_move_percentage': (no_move_days / total_days) * 100,

        'high_return_percentiles': {
            '25th': data['return_from_open_high'].quantile(0.25),
            '50th': data['return_from_open_high'].quantile(0.50),
            '75th': data['return_from_open_high'].quantile(0.75),
            '90th': data['return_from_open_high'].quantile(0.90),
            '95th': data['return_from_open_high'].quantile(0.95),
        },
        'low_return_percentiles': {
            '25th': data['return_from_open_low'].quantile(0.25),
            '50th': data['return_from_open_low'].quantile(0.50),
            '75th': data['return_from_open_low'].quantile(0.75),
            '90th': data['return_from_open_low'].quantile(0.90),
            '95th': data['return_from_open_low'].quantile(0.95),
        }
    }

    return stats


def plot_intraday_metrics(data, title="S&P 500 Intraday Analysis", threshold=0.5):
    """
    Plot intraday metrics including returns from open.

    Args:
        data (pandas.DataFrame): DataFrame containing the metrics to plot
        title (str): Title for the plot
    """
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(
        4, 1, figsize=(15, 15), sharex=True)

    # Plot 1: Returns from Open
    ax1.fill_between(data.index, data['return_from_open_high'],
                     data['return_from_open_low'], alpha=0.3, color='gray')
    ax1.axhline(y=threshold, color='red', linestyle='--',
                alpha=0.5, label=f'+{threshold}% Threshold')
    ax1.axhline(y=-threshold, color='red', linestyle='--',
                alpha=0.5, label=f'-{threshold}% Threshold')
    ax1.set_title(f"{title}\nIntraday Return Range from Open")
    ax1.set_ylabel('Return from Open (%)')
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Volatility
    ax2.plot(data.index, data['intraday_volatility'],
             alpha=0.5, label='Daily Volatility', color='gray')
    ax2.plot(data.index, data['volatility_ma5'],
             label='5-day MA', color='blue')
    ax2.plot(data.index, data['volatility_ma20'],
             label='20-day MA', color='red')
    ax2.set_title('Intraday Volatility')
    ax2.set_ylabel('Volatility (%)')
    ax2.legend()
    ax2.grid(True)

    # Plot 3: Normalized Volume
    ax3.plot(data.index, data['normalized_volume'],
             label='Normalized Volume', color='purple')
    ax3.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    ax3.set_title('Normalized Volume')
    ax3.set_ylabel('Volume (relative to 20-day MA)')
    ax3.legend()
    ax3.grid(True)

    # Plot 4: Price over time
    ax4.plot(data.index, data['Close'], label='Price', color='green')
    ax4.set_title('Price History')
    ax4.set_ylabel('Price ($)')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    return fig


def analyze_trading_strategy(data, target_gain=0.3, stop_loss=0.3, spread_cost=0.5, leverage=1):
    """
    Analyze a trading strategy that buys at open and sells when target is reached.

    Args:
        data (pandas.DataFrame): DataFrame with price data
        target_gain (float): Target gain percentage to trigger sell
        stop_loss (float): Stop loss percentage (use None for hold-till-close strategy)
        spread_cost (float): Trading cost as percentage of investment
        leverage (float): Leverage multiplier

    Returns:
        dict: Dictionary containing strategy performance metrics
    """
    # Create copy to avoid modifying original data
    analysis = data.copy()

    # Initialize results columns
    analysis['trade_return'] = 0.0
    analysis['trade_taken'] = False
    analysis['hit_target'] = False
    analysis['hit_stop'] = False

    # Calculate returns for each day
    for idx in analysis.index:
        day = analysis.loc[idx]

        # Skip days where price moves both up and down beyond thresholds
        if day['return_from_open_high'] > target_gain and day['return_from_open_low'] < -target_gain:
            analysis.loc[idx, 'trade_return'] = -spread_cost
            analysis.loc[idx, 'trade_taken'] = True
            continue

        # Check if target was hit
        if day['return_from_open_high'] >= target_gain:
            analysis.loc[idx, 'trade_return'] = (
                target_gain * leverage) - spread_cost
            analysis.loc[idx, 'trade_taken'] = True
            analysis.loc[idx, 'hit_target'] = True
            continue

        # Check stop loss if applicable
        if stop_loss is not None and day['return_from_open_low'] <= -stop_loss:
            analysis.loc[idx,
                         'trade_return'] = (-stop_loss * leverage) - spread_cost
            analysis.loc[idx, 'trade_taken'] = True
            analysis.loc[idx, 'hit_stop'] = True
            continue

        # If neither target nor stop was hit, return is close minus open (minus spread)
        if stop_loss is None:  # Hold-till-close strategy
            day_return = ((day['Close'] - day['Open']) / day['Open'] * 100)
            analysis.loc[idx, 'trade_return'] = (
                day_return * leverage) - spread_cost
            analysis.loc[idx, 'trade_taken'] = True

    # Calculate strategy metrics
    trades_taken = analysis['trade_taken'].sum()
    winning_trades = analysis[analysis['trade_taken']
                              ]['trade_return'].gt(0).sum()
    total_return = analysis['trade_return'].sum()
    avg_return_per_trade = analysis['trade_return'][analysis['trade_taken']].mean(
    )
    hit_target_count = analysis['hit_target'].sum()
    hit_stop_count = analysis['hit_stop'].sum()

    strategy_stats = {
        'total_trades': trades_taken,
        'winning_trades': winning_trades,
        'win_rate': (winning_trades / trades_taken * 100) if trades_taken > 0 else 0,
        'total_return': total_return,
        'avg_return_per_trade': avg_return_per_trade,
        'hit_target_count': hit_target_count,
        'hit_stop_count': hit_stop_count,
        'target_hit_rate': (hit_target_count / trades_taken * 100) if trades_taken > 0 else 0,
        'stop_hit_rate': (hit_stop_count / trades_taken * 100) if trades_taken > 0 else 0,
    }

    return strategy_stats


# Example usage:
if __name__ == "__main__":
    threshold = 0.2
    spread_cost = 0.3
    # Calculate metrics for S&P 500
    sp500_data = calculate_intraday_metrics(period='5y', threshold=threshold)

    # Analyze return distribution
    ret_stats = analyze_return_distribution(sp500_data)
    print("\nReturn Analysis:")
    print(f"Total trading days analyzed: {ret_stats['total_days']}")

    print("\nMove Categories (Exclusive):")
    print(
        f"Up moves only (>{threshold}%): {ret_stats['up_only_days']} days ({ret_stats['up_only_percentage']:.2f}%)")
    print(
        f"Down moves only (<-{threshold}%): {ret_stats['down_only_days']} days ({ret_stats['down_only_percentage']:.2f}%)")
    print(
        f"Both up and down moves: {ret_stats['both_direction_days']} days ({ret_stats['both_direction_percentage']:.2f}%)")
    print(
        f"No significant moves: {ret_stats['no_move_days']} days ({ret_stats['no_move_percentage']:.2f}%)")

    print("\nTotal Moves (Including Overlaps):")
    print(
        f"Days with >{threshold}% upward move: {ret_stats['up_move_days']} ({ret_stats['up_move_percentage']:.2f}%)")
    print(
        f"Days with >{threshold}% downward move: {ret_stats['down_move_days']} ({ret_stats['down_move_percentage']:.2f}%)")
    print(
        f"Total days with significant moves: {ret_stats['significant_move_days']} ({ret_stats['significant_move_percentage']:.2f}%)")

    print("\nUpward Move Percentiles:")
    for percentile, value in ret_stats['high_return_percentiles'].items():
        print(f"{percentile}: {value:.2f}%")

    print("\nDownward Move Percentiles:")
    for percentile, value in ret_stats['low_return_percentiles'].items():
        print(f"{percentile}: {value:.2f}%")

    # Create and show the plot
    fig = plot_intraday_metrics(sp500_data, threshold=threshold)
    plt.show()

    print("\nTrading Strategy Analysis:")
    print(
        f"\nStrategy 1: Hold till close if target not hit (target gain: {threshold}%)")
    strategy1_stats = analyze_trading_strategy(
        sp500_data,
        target_gain=threshold,
        stop_loss=None,
        spread_cost=spread_cost,
        leverage=15
    )
    print(f"Total trades: {strategy1_stats['total_trades']}")
    print(f"Win rate: {strategy1_stats['win_rate']:.2f}%")
    print(f"Total return: {strategy1_stats['total_return']:.2f}%")
    print(
        f"Average return per trade: {strategy1_stats['avg_return_per_trade']:.2f}%")
    print(f"Target hit rate: {strategy1_stats['target_hit_rate']:.2f}%")

    print(
        f"\nStrategy 2: Stop loss at -{threshold}%")
    strategy2_stats = analyze_trading_strategy(
        sp500_data,
        target_gain=threshold,
        stop_loss=threshold,
        spread_cost=spread_cost,
        leverage=15
    )
    print(f"Total trades: {strategy2_stats['total_trades']}")
    print(f"Win rate: {strategy2_stats['win_rate']:.2f}%")
    print(f"Total return: {strategy2_stats['total_return']:.2f}%")
    print(
        f"Average return per trade: {strategy2_stats['avg_return_per_trade']:.2f}%")
    print(f"Target hit rate: {strategy2_stats['target_hit_rate']:.2f}%")
    print(f"Stop loss hit rate: {strategy2_stats['stop_hit_rate']:.2f}%")
