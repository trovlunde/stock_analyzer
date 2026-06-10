import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import numpy as np
from collections import Counter
from plottable import Table, ColDef, ColumnDefinition
from matplotlib.cm import ScalarMappable  # Red-Yellow-Green colormap
from matplotlib.colors import LinearSegmentedColormap


def find_undervalued_sectors(marketTicker='^GSPC'):
    stocks = yf.Tickers(market=marketTicker)
    info = stocks.info
    print(info)


def analyze_and_display_sector_metrics(stocks_df, min_market_cap=1e9, show_summary_table=True):
    """
    Analyze and visualize key metrics for each sector

    Args:
        stocks_df (pd.DataFrame): DataFrame containing stock data
        min_market_cap (float): Minimum market cap filter
        show_summary_table (bool): Whether to print summary statistics table
    """
    # Convert numeric columns and handle non-numeric values
    numeric_columns = ['market_cap', 'forward_pe', 'trailing_pe', 'price_to_book',
                       'profit_margins', 'operating_margins']

    df = stocks_df.copy()
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Filter by market cap
    df = df[df['market_cap'] >= min_market_cap].copy()

    if df.empty:
        print("No stocks found meeting the minimum market cap requirement")
        return None, None

    # Create visualizations
    metrics = {
        'forward_pe': 'Forward P/E',
        'trailing_pe': 'Trailing P/E',
        'price_to_book': 'Price to Book',
        'profit_margins': 'Profit Margins',
        'operating_margins': 'Operating Margins'
    }

    # Create subplot grid
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2

    fig = plt.figure(figsize=(16, 6 * n_rows))

    # Create box plots for each metric
    for idx, (metric, title) in enumerate(metrics.items(), 1):
        if metric not in df.columns:
            continue

        # Skip if no valid numeric data
        if df[metric].notna().sum() == 0:
            continue

        ax = plt.subplot(n_rows, n_cols, idx)

        # Filter out extreme outliers for cleaner visualization
        metric_data = df[['sector', metric]].dropna()
        metric_data = metric_data[~metric_data[metric].isin(
            [float('inf'), float('-inf')])]

        # Cap outliers at 95th percentile for better visualization
        if len(metric_data) > 0:
            q95 = metric_data[metric].quantile(0.95)
            q5 = metric_data[metric].quantile(0.05)
            # Only cap if there are extreme outliers
            if q95 - q5 > 0:
                metric_data = metric_data[
                    (metric_data[metric] >= q5 - 3 * (q95 - q5)) &
                    (metric_data[metric] <= q95 + 3 * (q95 - q5))
                ]

        # Calculate sector averages (excluding NaN values)
        sector_stats = metric_data.groupby('sector')[metric].agg(
            ['mean', 'median', 'std']).round(2)

        # Create box plot with outlier handling
        sns.boxplot(data=metric_data, x='sector', y=metric, ax=ax,
                    showfliers=True, fliersize=3)

        # Customize plot
        ax.set_title(f'{title} by Sector', fontsize=12, pad=10)
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=45, ha='right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.set_xlabel('')

        # Add sector averages as text (only if there's space)
        y_max = ax.get_ylim()[1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        text_y = y_max + y_range * 0.05

        for i, sector in enumerate(sector_stats.index):
            stats = sector_stats.loc[sector]
            if not pd.isna(stats['mean']):
                # Only add text if it won't cause overlap
                try:
                    ax.text(i, text_y, f'μ:{stats["mean"]:.1f}',
                            ha='center', va='bottom', fontsize=7, alpha=0.7)
                except:
                    pass  # Skip if text can't be placed

    plt.tight_layout()
    plt.show()

    # Print summary statistics table if requested
    if show_summary_table:
        print("\n" + "="*80)
        print("SECTOR SUMMARY STATISTICS")
        print("="*80)
        for metric, title in metrics.items():
            if metric in df.columns:
                metric_data = df[['sector', metric]].dropna()
                metric_data = metric_data[~metric_data[metric].isin(
                    [float('inf'), float('-inf')])]
                if len(metric_data) > 0:
                    sector_summary = metric_data.groupby('sector')[metric].agg(
                        ['mean', 'median', 'std', 'count']).round(2)
                    print(f"\n{title}:")
                    print(sector_summary.to_string())
        print("\n" + "="*80)

    # Create summary table
    summary_stats = []

    # Filter out stocks with no sector information
    valid_sectors = df['sector'].dropna().unique()

    for sector in valid_sectors:
        # Skip 'N/A' sectors
        if sector == 'N/A':
            continue

        sector_data = df[df['sector'] == sector]

        stats = {
            'Sector': sector,
            'Number of Companies': len(sector_data),
            'Avg Market Cap ($B)': sector_data['market_cap'].mean() / 1e9
        }

        for metric in metrics:
            if metric in sector_data.columns:
                stats[f'Avg {metrics[metric]}'] = sector_data[metric].mean()
                stats[f'Median {metrics[metric]}'] = sector_data[metric].median()

        summary_stats.append(stats)

    summary_df = pd.DataFrame(summary_stats)

    # Format float values to 2 decimal places
    float_cols = summary_df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        summary_df[col] = summary_df[col].round(2)

    # Create visualization with column-specific coloring
    fig, ax = plt.subplots(figsize=(15, len(summary_df) * 0.5))

    # Define limits for each metric (you might want to adjust these)
    metric_limits = {
        'Number of Companies': [0, summary_df['Number of Companies'].max() * 1.2],
        'Avg Market Cap ($B)': [0, summary_df['Avg Market Cap ($B)'].max() * 1.2],
        'Avg Forward P/E': [0, 30],
        'Median Forward P/E': [0, 30],
        'Avg Trailing P/E': [0, 35],
        'Median Trailing P/E': [0, 35],
        'Avg Price to Book': [0, 5],
        'Median Price to Book': [0, 5],
        'Avg Profit Margins': [-0.1, 0.3],
        'Median Profit Margins': [-0.1, 0.3],
        'Avg Operating Margins': [-0.1, 0.3],
        'Median Operating Margins': [-0.1, 0.3]
    }

    # Create a numeric DataFrame for coloring and keep original for display
    display_df = summary_df.copy()
    df_norm = pd.DataFrame(index=summary_df.index, columns=summary_df.columns)

    for col in summary_df.columns:
        if col in metric_limits:
            # Remove inf values for normalization
            valid_data = summary_df[col].replace([np.inf, -np.inf], np.nan)
            col_min, col_max = metric_limits[col]
            df_norm[col] = (valid_data - col_min) / (col_max - col_min)
        else:
            df_norm[col] = np.nan  # Use NaN for non-numeric columns

    # Create heatmap for each column
    for col in df_norm.columns:
        if col == 'Sector':  # Skip non-numeric column
            continue

        colors = [
            [0.0, '#ff4444'],  # red
            [0.3, '#ffcccc'],  # light red
            [0.7, '#ccffcc'],  # light green
            [1.0, '#44ff44']   # green
        ]
        cmap = LinearSegmentedColormap.from_list('', colors)

        mask = pd.DataFrame(True, index=df_norm.index, columns=df_norm.columns)
        mask[col] = False

        sns.heatmap(data=df_norm,
                    annot=display_df,  # Use original values for display
                    annot_kws={'size': 8},
                    fmt='.2f',
                    mask=mask,
                    cmap=cmap,
                    vmin=0,
                    vmax=1,
                    cbar=False,
                    ax=ax)

    ax.set_facecolor('white')
    ax.set_yticklabels(summary_df['Sector'], rotation=0)

    # Add colorbar
    colors = [[0, '#ff4444'],
              [0.3, '#ffcccc'],
              [0.7, '#ccffcc'],
              [1, '#44ff44']]
    cmap = LinearSegmentedColormap.from_list('', colors)
    cbar = plt.colorbar(ScalarMappable(cmap=cmap),
                        ax=ax, ticks=[0, 0.3, 0.7, 1])
    cbar.ax.yaxis.set_ticklabels(['min\nlimit', 'min', 'max', 'max\nlimit'])

    plt.tight_layout()
    plt.show()

    # Save summary to CSV
    filename = f"sector_analysis_{time.strftime('%Y%m%d')}.csv"
    summary_df.to_csv(filename, index=False)
    print(f"\nSector analysis saved to {filename}")

    return summary_df, df


def display_sector_comparison(sector_data, metrics=['forward_pe'], filter_outliers=True):
    """
    Create a detailed comparison visualization for multiple metrics across sectors

    Args:
        sector_data (pd.DataFrame): DataFrame containing sector data
        metrics (list): List of metrics to visualize
        filter_outliers (bool): Whether to filter extreme outliers for cleaner plots
    """
    # Filter out stocks with no sector or 'N/A' sector
    sector_data = sector_data[sector_data['sector'].notna() & (
        sector_data['sector'] != 'N/A')].copy()

    # Calculate number of rows and columns for subplots
    n_metrics = len(metrics)
    n_cols = min(2, n_metrics)
    n_rows = (n_metrics + 1) // 2

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(16 * n_cols // 2, 9 * n_rows))
    axes = np.array(axes).flatten() if n_metrics > 1 else [axes]

    for idx, (metric, ax) in enumerate(zip(metrics, axes)):
        # Create clean subset of data for this metric, dropping both NaN and inf values
        metric_df = sector_data[['sector', metric]].dropna()
        metric_df = metric_df[~metric_df[metric].isin(
            [float('inf'), float('-inf')])]

        # Skip if no valid data for this metric
        if len(metric_df) == 0:
            ax.text(0.5, 0.5, f'No valid data for {metric}',
                    ha='center', va='center')
            continue

        # Filter extreme outliers for cleaner visualization
        if filter_outliers and len(metric_df) > 10:
            q95 = metric_df[metric].quantile(0.95)
            q5 = metric_df[metric].quantile(0.05)
            iqr = q95 - q5
            if iqr > 0:
                # Cap outliers at reasonable range (less aggressive than before)
                lower_bound = q5 - 3 * iqr
                upper_bound = q95 + 3 * iqr
                original_count = len(metric_df)
                metric_df = metric_df[
                    (metric_df[metric] >= lower_bound) &
                    (metric_df[metric] <= upper_bound)
                ]
                filtered_count = len(metric_df)
                if original_count > filtered_count:
                    print(
                        f"Note: Filtered {original_count - filtered_count} outliers for {metric}")

        # Use violin plot with quartiles instead of swarm plot for less clutter
        sns.violinplot(data=metric_df, x='sector',
                       y=metric, inner='quartile', ax=ax, cut=0)

        # Optionally add a sample of points if dataset is small enough
        if len(metric_df) < 500:
            # Use stripplot with jitter for smaller datasets
            try:
                sns.stripplot(data=metric_df, x='sector', y=metric,
                              color='black', alpha=0.3, size=1.5, ax=ax, jitter=0.2)
            except:
                pass  # Skip if stripplot fails

        ax.set_title(f'{metric.replace("_", " ").title()} Distribution by Sector',
                     fontsize=11, pad=10)
        ax.set_xticklabels(ax.get_xticklabels(),
                           rotation=45, ha='right', fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.3, axis='y')
        ax.set_xlabel('')

        # Add sector averages in a cleaner way
        sector_means = metric_df.groupby('sector')[metric].mean()
        y_max = ax.get_ylim()[1]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        text_y = y_max + y_range * 0.03

        for i, (sector, mean) in enumerate(sector_means.items()):
            if not pd.isna(mean):
                try:
                    ax.text(i, text_y, f'μ:{mean:.1f}',
                            ha='center', va='bottom', fontsize=7, alpha=0.7)
                except:
                    pass  # Skip if text can't be placed

    # Hide empty subplots if any
    for idx in range(len(metrics), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()
