import warnings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..helpers import get_sp500_data, get_significant_changes


def linear_reg_sig_days(filter_consecutive=True, show_filtered=False):
    print("Fetching SP500 data...")
    try:
        sp500 = get_sp500_data()
    except Exception as e:
        print(f"Error fetching SP500 data: {e}")
        return None
    sp500 = sp500.drop(columns=['Dividends', 'Stock Splits'])
    sp500.info()

    # Calculate returns
    sp500['return'] = sp500['Close'].pct_change(fill_method=None)

    # Get significant changes
    if filter_consecutive:
        significant_changes, filtered_points = get_significant_changes(
            sp500, filter_consecutive)
    else:
        significant_changes = get_significant_changes(
            sp500, filter_consecutive)
        filtered_points = None

    # Create subplots
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=('Prior Day vs Significant Change',
                                        'Prior Week vs Significant Change',
                                        'Prior Month vs Significant Change'))

    periods = ['prior_day_return', 'prior_week_return', 'prior_month_return']

    for i, period in enumerate(periods, 1):
        # Calculate linear regression
        x = significant_changes[period].values
        y = significant_changes['return'].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x = x[mask]
        y = y[mask]

        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.array([min(x), max(x)])
        line_y = slope * line_x + intercept

        # Add main scatter plot
        fig.add_trace(
            go.Scatter(x=x, y=y, mode='markers',
                       name='Included Points',
                       marker=dict(size=8, opacity=0.6, color='blue'),
                       hovertemplate="<br>".join([
                           "Prior Return: %{x:.2%}",
                           "Significant Change: %{y:.2%}",
                           "<extra></extra>"
                       ])),
            row=1, col=i
        )

        # Add filtered points if requested
        if filter_consecutive and show_filtered:
            fig.add_trace(
                go.Scatter(x=filtered_points[period],
                           y=filtered_points['return'],
                           mode='markers',
                           name='Filtered Points',
                           marker=dict(size=8, opacity=0.6, color='red'),
                           hovertemplate="<br>".join([
                               "Prior Return: %{x:.2%}",
                               "Significant Change: %{y:.2%}",
                               "<extra></extra>"
                           ])),
                row=1, col=i
            )

        # Add regression line
        fig.add_trace(
            go.Scatter(x=line_x, y=line_y, mode='lines',
                       name='Regression',
                       line=dict(color='black'),
                       hovertemplate=f"Slope: {slope:.4f}<br>Intercept: {intercept:.4f}<extra></extra>"),
            row=1, col=i
        )

    # Update layout
    title_prefix = "Filtered " if filter_consecutive else ""
    fig.update_layout(height=500, width=1500,
                      showlegend=True if show_filtered else False,
                      title_text=f"{title_prefix}Relationship between Prior Returns and Significant Changes")

    # Update axes labels
    fig.update_xaxes(title_text="Prior Return", row=1, col=1)
    fig.update_xaxes(title_text="Prior Return", row=1, col=2)
    fig.update_xaxes(title_text="Prior Return", row=1, col=3)
    fig.update_yaxes(title_text="Significant Change", row=1, col=1)
    fig.update_yaxes(title_text="Significant Change", row=1, col=2)
    fig.update_yaxes(title_text="Significant Change", row=1, col=3)

    fig.show()

    print(f"Number of significant days: {len(significant_changes)}")
    return significant_changes


# Run both versions
print("\nWith consecutive days:")
with_consecutive = linear_reg_sig_days(filter_consecutive=False)
print("\nWithout consecutive days (showing filtered):")
filtered = linear_reg_sig_days(filter_consecutive=True, show_filtered=True)
