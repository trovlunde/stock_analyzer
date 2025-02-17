import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..helpers import get_significant_changes, get_index_data


def pca_sig_days(significant_changes):
    """Analyze significant market changes using PCA."""

    # Prepare data for PCA
    features = ['return', 'prior_day_return', 'prior_5day_return',
                'prior_month_return', 'RSI_5d', 'RSI_5d_alt']
    X = significant_changes[features].copy()

    # Remove rows with NaN values
    X = X.dropna()

    # Create correlation matrix plot
    corr_matrix = X.corr()

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=features,
        y=features,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu',
        zmid=0
    ))

    fig_corr.update_layout(
        title='Correlation Matrix of Features',
        height=800,
        width=800,
        xaxis_tickangle=-45
    )

    fig_corr.show()

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # Create explained variance plot with cumulative plot
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar plot for individual explained variance ratios
    fig1.add_trace(
        go.Bar(
            x=[f'PC{i+1}' for i in range(len(features))],
            y=pca.explained_variance_ratio_,
            text=[f'{x:.1%}' for x in pca.explained_variance_ratio_],
            textposition='auto',
            name='Individual'
        ),
        secondary_y=False
    )

    # Line plot for cumulative explained variance
    cumulative_var_ratio = np.cumsum(pca.explained_variance_ratio_)
    fig1.add_trace(
        go.Scatter(
            x=[f'PC{i+1}' for i in range(len(features))],
            y=cumulative_var_ratio,
            mode='lines+markers',
            name='Cumulative',
            text=[f'{x:.1%}' for x in cumulative_var_ratio],
            textposition='top center'
        ),
        secondary_y=True
    )

    fig1.update_layout(
        title='Explained Variance Ratio by Principal Component',
        xaxis_title='Principal Component',
        yaxis_title='Individual Explained Variance Ratio',
        yaxis2_title='Cumulative Explained Variance Ratio'
    )

    # Create component loadings plot
    loadings = pca.components_
    n_components = len(loadings)  # Get actual number of components

    fig2 = go.Figure()

    for i in range(min(4, n_components)):  # Only plot first 4 PCAs
        fig2.add_trace(go.Bar(
            name=f'PC{i+1}',
            x=features,
            y=loadings[i],
            text=[f'{x:.3f}' for x in loadings[i]],
            textposition='outside',
        ))

    fig2.update_layout(
        title='PCA Component Loadings',
        xaxis_title='Features',
        yaxis_title='Loading Value',
        barmode='group',
        height=1200,
        width=1600,
        margin=dict(t=100, b=100),
        bargap=0.2,
        bargroupgap=0.1
    )

    # Create biplot of first two components
    fig3 = go.Figure()

    # Plot scores
    fig3.add_trace(go.Scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        mode='markers',
        name='Scores',
        marker=dict(
            size=8,
            color=X['return'],
            colorscale='RdYlBu',
            showscale=True,
            colorbar=dict(
                title='Return',
                x=1.2  # Move colorbar further right
            )
        ),
        hovertemplate="<br>".join([
            "PC1: %{x:.2f}",
            "PC2: %{y:.2f}",
            "Return: %{marker.color:.2%}",
            "<extra></extra>"
        ])
    ))

    # Add feature vectors
    scaleFactor = 3
    for i, feature in enumerate(features):
        fig3.add_trace(go.Scatter(
            x=[0, pca.components_[0, i] * scaleFactor],
            y=[0, pca.components_[1, i] * scaleFactor],
            mode='lines+text',
            name=feature,
            text=['', feature],
            textposition='top center',
            line=dict(color='red', width=1),
            hoverinfo='skip'
        ))

    fig3.update_layout(
        title='PCA Biplot (PC1 vs PC2)',
        xaxis_title='First Principal Component',
        yaxis_title='Second Principal Component',
        xaxis=dict(zeroline=True),
        yaxis=dict(zeroline=True),
    )

    # Create biplot components 1 and 3
    fig4 = go.Figure()

    # Plot scores
    fig4.add_trace(go.Scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 2],
        mode='markers',
        name='Scores',
        marker=dict(
            size=8,
            color=X['return'],
            colorscale='RdYlBu',
            showscale=True,
            colorbar=dict(title='Return', x=1.2)
        ),
        hovertemplate="<br>".join([
            "PC1: %{x:.2f}",
            "PC2: %{y:.2f}",
            "Return: %{marker.color:.2%}",
            "<extra></extra>"
        ])
    ))

    # Add feature vectors
    scaleFactor = 3  # Adjust this to change the length of the arrows
    for i, feature in enumerate(features):
        fig4.add_trace(go.Scatter(
            x=[0, pca.components_[0, i] * scaleFactor],
            y=[0, pca.components_[2, i] * scaleFactor],
            mode='lines+text',
            name=feature,
            text=['', feature],
            textposition='top right',
            line=dict(color='red', width=1),
            hoverinfo='skip'
        ))

    fig4.update_layout(
        title='PCA Biplot (PC1 vs PC3)',
        xaxis_title='First Principal Component',
        yaxis_title='Third Principal Component',
        xaxis=dict(zeroline=True),
        yaxis=dict(zeroline=True)
    )

    # Create return vs PC1/PC2 subplots
    fig5 = make_subplots(rows=1, cols=2,
                         subplot_titles=('Returns vs PC1', 'Returns vs PC2'))

    # Plot Returns vs PC1
    fig5.add_trace(
        go.Scatter(
            x=X_pca[:, 0],
            y=X['return'],
            mode='markers',
            marker=dict(
                size=8,
                color=X['return'],
                colorscale='RdYlBu',
                showscale=True,
                colorbar=dict(
                    title='Return',
                    x=0.46  # Position colorbar between subplots
                )
            ),
            name='PC1'
        ),
        row=1, col=1
    )

    # Plot Returns vs PC2
    fig5.add_trace(
        go.Scatter(
            x=X_pca[:, 1],
            y=X['return'],
            mode='markers',
            marker=dict(
                size=8,
                color=X['return'],
                colorscale='RdYlBu',
                showscale=False  # Don't show second colorbar
            ),
            name='PC2'
        ),
        row=1, col=2
    )

    # Update layout
    fig5.update_layout(
        title='Returns vs Principal Components',
        height=600,
        width=1200,
        showlegend=False,
        plot_bgcolor='white'
    )

    # Update axes
    fig5.update_xaxes(title_text='PC1', row=1, col=1,
                      gridcolor='lightgrey', zeroline=True, zerolinecolor='grey')
    fig5.update_xaxes(title_text='PC2', row=1, col=2,
                      gridcolor='lightgrey', zeroline=True, zerolinecolor='grey')
    fig5.update_yaxes(title_text='Return', row=1, col=1,
                      gridcolor='lightgrey', zeroline=True, zerolinecolor='grey')
    fig5.update_yaxes(title_text='Return', row=1, col=2,
                      gridcolor='lightgrey', zeroline=True, zerolinecolor='grey')

    # Show all plots
    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()
    fig5.show()

    # Print explained variance ratios
    print("\nExplained Variance Ratios:")
    for i, ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {ratio:.1%}")

    return pca, X_pca


def sig_days_analysis(significant_changes):
    """Analyze and plot market behavior around significant market changes."""
    # Create a window of -5 to +5 days
    days = list(range(-5, 6))  # Convert range to list

    # Separate positive and negative significant changes
    pos_changes = significant_changes[significant_changes['return'] > 0]
    neg_changes = significant_changes[significant_changes['return'] < 0]

    # Create subplots for positive and negative changes
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Behavior Around Positive Significant Changes',
                                        'Behavior Around Negative Significant Changes'))

    # Function to get returns around a date
    def get_window_returns(date, sp500):
        start_idx = sp500.index.get_loc(date) - 5
        end_idx = sp500.index.get_loc(date) + 6
        if start_idx < 0 or end_idx > len(sp500):
            return None
        window_prices = sp500['Close'][start_idx:end_idx]
        base_price = window_prices.iloc[5]  # Price at the significant change
        return (window_prices / base_price - 1) * 100  # Convert to percentage

    # Plot positive changes
    for date in pos_changes.index:
        returns = get_window_returns(date, sp500)
        if returns is not None:
            fig.add_trace(
                go.Scatter(x=days, y=returns,
                           mode='lines', line=dict(color='rgba(0,255,0,0.2)'),
                           showlegend=False),
                row=1, col=1
            )

    # Plot negative changes
    for date in neg_changes.index:
        returns = get_window_returns(date, sp500)
        if returns is not None:
            fig.add_trace(
                go.Scatter(x=days, y=returns,
                           mode='lines', line=dict(color='rgba(255,0,0,0.2)'),
                           showlegend=False),
                row=2, col=1
            )

    # Calculate and plot average lines
    def plot_average(changes, color, row):
        all_returns = []
        for date in changes.index:
            returns = get_window_returns(date, sp500)
            if returns is not None:
                # Convert to numpy array to ensure proper alignment
                all_returns.append(returns.values)

        if all_returns:
            # Use numpy mean instead of pandas
            avg_returns = np.mean(all_returns, axis=0)
            fig.add_trace(
                go.Scatter(x=days, y=avg_returns,
                           mode='lines', line=dict(color=color, width=3),
                           name='Average' if row == 1 else 'Average',
                           showlegend=True),
                row=row, col=1
            )

    plot_average(pos_changes, 'black', 1)
    plot_average(neg_changes, 'black', 2)

    # Update layout
    fig.update_layout(
        height=800,
        title_text="Market Behavior Around Significant Changes",
        showlegend=True
    )

    # Update axes
    for row in [1, 2]:
        fig.update_xaxes(title_text='Days from Event', row=row, col=1)
        fig.update_yaxes(title_text='Return (%)', row=row, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="gray", row=row, col=1)

    fig.show()


sp500 = get_index_data('^GSPC')
significant_changes = get_significant_changes(
    sp500, filter_return=0.02, filter_consecutive=False)


sig_days_analysis(significant_changes)

# Perform PCA analysis
# pca, X_pca = pca_sig_days(significant_changes)
