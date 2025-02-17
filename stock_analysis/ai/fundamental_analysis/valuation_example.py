from .valuation_methods import StockValuation, format_valuation_results, compare_stocks_valuation
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np


def create_valuation_summary_table(valuations: dict, ticker: str) -> go.Figure:
    """Create a summary table comparing different valuation methods to current price."""
    try:
        # Check if we have the required data
        if not valuations.get('multiples') or not valuations['multiples'].get('current_price'):
            print(f"Warning: Missing price data for {ticker}")
            return None

        current_price = valuations['multiples']['current_price']
        market_cap = valuations['multiples'].get('market_cap', 0)
        shares_outstanding = valuations['multiples'].get(
            'shares_outstanding', 0)

        if current_price == 0 or shares_outstanding == 0:
            print(f"Warning: Invalid price or shares data for {ticker}")
            return None

        # Extract main values from each valuation method
        valuation_metrics = {
            'Current Market': market_cap,
            'Asset Based': valuations['asset_based'].get('current_value', 0),
            'Income Based': valuations['income_based'].get('present_value', 0),
            'DCF': valuations['dcf'].get('total_value', 0),
            'Book Value': valuations['book_value'].get('book_value', 0),
            'Liquidation': valuations['liquidation'].get('liquidation_value', 0),
            'Break-up': valuations['breakup'].get('breakup_value', 0)
        }

        # Calculate implied share prices
        implied_prices = {k: v/shares_outstanding if shares_outstanding != 0 else 0
                          for k, v in valuation_metrics.items()}

        # Calculate premium/discount to current price (excluding Current Market from premium calc)
        premiums = {}
        for k, v in implied_prices.items():
            if k == 'Current Market':
                # Current market price has no premium to itself
                premiums[k] = 0
            else:
                premiums[k] = ((v/current_price) - 1) * \
                    100 if current_price != 0 else 0

        # Create figure
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Valuation Method', 'Enterprise Value',
                        'Implied Share Price', 'Premium/Discount'],
                font=dict(size=12, color='white'),
                fill_color='darkblue',
                align='left'
            ),
            cells=dict(
                values=[
                    list(valuation_metrics.keys()),
                    [f"${v:,.0f}" for v in valuation_metrics.values()],
                    [f"${v:,.2f}" for v in implied_prices.values()],
                    [f"{v:+.1f}%" for v in premiums.values()]
                ],
                font=dict(size=11),
                fill_color=[[
                    'white' if k == 'Current Market' else
                    'lightgrey' if abs(p) < 10 else
                    # Red for overvaluation (premium)
                    'lightcoral' if p < 0 else
                    # Green for undervaluation (discount)
                    'lightgreen' for k, p in premiums.items()
                ]],
                align='left'
            )
        )])

        fig.update_layout(
            title=f"Valuation Summary for {ticker}",
            width=800,
            height=400
        )

        return fig
    except Exception as e:
        print(f"Error creating valuation summary for {ticker}: {str(e)}")
        return None


def create_multiples_comparison_chart(tickers: list) -> go.Figure:
    """Create a visual comparison of key multiples across companies."""
    results = {}
    for ticker in tickers:
        try:
            valuation = StockValuation(ticker)
            multiples = valuation.calculate_multiples()
            if multiples:
                results[ticker] = multiples
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

    # Create subplots for different multiple categories
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Price Multiples',
            'Enterprise Value Multiples',
            'Returns',
            'Margins'
        )
    )

    # Define metrics for each subplot
    subplot_metrics = {
        (1, 1): ['pe_ratio', 'forward_pe', 'price_to_book', 'price_to_sales'],
        (1, 2): ['ev_to_ebitda', 'ev_to_sales', 'ev_to_fcf'],
        (2, 1): ['return_on_equity', 'return_on_assets', 'return_on_invested_capital'],
        (2, 2): ['gross_margin', 'operating_margin', 'net_margin']
    }

    # Colors for each company
    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)',
              'rgb(44, 160, 44)', 'rgb(214, 39, 40)']

    for (row, col), metrics in subplot_metrics.items():
        for ticker_idx, ticker in enumerate(results.keys()):
            values = [results[ticker][m] for m in metrics]
            fig.add_trace(
                go.Bar(
                    name=ticker,
                    x=metrics,
                    y=values,
                    marker_color=colors[ticker_idx % len(colors)]
                ),
                row=row, col=col
            )

    # Update layout
    fig.update_layout(
        height=800,
        width=1200,
        showlegend=True,
        title_text="Valuation Multiples Comparison",
        barmode='group'
    )

    # Update axes labels
    for i in fig['layout']['annotations']:
        i['font'] = dict(size=12, color='black')

    return fig


def create_dcf_analysis_chart(dcf_results: dict, ticker: str) -> go.Figure:
    """Create a visual breakdown of DCF valuation components."""
    # Extract values
    projected_cash_flows = dcf_results['projected_cash_flows']
    terminal_value = dcf_results['terminal_value']
    total_value = dcf_results['total_value']

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add projected cash flows
    fig.add_trace(
        go.Bar(
            name="Projected Cash Flows",
            x=[f"Year {i+1}" for i in range(len(projected_cash_flows))],
            y=projected_cash_flows,
            marker_color='lightblue'
        ),
        secondary_y=False
    )

    # Add cumulative value line
    cumulative = np.cumsum(projected_cash_flows)
    fig.add_trace(
        go.Scatter(
            name="Cumulative Value",
            x=[f"Year {i+1}" for i in range(len(projected_cash_flows))],
            y=cumulative,
            line=dict(color='darkblue', width=2)
        ),
        secondary_y=True
    )

    # Add terminal value marker
    fig.add_trace(
        go.Scatter(
            name="Terminal Value",
            x=[f"Year {len(projected_cash_flows)}"],
            y=[terminal_value],
            mode='markers',
            marker=dict(
                color='red',
                size=15,
                symbol='star'
            )
        ),
        secondary_y=False
    )

    # Update layout
    fig.update_layout(
        title=f"DCF Analysis for {ticker}",
        xaxis_title="Projection Period",
        yaxis_title="Cash Flow Value",
        yaxis2_title="Cumulative Value",
        height=500,
        width=800,
        showlegend=True
    )

    return fig


def create_valuation_premiums_chart(valuations_dict: dict) -> go.Figure:
    """Create a chart comparing valuation premiums across companies."""
    companies = []
    methods = []
    premiums = []
    implied_prices = []
    current_prices = []

    # Extract data for each company
    for company, vals in valuations_dict.items():
        current_price = vals['multiples']['current_price']
        shares_outstanding = vals['multiples']['shares_outstanding']

        # Get valuations for each method (excluding Current Market)
        valuation_metrics = {
            'Asset Based': vals['asset_based'].get('current_value', 0),
            'Income Based': vals['income_based'].get('present_value', 0),
            'DCF': vals['dcf'].get('total_value', 0),
            'Book Value': vals['book_value'].get('book_value', 0),
            'Liquidation': vals['liquidation'].get('liquidation_value', 0),
            'Break-up': vals['breakup'].get('breakup_value', 0)
        }

        # Calculate implied prices and premiums for each method
        for method, value in valuation_metrics.items():
            implied_price = value / shares_outstanding if shares_outstanding != 0 else 0
            premium = ((implied_price / current_price) - 1) * \
                100 if current_price != 0 else 0

            companies.append(company)
            methods.append(method)
            premiums.append(premium)
            implied_prices.append(implied_price)
            current_prices.append(current_price)

    # Create figure
    fig = go.Figure()

    # Add premium/discount bars
    fig.add_trace(go.Bar(
        name='Premium/Discount to Current Price',
        x=[f"{company} - {method}" for company,
            method in zip(companies, methods)],
        y=premiums,
        marker_color=[
            'lightcoral' if p < -10 else  # Red for overvaluation (premium)
            'lightgreen' if p > 10 else  # Green for undervaluation (discount)
            'lightgrey' for p in premiums
        ],
        text=[f"{p:+.1f}%" for p in premiums],
        textposition='auto',
    ))

    # Add price comparison table
    price_table = go.Table(
        header=dict(
            values=['Company', 'Valuation Method', 'Current Price',
                    'Implied Price', 'Premium/Discount'],
            font=dict(size=12, color='white'),
            fill_color='darkblue',
            align='left'
        ),
        cells=dict(
            values=[
                companies,
                methods,
                [f"${p:,.2f}" for p in current_prices],
                [f"${p:,.2f}" for p in implied_prices],
                [f"{p:+.1f}%" for p in premiums]
            ],
            font=dict(size=11),
            fill_color=[[
                'lightgrey' if abs(p) < 10 else
                'lightcoral' if p > 10 else  # Red for overvaluation (premium)
                # Green for undervaluation (discount)
                'lightgreen' for p in premiums
            ]],
            align='left'
        ),
        domain=dict(x=[0, 1], y=[0, 0.3])
    )

    # Update layout to accommodate both chart and table
    fig.add_trace(price_table)
    fig.update_layout(
        title="Valuation Premiums Comparison Across Companies",
        xaxis_title="Company - Valuation Method",
        yaxis_title="Premium/Discount (%)",
        height=1000,  # Increased height to accommodate table
        width=1200,
        showlegend=False,
        # Adjust the bar chart position to leave room for table
        yaxis=dict(domain=[0.35, 1])
    )

    return fig


def main():
    # Example 1: Single stock comprehensive valuation
    print("\nExample 1: Comprehensive Stock Valuation")
    print("=======================================")

    # List of stocks to analyze
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'META',
               'AMZN', 'TSLA', 'JPM', 'JNJ', 'PG', 'XOM']
    valuations = {}

    # Get valuations for each stock
    for ticker in tickers:
        try:
            stock = StockValuation(ticker)
            vals = stock.get_all_valuations(
                discount_rate=0.10,
                growth_rate=0.03,
                projection_years=5
            )
            if vals.get('multiples') and vals['multiples'].get('current_price'):
                valuations[ticker] = vals
            else:
                print(f"Warning: Skipping {ticker} due to missing data")
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue

    # Create and display valuation summary tables
    for ticker, vals in valuations.items():
        summary_table = create_valuation_summary_table(vals, ticker)
        if summary_table:
            summary_table.show()

    # Create and display premium comparison across companies
    if len(valuations) > 1:
        premiums_chart = create_valuation_premiums_chart(valuations)
        premiums_chart.show()

    # Example 2: Compare multiple stocks
    print("\nExample 2: Multiple Stock Comparison")
    print("====================================")

    # Compare tech giants
    multiples_chart = create_multiples_comparison_chart(tickers)
    multiples_chart.show()

    # Example 3: Detailed DCF Analysis
    print("\nExample 3: Detailed DCF Analysis")
    print("================================")

    # Get detailed DCF valuation for Microsoft
    if 'MSFT' in valuations:
        msft_dcf = valuations['MSFT']['dcf']
        if msft_dcf:
            dcf_chart = create_dcf_analysis_chart(msft_dcf, 'MSFT')
            dcf_chart.show()
        else:
            print("Warning: DCF data not available for MSFT")


if __name__ == "__main__":
    main()
