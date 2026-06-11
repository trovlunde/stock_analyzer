import pandas as pd
import numpy as np
from ..helpers import get_index_data, get_ticker
import plotly.graph_objects as go
from ...market_indices import MarketIndices
import time


def prepare_classification_data(ticker):
    """Prepare data for three-class classification using financial statements."""
    # Get stock price history, quarterly financials, balance sheet and earnings history
    stock_prices = ticker.history(period="2y")
    quarterly_fins = ticker.quarterly_financials
    quarterly_balance = ticker.quarterly_balance_sheet
    earnings_history = ticker.get_earnings_history()

    print("\nDebug: Stock price range:",
          stock_prices.index[0], "to", stock_prices.index[-1])

    # Convert earnings history to DataFrame and print structure
    earnings_df = pd.DataFrame(earnings_history)
    print("\nDebug: Earnings DataFrame columns:", earnings_df.columns)

    # Calculate revenue growth series
    revenue_series = quarterly_fins.loc['Total Revenue']
    revenue_growth = revenue_series.pct_change(fill_method=None)

    # Create financial features
    fin_features = []
    for date in quarterly_fins.columns:
        try:
            print(f"\nProcessing date: {date}")
            # Convert dates to timezone-naive for comparison
            report_date = pd.to_datetime(date).tz_localize(None)
            latest_price_date = stock_prices.index[-1].tz_localize(None)

            # Skip future dates
            if report_date > latest_price_date:
                print(f"Skipping future date: {date}")
                continue

            # Get next trading day after report
            next_trading_day = report_date

            # Find next available trading day
            attempts = 0
            while next_trading_day.tz_localize(stock_prices.index.tz) not in stock_prices.index and attempts < 10:
                next_trading_day += pd.Timedelta(days=1)
                attempts += 1

            if next_trading_day.tz_localize(stock_prices.index.tz) not in stock_prices.index:
                print(f"No trading day found near {report_date}")
                continue

            # Convert next_trading_day to match stock_prices index timezone
            next_trading_day = next_trading_day.tz_localize(
                stock_prices.index.tz)
            print(f"Found trading day: {next_trading_day}")

            # Calculate returns from this trading day
            next_day_idx = stock_prices.index.get_loc(next_trading_day)
            if next_day_idx + 1 >= len(stock_prices) or next_day_idx + 5 >= len(stock_prices):
                print(f"Not enough future data for returns calculation")
                continue

            daily_return = (stock_prices['Close'].iloc[next_day_idx + 1] -
                            stock_prices['Close'].iloc[next_day_idx]) / stock_prices['Close'].iloc[next_day_idx]
            weekly_return = (stock_prices['Close'].iloc[next_day_idx + 5] -
                             stock_prices['Close'].iloc[next_day_idx]) / stock_prices['Close'].iloc[next_day_idx]

            quarter_data = quarterly_fins[date]
            balance_data = quarterly_balance[date]

            # Get earnings expectations data if available
            date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
            matching_earnings = earnings_df[earnings_df['startdatetime'].str[:10] ==
                                            date_str] if 'startdatetime' in earnings_df.columns else pd.DataFrame()

            if not matching_earnings.empty:
                row = matching_earnings.iloc[0]
                eps_estimate = row.get('epsEstimate', 0)
                eps_actual = row.get('epsActual', 0)
                eps_difference = eps_actual - \
                    eps_estimate if eps_actual is not None and eps_estimate is not None else 0
                surprise_percent = row.get('surprisePercent', 0)
            else:
                eps_estimate = eps_actual = eps_difference = surprise_percent = 0

            # Calculate financial ratios using both income statement and balance sheet data
            features = {
                'report_date': next_trading_day,
                'gross_margin': quarter_data['Gross Profit'] / quarter_data['Total Revenue'],
                'profit_margin': quarter_data['Net Income'] / quarter_data['Total Revenue'],
                'revenue_growth': revenue_growth[date] if date in revenue_growth.index else 0,
                'eps_estimate': eps_estimate,
                'eps_actual': eps_actual,
                'eps_difference': eps_difference,
                'surprise_percent': surprise_percent,
                'daily_return': daily_return,
                'weekly_return': weekly_return
            }

            # Add balance sheet ratios if available
            try:
                features.update({
                    'current_ratio': balance_data['Current Assets'] / balance_data['Current Liabilities'],
                    'debt_to_equity': balance_data['Total Liabilities Net Minority Interest'] / balance_data['Stockholders Equity']
                })
            except (KeyError, ZeroDivisionError) as e:
                print(
                    f"Warning: Could not calculate some balance sheet ratios: {e}")
                features.update({
                    'current_ratio': 1.0,  # Default values
                    'debt_to_equity': 1.0
                })

            fin_features.append(features)
            print(f"Successfully added features for {date}")

        except (KeyError, ZeroDivisionError) as e:
            print(f"Error processing date {date}: {str(e)}")
            continue

    if not fin_features:
        raise ValueError("No financial features were successfully processed")

    # Convert to DataFrame and set index
    data = pd.DataFrame(fin_features)
    data = data.set_index('report_date')

    # Create categorical targets
    data['daily_target'] = pd.cut(data['daily_return'],
                                  bins=[-np.inf, -0.01, 0.01, np.inf],
                                  labels=['negative', 'neutral', 'positive'])

    data['weekly_target'] = pd.cut(data['weekly_return'],
                                   bins=[-np.inf, -0.02, 0.02, np.inf],
                                   labels=['negative', 'neutral', 'positive'])

    # Select features and targets
    features = ['gross_margin', 'profit_margin', 'current_ratio',
                'debt_to_equity', 'revenue_growth', 'eps_estimate',
                'eps_actual', 'eps_difference', 'surprise_percent']

    returns_data = data[['daily_return', 'weekly_return']]

    return data[features], data[['daily_target', 'weekly_target']], returns_data


def prepare_classification_data_cache(ticker, _cache_dir=None):
    """Prepare classification data with file-based caching."""
    import os

    if _cache_dir is None:
        _cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(__file__)))), 'data', 'fin_classifier_cache')
    os.makedirs(_cache_dir, exist_ok=True)

    # Create cache filename based on ticker symbol
    cache_file = os.path.join(_cache_dir, f"{ticker.ticker}_data.csv")
    target_file = os.path.join(_cache_dir, f"{ticker.ticker}_targets.csv")

    # Check if cache exists and is recent (less than 30 days old)
    if os.path.exists(cache_file) and os.path.exists(target_file):
        if (time.time() - os.path.getmtime(cache_file)) < 30*24*60*60:
            X = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            y = pd.read_csv(target_file, index_col=0)
            print(f"Using cached data for {ticker.ticker}")
            return X, y

    try:
        # If no cache or cache is old, calculate the data
        X, y, returns = prepare_classification_data(ticker)

        # Save to cache
        X.to_csv(cache_file)
        y.to_csv(target_file)
        print(f"Cached data for {ticker.ticker}")

        return X, y
    except Exception as e:
        print(f"Error processing {ticker.ticker}: {str(e)}")
        return None


def train_classifier(tickers):
    """Train daily and weekly classifiers using financial data from multiple tickers."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    # Initialize lists to store features and targets from all tickers
    all_features = []
    all_targets = []

    # Process each ticker
    for ticker in tickers:
        try:
            # Get features and targets for this ticker using cache
            result = prepare_classification_data_cache(ticker)
            if result is not None:
                X, y = result
                # Store the data
                all_features.append(X)
                all_targets.append(y)
                print(f"\nProcessed {ticker.ticker}")
                print(f"Added {len(X)} samples")

        except Exception as e:
            print(f"Error processing {ticker.ticker}: {str(e)}")
            continue

    if not all_features:
        raise ValueError("No data was successfully processed")

    # Combine data from all tickers
    X = pd.concat(all_features, axis=0)
    y = pd.concat(all_targets, axis=0)

    print(f"\nTotal samples gathered: {len(X)}")

    # Print class distribution
    print("\nDaily target class distribution:")
    print(y['daily_target'].value_counts(normalize=True))
    print("\nWeekly target class distribution:")
    print(y['weekly_target'].value_counts(normalize=True))

    # Split the data with stratification if enough samples
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y['daily_target'] if len(X) >= 10 else None
    )

    # Initialize classifiers with balanced class weights
    daily_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )

    weekly_clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=3,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight='balanced',
        random_state=42
    )

    # Train classifiers
    daily_clf.fit(X_train, y_train['daily_target'])
    weekly_clf.fit(X_train, y_train['weekly_target'])

    # Evaluate on test set
    daily_pred_test = daily_clf.predict(X_test)
    weekly_pred_test = weekly_clf.predict(X_test)

    print("\nTest Set Evaluation:")
    print("\nDaily Classifier Accuracy:", accuracy_score(
        y_test['daily_target'], daily_pred_test))
    print("\nDaily Classification Report:")
    print(classification_report(y_test['daily_target'], daily_pred_test))

    print("\nWeekly Classifier Accuracy:", accuracy_score(
        y_test['weekly_target'], weekly_pred_test))
    print("\nWeekly Classification Report:")
    print(classification_report(y_test['weekly_target'], weekly_pred_test))

    return daily_clf, weekly_clf


def evaluate_classifier(ticker_list, daily_clf, weekly_clf):
    """Evaluate classifier performance for individual stocks."""
    for i, ticker in enumerate(ticker_list[:5]):  # Only first 5 tickers
        try:
            print(f"\nEvaluating {ticker.ticker}")

            # Get features, targets, and returns for this ticker
            X, y, returns = prepare_classification_data(ticker)

            # Make predictions
            daily_pred = daily_clf.predict(X)
            weekly_pred = weekly_clf.predict(X)

            # Plot results for this ticker using returns data
            plot_results(ticker, daily_pred, weekly_pred, X, y, returns)

        except Exception as e:
            print(f"Error evaluating {ticker.ticker}: {str(e)}")
            continue


def plot_results(ticker, daily_pred, weekly_pred, X_test, y_test, returns):
    """Plot stock price with predictions as triangles and prediction table."""
    ticker_data = ticker.history(period="2y")

    # Create figure with price data
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=ticker_data.index,
        open=ticker_data['Open'],
        high=ticker_data['High'],
        low=ticker_data['Low'],
        close=ticker_data['Close'],
        name='Price'
    ))

    # Add daily prediction triangles
    for date, pred in zip(X_test.index, daily_pred):
        color = 'green' if pred == 'positive' else 'red' if pred == 'negative' else 'white'
        fig.add_trace(go.Scatter(
            x=[date],
            # Place slightly above the high
            y=[ticker_data.loc[date, 'High'] * 1.02],
            mode='markers',
            marker=dict(
                symbol='triangle-down' if pred == 'negative' else 'triangle-up' if pred == 'positive' else 'circle',
                size=15,
                color=color
            ),
            name=f'Daily Pred: {pred}',
            showlegend=False
        ))

    # Add weekly prediction triangles (slightly offset)
    for date, pred in zip(X_test.index, weekly_pred):
        color = 'green' if pred == 'positive' else 'red' if pred == 'negative' else 'white'
        fig.add_trace(go.Scatter(
            x=[date],
            # Place higher than daily predictions
            y=[ticker_data.loc[date, 'High'] * 1.04],
            mode='markers',
            marker=dict(
                symbol='triangle-down' if pred == 'negative' else 'triangle-up' if pred == 'positive' else 'circle',
                size=12,
                color=color,
                line=dict(color='black', width=1)
            ),
            name=f'Weekly Pred: {pred}',
            showlegend=False
        ))

    # Update layout
    fig.update_layout(
        title=f'{ticker.ticker} Stock Price with Predictions',
        yaxis_title='Price',
        xaxis_title='Date',
        width=1200,
        height=800,
        showlegend=True
    )

    fig.show()

    # Create results table with actual return percentages
    results_df = pd.DataFrame({
        'Date': X_test.index.strftime('%Y-%m-%d'),
        'Daily Prediction': daily_pred,
        'Weekly Prediction': weekly_pred,
        'Daily Return': returns['daily_return'].map('{:.2%}'.format),
        'Weekly Return': returns['weekly_return'].map('{:.2%}'.format)
    })

    print("results_df", results_df)

    # Add color coding for predictions vs actual returns
    daily_correct = (daily_pred == 'positive') & (returns['daily_return'] > 0.01) | \
        (daily_pred == 'negative') & (returns['daily_return'] < -0.01) | \
        (daily_pred == 'neutral') & (abs(returns['daily_return']) <= 0.01)

    weekly_correct = (weekly_pred == 'positive') & (returns['weekly_return'] > 0.02) | \
        (weekly_pred == 'negative') & (returns['weekly_return'] < -0.02) | \
        (weekly_pred == 'neutral') & (abs(returns['weekly_return']) <= 0.02)

    print("\nPredictions and Returns Table:")
    print(results_df)  # Keep the print for console output

    # Create and display table using plotly
    table_fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(results_df.columns),
            fill_color='paleturquoise',
            align='left',
            font=dict(size=12, color='black'),
            line_color='darkslategray'
        ),
        cells=dict(
            values=[results_df[col].tolist() for col in results_df.columns],
            fill_color=[
                ['white'] * len(results_df),
                ['lightgreen' if c else 'lightpink' for c in daily_correct],
                ['lightgreen' if c else 'lightpink' for c in weekly_correct],
                ['white'] * len(results_df),
                ['white'] * len(results_df)
            ],
            align='left',
            font=dict(size=11, color='black'),
            line_color='darkslategray'
        )
    )])

    table_fig.update_layout(
        title=dict(
            text=f'{ticker.ticker} Predictions and Returns',
            x=0.5,
            font=dict(size=16)
        ),
        width=1200,
        height=max(400, len(results_df) * 30),
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='white',
        plot_bgcolor='white'
    )

    # Use the same display method as the graphs
    table_fig.show(renderer="browser")

    # Print accuracy metrics
    daily_accuracy = daily_correct.mean()
    weekly_accuracy = weekly_correct.mean()

    print(f"\nPrediction Accuracy for {ticker.ticker}:")
    print(f"Daily Prediction Accuracy: {daily_accuracy:.2%}")
    print(f"Weekly Prediction Accuracy: {weekly_accuracy:.2%}")


if __name__ == "__main__":
    tickers = MarketIndices.get_sp500_tickers()
    yf_tickers = []
    for ticker in tickers:
        ticker = get_ticker(ticker)
        yf_tickers.append(ticker)

    # Train classifiers on all data
    daily_clf, weekly_clf = train_classifier(yf_tickers)

    # Evaluate and plot for first 5 tickers
    evaluate_classifier(yf_tickers[:5], daily_clf, weekly_clf)
