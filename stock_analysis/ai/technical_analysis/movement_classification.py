import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve, train_test_split, StratifiedKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import os
import time

from stock_analysis.ai.technical_analysis.ensemble_classifier import test_ensemble, EnsembleClassifier
from ..helpers import get_index_data, get_ticker, RSI, get_features
from ...market_indices import MarketIndices
from ...portfolio_evaluating import analyze_trading_strategies
from ...ai.technical_analysis.prepare_classification_data import prepare_classification_data_cache, prepare_classification_data, prepare_classification_data_enhanced
from ...ai.technical_analysis.classifier_comparison import compare_classifiers, plot_classifier_comparison, print_best_configs
from sklearn.utils.class_weight import compute_class_weight
from ...ai.technical_analysis.rsi_signals import generate_rsi_signals
from ...ai.technical_analysis.ma_signals import generate_multi_ma_signals
from ..helpers import classifiers


def temporal_holdout_split(data, holdout_months=None, test_size=0.2):
    """Split labeled time-series rows into chronological train and holdout sets."""
    training_data = data[data['Target'].notna()]
    if holdout_months is not None and holdout_months > 0:
        cutoff = training_data.index.max() - pd.DateOffset(months=int(holdout_months))
        train_data = training_data[training_data.index < cutoff]
        holdout_data = training_data[training_data.index >= cutoff]
    else:
        split_idx = int(len(training_data) * (1 - test_size))
        train_data = training_data.iloc[:split_idx]
        holdout_data = training_data.iloc[split_idx:]
    return train_data, holdout_data


def train_classifier_single_stock(
    stock_data,
    use_extra_features=False,
    predict_weekly=False,
    threshold=0.005,
    plot=False,
    overfit_check=False,
    classifier=RandomForestClassifier(random_state=42),
    eval_split=0.2,
    verbose=True,
):
    """Train a three-class classifier."""
    # Get model name and initialize model correctly
    if isinstance(classifier, type):  # If it's a class (not instance)
        model = classifier(random_state=42)
        model_name = classifier.__name__
    else:  # If it's already an instance
        model = classifier
        model_name = classifier.__class__.__name__

    prediction_type = "Weekly" if predict_weekly else "Daily"
    if verbose:
        print(f"\n{'='*50}")
        print(f"Training {model_name} for {prediction_type} predictions")
        print(f"{'='*50}")

    # Prepare features and target
    features = get_features(use_extra_features)

    contains_new_data = False
    # Split into training data (where we have targets) and prediction data (where we don't)
    training_data = stock_data[stock_data['Target'].notna()]
    prediction_data = stock_data[stock_data['Target'].isna()]
    if len(prediction_data) != 0:
        contains_new_data = True

    # Prepare training sets
    X = training_data[features]
    y = training_data['Target']

    use_eval_split = eval_split and eval_split > 0
    if use_eval_split:
        # Chronological split — never shuffle time-series data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=eval_split, shuffle=False)
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None

    # Fit scaler on training data only to avoid leakage
    scaler = StandardScaler()
    scaler.fit(X_train)

    # Transform all sets
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test) if use_eval_split else None

    if contains_new_data:
        # Also prepare the prediction data (rows with NaN targets)
        X_predict = prediction_data[features]
        X_predict_scaled = scaler.transform(X_predict)

        if use_eval_split:
            test_indices = np.concatenate([X_test.index, X_predict.index])
        else:
            test_indices = X_predict.index
    elif use_eval_split:
        test_indices = X_test.index
    else:
        test_indices = pd.Index([])

    # Train model
    model.fit(X_train_scaled, y_train)

    # Add after model training but before predictions
    if overfit_check and use_eval_split:
        check_overfitting(model, X_train_scaled, X_test_scaled,
                          y_train, y_test, model_name, features)

    # Make predictions on internal eval split when present
    y_pred = model.predict(X_test_scaled) if use_eval_split else None

    # Plot results
    if plot and verbose and hasattr(model, 'feature_importances_'):
        # Create a simpler, more efficient feature importance plot
        importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        fig = go.Figure()

        # Add single bar trace
        fig.add_trace(go.Bar(
            x=importance['feature'],
            y=importance['importance'],
            text=[f"{val:.3f}" for val in importance['importance']],
            textposition='auto',
            marker_color='lightblue'  # Single color for better performance
        ))

        # Simplified layout
        fig.update_layout(
            title=f'Feature Importance - {model_name} ({prediction_type} Predictions)',
            xaxis_title='Features',
            yaxis_title='Importance Score',
            height=400,  # Reduced height
            width=800,   # Fixed width
            margin=dict(l=50, r=50, t=50, b=50),  # Optimized margins
            showlegend=False,
            template='simple_white'  # Simpler template
        )

        # Optimize x-axis labels
        fig.update_xaxes(
            tickangle=45,
            tickmode='array',
            ticktext=importance['feature'],
            tickvals=list(range(len(importance['feature'])))
        )

        # Use lower animation frame rate and optimize rendering
        fig.show(config={
            'staticPlot': False,
            'scrollZoom': False,
            'displayModeBar': False
        })

    if verbose:
        print(f"\nClass Distribution for {model_name}:")
        print(y.value_counts(normalize=True))
        if use_eval_split:
            print(f"\nClassification Report for {model_name}:")
            print(classification_report(y_test, y_pred))

    # Mark train vs internal-test rows for downstream analysis
    full_data = stock_data.copy()
    if len(test_indices) > 0:
        full_data.loc[test_indices, 'test_set'] = True
    full_data.loc[X_train.index, 'test_set'] = False

    return model, scaler, full_data


def train_classifier_tickers(
    tickers,
    predict_weekly=False,
    threshold=0.005,
    plot=False,
    overfit_check=False,
    classifier=RandomForestClassifier(random_state=42),
    use_extra_features=False,
    cutoff_date=None,
    eval_split=0.2,
    verbose=True,
):
    """Train a classifier on multiple tickers simultaneously."""
    all_data_frames = []

    if verbose:
        print(f"\nPreparing data for {len(tickers)} tickers...")
        if cutoff_date is not None:
            print(f"Training cutoff: rows before {pd.Timestamp(cutoff_date).date()}")

    # Collect data from all tickers
    for ticker in tickers:
        try:
            stock_data = get_index_data(ticker)
            data = prepare_classification_data_cache(
                stock_data, predict_weekly, threshold, use_extra_features)

            if data is not None:
                if cutoff_date is not None:
                    data = data[data.index < cutoff_date]
                if len(data) == 0:
                    continue
                data = data.copy()
                data['ticker'] = ticker
                all_data_frames.append(data)
                if verbose:
                    print(f"Successfully processed {ticker} ({len(data)} rows)")

        except Exception as e:
            if verbose:
                print(f"Error processing {ticker}: {e}")
            continue

    if not all_data_frames:
        raise ValueError("No valid data collected from any ticker")

    combined_data = pd.concat(all_data_frames, axis=0)
    if verbose:
        print(f"\nCombined dataset shape: {combined_data.shape}")

    # Get model name
    if isinstance(classifier, type):  # If it's a class (not instance)
        model = classifier(random_state=42)
        model_name = classifier.__name__
    else:  # If it's already an instance
        model = classifier
        model_name = classifier.__class__.__name__

    prediction_type = "Weekly" if predict_weekly else "Daily"
    if verbose:
        print(f"\n{'='*50}")
        print(f"Training {model_name} for {prediction_type} predictions")
        print(f"{'='*50}")

    features = get_features(use_extra_features)
    training_data = combined_data[combined_data['Target'].notna()]
    X = training_data[features]
    y = training_data['Target']

    use_eval_split = eval_split and eval_split > 0 and cutoff_date is None
    if use_eval_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=eval_split, shuffle=False)
    else:
        X_train, y_train = X, y
        X_test, y_test = None, None

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test) if use_eval_split else None

    model.fit(X_train_scaled, y_train)

    if overfit_check and use_eval_split:
        check_overfitting(model, X_train_scaled, X_test_scaled,
                          y_train, y_test, model_name, features)

    if verbose and use_eval_split:
        y_pred = model.predict(X_test_scaled)
        print(f"\nClass Distribution:")
        print(y.value_counts(normalize=True))
        print(f"\nInternal chronological classification report:")
        print(classification_report(y_test, y_pred))
    elif verbose:
        print(f"\nClass Distribution:")
        print(y.value_counts(normalize=True))
        print(
            "\nTrained on all pre-cutoff rows; evaluate on the analysis "
            "ticker holdout for out-of-sample metrics."
        )

    if plot and use_eval_split:
        y_pred = model.predict(X_test_scaled)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred),
                    annot=True,
                    fmt='d',
                    xticklabels=['negative', 'neutral', 'positive'],
                    yticklabels=['negative', 'neutral', 'positive'])
        plt.title(
            f'Confusion Matrix - {model_name} ({prediction_type} Predictions)')
        plt.show()

        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)

            plt.figure(figsize=(14, 4))
            sns.barplot(data=importance, x='importance', y='feature')
            plt.title(
                f'Feature Importance - {model_name} ({prediction_type} Predictions)')
            plt.show()

    return model, scaler, combined_data


def resolve_backtest_period(
    prepared_daily,
    prepared_weekly,
    stock_data,
    *,
    holdout_months=0,
    cutoff_date=None,
    train_meta=None,
):
    """Resolve out-of-sample rows for trading backtests."""
    if holdout_months and cutoff_date is not None:
        eval_daily = prepared_daily[prepared_daily.index >= cutoff_date]
        eval_weekly = prepared_weekly[prepared_weekly.index >= cutoff_date]
        eval_stock = stock_data[stock_data.index >= cutoff_date]
        description = f"temporal holdout ({holdout_months} months)"
    else:
        if train_meta is not None and 'test_set' in train_meta.columns:
            test_index = train_meta.index[train_meta['test_set'] == True]
        else:
            _, holdout = temporal_holdout_split(prepared_daily, test_size=0.2)
            test_index = holdout.index

        eval_daily = prepared_daily.loc[prepared_daily.index.isin(test_index)]
        eval_weekly = prepared_weekly.loc[prepared_weekly.index.isin(test_index)]
        eval_stock = stock_data.loc[stock_data.index.isin(test_index)]
        description = f"internal chronological test set ({len(test_index)} days)"

    if len(eval_daily) == 0 or len(eval_stock) == 0:
        raise ValueError("No out-of-sample rows available for backtesting")

    return eval_daily, eval_weekly, eval_stock, description


def run_movement_backtests(
    stock_data,
    daily_model,
    daily_scaler,
    prepared_daily,
    weekly_model,
    weekly_scaler,
    prepared_weekly,
    threshold,
    use_extra_features,
    eval_daily,
    eval_weekly,
    eval_stock,
    backtest_label,
):
    """Run trading backtests only on out-of-sample evaluation data."""
    print(f"\n{'='*60}")
    print(f"Out-of-sample trading backtest: {backtest_label}")
    print(
        f"Evaluation window: {eval_stock.index.min().date()} to "
        f"{eval_stock.index.max().date()} ({len(eval_stock)} days)"
    )
    print(f"{'='*60}")

    daily_simulation = simulate_trading(
        daily_model,
        daily_scaler,
        eval_daily,
        threshold=threshold,
        stock_data=eval_stock,
        predict_weekly=False,
        use_extra_features=use_extra_features,
        verbose=False,
    )
    weekly_simulation = simulate_trading(
        weekly_model,
        weekly_scaler,
        eval_weekly,
        threshold=threshold,
        stock_data=eval_stock,
        predict_weekly=True,
        use_extra_features=use_extra_features,
        verbose=False,
    )
    combined_signals = full_analysis(
        eval_stock,
        eval_daily,
        daily_model,
        daily_scaler,
        eval_weekly,
        weekly_model,
        weekly_scaler,
        threshold=threshold,
        use_extra_features=use_extra_features,
        plot=False,
    )

    analyze_trading_strategies(
        eval_stock,
        daily_simulation,
        initial_investment=10000,
        leverage=10,
        signal_type="Daily Model (OOS)",
    )
    analyze_trading_strategies(
        eval_stock,
        weekly_simulation,
        initial_investment=10000,
        leverage=10,
        signal_type="Weekly Model (OOS)",
    )
    analyze_trading_strategies(
        eval_stock,
        combined_signals,
        initial_investment=10000,
        leverage=10,
        signal_type="Combined Model (OOS)",
    )

    return {
        'daily_simulation': daily_simulation,
        'weekly_simulation': weekly_simulation,
        'combined_signals': combined_signals,
    }


def simulate_trading(model, scaler, data, investment_amount=100, stock_data=None, threshold=0.005, predict_weekly=False, use_extra_features=False, verbose=True):
    """Simulate trading based on model predictions."""
    # Get model name
    model_name = model.__class__.__name__
    prediction_type = "Weekly" if predict_weekly else "Daily"

    features = get_features(use_extra_features)

    # Prepare features for prediction
    X = data[features]
    X_scaled = scaler.transform(X)

    if verbose:
        print("Simulate trading X", X)
        print("\nDebug - Feature order:", features)
        print("Debug - First row scaled:", X_scaled[0])
        print("Debug - Model random_state:", model.random_state)

    # Get predictions and probabilities for all data points
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)

    if verbose:
        print(f"\n{'='*50}")
        print(
            f"Trading Simulation Results for {model_name} ({prediction_type} Predictions)")
        print(f"{'='*50}")

    # Get the Close prices as a 1D array
    close_prices = stock_data['Close'].values.squeeze()

    # Create simulation DataFrame indexed by date for downstream backtests
    simulation = pd.DataFrame({
        'actual_return': data['return'],
        'prediction': predictions,
        'price': close_prices[stock_data.index.isin(data.index)],
    }, index=data.index)
    simulation.index.name = 'date'

    # Calculate actual class only where return data is available
    simulation['actual_class'] = pd.cut(simulation['actual_return'],
                                        bins=[-np.inf, -threshold,
                                              threshold, np.inf],
                                        labels=['negative', 'neutral', 'positive'])

    # Calculate trading metrics only where we have actual returns
    has_returns = simulation['actual_return'].notna()
    if has_returns.any():
        # Calculate returns for both long and short positions
        simulation['long_investment'] = (
            predictions == 'positive') * investment_amount
        simulation['short_investment'] = (
            predictions == 'negative') * investment_amount

        # For long positions: gain when price goes up
        simulation['long_return'] = simulation['long_investment'] * \
            simulation['actual_return']

        # For short positions: gain when price goes down (negative return becomes positive)
        simulation['short_return'] = simulation['short_investment'] * \
            (-simulation['actual_return'])

        # Combine returns
        simulation['total_return'] = simulation['long_return'] + \
            simulation['short_return']

        # Calculate results
        total_invested = simulation['long_investment'].sum(
        ) + simulation['short_investment'].sum()
        total_trades = sum((simulation['long_investment'] > 0) | (
            simulation['short_investment'] > 0))
        total_return = simulation['total_return'].sum()

        # Calculate win rates for different positions
        long_correct = sum((simulation['prediction'] == 'positive') &
                           (simulation['actual_class'] == 'positive'))
        total_long = sum(simulation['prediction'] == 'positive')

        short_correct = sum((simulation['prediction'] == 'negative') &
                            (simulation['actual_class'] == 'negative'))
        total_short = sum(simulation['prediction'] == 'negative')

        if not verbose:
            return simulation

        # Print results
        print("\nTrading Performance (for known returns):")
        print(f"Total number of trading days: {len(simulation)}")
        print(f"Number of trades: {total_trades}")
        print(f"  Long positions: {total_long}")
        print(f"  Short positions: {total_short}")
        print(f"Total amount invested: ${total_invested:,.2f}")
        print(f"Total return: ${total_return:,.2f}")
        print(f"Return percentage: {(total_return/total_invested)*100:.2f}%")

        if total_long > 0:
            print(
                f"Long position win rate: {(long_correct/total_long)*100:.2f}%")
        if total_short > 0:
            print(
                f"Short position win rate: {(short_correct/total_short)*100:.2f}%")

        print("\nPosition Analysis:")
        print(f"\nLong Positions Outcomes ({model_name}):")
        long_predictions = simulation[simulation['prediction'] == 'positive']
        print(long_predictions['actual_class'].value_counts(
            normalize=True).round(3) * 100)

        print("\nShort Positions Outcomes:")
        short_predictions = simulation[simulation['prediction'] == 'negative']
        print(short_predictions['actual_class'].value_counts(
            normalize=True).round(3) * 100)

        # Create figure with 3 subplots (price with long signals, price with short signals, RSI)
        fig = make_subplots(rows=3, cols=1,
                            vertical_spacing=0.05,
                            subplot_titles=(f'Long Positions ({model_name} {prediction_type})',
                                            f'Short Positions ({model_name} {prediction_type})',
                                            '21-day RSI'),
                            row_heights=[0.4, 0.4, 0.2])  # Adjust heights as needed

        # Add traces for first subplot (Long Positions)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=simulation['price'],
                mode='lines',
                name='Price',
                line=dict(color='rgba(0, 0, 255, 0.5)', width=1)
            ),
            row=1, col=1
        )

        # Add correct positive predictions
        correct_long = simulation[
            (simulation['prediction'] == 'positive') &
            (simulation['actual_class'] == 'positive')
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
                    hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<br>Next Day Return: %{customdata:.2%}<br>Prediction: Correct Long<extra></extra>",
                    customdata=simulation.loc[correct_long, 'actual_return']
                ),
                row=1, col=1
            )

        # Add wrong long predictions
        for actual, color, symbol in [
            ('neutral', 'white', 'triangle-up'),
            ('negative', 'red', 'triangle-up')
        ]:
            wrong_long = simulation[
                (simulation['prediction'] == 'positive') &
                (simulation['actual_class'] == actual)
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
                        hovertemplate=f"Date: %{{x}}<br>Price: %{{y:.2f}}<br>Next Day Return: %{{customdata:.2%}}<br>Prediction: Wrong Long ({actual})<extra></extra>",
                        customdata=simulation.loc[wrong_long, 'actual_return']
                    ),
                    row=1, col=1
                )

        # Add future predictions (no actual class) for long positions
        future_long = simulation[
            (simulation['prediction'] == 'positive') &
            (simulation['actual_class'].isna())
        ].index

        if len(future_long) > 0:
            fig.add_trace(
                go.Scatter(
                    x=future_long,
                    y=simulation.loc[future_long, 'price'],
                    mode='markers',
                    name='Future Long',
                    marker=dict(symbol='triangle-up', size=10,
                                color='blue', opacity=0.7),
                    hovertemplate=f"Date: %{{x}}<br>Price: %{{y:.2f}}<br>Prediction: Future Long<extra></extra>"
                ),
                row=1, col=1
            )

        # Add traces for second subplot (Short Positions)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=simulation['price'],
                mode='lines',
                name='Price',
                line=dict(color='rgba(0, 0, 255, 0.5)', width=1),
                showlegend=False
            ),
            row=2, col=1
        )

        # Add correct short predictions
        correct_short = simulation[
            (simulation['prediction'] == 'negative') &
            (simulation['actual_class'] == 'negative')
        ].index

        if len(correct_short) > 0:
            fig.add_trace(
                go.Scatter(
                    x=correct_short,
                    y=simulation.loc[correct_short, 'price'],
                    mode='markers',
                    name='Correct Short',
                    marker=dict(symbol='triangle-down', size=10,
                                color='red', opacity=0.7),
                    hovertemplate=f"Date: %{{x}}<br>Price: %{{y:.2f}}<br>Next Day Return: %{{customdata:.2%}}<br>Prediction: Correct Short<extra></extra>",
                    customdata=simulation.loc[correct_short, 'actual_return']
                ),
                row=2, col=1
            )

        # Add wrong short predictions
        for actual, color, symbol in [
            ('neutral', 'white', 'triangle-down'),
            ('positive', 'green', 'triangle-down')
        ]:
            wrong_short = simulation[
                (simulation['prediction'] == 'negative') &
                (simulation['actual_class'] == actual)
            ].index

            if len(wrong_short) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=wrong_short,
                        y=simulation.loc[wrong_short, 'price'],
                        mode='markers',
                        name=f'Wrong Short ({actual})',
                        marker=dict(
                            symbol=symbol,
                            size=10,
                            color=color,
                            line=dict(color='black',
                                      width=1) if color == 'white' else None,
                            opacity=0.7
                        ),
                        hovertemplate=f"Date: %{{x}}<br>Price: %{{y:.2f}}<br>Next Day Return: %{{customdata:.2%}}<br>Prediction: Wrong Short ({actual})<extra></extra>",
                        customdata=simulation.loc[wrong_short, 'actual_return']
                    ),
                    row=2, col=1
                )

        # Add future predictions (no actual class) for short positions
        future_short = simulation[
            (simulation['prediction'] == 'negative') &
            (simulation['actual_class'].isna())
        ].index

        if len(future_short) > 0:
            fig.add_trace(
                go.Scatter(
                    x=future_short,
                    y=simulation.loc[future_short, 'price'],
                    mode='markers',
                    name='Future Short',
                    marker=dict(symbol='triangle-down', size=10,
                                color='blue', opacity=0.7),
                    hovertemplate=f"Date: %{{x}}<br>Price: %{{y:.2f}}<br>Prediction: Future Short<extra></extra>"
                ),
                row=2, col=1
            )

        # Get today's prediction and probabilities
        today_features = pd.DataFrame({
            'prev_day_return': [stock_data['Close'].pct_change(fill_method=None).iloc[-1]],
            'prev_2day_return': [stock_data['Close'].pct_change(fill_method=None).shift(1).iloc[-1]],
            'prev_week_return': [stock_data['Close'].pct_change(periods=5, fill_method=None).iloc[-1]]
        })
        if use_extra_features:
            # Calculate volatility
            today_features['volatility_5d'] = [
                stock_data['Close'].pct_change(fill_method=None).rolling(5).std().iloc[-1] * np.sqrt(252/5)]
            today_features['volatility_21d'] = [
                stock_data['Close'].pct_change(fill_method=None).rolling(21).std().iloc[-1] * np.sqrt(252/21)]

            # Moving averages relative to current price
            today_features['ma_5d'] = [stock_data['Close'].rolling(
                5).mean().iloc[-1] / stock_data['Close'].iloc[-1]]
            today_features['ma_21d'] = [stock_data['Close'].rolling(
                21).mean().iloc[-1] / stock_data['Close'].iloc[-1]]

            # Volume features
            today_features['volume_1d'] = [
                stock_data['Volume'].iloc[-1] / stock_data['Volume'].rolling(21).mean().iloc[-1]]
            today_features['volume_5d'] = [stock_data['Volume'].rolling(
                5).mean().iloc[-1] / stock_data['Volume'].rolling(21).mean().iloc[-1]]

            # RSI features
            today_features['rsi_5d'] = [RSI(stock_data, window=5).iloc[-1]]
            today_features['rsi_21d'] = [
                RSI(stock_data, window=21).iloc[-1]]

        today_scaled = scaler.transform(today_features)
        today_probs = model.predict_proba(today_scaled)[0]

        # Add today's prediction to both plots with special marker
        today_price = stock_data['Close'].iloc[-1]
        today_date = stock_data.index[-1]
        current_prediction = simulation.loc[today_date, 'prediction']

        if current_prediction == 'neutral':
            fig.add_trace(
                go.Scatter(
                    x=[today_date],
                    y=[today_price],
                    mode='markers',
                    name="Today's Prediction",
                    marker=dict(symbol='circle', size=6, color='blue',
                                line=dict(color='black', width=1), opacity=1),
                    hovertemplate=(
                        f"Date: %{{x}}<br>Price: %{{y:.2f}}<br>Prediction: neutral<br><extra></extra>")
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=[today_date],
                    y=[today_price],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        size=6,
                        color='blue',
                        line=dict(color='black', width=1),
                        opacity=1
                    ),
                    hovertemplate=(
                        f"Date: %{{x}}<br>"
                        f"Price: %{{y:.2f}}<br>"
                        f"Prediction: neutral<br>"
                    ),
                    showlegend=False
                ),
                row=2, col=1
            )

        end_date = stock_data.index.max()
        start_date = end_date - pd.DateOffset(months=6)

        # Update layout to ensure today's prediction is visible
        fig.update_xaxes(
            range=[start_date, today_date + pd.DateOffset(days=7)], row=1, col=1)
        fig.update_xaxes(
            range=[start_date, today_date + pd.DateOffset(days=7)], row=2, col=1)

        # Get price range for zoom period
        zoom_data = stock_data[start_date:end_date]
        low_price = zoom_data['Close'].min() - \
            zoom_data['Close'].min() * 0.1  # Extract numeric value
        high_price = zoom_data['Close'].max() + \
            zoom_data['Close'].max() * 0.1  # Extract numeric value

        # Update layout
        fig.update_layout(
            height=1200,  # Increased height to accommodate RSI
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            hovermode='x unified',
        )

        # Update both subplots with the same zoom settings
        fig.update_xaxes(range=[start_date, end_date +
                         pd.DateOffset(days=7)], row=1, col=1)
        fig.update_xaxes(range=[start_date, end_date +
                         pd.DateOffset(days=7)], row=2, col=1)
        fig.update_xaxes(range=[start_date, end_date +
                         pd.DateOffset(days=7)], row=3, col=1)
        fig.update_yaxes(range=[low_price, high_price], row=1, col=1)
        fig.update_yaxes(range=[low_price, high_price], row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=3, col=1)

        # Update axes labels
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Price ($)", row=2, col=1)

        # Add RSI subplot
        rsi_21 = RSI(stock_data, window=21)
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=rsi_21,
                mode='lines',
                name='21-day RSI',
                line=dict(color='purple', width=1)
            ),
            row=3, col=1
        )

        # Add RSI reference lines at 70 and 30
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.add_hline(y=50, line_dash="dash", line_color="gray", row=3, col=1)

        # Update RSI y-axis range
        fig.update_yaxes(range=[0, 100], row=3, col=1)

        # Show the combined figure
        fig.show()

    if verbose:
        print("\nDebug - Latest predictions:")
        print(f"Latest date: {data.index[-1]}")
        print(f"Prediction: {predictions[-1]}")
        print(f"Probabilities: {probabilities[-1]}")

    return simulation


def get_recent_predictions(stock_data, daily_data, daily_model, daily_scaler, weekly_data, weekly_model, weekly_scaler, days=5, threshold=0.005, use_extra_features=False):
    """
    Generate predictions for the most recent trading days using both daily and weekly models.
    Include probability predictions for each class.
    """
    import os
    # Get features for recent days with proper index

    features = get_features(use_extra_features)
    last_n_days_daily_features = daily_data[features].tail(days)
    last_n_days_weekly_features = weekly_data[features].tail(days)

    # Get predictions and probabilities
    daily_scaled = daily_scaler.transform(last_n_days_daily_features)
    weekly_scaled = weekly_scaler.transform(last_n_days_weekly_features)

    daily_predictions = daily_model.predict(daily_scaled)
    weekly_predictions = weekly_model.predict(weekly_scaled)

    daily_probs = daily_model.predict_proba(daily_scaled)
    weekly_probs = weekly_model.predict_proba(weekly_scaled)

    # Get the numeric values first
    close_values = stock_data['Close'].tail(days).values
    returns = stock_data['Close'].pct_change(fill_method=None).tail(days).values

    # Create results table with formatted values and probabilities
    results = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in stock_data.index[-days:]],
        # Ensure float conversion
        'Close': [f"${float(x):,.2f}" for x in close_values],
        # Ensure float conversion
        'Daily Return': [f"{float(x):.2%}" for x in returns],
        'Daily Model': daily_predictions,
        'Weekly Model': weekly_predictions,
        'Daily Neg%': [f"{p[0]:.1%}" for p in daily_probs],
        'Daily Neu%': [f"{p[1]:.1%}" for p in daily_probs],
        'Daily Pos%': [f"{p[2]:.1%}" for p in daily_probs],
        'Weekly Neg%': [f"{p[0]:.1%}" for p in weekly_probs],
        'Weekly Neu%': [f"{p[1]:.1%}" for p in weekly_probs],
        'Weekly Pos%': [f"{p[2]:.1%}" for p in weekly_probs],
    })

    if threshold == 0.01:
        # Save results to CSV if threshold is 0.01
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(__file__)))), 'data', 'predictions')
        os.makedirs(cache_dir, exist_ok=True)

        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        file_location = f"{cache_dir}/{today}"
        os.makedirs(file_location, exist_ok=True)
        filename = f'{file_location}/prediction_{threshold}.csv'

        # Check if file already exists
        if not os.path.exists(filename):
            results.to_csv(filename, index=False)
            print(f"Saved predictions to {filename}")
        else:
            print(f"File {filename} already exists, skipping save")

    # Add debug prints
    print("\nDebug - Model predictions:")
    print(f"Daily predictions: {daily_predictions}")
    print(f"Weekly predictions: {weekly_predictions}")
    print(f"Daily probabilities: {daily_probs}")
    print(f"Weekly probabilities: {weekly_probs}")

    # Create color arrays for returns
    def get_color(value_str):
        try:
            value = float(value_str.strip('%'))/100
            if value > threshold:  # More than 1%
                return '#c8e6c9'  # green
            elif value < -threshold:  # Less than -1%
                return '#ffcdd2'  # red
            else:  # Between -1% and 1%
                return 'white'
        except:
            return 'white'

    def get_prediction_color(value):
        if value == 'positive':
            return '#c8e6c9'  # green
        elif value == 'negative':
            return '#ffcdd2'  # red
        else:
            return 'white'

    color_arrays = {
        'Date': ['white'] * days,
        'Close': ['white'] * days,
        'Daily Return': [get_color(x) for x in results['Daily Return']],
        'Daily Model': [get_prediction_color(x) for x in results['Daily Model']],
        'Weekly Model': [get_prediction_color(x) for x in results['Weekly Model']]
    }

    # Display table using plotly
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=list(results.columns),
            fill_color='paleturquoise',
            align='left'
        ),
        cells=dict(
            values=[results[col] for col in results.columns],
            fill_color=[color_arrays[col] if col in color_arrays else [
                'white'] * days for col in results.columns],
            align='left'
        )
    )])

    fig.update_layout(
        title='Recent Trading Days - Model Predictions with Probabilities',
        height=400
    )

    fig.show()

    return results


def full_analysis(stock_data, daily_data, daily_model, daily_scaler, weekly_data, weekly_model, weekly_scaler, threshold=0.01, use_extra_features=False, plot=True):
    """Analyze stock using both daily and weekly models, showing only where they agree."""
    # Get predictions and probabilities from both models
    features = get_features(use_extra_features)
    daily_features = daily_data[features]
    weekly_features = weekly_data[features]

    daily_scaled = daily_scaler.transform(daily_features)
    weekly_scaled = weekly_scaler.transform(weekly_features)

    daily_predictions = daily_model.predict(daily_scaled)
    weekly_predictions = weekly_model.predict(weekly_scaled)

    # Get prediction probabilities
    daily_probs = daily_model.predict_proba(daily_scaled)
    weekly_probs = weekly_model.predict_proba(weekly_scaled)

    # Get the Close prices as a 1D array
    close_prices = stock_data['Close'].values.squeeze()

    # Create combined analysis DataFrame with probabilities
    combined_analysis = pd.DataFrame({
        'daily_pred': daily_predictions,
        'weekly_pred': weekly_predictions,
        'return': daily_data['return'],
        'price': close_prices,
        # Get the actual probability values (already between 0 and 1)
        'daily_prob': [probs[list(daily_model.classes_).index(pred)] for pred, probs in zip(daily_predictions, daily_probs)],
        'weekly_prob': [probs[list(weekly_model.classes_).index(pred)] for pred, probs in zip(weekly_predictions, weekly_probs)]
    }, index=daily_data.index)

    # Find where models agree
    agreement_mask = combined_analysis['daily_pred'] == combined_analysis['weekly_pred']
    agreement_signals = combined_analysis[agreement_mask].copy()

    # Add prediction column for trading strategy analysis
    # Since they agree, we can use either one
    agreement_signals['prediction'] = agreement_signals['daily_pred']

    if not plot:
        return agreement_signals

    # Calculate average probability for sizing
    agreement_signals['avg_prob'] = (
        agreement_signals['daily_prob'] + agreement_signals['weekly_prob']) / 2

    # Scale probabilities to marker sizes (between 8 and 24)
    min_size = 5
    max_size = 30
    # Ensure probabilities are properly scaled
    min_prob = agreement_signals['avg_prob'].min()
    max_prob = agreement_signals['avg_prob'].max()
    agreement_signals['marker_size'] = (agreement_signals['avg_prob'] - min_prob) / (
        max_prob - min_prob) * (max_size - min_size) + min_size

    # Create figure with 3 subplots (price with signals and RSI)
    fig = make_subplots(rows=2, cols=1,
                        vertical_spacing=0.05,
                        subplot_titles=('Combined Model Analysis - Signals Where Daily and Weekly Models Agree',
                                        '21-day RSI'),
                        row_heights=[0.8, 0.2])

    # Update the yaxis range in the layout
    fig.update_layout(
        height=1000,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified',
    )

    # Add price series
    fig.add_trace(go.Scatter(
        x=stock_data.index,
        y=close_prices,
        mode='lines',
        name='Price',
        line=dict(color='rgba(0, 0, 255, 0.5)', width=1)
    ), row=1, col=1)

    # Add markers for agreement on positive
    positive_agreement = agreement_signals[agreement_signals['daily_pred'] == 'positive']
    if not positive_agreement.empty:
        fig.add_trace(go.Scatter(
            x=positive_agreement.index,
            y=positive_agreement['price'],
            mode='markers',
            name='Both Models Long',
            marker=dict(
                symbol='triangle-up',
                size=positive_agreement['marker_size'],
                color='green',
                opacity=0.7
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Price: $%{y:.2f}<br>"
                "Return: %{customdata[0]:.2%}<br>"
                "Confidence: %{customdata[1]:.1%}<br>"
                "Signal: Both Daily & Weekly Positive<extra></extra>"
            ),
            customdata=np.column_stack((
                positive_agreement['return'],
                positive_agreement['avg_prob']
            ))
        ))

    # Add markers for agreement on negative
    negative_agreement = agreement_signals[agreement_signals['daily_pred'] == 'negative']
    if not negative_agreement.empty:
        fig.add_trace(go.Scatter(
            x=negative_agreement.index,
            y=negative_agreement['price'],
            mode='markers',
            name='Both Models Short',
            marker=dict(
                symbol='triangle-down',
                size=negative_agreement['marker_size'],
                color='red',
                opacity=0.7
            ),
            hovertemplate=(
                "Date: %{x|%Y-%m-%d}<br>"
                "Price: $%{y:.2f}<br>"
                "Return: %{customdata[0]:.2%}<br>"
                "Confidence: %{customdata[1]:.1%}<br>"
                "Signal: Both Daily & Weekly Negative<extra></extra>"
            ),
            customdata=np.column_stack((
                negative_agreement['return'],
                negative_agreement['avg_prob']
            ))
        ))

    # Print summary statistics
    print("\nDate Ranges:")
    print(f"Stock data: {stock_data.index.min()} to {stock_data.index.max()}")
    print(f"Number of agreement signals: {len(agreement_signals)}")
    print(f"Number of positive signals: {len(positive_agreement)}")
    print(f"Number of negative signals: {len(negative_agreement)}")

    # Print average confidence levels
    if len(positive_agreement) > 0:
        print(
            f"Average confidence for positive signals: {positive_agreement['avg_prob'].mean():.1%}")
    if len(negative_agreement) > 0:
        print(
            f"Average confidence for negative signals: {negative_agreement['avg_prob'].mean():.1%}")

    # Calculate default view range (last 6 months)
    end_date = stock_data.index.max()
    start_date = end_date - pd.DateOffset(months=6)
    # Get price range for zoom period
    zoom_data = stock_data[start_date:end_date]
    low_price = zoom_data['Close'].min(
    ) - zoom_data['Close'].min() * 0.1  # Extract numeric value
    high_price = zoom_data['Close'].max(
    ) + zoom_data['Close'].max() * 0.1  # Extract numeric value
    print(f"Low price: {low_price}, High price: {high_price}")

    end_date = stock_data.index.max()
    start_date = end_date - pd.DateOffset(months=6)

    # Update layout to ensure today's prediction is visible
    fig.update_xaxes(
        range=[start_date, end_date + pd.DateOffset(days=7)], row=1, col=1)
    fig.update_xaxes(
        range=[start_date, end_date + pd.DateOffset(days=7)], row=2, col=1)

    # Get price range for zoom period
    zoom_data = stock_data[start_date:end_date]
    low_price = zoom_data['Close'].min() - \
        zoom_data['Close'].min() * 0.1  # Extract numeric value
    high_price = zoom_data['Close'].max() + \
        zoom_data['Close'].max() * 0.1  # Extract numeric value

    # Update layout
    fig.update_layout(
        height=1200,  # Increased height to accommodate RSI
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified',
    )

    # Update both subplots with the same zoom settings
    fig.update_xaxes(range=[start_date, end_date +
                            pd.DateOffset(days=7)], row=1, col=1)
    fig.update_xaxes(range=[start_date, end_date +
                            pd.DateOffset(days=7)], row=2, col=1)
    fig.update_yaxes(range=[low_price, high_price], row=1, col=1)
    fig.update_yaxes(range=[low_price, high_price], row=2, col=1)

    # Update axes labels
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)

    # Add RSI subplot
    rsi_21 = RSI(stock_data, window=21)
    fig.add_trace(
        go.Scatter(
            x=stock_data.index,
            y=rsi_21,
            mode='lines',
            name='21-day RSI',
            line=dict(color='purple', width=1)
        ),
        row=2, col=1
    )

    # Add RSI reference lines at 70 and 30
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_hline(y=50, line_dash="dash", line_color="gray", row=2, col=1)

    # Update layout
    fig.update_layout(
        height=1000,  # Increased height to accommodate RSI
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified',
        xaxis=dict(
            type='date',
            tickformat='%Y-%m-%d',
            rangeslider=dict(visible=False),
            range=[start_date, end_date + pd.DateOffset(days=7)]
        ),
        yaxis=dict(
            title='Price ($)',
            tickprefix='$',
            range=[low_price, high_price]
        )
    )

    # Update axes labels
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1)

    fig.show()

    return agreement_signals


def check_overfitting(model, X_train, X_test, y_train, y_test, model_name="Model", feature_names=None):
    """
    Test a model for overfitting using multiple methods:
    1. Compare training vs testing accuracy
    2. Learning curves (if supported)
    3. Cross-validation scores
    4. Feature importance analysis (if supported)
    """
    print(f"\n{'='*50}")
    print(f"Overfitting Analysis for {model_name}")
    print(f"{'='*50}")

    # 1. Compare training vs testing accuracy
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print("\n1. Training vs Testing Accuracy:")
    print(f"Training Accuracy: {train_score:.3f}")
    print(f"Testing Accuracy:  {test_score:.3f}")
    print(f"Difference:        {train_score - test_score:.3f}")

    if train_score - test_score > 0.05:
        print("WARNING: Training accuracy is significantly higher than testing accuracy,")
        print("         suggesting possible overfitting.")

    # 2. Learning curves
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        n_jobs=-1
    )

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score',
             color='blue', marker='o')
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, label='Cross-validation score',
             color='red', marker='o')
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.15, color='red')
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title(f'Learning Curves - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # 3. Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    print("\n2. Cross-validation Scores:")
    print(
        f"Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

    # 4. Feature importance analysis (if available)
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        print("\n3. Feature Importance Analysis:")
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        plt.figure(figsize=(10, 4))
        plt.bar(importance['feature'], importance['importance'])
        plt.title(f'Feature Importance - {model_name}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print("\nFeature Importance:")
        for idx, row in importance.iterrows():
            print(f"{row['feature']}: {row['importance']:.3f}")

    # Provide recommendations
    print("\nRecommendations:")
    if train_score - test_score > 0.05:
        print("- Consider using regularization")
        print("- Reduce model complexity")
        print("- Gather more training data")
        print("- Feature selection/reduction")
    else:
        print("- Model shows good balance between training and testing performance")
        print("- Continue monitoring for changes in production")


def calculate_rmse(group_returns, group_mean):
    """Calculate RMSE for a group of returns."""
    return np.sqrt(np.mean((group_returns - group_mean) ** 2))


def analyze_prediction_probabilities(stock_data, daily_model, daily_scaler, daily_data,
                                     weekly_model, weekly_scaler, weekly_data, threshold=0.01, use_extra_features=False):
    """Analyze how prediction probabilities correlate with actual returns."""

    features = get_features(use_extra_features)
    daily_features = daily_data[features]
    weekly_features = weekly_data[features]

    daily_scaled = daily_scaler.transform(daily_features)
    weekly_scaled = weekly_scaler.transform(weekly_features)

    # Get predictions and probabilities
    daily_preds = daily_model.predict(daily_scaled)
    weekly_preds = weekly_model.predict(weekly_scaled)
    daily_probs = daily_model.predict_proba(daily_scaled)
    weekly_probs = weekly_model.predict_proba(weekly_scaled)

    # Create analysis DataFrame
    analysis = pd.DataFrame({
        'actual_return': daily_data['return'],
        'daily_pred': daily_preds,
        'weekly_pred': weekly_preds,
        'daily_neg_prob': daily_probs[:, 0],
        'daily_neu_prob': daily_probs[:, 1],
        'daily_pos_prob': daily_probs[:, 2],
        'weekly_neg_prob': weekly_probs[:, 0],
        'weekly_neu_prob': weekly_probs[:, 1],
        'weekly_pos_prob': weekly_probs[:, 2],
    })

    # Filter probabilities to only include when actual classifications are made
    analysis['daily_neg_prob'] = np.where(analysis['daily_pred'] == 'negative',
                                          analysis['daily_neg_prob'], np.nan)
    analysis['daily_pos_prob'] = np.where(analysis['daily_pred'] == 'positive',
                                          analysis['daily_pos_prob'], np.nan)
    analysis['weekly_neg_prob'] = np.where(analysis['weekly_pred'] == 'negative',
                                           analysis['weekly_neg_prob'], np.nan)
    analysis['weekly_pos_prob'] = np.where(analysis['weekly_pred'] == 'positive',
                                           analysis['weekly_pos_prob'], np.nan)

    # Add combined probabilities only where models agree
    analysis['combined_neg_prob'] = np.where(
        (analysis['daily_pred'] == 'negative') & (
            analysis['weekly_pred'] == 'negative'),
        (analysis['daily_neg_prob'] + analysis['weekly_neg_prob']) / 2,
        np.nan
    )

    analysis['combined_pos_prob'] = np.where(
        (analysis['daily_pred'] == 'positive') & (
            analysis['weekly_pred'] == 'positive'),
        (analysis['daily_pos_prob'] + analysis['weekly_pos_prob']) / 2,
        np.nan
    )

    # Updated confidence bins to start at 0.333
    confidence_bins = [0.333, 0.555, 0.777, 1.0]
    confidence_labels = ['Low', 'Medium', 'High']

    # Create subplots
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=('Daily Model (SEM)', 'Daily Model (RMSE)', 'Daily Model Returns Distribution',
                        'Weekly Model (SEM)', 'Weekly Model (RMSE)', 'Weekly Model Returns Distribution',
                        'Combined Model (SEM)', 'Combined Model (RMSE)', 'Combined Model Returns Distribution'),
        column_widths=[0.33, 0.33, 0.34]
    )

    models = {
        'Daily': ['daily_neg_prob', 'daily_pos_prob'],
        'Weekly': ['weekly_neg_prob', 'weekly_pos_prob'],
        'Combined': ['combined_neg_prob', 'combined_pos_prob']
    }

    row = 1
    for model_name, probs in models.items():
        neg_prob, pos_prob = probs

        # Create confidence categories (only for non-NaN values)
        neg_valid = analysis[neg_prob].dropna()
        pos_valid = analysis[pos_prob].dropna()

        if len(neg_valid) > 0:
            analysis[f'{model_name.lower()}_neg_conf'] = pd.cut(
                neg_valid,
                bins=confidence_bins,
                labels=confidence_labels
            )

        if len(pos_valid) > 0:
            analysis[f'{model_name.lower()}_pos_conf'] = pd.cut(
                pos_valid,
                bins=confidence_bins,
                labels=confidence_labels
            )

        # Calculate statistics for each confidence level
        neg_returns_by_conf = {}
        pos_returns_by_conf = {}

        for conf in confidence_labels:
            conf_col_neg = f'{model_name.lower()}_neg_conf'
            conf_col_pos = f'{model_name.lower()}_pos_conf'

            if conf_col_neg in analysis.columns:
                neg_mask = (analysis[conf_col_neg] ==
                            conf) & analysis['actual_return'].notna()
                neg_group = analysis[neg_mask]['actual_return']

                if len(neg_group) > 0:
                    neg_mean = neg_group.mean()
                    neg_returns_by_conf[conf] = {
                        'mean': neg_mean,
                        'count': len(neg_group),
                        'std': neg_group.std(),
                        'rmse': calculate_rmse(neg_group, neg_mean)
                    }

            if conf_col_pos in analysis.columns:
                pos_mask = (analysis[conf_col_pos] ==
                            conf) & analysis['actual_return'].notna()
                pos_group = analysis[pos_mask]['actual_return']

                if len(pos_group) > 0:
                    pos_mean = pos_group.mean()
                    pos_returns_by_conf[conf] = {
                        'mean': pos_mean,
                        'count': len(pos_group),
                        'std': pos_group.std(),
                        'rmse': calculate_rmse(pos_group, pos_mean)
                    }

        # Convert to DataFrame only if we have data
        if neg_returns_by_conf:
            neg_returns = pd.DataFrame(neg_returns_by_conf).T
        else:
            neg_returns = pd.DataFrame()

        if pos_returns_by_conf:
            pos_returns = pd.DataFrame(pos_returns_by_conf).T
        else:
            pos_returns = pd.DataFrame()

        # Plot with SEM error bars
        if not neg_returns.empty:
            fig.add_trace(
                go.Bar(
                    name='Negative Signals (SEM)',
                    x=neg_returns.index,
                    y=neg_returns['mean'] * 100,
                    error_y=dict(
                        type='data',
                        array=(neg_returns['std'] /
                               np.sqrt(neg_returns['count'])) * 100,
                        visible=True
                    ),
                    marker_color='red',
                ),
                row=row, col=1
            )

        if not pos_returns.empty:
            fig.add_trace(
                go.Bar(
                    name='Positive Signals (SEM)',
                    x=pos_returns.index,
                    y=pos_returns['mean'] * 100,
                    error_y=dict(
                        type='data',
                        array=(pos_returns['std'] /
                               np.sqrt(pos_returns['count'])) * 100,
                        visible=True
                    ),
                    marker_color='green',
                ),
                row=row, col=1
            )

        # Plot with RMSE error bars
        if not neg_returns.empty:
            fig.add_trace(
                go.Bar(
                    name='Negative Signals (RMSE)',
                    x=neg_returns.index,
                    y=neg_returns['mean'] * 100,
                    error_y=dict(
                        type='data',
                        array=neg_returns['rmse'] * 100,
                        visible=True
                    ),
                    marker_color='red',
                ),
                row=row, col=2
            )

        if not pos_returns.empty:
            fig.add_trace(
                go.Bar(
                    name='Positive Signals (RMSE)',
                    x=pos_returns.index,
                    y=pos_returns['mean'] * 100,
                    error_y=dict(
                        type='data',
                        array=pos_returns['rmse'] * 100,
                        visible=True
                    ),
                    marker_color='green',
                ),
                row=row, col=2
            )

        # Distribution plots
        for conf_level in confidence_labels:
            conf_col_neg = f'{model_name.lower()}_neg_conf'
            conf_col_pos = f'{model_name.lower()}_pos_conf'

            # Negative signals
            if conf_col_neg in analysis.columns:
                neg_mask = (analysis[conf_col_neg] ==
                            conf_level) & analysis['actual_return'].notna()
                neg_dist = analysis[neg_mask]['actual_return']

                if len(neg_dist) > 0:
                    neg_dist = neg_dist.replace(
                        [np.inf, -np.inf], np.nan).dropna()
                    if len(neg_dist) > 0:
                        if len(neg_dist) > 1000:
                            neg_dist = neg_dist.sample(n=1000, random_state=42)

                        try:
                            kernel = gaussian_kde(neg_dist * 100)
                            x_range = np.linspace(
                                neg_dist.min() * 100, neg_dist.max() * 100, 100)

                            line_styles = {
                                'High': 'solid',
                                'Medium': 'dash',
                                'Low': 'dot'
                            }

                            fig.add_trace(
                                go.Scatter(
                                    x=x_range,
                                    y=kernel(x_range),
                                    name=f'Negative {conf_level}',
                                    line=dict(
                                        color='red',
                                        dash=line_styles[conf_level],
                                        width=2 if conf_level == 'High' else 1
                                    ),
                                    opacity=0.7 if conf_level == 'High' else 0.4,
                                    showlegend=True
                                ),
                                row=row, col=3
                            )
                        except Exception as e:
                            print(
                                f"Warning: Could not create KDE for negative {conf_level} signals: {e}")

            # Positive signals
            if conf_col_pos in analysis.columns:
                pos_mask = (analysis[conf_col_pos] ==
                            conf_level) & analysis['actual_return'].notna()
                pos_dist = analysis[pos_mask]['actual_return']

                if len(pos_dist) > 0:
                    pos_dist = pos_dist.replace(
                        [np.inf, -np.inf], np.nan).dropna()
                    if len(pos_dist) > 0:
                        if len(pos_dist) > 1000:
                            pos_dist = pos_dist.sample(n=1000, random_state=42)

                        try:
                            kernel = gaussian_kde(pos_dist * 100)
                            x_range = np.linspace(
                                pos_dist.min() * 100, pos_dist.max() * 100, 100)

                            fig.add_trace(
                                go.Scatter(
                                    x=x_range,
                                    y=kernel(x_range),
                                    name=f'Positive {conf_level}',
                                    line=dict(
                                        color='green',
                                        dash=line_styles[conf_level],
                                        width=2 if conf_level == 'High' else 1
                                    ),
                                    opacity=0.7 if conf_level == 'High' else 0.4,
                                    showlegend=True
                                ),
                                row=row, col=3
                            )
                        except Exception as e:
                            print(
                                f"Warning: Could not create KDE for positive {conf_level} signals: {e}")

        row += 1

    # Update layout
    fig.update_layout(
        height=1200,
        width=1500,
        showlegend=True,
        title_text="Return Analysis by Model and Confidence Level",
    )

    # Update axes labels
    for i in range(1, 4):
        fig.update_xaxes(title_text="Confidence Level", row=i, col=1)
        fig.update_xaxes(title_text="Confidence Level", row=i, col=2)
        fig.update_yaxes(title_text="Mean Return (%)", row=i, col=1)
        fig.update_yaxes(title_text="Mean Return (%)", row=i, col=2)
        fig.update_xaxes(title_text="Return (%)", row=i, col=3)
        fig.update_yaxes(title_text="Density", row=i, col=3)

    fig.show()

    # Calculate correlations only for non-NaN values
    correlations = pd.DataFrame({
        'Daily Negative': analysis['actual_return'].corr(analysis['daily_neg_prob'].dropna()),
        'Daily Positive': analysis['actual_return'].corr(analysis['daily_pos_prob'].dropna()),
        'Weekly Negative': analysis['actual_return'].corr(analysis['weekly_neg_prob'].dropna()),
        'Weekly Positive': analysis['actual_return'].corr(analysis['weekly_pos_prob'].dropna()),
        'Combined Negative': analysis['actual_return'].corr(analysis['combined_neg_prob'].dropna()),
        'Combined Positive': analysis['actual_return'].corr(analysis['combined_pos_prob'].dropna()),
    }, index=['Correlation with Returns'])

    print("\nCorrelations between Probabilities and Actual Returns:")
    print(correlations.round(4))

    # Print additional statistics
    for model_name, probs in models.items():
        neg_prob, pos_prob = probs
        print(f"\n{model_name} Model Statistics:")
        print(f"Negative signals: {analysis[neg_prob].notna().sum()}")
        print(f"Positive signals: {analysis[pos_prob].notna().sum()}")

        # Distribution of confidence levels
        for direction in ['neg', 'pos']:
            conf_col = f'{model_name.lower()}_{direction}_conf'
            if conf_col in analysis:
                print(
                    f"\n{model_name} {direction.capitalize()} Confidence Distribution:")
                print(analysis[conf_col].value_counts(
                    normalize=True).round(3) * 100)

    return analysis


def test_model_on_new_ticker(trained_daily_model, trained_daily_scaler, trained_weekly_model, trained_weekly_scaler,
                             test_ticker, period='5y', threshold=0.005, use_extra_features=False):
    """
    Test pre-trained models on a new ticker symbol.

    Parameters:
    -----------
    trained_daily_model : sklearn model
        Pre-trained daily prediction model
    trained_daily_scaler : sklearn.preprocessing.StandardScaler
        Scaler used with daily model
    trained_weekly_model : sklearn model
        Pre-trained weekly prediction model
    trained_weekly_scaler : sklearn.preprocessing.StandardScaler
        Scaler used with weekly model
    test_ticker : str
        Ticker symbol to test models on
    period : str, default='5y'
        Period of historical data to test on
    threshold : float, default=0.005
        Threshold for classifying returns

    Returns:
    --------
    dict
        Dictionary containing test results and data
    """
    print(f"\n{'='*50}")
    print(f"Testing models on {test_ticker}")
    print(f"{'='*50}")

    # Get test data
    test_data = get_index_data(test_ticker, period)
    print(f"\nLoaded {len(test_data)} days of data for {test_ticker}")

    # Prepare test data
    test_daily_data = prepare_classification_data(
        test_data, predict_weekly=False, threshold=threshold, use_extra_features=use_extra_features)
    test_weekly_data = prepare_classification_data(
        test_data, predict_weekly=True, threshold=threshold, use_extra_features=use_extra_features)

    # Create visualization
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=(
                            f'Daily Model Signals - {test_ticker}',
                            f'Weekly Model Signals - {test_ticker}'
                        ),
                        vertical_spacing=0.1,
                        shared_xaxes=True)

    # Add price series to both subplots
    for row in [1, 2]:
        fig.add_trace(
            go.Scatter(
                x=test_data.index,
                y=test_data['Close'],
                mode='lines',
                name='Price',
                line=dict(color='rgba(0, 0, 255, 0.5)', width=1),
                showlegend=False if row == 2 else True
            ),
            row=row, col=1
        )

    # Get predictions for daily model
    features = get_features(use_extra_features)
    daily_features = test_daily_data[features]

    daily_scaled = trained_daily_scaler.transform(daily_features)
    daily_preds = trained_daily_model.predict(daily_scaled)
    daily_probs = trained_daily_model.predict_proba(daily_scaled)

    # Get predictions for weekly model
    if use_extra_features:
        weekly_features = test_weekly_data[[
            'prev_day_return', 'prev_2day_return', 'prev_week_return', 'volatility_5d', 'volatility_21d', 'ma_5d', 'ma_21d', 'volume_1d', 'volume_5d']]
    else:
        weekly_features = test_weekly_data[[
            'prev_day_return', 'prev_2day_return', 'prev_week_return']]

    weekly_scaled = trained_weekly_scaler.transform(weekly_features)
    weekly_preds = trained_weekly_model.predict(weekly_scaled)
    weekly_probs = trained_weekly_model.predict_proba(weekly_scaled)

    # Add signals to plots
    for row, (preds, probs) in enumerate([(daily_preds, daily_probs), (weekly_preds, weekly_probs)], 1):
        # Add positive signals
        pos_mask = preds == 'positive'
        if pos_mask.any():
            # Get max probability for sizing
            pos_probs = [max(p) for p in probs[pos_mask]]
            pos_dates = test_data.index[pos_mask]
            pos_prices = test_data.loc[pos_dates, 'Close']

            fig.add_trace(
                go.Scatter(
                    x=pos_dates,
                    y=pos_prices,
                    mode='markers',
                    name='Buy Signal' if row == 1 else 'Weekly Buy',
                    marker=dict(
                        symbol='triangle-up',
                        # Scale marker size by probability
                        size=[p * 20 for p in pos_probs],
                        color='green',
                        opacity=0.7
                    ),
                    hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<br>Confidence: %{text:.1%}<extra></extra>",
                    text=pos_probs
                ),
                row=row, col=1
            )

        # Add negative signals
        neg_mask = preds == 'negative'
        if neg_mask.any():
            neg_probs = [max(p) for p in probs[neg_mask]]
            neg_dates = test_data.index[neg_mask]
            neg_prices = test_data.loc[neg_dates, 'Close']

            fig.add_trace(
                go.Scatter(
                    x=neg_dates,
                    y=neg_prices,
                    mode='markers',
                    name='Sell Signal' if row == 1 else 'Weekly Sell',
                    marker=dict(
                        symbol='triangle-down',
                        size=[p * 20 for p in neg_probs],
                        color='red',
                        opacity=0.7
                    ),
                    hovertemplate="Date: %{x}<br>Price: $%{y:.2f}<br>Confidence: %{text:.1%}<extra></extra>",
                    text=neg_probs
                ),
                row=row, col=1
            )

    # Update layout
    fig.update_layout(
        height=1000,
        title_text=f"Model Signals for {test_ticker} (Marker Size = Confidence)",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        hovermode='x unified'
    )

    # Update axes labels
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)

    # Show the plot
    fig.show()

    # Get recent predictions
    recent_predictions = get_recent_predictions(
        test_data,
        trained_daily_model,
        trained_daily_scaler,
        trained_weekly_model,
        trained_weekly_scaler
    )

    print("\nRecent Predictions:")
    print(recent_predictions.to_string())

    return {
        'test_data': test_data,
        'daily_predictions': daily_preds,
        'weekly_predictions': weekly_preds,
        'daily_probabilities': daily_probs,
        'weekly_probabilities': weekly_probs,
        'recent_predictions': recent_predictions
    }


def test_model_performance(daily_data, daily_model, daily_scaler, weekly_data, weekly_model, weekly_scaler, threshold=0.005, use_extra_features=False):
    """
    Test model performance on holdout data and display results in a table with granular return buckets.
    Returns predictions DataFrame for use in trading strategy analysis.
    """
    # Validate test data

    # Prepare test data for both models

    # Validate prepared data
    if len(daily_data) == 0 or len(weekly_data) == 0:
        print("Error: No valid samples after data preparation")
        return None

    # Get features
    features = get_features(use_extra_features)
    daily_features = daily_data[features]
    weekly_features = weekly_data[features]

    # Scale features
    daily_scaled = daily_scaler.transform(daily_features)
    weekly_scaled = weekly_scaler.transform(weekly_features)

    # Get predictions
    daily_preds = daily_model.predict(daily_scaled)
    weekly_preds = weekly_model.predict(weekly_scaled)

    # Create results DataFrame
    results = pd.DataFrame({
        'date': daily_data.index,
        'daily_pred': daily_preds,
        'weekly_pred': weekly_preds,
        'next_day_return': daily_data['return'],
        'next_week_return': weekly_data['return'],
        'prediction': 'neutral'  # Initialize with neutral
    })

    # Set predictions for agreement signals
    results.loc[(results['daily_pred'] == 'positive') &
                (results['weekly_pred'] == 'positive'), 'prediction'] = 'positive'
    results.loc[(results['daily_pred'] == 'negative') &
                (results['weekly_pred'] == 'negative'), 'prediction'] = 'negative'

    # Create detailed return buckets
    def categorize_return_detailed(x):
        if pd.isna(x):
            return 'unknown'
        elif x < -0.02:
            return 'strong_negative'
        elif x < -0.005:
            return 'negative'
        elif x <= 0.005:
            return 'neutral'
        elif x <= 0.02:
            return 'positive'
        else:
            return 'strong_positive'

    results['actual_daily_detailed'] = results['next_day_return'].apply(
        categorize_return_detailed)
    results['actual_weekly_detailed'] = results['next_week_return'].apply(
        categorize_return_detailed)

    # Calculate performance metrics
    print(f"\n{'='*50}")
    print("Model Performance on Test Data")
    print(f"{'='*50}")

    # Create confusion matrices and visualizations for both models
    models = {
        'Daily': ('daily_pred', 'actual_daily_detailed', 'next_day_return'),
        'Weekly': ('weekly_pred', 'actual_weekly_detailed', 'next_week_return'),
        # Added Combined model
        'Combined': ('prediction', 'actual_daily_detailed', 'next_day_return')
    }

    for model_name, (pred_col, actual_col, return_col) in models.items():
        print(f"\n{model_name} Model Performance:")
        print("-" * 30)

        # Skip if no predictions for this model
        if results[pred_col].value_counts().empty:
            print(f"No predictions available for {model_name} model")
            continue

        # Calculate distribution of actual returns
        actual_dist = results[actual_col].value_counts(normalize=True)

        # Create Plotly figure with subplots
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'{model_name} Return Distribution',
                            f'{model_name} Average Returns by Prediction'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )

        # Add distribution plot
        colors = {
            'strong_negative': 'red',
            'negative': 'lightcoral',
            'neutral': 'grey',
            'positive': 'lightgreen',
            'strong_positive': 'green',
            'unknown': 'lightgrey'
        }

        # Plot 1: Return Distribution
        fig.add_trace(
            go.Bar(
                x=actual_dist.index,
                y=actual_dist.values * 100,
                name='Actual Returns',
                marker_color=[colors[cat] for cat in actual_dist.index],
                text=[f'{val:.1f}%' for val in (actual_dist.values * 100)],
                textposition='auto',
            ),
            row=1, col=1
        )

        # Calculate average returns by prediction
        avg_returns = results.groupby(pred_col)[return_col].agg([
            'count',
            'mean',
            'std',
            ('win_rate', lambda x: (x > 0).mean())
        ]).round(4)

        # Plot 2: Average Returns by Prediction
        fig.add_trace(
            go.Bar(
                x=avg_returns.index,
                y=avg_returns['mean'] * 100,
                name='Average Return',
                error_y=dict(
                    type='data',
                    array=(avg_returns['std'] /
                           np.sqrt(avg_returns['count'])) * 100,
                    visible=True
                ),
                text=[f'{val:.1f}%' for val in (avg_returns['mean'] * 100)],
                textposition='auto',
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            height=500,
            width=1200,
            title_text=f"{model_name} Model Performance Analysis",
            showlegend=False,
            plot_bgcolor='white'
        )

        # Update axes
        fig.update_yaxes(title_text='Percentage', ticksuffix='%',
                         gridcolor='lightgrey', gridwidth=0.5)
        fig.update_xaxes(title_text='Return Category',
                         gridcolor='lightgrey', gridwidth=0.5)

        fig.show()

        # Print numerical results
        print("\nConfusion Matrix:")
        confusion = pd.crosstab(
            results[pred_col], results[actual_col], margins=True)
        print(confusion)

        print("\nReturn Statistics by Prediction:")
        stats_df = avg_returns.copy()
        stats_df['mean'] = stats_df['mean'].map('{:.2%}'.format)
        stats_df['win_rate'] = stats_df['win_rate'].map('{:.2%}'.format)
        print(stats_df)

        # Calculate accuracy metrics
        valid_mask = results[actual_col] != 'unknown'
        accuracy = (results[pred_col][valid_mask] ==
                    results[actual_col][valid_mask]).mean()
        print(f"\nOverall Accuracy: {accuracy:.2%}")

    return results


DEFAULT_VARIANT_SPECS = [
    {
        'name': 'baseline',
        'threshold': 0.005,
        'extra_features': False,
        'n_estimators': 100,
    },
    {
        'name': 'extra_features',
        'threshold': 0.005,
        'extra_features': True,
        'n_estimators': 100,
    },
    {
        'name': 'threshold_1pct',
        'threshold': 0.01,
        'extra_features': False,
        'n_estimators': 100,
    },
    {
        'name': 'threshold_1pct_extra',
        'threshold': 0.01,
        'extra_features': True,
        'n_estimators': 100,
    },
    {
        'name': 'rf_deep',
        'threshold': 0.005,
        'extra_features': True,
        'n_estimators': 200,
        'max_depth': 10,
    },
    {
        'name': 'rf_shallow',
        'threshold': 0.005,
        'extra_features': False,
        'n_estimators': 50,
        'max_depth': 3,
    },
]


def _actual_movement_class(return_val, threshold):
    if pd.isna(return_val):
        return None
    if return_val < -threshold:
        return 'negative'
    if return_val > threshold:
        return 'positive'
    return 'neutral'


def _prediction_accuracy(predictions, actuals):
    pairs = [(pred, actual) for pred, actual in zip(predictions, actuals) if actual is not None]
    if not pairs:
        return None
    return sum(pred == actual for pred, actual in pairs) / len(pairs)


def compute_holdout_metrics(
    daily_holdout,
    daily_model,
    daily_scaler,
    weekly_holdout,
    weekly_model,
    weekly_scaler,
    threshold,
    use_extra_features,
):
    """Compute holdout metrics for the dual daily+weekly movement classifier pipeline."""
    features = get_features(use_extra_features)
    daily_scaled = daily_scaler.transform(daily_holdout[features])
    weekly_scaled = weekly_scaler.transform(weekly_holdout[features])

    daily_preds = daily_model.predict(daily_scaled)
    weekly_preds = weekly_model.predict(weekly_scaled)
    actual_daily = [
        _actual_movement_class(value, threshold) for value in daily_holdout['return']
    ]
    actual_weekly = [
        _actual_movement_class(value, threshold) for value in weekly_holdout['return']
    ]

    combined_preds = []
    for daily_pred, weekly_pred in zip(daily_preds, weekly_preds):
        if daily_pred == 'positive' and weekly_pred == 'positive':
            combined_preds.append('positive')
        elif daily_pred == 'negative' and weekly_pred == 'negative':
            combined_preds.append('negative')
        else:
            combined_preds.append('neutral')

    daily_returns = daily_holdout['return'].tolist()
    combined_positive_returns = [
        daily_returns[index]
        for index, pred in enumerate(combined_preds)
        if pred == 'positive' and not pd.isna(daily_returns[index])
    ]
    combined_negative_returns = [
        daily_returns[index]
        for index, pred in enumerate(combined_preds)
        if pred == 'negative' and not pd.isna(daily_returns[index])
    ]

    combined_signal_count = sum(pred != 'neutral' for pred in combined_preds)
    holdout_samples = len(daily_holdout)

    return {
        'holdout_samples': holdout_samples,
        'daily_accuracy': _prediction_accuracy(daily_preds, actual_daily),
        'weekly_accuracy': _prediction_accuracy(weekly_preds, actual_weekly),
        'combined_accuracy': _prediction_accuracy(combined_preds, actual_daily),
        'combined_signal_rate': (
            combined_signal_count / holdout_samples if holdout_samples else None
        ),
        'combined_positive_avg_return': (
            float(np.mean(combined_positive_returns))
            if combined_positive_returns else None
        ),
        'combined_negative_avg_return': (
            float(np.mean(combined_negative_returns))
            if combined_negative_returns else None
        ),
    }


def _build_variant_classifier(variant):
    params = {
        'n_estimators': variant.get('n_estimators', 100),
        'random_state': 42,
    }
    if 'max_depth' in variant:
        params['max_depth'] = variant['max_depth']
    return RandomForestClassifier(**params)


def compare_variants(
    ticker='^GSPC',
    period='20y',
    holdout_months=12,
    variants=None,
    verbose=True,
):
    """
    Train multiple movement-classifier pipeline variants on the same train period
    and compare them on a shared temporal holdout.
    """
    if variants is None:
        variants = DEFAULT_VARIANT_SPECS

    data = get_index_data(ticker, period)
    if data is None or data.empty:
        raise ValueError(f"No data for {ticker}")

    rows = []
    holdout_start = None

    for variant in variants:
        name = variant['name']
        threshold = variant['threshold']
        use_extra_features = variant['extra_features']

        if verbose:
            print(f"\n{'='*60}")
            print(f"Variant: {name}")
            print(f"{'='*60}")

        prepared_daily = prepare_classification_data(
            data,
            predict_weekly=False,
            threshold=threshold,
            use_extra_features=use_extra_features,
        ).dropna()
        prepared_weekly = prepare_classification_data(
            data,
            predict_weekly=True,
            threshold=threshold,
            use_extra_features=use_extra_features,
        ).dropna()

        daily_train, daily_holdout = temporal_holdout_split(
            prepared_daily, holdout_months=holdout_months
        )
        weekly_train, weekly_holdout = temporal_holdout_split(
            prepared_weekly, holdout_months=holdout_months
        )
        holdout_start = daily_holdout.index.min()

        classifier = _build_variant_classifier(variant)
        daily_model, daily_scaler, _ = train_classifier_single_stock(
            daily_train,
            use_extra_features=use_extra_features,
            threshold=threshold,
            classifier=classifier,
            eval_split=0,
            verbose=False,
        )
        weekly_model, weekly_scaler, _ = train_classifier_single_stock(
            weekly_train,
            predict_weekly=True,
            use_extra_features=use_extra_features,
            threshold=threshold,
            classifier=_build_variant_classifier(variant),
            eval_split=0,
            verbose=False,
        )

        metrics = compute_holdout_metrics(
            daily_holdout,
            daily_model,
            daily_scaler,
            weekly_holdout,
            weekly_model,
            weekly_scaler,
            threshold,
            use_extra_features,
        )

        rows.append({
            'variant': name,
            'threshold': threshold,
            'extra_features': use_extra_features,
            'n_estimators': variant.get('n_estimators', 100),
            'max_depth': variant.get('max_depth', 'default'),
            'train_samples': len(daily_train),
            **metrics,
        })

    results = pd.DataFrame(rows).sort_values(
        'combined_accuracy', ascending=False, na_position='last'
    ).reset_index(drop=True)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Variant comparison for {ticker} (holdout={holdout_months} months)")
        if holdout_start is not None:
            print(f"Holdout starts: {holdout_start.date()}")
        print(f"{'='*60}")

        display_cols = [
            'variant', 'threshold', 'extra_features', 'n_estimators', 'max_depth',
            'daily_accuracy', 'weekly_accuracy', 'combined_accuracy',
            'combined_signal_rate', 'combined_positive_avg_return',
            'combined_negative_avg_return', 'holdout_samples',
        ]
        printable = results[display_cols].copy()
        for col in [
            'daily_accuracy', 'weekly_accuracy', 'combined_accuracy',
            'combined_signal_rate', 'combined_positive_avg_return',
            'combined_negative_avg_return',
        ]:
            printable[col] = printable[col].map(
                lambda value: f"{value:.2%}" if pd.notna(value) else 'n/a'
            )
        print(printable.to_string(index=False))

        best = results.iloc[0]
        print(
            f"\nBest variant by combined accuracy: {best['variant']} "
            f"({best['combined_accuracy']:.2%})"
        )

    return results


def evaluate_all_classifiers(
    prepared_data,
    threshold=0.005,
    use_extra_features=False,
    test_size=0.2,
    holdout_months=None,
    plot=True,
    classifier_names=None,
):
    """
    Compare classifiers with time-series CV on the training period,
    then evaluate the best model on a temporal holdout set.
    """
    features = get_features(use_extra_features)
    train_data, holdout_data = temporal_holdout_split(
        prepared_data, holdout_months=holdout_months, test_size=test_size
    )

    if len(train_data) < 30:
        raise ValueError(f"Insufficient training data: {len(train_data)} samples")
    if len(holdout_data) < 10:
        raise ValueError(f"Insufficient holdout data: {len(holdout_data)} samples")

    print(
        f"\nTemporal split: train={len(train_data)}, holdout={len(holdout_data)}"
    )
    if holdout_months:
        cutoff = holdout_data.index.min()
        print(f"Holdout starts at: {cutoff.date()}")

    X_train = train_data[features]
    y_train = train_data['Target'].map({'negative': 0, 'neutral': 1, 'positive': 2})
    X_holdout = holdout_data[features]
    y_holdout = holdout_data['Target'].map({'negative': 0, 'neutral': 1, 'positive': 2})

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_holdout_scaled = scaler.transform(X_holdout)

    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))

    results = compare_classifiers(
        X_train_scaled, y_train, class_weights=class_weight_dict,
        classifier_names=classifier_names,
    )
    print_best_configs(results)
    if plot:
        plot_classifier_comparison(results)

    best_name = max(results.keys(), key=lambda n: results[n]['cv_results']['f1'])
    best_params = results[best_name]['best_params']
    print(
        f"\nBest classifier: {best_name} "
        f"(CV F1={results[best_name]['cv_results']['f1']:.4f})"
    )

    best_clf = deepcopy(classifiers[best_name])
    best_clf.set_params(**best_params)
    best_clf.fit(X_train_scaled, y_train)

    y_pred = best_clf.predict(X_holdout_scaled)
    print(f"\n{'='*50}")
    print(f"Holdout evaluation: {best_name}")
    print(f"{'='*50}")
    print(classification_report(
        y_holdout, y_pred, target_names=['negative', 'neutral', 'positive']
    ))

    holdout_metrics = {
        'accuracy': float(accuracy_score(y_holdout, y_pred)),
        'f1_weighted': float(f1_score(
            y_holdout, y_pred, average='weighted', zero_division=0
        )),
    }

    return {
        'comparison_results': results,
        'best_classifier': best_name,
        'best_params': best_params,
        'best_model': best_clf,
        'scaler': scaler,
        'holdout_metrics': holdout_metrics,
        'train_size': len(train_data),
        'holdout_size': len(holdout_data),
    }


def test_classifiers():
    """Test and compare different classifiers including ensemble (binary and combined), RSI and MA signals."""
    print("Testing classifier comparison")
    data = get_index_data("^GSPC", "20y")
    prepared_data = prepare_classification_data_enhanced(
        data,
        predict_weekly=False,
        threshold=0.005,
        use_extra_features=True
    )

    # Remove rows with NaN values in features or target
    prepared_data = prepared_data.dropna()

    # Separate features and target
    X = prepared_data.drop(columns=['Target', 'return'])
    y = prepared_data['Target']

    # Convert target to numeric
    target_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    y = y.map(target_map)

    # Use the last TimeSeriesSplit fold (largest train, most recent test)
    tscv = TimeSeriesSplit(n_splits=5)
    train_index, test_index = None, None
    for train_index, test_index in tscv.split(X):
        pass

    X_train_raw = X.iloc[train_index]
    X_test_raw = X.iloc[test_index]
    y_train = y.iloc[train_index]
    y_test = y.iloc[test_index]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # Train both ensemble approaches
    print("\nTraining Binary Ensemble Classifier...")
    binary_ensemble = EnsembleClassifier(mode='binary')
    binary_ensemble.fit(X_train, y_train)

    print("\nTraining Combined Ensemble Classifier...")
    combined_ensemble = EnsembleClassifier(mode='combined')
    combined_ensemble.fit(X_train, y_train)

    # Get predictions for both approaches
    binary_preds = binary_ensemble.predict(X_test)
    binary_probs = binary_ensemble.predict_proba(X_test)

    combined_preds = combined_ensemble.predict(X_test)
    combined_probs = combined_ensemble.predict_proba(X_test)

    # Convert numeric predictions back to labels
    reverse_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    binary_signals = pd.Series(
        [reverse_map[p] for p in binary_preds], index=data.index[test_index])
    combined_signals = pd.Series(
        [reverse_map[p] for p in combined_preds], index=data.index[test_index])

    # Generate RSI signals
    print("\nGenerating RSI signals...")
    rsi_data = generate_rsi_signals(data)
    rsi_signals = pd.Series(rsi_data['Signal'].map({1: 'positive', -1: 'negative', 0: 'neutral'}),
                            index=data.index)

    # Generate MA signals
    print("\nGenerating MA signals...")
    ma_data = generate_multi_ma_signals(data)
    ma_signals = pd.Series(ma_data['Combined_Signal'].map({1: 'positive', -1: 'negative', 0: 'neutral'}),
                           index=data.index)

    # Prepare test period data
    test_period = data.index[test_index]
    test_data = data.loc[test_period]

    # Create DataFrames for each signal type with proper format for analyze_trading_strategies
    signal_dfs = {
        'Binary Ensemble': pd.DataFrame({'prediction': binary_signals}),
        'Combined Ensemble': pd.DataFrame({'prediction': combined_signals}),
        'RSI': pd.DataFrame({'prediction': rsi_signals.loc[test_period]}),
        'MA': pd.DataFrame({'prediction': ma_signals.loc[test_period]})
    }

    # Analysis parameters
    initial_investment = 10000
    leverage = 1

    print("\nAnalyzing trading performance:")

    results = {}
    for signal_name, signal_df in signal_dfs.items():
        print(f"\n{signal_name} Strategy Analysis:")
        try:
            results[signal_name] = analyze_trading_strategies(
                test_data,
                signal_df,
                initial_investment=initial_investment,
                leverage=leverage,
                signal_type=signal_name
            )
        except Exception as e:
            print(f"Error analyzing {signal_name} strategy: {str(e)}")
            continue

    # Compare strategy performances
    print("\nStrategy Performance Comparison:")
    print("=" * 50)

    comparison = pd.DataFrame(
        columns=['Total Return', 'Max Drawdown', 'Sharpe Ratio'])

    for strategy_name, result in results.items():
        metrics = result['metrics']
        comparison.loc[f"{strategy_name} (Day Trading)"] = [
            f"{metrics['Day_Trading']['Total_Return']*100:.2f}%",
            f"{metrics['Day_Trading']['Max_Drawdown']*100:.2f}%",
            f"{metrics['Day_Trading']['Sharpe_Ratio']:.2f}"
        ]
        comparison.loc[f"{strategy_name} (Position Trading)"] = [
            f"{metrics['Position_Trading']['Total_Return']*100:.2f}%",
            f"{metrics['Position_Trading']['Max_Drawdown']*100:.2f}%",
            f"{metrics['Position_Trading']['Sharpe_Ratio']:.2f}"
        ]

    print(comparison)

    # Plot combined signals
    fig = make_subplots(rows=5, cols=1,
                        subplot_titles=('Price', 'Binary Ensemble Signals',
                                        'Combined Ensemble Signals', 'RSI Signals', 'MA Signals'),
                        vertical_spacing=0.05,
                        row_heights=[0.3, 0.175, 0.175, 0.175, 0.175])

    # Add price plot
    fig.add_trace(
        go.Scatter(x=test_data.index, y=test_data['Close'], name='Price'),
        row=1, col=1
    )

    # Add signal plots
    signal_data = {
        'Binary Ensemble': binary_signals,
        'Combined Ensemble': combined_signals,
        'RSI': rsi_signals.loc[test_period],
        'MA': ma_signals.loc[test_period]
    }

    for i, (signal_name, signals) in enumerate(signal_data.items(), start=2):
        # Convert signals to numeric for plotting
        numeric_signals = signals.map(
            {'positive': 1, 'neutral': 0, 'negative': -1})

        fig.add_trace(
            go.Scatter(
                x=test_data.index,
                y=numeric_signals,
                name=f'{signal_name} Signals',
                line=dict(color=['blue', 'purple', 'green', 'red'][i-2])
            ),
            row=i, col=1
        )

    fig.update_layout(
        height=1500,  # Increased height for 5 subplots
        title_text="Signal Comparison",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    # Update axes labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    for i in range(2, 6):
        fig.update_yaxes(title_text="Signal", row=i, col=1)
    fig.update_xaxes(title_text="Date", row=5, col=1)

    fig.show()

    return {
        'results': results,
        'binary_ensemble': binary_ensemble,
        'combined_ensemble': combined_ensemble,
        'binary_signals': binary_signals,
        'combined_signals': combined_signals
    }


if __name__ == "__main__":
    while True:
        # Get mode
        print("\nAvailable modes:")
        print("1. Single stock train (train and analyze one stock)")
        print("2. SP500 all tickers trained, single stock analysis")
        print("3. Quit")
        print("4. Test strategy comparison (ensemble vs RSI vs MA)")
        print("5. Compare ML classifiers (temporal holdout)")
        print("6. Compare pipeline variants (dual-RF configs)")

        mode = input("\nEnter mode (1/2/3/4/5/6) [1]: ") or "1"

        if mode == "3":
            print("\nExiting program...")
            break

        if mode == "2":
            analysis_ticker = input("\nEnter ticker to analyze [^GSPC]: ") or "^GSPC"
            data = get_index_data(analysis_ticker, '20y')
            try:
                sector = get_ticker(analysis_ticker).info['sector']
            except Exception:
                sector = None
            universe_tickers = MarketIndices.get_sp500_tickers()
            if sector is not None:
                training_filter = input(
                    f"Enter training filter ({sector})? [y/n] ") or "y"
                if training_filter == "y":
                    sector_tickers = []
                    for symbol in universe_tickers:
                        try:
                            info = get_ticker(symbol).info
                            if 'sector' in info and info['sector'] == sector:
                                sector_tickers.append(symbol)
                        except Exception:
                            continue
                    universe_tickers = sector_tickers
                    print(
                        f"\nFiltered to {len(universe_tickers)} {sector} tickers")

            threshold = float(input("Enter threshold [0.01]: ") or 0.01)
            use_extra_features = input("Use extra features? [y/n]: ") or "n"
            use_extra_features = use_extra_features.lower() == "y"
            holdout_months = int(input("Holdout months for training and eval [12]: ") or 12)
            overfit_check = False
            plot = False

            analysis_symbol = analysis_ticker.lstrip('^')
            training_tickers = [t for t in universe_tickers if t != analysis_symbol]
            if len(training_tickers) < len(universe_tickers):
                print(f"Excluding {analysis_symbol} from SP500 training set.")
            if not training_tickers:
                print("No training tickers left after exclusion — using full universe.")
                training_tickers = universe_tickers

            prepared_daily = prepare_classification_data(
                data, predict_weekly=False, threshold=threshold, use_extra_features=use_extra_features)
            prepared_weekly = prepare_classification_data(
                data, predict_weekly=True, threshold=threshold, use_extra_features=use_extra_features)
            cutoff_date = prepared_daily.index.max() - pd.DateOffset(months=holdout_months)

            print(f"\nAnalyzing {analysis_ticker} with SP500-trained model...")
            print(
                f"Temporal holdout: train all tickers before {cutoff_date.date()}, "
                f"evaluate {analysis_ticker} from {cutoff_date.date()} onward."
            )
            print(
                "Other SP500 names are truncated at the same cutoff to limit "
                "correlated recent-market leakage."
            )

            try:
                daily_model, daily_scaler, _ = train_classifier_tickers(
                    training_tickers,
                    predict_weekly=False,
                    use_extra_features=use_extra_features,
                    threshold=threshold,
                    plot=plot,
                    overfit_check=overfit_check,
                    classifier=RandomForestClassifier(n_estimators=100, random_state=42),
                    cutoff_date=cutoff_date,
                    eval_split=0,
                )

                weekly_model, weekly_scaler, _ = train_classifier_tickers(
                    training_tickers,
                    predict_weekly=True,
                    use_extra_features=use_extra_features,
                    threshold=threshold,
                    plot=plot,
                    overfit_check=overfit_check,
                    classifier=RandomForestClassifier(n_estimators=100, random_state=42),
                    cutoff_date=cutoff_date,
                    eval_split=0,
                )

                eval_daily, eval_weekly, eval_stock, backtest_label = resolve_backtest_period(
                    prepared_daily,
                    prepared_weekly,
                    data,
                    holdout_months=holdout_months,
                    cutoff_date=cutoff_date,
                )

                recent_predictions = get_recent_predictions(
                    data, prepared_daily, daily_model, daily_scaler, prepared_weekly, weekly_model, weekly_scaler, use_extra_features=use_extra_features
                )
                print("\nLast 5 Trading Days Predictions:")
                print(recent_predictions.to_string())

                full_analysis(
                    data, prepared_daily, daily_model, daily_scaler,
                    prepared_weekly, weekly_model, weekly_scaler,
                    threshold=threshold, use_extra_features=use_extra_features,
                )

                analyze_prediction_probabilities(
                    data,
                    daily_model,
                    daily_scaler,
                    prepared_daily,
                    weekly_model,
                    weekly_scaler,
                    prepared_weekly,
                    use_extra_features=use_extra_features
                )
                test_model_performance(
                    eval_daily, daily_model, daily_scaler, eval_weekly, weekly_model, weekly_scaler,
                    threshold=threshold, use_extra_features=use_extra_features)

                run_movement_backtests(
                    data,
                    daily_model,
                    daily_scaler,
                    prepared_daily,
                    weekly_model,
                    weekly_scaler,
                    prepared_weekly,
                    threshold,
                    use_extra_features,
                    eval_daily,
                    eval_weekly,
                    eval_stock,
                    backtest_label,
                )

            except FileNotFoundError:
                print("\nError: Pre-trained SP500 models not found.")
                print(
                    "Please run the SP500 training script first to generate the models.")
                continue
            except Exception as e:
                print(f"\nError analyzing ticker: {str(e)}")
                continue

        elif mode == "1":  # mode == "1"
            # Get ticker for single stock analysis
            ticker = input("\nEnter ticker [^GSPC]: ") or "^GSPC"
            date_input = input(
                "Specify alternate start date on format dd.mm.yyyy: ")
            alternate_start_date = None
            if date_input:
                try:
                    alternate_start_date = pd.to_datetime(
                        date_input, format='%d.%m.%Y')
                except ValueError:
                    print("Invalid date format. Using default date.")
                start_date = pd.Timestamp.now() if alternate_start_date is None else alternate_start_date
                period = input("Enter period [20y]: ") or "20y"
                data = get_index_data(ticker, period, start_date)
            else:
                start_date = pd.Timestamp.now()
                period = input("Enter period [20y]: ") or "20y"
                data = get_index_data(ticker, period)

            # Use pandas Timestamp consistently

            print(f"\nAnalyzing {ticker} data:")

            threshold = float(input("Enter threshold [0.005]: ") or 0.005)
            use_extra_features = True if (
                input("Use extra features? [y/n]: ") or "n").lower() == "y" else False
            overfit_check = False
            plot = True

            test_last_n_months = input(
                "Enter number of months to test [0]: ") or 0

            # Use pandas Timestamp for cutoff_date
            cutoff_date = start_date

            daily_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42)
            weekly_classifier = RandomForestClassifier(
                n_estimators=100, random_state=42)

            if test_last_n_months != 0:
                # Convert test_last_n_months to integer
                test_last_n_months = int(test_last_n_months)

                # Calculate cutoff date and print it for debugging
                cutoff_date = start_date - \
                    pd.DateOffset(months=test_last_n_months)
                print(f"\nCutoff date: {cutoff_date}")
                print(
                    f"Data date range: {data.index.min()} to {data.index.max()}")

                if cutoff_date < data.index.min():
                    print("Warning: Cutoff date is before the start of the data")
                    # Set to middle index
                    cutoff_date = data.index[len(data.index)//2]
                    print(f"Cutoff date set to: {cutoff_date}")

                prepared_daily_data = prepare_classification_data(
                    data, predict_weekly=False, threshold=threshold, use_extra_features=use_extra_features)
                prepared_weekly_data = prepare_classification_data(
                    data, predict_weekly=True, threshold=threshold, use_extra_features=use_extra_features)

            else:
                prepared_daily_data = prepare_classification_data(
                    data, predict_weekly=False, threshold=threshold, use_extra_features=use_extra_features)
                prepared_weekly_data = prepare_classification_data(
                    data, predict_weekly=True, threshold=threshold, use_extra_features=use_extra_features)

            print(prepared_daily_data.tail())

            test_last_n_months = int(test_last_n_months) if test_last_n_months else 0
            use_holdout = test_last_n_months > 0
            eval_split = 0 if use_holdout else 0.2
            train_daily = (
                prepared_daily_data[prepared_daily_data.index < cutoff_date]
                if use_holdout else prepared_daily_data
            )
            train_weekly = (
                prepared_weekly_data[prepared_weekly_data.index < cutoff_date]
                if use_holdout else prepared_weekly_data
            )

            if not use_holdout:
                print(
                    "\nNo holdout months set — training uses an internal 80/20 "
                    "chronological split; backtests run on the held-out 20% only."
                )

            print("\nTraining daily return classifier:")
            daily_model, daily_scaler, daily_train_meta = train_classifier_single_stock(
                train_daily,
                predict_weekly=False,
                use_extra_features=use_extra_features,
                threshold=threshold,
                plot=plot,
                overfit_check=overfit_check,
                classifier=daily_classifier,
                eval_split=eval_split,
            )
            print("\nTraining weekly return classifier:")
            weekly_model, weekly_scaler, weekly_train_meta = train_classifier_single_stock(
                train_weekly,
                predict_weekly=True,
                use_extra_features=use_extra_features,
                threshold=threshold,
                plot=plot,
                overfit_check=overfit_check,
                classifier=weekly_classifier,
                eval_split=eval_split,
            )

            eval_daily, eval_weekly, eval_stock, backtest_label = resolve_backtest_period(
                prepared_daily_data,
                prepared_weekly_data,
                data,
                holdout_months=test_last_n_months if use_holdout else 0,
                cutoff_date=cutoff_date if use_holdout else None,
                train_meta=daily_train_meta,
            )

            recent_predictions = get_recent_predictions(
                data, prepared_daily_data, daily_model, daily_scaler,
                prepared_weekly_data, weekly_model, weekly_scaler,
                threshold=threshold, use_extra_features=use_extra_features,
            )
            print("\nLast 5 Trading Days Predictions:")
            print(recent_predictions.to_string())

            full_analysis(
                data, prepared_daily_data, daily_model, daily_scaler,
                prepared_weekly_data, weekly_model, weekly_scaler,
                threshold=threshold, use_extra_features=use_extra_features,
            )

            analyze_prediction_probabilities(
                data,
                daily_model,
                daily_scaler,
                prepared_daily_data,
                weekly_model,
                weekly_scaler,
                prepared_weekly_data,
                use_extra_features=use_extra_features,
            )

            print("\nClassification metrics on out-of-sample data:")
            test_model_performance(
                eval_daily,
                daily_model,
                daily_scaler,
                eval_weekly,
                weekly_model,
                weekly_scaler,
                threshold=float(threshold),
                use_extra_features=use_extra_features,
            )

            run_movement_backtests(
                data,
                daily_model,
                daily_scaler,
                prepared_daily_data,
                weekly_model,
                weekly_scaler,
                prepared_weekly_data,
                threshold,
                use_extra_features,
                eval_daily,
                eval_weekly,
                eval_stock,
                backtest_label,
            )

        elif mode == "4":
            test_classifiers()

        elif mode == "5":
            ticker = input("\nEnter ticker [^GSPC]: ") or "^GSPC"
            period = input("Enter period [20y]: ") or "20y"
            holdout_months = int(input("Holdout months [12]: ") or 12)
            threshold = float(input("Enter threshold [0.005]: ") or 0.005)
            use_extra_features = (
                input("Use extra features? [y/n]: ") or "y"
            ).lower() == "y"

            data = get_index_data(ticker, period)
            prepared = prepare_classification_data_enhanced(
                data,
                predict_weekly=False,
                threshold=threshold,
                use_extra_features=use_extra_features,
            ).dropna()

            evaluate_all_classifiers(
                prepared,
                threshold=threshold,
                use_extra_features=use_extra_features,
                holdout_months=holdout_months,
            )

        elif mode == "6":
            ticker = input("\nEnter ticker [^GSPC]: ") or "^GSPC"
            period = input("Enter period [20y]: ") or "20y"
            holdout_months = int(input("Holdout months [12]: ") or 12)
            compare_variants(
                ticker=ticker,
                period=period,
                holdout_months=holdout_months,
            )

        else:
            print("Invalid mode. Please enter 1, 2, 3, 4, 5, or 6.")
