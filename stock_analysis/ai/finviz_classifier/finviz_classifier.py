import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from stock_analysis.ai.finviz_classifier.data_fetching import fetch_stock_returns, load_historical_signals
import plotly.graph_objects as go
import gc


def prepare_features(signals_df, returns_df, threshold=0.05, use_technical_features=False, use_market_features=False):
    """Prepare features for machine learning"""
    # Normalize timestamps
    signals_df = signals_df.copy()
    returns_df = returns_df.copy()

    # Convert timestamps to datetime and remove timezone info
    signals_df['timestamp'] = pd.to_datetime(
        signals_df['timestamp']).dt.tz_localize(None).dt.normalize()
    returns_df['timestamp'] = pd.to_datetime(
        returns_df['timestamp']).dt.normalize()

    # Merge signals with returns
    df = pd.merge(
        signals_df,
        returns_df,
        on=['Ticker', 'timestamp'],
        how='inner'
    )

    return_columns = ['next_day_return', 'next_week_return',
                      'next_day_high_return', 'next_week_high_return']
    for col in return_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Create binary target variables - only mark as None if ALL return values are missing
    # Create binary target variables
    df['next_day_positive'] = df['next_day_return'].apply(
        lambda x: 1 if pd.notna(x) and x > threshold else 0 if pd.notna(x) else None)
    df['next_week_positive'] = df['next_week_return'].apply(
        lambda x: 1 if pd.notna(x) and x > threshold else 0 if pd.notna(x) else None)
    df['next_day_high_positive'] = df['next_day_high_return'].apply(
        lambda x: 1 if pd.notna(x) and x > threshold else 0 if pd.notna(x) else None)
    df['next_week_high_positive'] = df['next_week_high_return'].apply(
        lambda x: 1 if pd.notna(x) and x > threshold else 0 if pd.notna(x) else None)

    # Select features for training
    feature_columns = ['Price', 'Change', 'Volume', 'Market Cap']

    # Basic feature
    df['volume_price_ratio'] = df['Volume'] * df['Price']
    feature_columns.append('volume_price_ratio')

    # Technical features
    if use_technical_features:
        # Previous features
        df['price_to_volume'] = df['Price'] / df['Volume']
        df['market_cap_to_volume'] = df['Market Cap'] / df['Volume']
        df['log_price'] = np.log1p(df['Price'])
        df['log_volume'] = np.log1p(df['Volume'])
        df['log_market_cap'] = np.log1p(df['Market Cap'])

        # Add previous day return (already in Change column as percentage)
        # This is already in decimal form from earlier cleaning
        df['prev_day_return'] = df['Change']

        # Calculate 14-day RSI using the Change column
        # First convert Change to price movement series
        df['price_change'] = df['Price'] * df['Change']

        # Calculate gains and losses
        df['gains'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
        df['losses'] = df['price_change'].apply(
            lambda x: abs(x) if x < 0 else 0)

        # Calculate average gains and losses over 14 periods
        # Note: This is a simplified RSI calculation as we don't have full price history
        avg_gain = df['gains'].rolling(window=14, min_periods=1).mean()
        avg_loss = df['losses'].rolling(window=14, min_periods=1).mean()

        # Calculate RSI
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Clean up intermediate columns
        df = df.drop(['price_change', 'gains', 'losses'], axis=1)

        feature_columns.extend([
            'price_to_volume',
            'market_cap_to_volume',
            'log_price',
            'log_volume',
            'log_market_cap',
            'prev_day_return',
            'rsi'
        ])

    # Market context features
    if use_market_features:
        # Relative metrics
        df['relative_volume'] = df['Volume'] / df['Volume'].mean()
        df['relative_market_cap'] = df['Market Cap'] / df['Market Cap'].mean()
        df['price_percentile'] = df['Price'].rank(pct=True)

        # Interaction terms
        df['price_change_interaction'] = df['Price'] * df['Change']
        df['volume_change_interaction'] = df['Volume'] * df['Change']

        feature_columns.extend([
            'relative_volume',
            'relative_market_cap',
            'price_percentile',
            'price_change_interaction',
            'volume_change_interaction'
        ])

    return df, feature_columns


def train_classifier(df, feature_columns, target='next_day_positive'):
    """Train a classifier on the prepared data"""
    if len(df) == 0:
        print("No data available for training")
        return None, None, None

    # Get the latest date in the dataset
    latest_date = df['timestamp'].max()

    # Split into training (all dates except latest) and prediction (only latest date)
    # Also ensure we only include rows where we have return data for training
    train_mask = (df['timestamp'] < latest_date) & (df[target].notna())
    predict_mask = df['timestamp'] == latest_date

    df_train = df[train_mask].copy()
    df_predict = df[predict_mask].copy()

    if len(df) > 0:
        print(f" ({len(df_train)/len(df)*100:.1f}%)")
    else:
        print("")
    print(f"Prediction samples: {len(df_predict)}", end='')
    if len(df) > 0:
        print(f" ({len(df_predict)/len(df)*100:.1f}%)")
    else:
        print("")

    # Check if we have enough training data
    if len(df_train) == 0:
        print("No training data available")
        return None, None, df_predict

    # Prepare features and target for training
    X_train = df_train[feature_columns]
    # Now safe to convert to int as we've filtered NaN values
    y_train = df_train[target].astype(int)

    # Prepare features for prediction
    X_predict = df_predict[feature_columns] if len(df_predict) > 0 else None

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(
        X_predict) if X_predict is not None else None

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Training accuracy
    train_score = clf.score(X_train_scaled, y_train)
    print(f"Training accuracy: {train_score:.3f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nFeature Importance:")
    print(feature_importance)

    # Make predictions on new data if available
    if X_predict is not None and len(X_predict) > 0:
        predictions = clf.predict_proba(X_predict_scaled)
        # Probability of positive class
        df_predict['predicted_prob'] = predictions[:, 1]

    return clf, scaler, df_predict if X_predict is not None else None


def train_multiclass_classifier(df, feature_columns, return_type='next_day_return'):
    """Train a multiclass classifier on the prepared data"""
    # Define return classes with wider thresholds
    def get_return_class(return_val):
        if return_val is None:
            return None
        elif return_val <= -0.05:  # <-10%
            return 0  # Strong negative
        elif return_val < 0.05:  # -5% to -10%
            return 1  # Neutral
        elif return_val < 0.3:  # -5% to 5%
            return 2  # "Mild" positive
        elif return_val < 0.6:  # 5% to 10%
            return 3  # Strong positive
        else:  # >10%
            return 4  # Exceptional

    # Create multiclass target
    df['return_class'] = df[return_type].apply(get_return_class)

    # Split into training and prediction sets
    latest_date = df['timestamp'].max()
    train_mask = df['timestamp'] < latest_date
    df_train = df[train_mask].copy()
    df_predict = df[~train_mask].copy()
    # Class distribution
    if len(df_train) > 0:
        class_dist = df_train['return_class'].value_counts().sort_index()
        print("\nClass distribution:")
        for class_label, count in class_dist.items():
            print(
                f"Class {class_label}: {count} ({count/len(df_train)*100:.1f}%)")

    # Check if we have enough training data
    if len(df_train) == 0:
        print("No training data available")
        return None, None, df_predict

    # Prepare features and target
    X_train = df_train[feature_columns]
    y_train = df_train['return_class'].astype(int)
    X_predict = df_predict[feature_columns] if len(df_predict) > 0 else None

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(
        X_predict) if X_predict is not None else None

    # Train classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Training metrics
    train_score = clf.score(X_train_scaled, y_train)
    print(f"\nTraining accuracy: {train_score:.3f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': clf.feature_importances_
    }).sort_values('importance', ascending=False)

    # Make predictions on new data
    if X_predict is not None and len(X_predict) > 0:
        predictions = clf.predict_proba(X_predict_scaled)
        df_predict['predicted_class'] = clf.predict(X_predict_scaled)
        # Store probabilities for each class
        for i in range(len(clf.classes_)):
            df_predict[f'prob_class_{i}'] = predictions[:, i]

    # Print accuracy metrics
    # print("\nClassification Report:")
    # if X_predict is not None and len(X_predict) > 0:
        # print(f"Number of predictions: {len(df_predict)}")
        # print("\nPredicted class distribution:")
        # pred_dist = df_predict['predicted_class'].value_counts().sort_index()
        # for class_label, count in pred_dist.items():
            # print(f"Class {class_label}: {count} ({count/len(df_predict)*100:.1f}%)")

    return clf, scaler, df_predict


def train_regressor(df, feature_columns, return_type='next_day_return'):
    """Train a regression model to predict actual returns"""
    if len(df) == 0:
        print("No data available for training")
        return None, None, None

    # Get the latest date in the dataset
    latest_date = df['timestamp'].max()

    # Split into training (all dates except latest) and prediction (only latest date)
    # Also ensure we only include rows where we have return data for training
    train_mask = (df['timestamp'] < latest_date) & (df[return_type].notna())
    predict_mask = df['timestamp'] == latest_date

    df_train = df[train_mask].copy()
    df_predict = df[predict_mask].copy()

    if len(df) > 0:
        print(f" ({len(df_train)/len(df)*100:.1f}%)")
    else:
        print("")
    print(f"Prediction samples: {len(df_predict)}", end='')
    if len(df) > 0:
        print(f" ({len(df_predict)/len(df)*100:.1f}%)")
    else:
        print("")

    # Return distribution statistics
    if len(df_train) > 0:
        returns = df_train[return_type]
        print("\nReturn distribution:")
        print(f"Mean: {returns.mean():.3f}")
        print(f"Median: {returns.median():.3f}")
        print(f"Std: {returns.std():.3f}")
        print(f"Min: {returns.min():.3f}")
        print(f"Max: {returns.max():.3f}")

        # Print percentiles
        percentiles = [1, 5, 10, 25, 75, 90, 95, 99]
        for p in percentiles:
            print(f"{p}th percentile: {returns.quantile(p/100):.3f}")

    # Check if we have enough training data
    if len(df_train) == 0:
        print("No training data available")
        return None, None, df_predict

    # Prepare features and target
    X_train = df_train[feature_columns]
    y_train = df_train[return_type]  # Now guaranteed to have no NaN values
    X_predict = df_predict[feature_columns] if len(df_predict) > 0 else None

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_predict_scaled = scaler.transform(
        X_predict) if X_predict is not None else None

    # Train regressor
    reg = RandomForestRegressor(n_estimators=100, random_state=42)
    reg.fit(X_train_scaled, y_train)

    # Training metrics
    train_pred = reg.predict(X_train_scaled)
    mse = ((train_pred - y_train) ** 2).mean()
    r2 = reg.score(X_train_scaled, y_train)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': reg.feature_importances_
    }).sort_values('importance', ascending=False)

    # Make predictions on new data
    if X_predict is not None and len(X_predict) > 0:
        predictions = reg.predict(X_predict_scaled)
        df_predict['predicted_return'] = np.round(
            predictions, 3)  # Round to 3 decimals

    return reg, scaler, df_predict


def main():
    # Load historical signals
    signals_df = load_historical_signals()
    if signals_df is None:
        return

    print(f"\nInitial data sizes:")
    print(
        f"Signals DataFrame: {signals_df.shape}, Memory usage: {signals_df.memory_usage().sum() / 1024**2:.2f} MB")

    # Fetch returns using cached approach
    returns_df, detailed_returns_df = fetch_stock_returns(signals_df)
    print(
        f"Returns DataFrame: {returns_df.shape}, Memory usage: {returns_df.memory_usage().sum() / 1024**2:.2f} MB")

    # Check if returns_df is empty
    if returns_df.empty:
        print("No return data available. Please check the stock data fetching process.")
        return

    # Initialize empty DataFrame for predictions with only essential columns
    all_predictions = None

    # Track unique tickers and timestamps to avoid duplicates
    processed_pairs = set()

    # Test different feature combinations
    feature_combinations = [
        (False, False, "Base features only"),
        (True, False, "With technical features"),
        (False, True, "With market features"),
        (True, True, "With all features")
    ]

    # Train different classifiers
    classifiers = {
        'dualclass': ('next_day_positive', train_classifier),
        'dualclass_high': ('next_day_high_positive', train_classifier),
        'multiclass_day': ('next_day_return', train_multiclass_classifier),
        'multiclass_day_high': ('next_week_return', train_multiclass_classifier),
        'regression_day': ('next_day_return', train_regressor),
        'regression_day_high': ('next_week_return', train_regressor)
    }

    for tech_features, market_features, desc in feature_combinations:
        print(f"\nProcessing feature combination: {desc}")
        gc.collect()  # Force garbage collection before each iteration

        df, feature_columns = prepare_features(
            signals_df.copy(),
            returns_df.copy(),
            use_technical_features=tech_features,
            use_market_features=market_features
        )

        # Store predictions for this feature combination
        feature_predictions = None

        for model_name, (target, train_func) in classifiers.items():
            print(f"\nTraining {model_name}")
            gc.collect()  # Force garbage collection before training

            model, scaler, predictions = train_func(
                df.copy(), feature_columns, target)

            if predictions is not None and len(predictions) > 0:
                # Select only essential columns
                pred_cols = [col for col in predictions.columns if 'pred' in col.lower(
                ) or 'prob' in col.lower()]
                pred_subset = predictions[[
                    'Ticker', 'timestamp'] + pred_cols].copy()

                # Rename prediction columns to include feature combination
                for col in pred_cols:
                    pred_subset.rename(
                        columns={col: f'{model_name}_{desc}_{col}'}, inplace=True)

                # Update feature predictions
                if feature_predictions is None:
                    feature_predictions = pred_subset
                else:
                    feature_predictions = pd.merge(
                        feature_predictions,
                        pred_subset,
                        on=['Ticker', 'timestamp'],
                        how='outer'
                    )

                # Clear memory
                del predictions, pred_subset
                gc.collect()

        # After all models for this feature combination, merge with main predictions
        if all_predictions is None:
            all_predictions = feature_predictions
        elif feature_predictions is not None:
            # Only merge new unique ticker-timestamp pairs
            new_pairs = set(
                zip(feature_predictions['Ticker'], feature_predictions['timestamp']))
            existing_pairs = set(
                zip(all_predictions['Ticker'], all_predictions['timestamp']))

            if new_pairs - existing_pairs:  # If there are new pairs to add
                all_predictions = pd.merge(
                    all_predictions,
                    feature_predictions,
                    on=['Ticker', 'timestamp'],
                    how='outer'
                )

        # Clear memory
        del feature_predictions
        gc.collect()

    # Final processing of predictions
    if all_predictions is not None and len(all_predictions) > 0:
        # Separate different types of predictions
        binary_cols_day = [
            col for col in all_predictions.columns if 'dualclass_' in col.lower() and not 'high' in col.lower()]
        binary_cols_high = [
            col for col in all_predictions.columns if 'dualclass_' in col.lower() and 'high' in col.lower()]

        multiclass_cols_day = [
            col for col in all_predictions.columns if 'multiclass_day' in col.lower() and not 'high' in col.lower()]
        multiclass_cols_high = [
            col for col in all_predictions.columns if 'multiclass_day' in col.lower() and 'high' in col.lower()]

        regression_cols_day = [
            col for col in all_predictions.columns if 'regression_day_' in col.lower() and not 'high' in col.lower()]
        regression_cols_high = [
            col for col in all_predictions.columns if 'regression_day_' in col.lower() and 'high' in col.lower()]

        # Calculate separate averages for day and high predictions
        if binary_cols_day:
            all_predictions['avg_binary_pred_day'] = all_predictions[binary_cols_day].mean(
                axis=1)
        if binary_cols_high:
            all_predictions['avg_binary_pred_high'] = all_predictions[binary_cols_high].mean(
                axis=1)

        if multiclass_cols_day:
            all_predictions['avg_multiclass_pred_day'] = all_predictions[multiclass_cols_day].mean(
                axis=1)
        if multiclass_cols_high:
            all_predictions['avg_multiclass_pred_high'] = all_predictions[multiclass_cols_high].mean(
                axis=1)

        if regression_cols_day:
            # Calculate the average regression prediction first
            all_predictions['avg_regression_pred_day'] = all_predictions[regression_cols_day].mean(
                axis=1)

            # Sort by average regression prediction
            regression_display = all_predictions.sort_values(
                'avg_regression_pred_day', ascending=False)

            # Create a more detailed table showing all regression predictions
            display_columns = ['Ticker', 'timestamp']

            # Group regression columns by feature combination
            base_reg = [
                col for col in regression_cols_day if 'Base features only' in col]
            tech_reg = [
                col for col in regression_cols_day if 'With technical features' in col]
            market_reg = [
                col for col in regression_cols_day if 'With market features' in col]
            all_reg = [
                col for col in regression_cols_day if 'With all features' in col]

            # Calculate averages for each feature combination
            if base_reg:
                regression_display['avg_base'] = regression_display[base_reg].mean(
                    axis=1).round(3)
                display_columns.append('avg_base')
                display_columns.extend(base_reg)

            if tech_reg:
                regression_display['avg_tech'] = regression_display[tech_reg].mean(
                    axis=1).round(3)
                display_columns.append('avg_tech')
                display_columns.extend(tech_reg)

            if market_reg:
                regression_display['avg_market'] = regression_display[market_reg].mean(
                    axis=1).round(3)
                display_columns.append('avg_market')
                display_columns.extend(market_reg)

            if all_reg:
                regression_display['avg_all_features'] = regression_display[all_reg].mean(
                    axis=1).round(3)
                display_columns.append('avg_all_features')
                display_columns.extend(all_reg)

            display_columns.append('avg_regression_pred_day')

            regression_fig = go.Figure(data=[go.Table(
                header=dict(
                    values=display_columns,
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[regression_display[col]
                            for col in display_columns],
                    fill_color='lavender',
                    align='left'
                )
            )])

            regression_fig.update_layout(
                title='Regression Predictions by Feature Set (Expected Return %) - Day',
                width=2000,  # Increased width to accommodate more columns
                height=800
            )
            regression_fig.show(renderer="browser")

            # Print top 10 regression predictions with all averages
            print("\nTop 10 Regression Predictions - Day:")
            print(regression_display[display_columns].head(10))

        if regression_cols_high:
            # Similar structure for high predictions
            regression_display = all_predictions.sort_values(
                'avg_regression_pred_high', ascending=False)

            display_columns = ['Ticker', 'timestamp']

            base_reg = [
                col for col in regression_cols_high if 'Base features only' in col]
            tech_reg = [
                col for col in regression_cols_high if 'With technical features' in col]
            market_reg = [
                col for col in regression_cols_high if 'With market features' in col]
            all_reg = [
                col for col in regression_cols_high if 'With all features' in col]

            if base_reg:
                regression_display['avg_base'] = regression_display[base_reg].mean(
                    axis=1).round(3)
                display_columns.append('avg_base')
                display_columns.extend(base_reg)

            if tech_reg:
                regression_display['avg_tech'] = regression_display[tech_reg].mean(
                    axis=1).round(3)
                display_columns.append('avg_tech')
                display_columns.extend(tech_reg)

            if market_reg:
                regression_display['avg_market'] = regression_display[market_reg].mean(
                    axis=1).round(3)
                display_columns.append('avg_market')
                display_columns.extend(market_reg)

            if all_reg:
                regression_display['avg_all_features'] = regression_display[all_reg].mean(
                    axis=1).round(3)
                display_columns.append('avg_all_features')
                display_columns.extend(all_reg)

            display_columns.append('avg_regression_pred_high')

            regression_fig = go.Figure(data=[go.Table(
                header=dict(
                    values=display_columns,
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[regression_display[col]
                            for col in display_columns],
                    fill_color='lavender',
                    align='left'
                )
            )])

            regression_fig.update_layout(
                title='Regression Predictions by Feature Set (Expected Return %) - High',
                width=2000,
                height=800
            )
            regression_fig.show(renderer="browser")

            # Print top 10 regression predictions with all averages
            print("\nTop 10 Regression Predictions - High:")
            print(regression_display[display_columns].head(10))

    return df, all_predictions


if __name__ == "__main__":
    main()
