import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_error
import os
from ..helpers import RSI, get_ticker
import keras_tuner as kt


def create_nn_model(input_shape):
    """Create a neural network model for stock prediction."""
    model = keras.Sequential([
        keras.layers.Dense(16, activation='relu', input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(8, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(4, activation='relu'),
        keras.layers.Dense(1)  # Output layer for regression
    ])

    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )

    return model


def create_tunable_model(hp):
    """Create a tunable neural network model with hyperparameters."""
    model = keras.Sequential()

    # Tune number of layers and units
    for i in range(hp.Int('num_layers', 1, 4)):
        model.add(keras.layers.Dense(
            units=hp.Int(f'units_{i}', min_value=4, max_value=64, step=4),
            activation=hp.Choice(f'activation_{i}', ['relu', 'tanh']),
            kernel_regularizer=keras.regularizers.l2(hp.Float(
                'l2_reg', 1e-6, 1e-2, sampling='log'
            ))
        ))

        # Tune dropout rate
        model.add(keras.layers.Dropout(
            hp.Float(f'dropout_{i}', 0, 0.5, step=0.1)
        ))

    model.add(keras.layers.Dense(1))  # Output layer

    # Tune learning rate
    learning_rate = hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    return model


def prepare_data(data, predict_weekly=False):
    """Prepare data for neural network training."""
    X = pd.DataFrame(index=data.index)

    # Target variable
    if predict_weekly:
        X['target'] = data['Close'].pct_change(
            periods=5).shift(-4)  # Next week's return
    else:
        X['target'] = data['Close'].pct_change().shift(-1)  # Next day's return

    # Technical indicators as features
    X['return_1d'] = data['Close'].pct_change()
    X['return_2d'] = data['Close'].pct_change(2)
    X['return_5d'] = data['Close'].pct_change(5)
    # X['return_21d'] = data['Close'].pct_change(21)

    # Volume changes
    X['volume_1d'] = data['Volume'].pct_change()
    X['volume_5d'] = data['Volume'].pct_change(5)

    # Volatility
    X['volatility_5d'] = data['Close'].pct_change().rolling(5).std()
    X['volatility_21d'] = data['Close'].pct_change().rolling(21).std()

    # Moving averages
    X['ma_5d'] = data['Close'].rolling(5).mean() / data['Close'] - 1
    X['ma_21d'] = data['Close'].rolling(21).mean() / data['Close'] - 1

    X['rsi_5d'] = RSI(data['Close'], window=5)
    X['rsi_21d'] = RSI(data['Close'], window=21)

    # Clean data: remove infinities and extreme values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()

    # Remove extreme outliers (beyond 5 standard deviations)
    for col in X.columns:
        mean = X[col].mean()
        std = X[col].std()
        X = X[X[col].between(mean - 5*std, mean + 5*std)]

    return X


def train_nn_predictor(data, predict_weekly=False, test_size=0.2, do_tuning=True):
    """Train neural network model for stock prediction with optional hyperparameter tuning."""
    # Prepare data
    prepared_data = prepare_data(data, predict_weekly)

    # Split features and target
    y = prepared_data['target']
    X = prepared_data.drop('target', axis=1)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print((X_train.shape[1],))

    if do_tuning:
        print("Starting hyperparameter tuning...")

        # Create tuner
        tuner = kt.Hyperband(
            create_tunable_model,
            objective='val_loss',
            max_epochs=100,
            factor=3,
            directory='keras_tuner',
            project_name='stock_predictor'
        )

        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        # Search for best hyperparameters
        tuner.search(
            X_train_scaled, y_train,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        # Get best model
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print("\nBest Hyperparameters:", best_hps.values)
        model = tuner.hypermodel.build(best_hps)

    else:
        model = create_nn_model(input_shape=(X_train.shape[1],))

    # Early stopping to prevent overfitting
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Train model
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # Make predictions
    train_pred = model.predict(X_train_scaled).flatten()
    test_pred = model.predict(X_test_scaled).flatten()

    print(test_pred)
    print(tf.nn.softmax(test_pred))

    # Create visualization
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Training Loss', 'Prediction vs Actual (Train)',
                                        'Feature Importance', 'Prediction vs Actual (Test)'))

    # Plot training history
    fig.add_trace(
        go.Scatter(y=history.history['loss'], name='Train Loss'),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(y=history.history['val_loss'], name='Val Loss'),
        row=1, col=1
    )

    # Plot predictions vs actual (Train)
    fig.add_trace(
        go.Scatter(x=y_train, y=train_pred, mode='markers',
                   name='Train Predictions', marker=dict(color='blue')),
        row=1, col=2
    )

    # Calculate feature importance using permutation
    feature_importance = {}
    baseline_loss = model.evaluate(X_test_scaled, y_test, verbose=0)[0]

    for i, feature in enumerate(X.columns):
        X_test_permuted = X_test_scaled.copy()
        X_test_permuted[:, i] = np.random.permutation(X_test_permuted[:, i])
        new_loss = model.evaluate(X_test_permuted, y_test, verbose=0)[0]
        feature_importance[feature] = new_loss - baseline_loss

    # Plot feature importance
    importance_df = pd.Series(feature_importance).sort_values(ascending=True)
    fig.add_trace(
        go.Bar(y=importance_df.index, x=importance_df.values,
               orientation='h', name='Feature Importance'),
        row=2, col=1
    )

    # Plot predictions vs actual (Test)
    fig.add_trace(
        go.Scatter(x=y_test, y=test_pred, mode='markers',
                   name='Test Predictions', marker=dict(color='red')),
        row=2, col=2
    )

    # Update layout
    fig.update_layout(height=800, width=1200, showlegend=True,
                      title_text=f'{"Weekly" if predict_weekly else "Daily"} Return Prediction Model Analysis')

    # Add diagonal lines for prediction plots
    for row, col in [(1, 2), (2, 2)]:
        min_val = min(min(y_train), min(y_test))
        max_val = max(max(y_train), max(y_test))
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                       mode='lines', line=dict(dash='dash', color='gray'),
                       showlegend=False),
            row=row, col=col
        )

    fig.show()

    # Print model performance metrics
    print("\nModel Performance Metrics:")
    print(f"{'='*30}")
    print(f"Training MAE: {model.evaluate(X_train_scaled, y_train)[1]:.4f}")
    print(f"Test MAE: {model.evaluate(X_test_scaled, y_test)[1]:.4f}")

    return model, scaler, history


def predict_returns(model, scaler, data, predict_weekly=False):
    """Make predictions using the trained model."""
    prepared_data = prepare_data(data, predict_weekly)
    features = prepared_data.drop('target', axis=1)
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled).flatten()

    return pd.Series(predictions, index=features.index)


def plot_results(data, predictions, threshold=0.01):
    """
    Plot the stock price with buy/sell signals based on predicted returns.

    Args:
        data: DataFrame containing stock price data
        predictions: Series of predicted returns
        threshold: Return threshold for generating signals (default: 1%)
    """
    fig = go.Figure()

    # Ensure data and predictions are aligned on the same index
    aligned_data = data.loc[predictions.index]

    # Plot stock price
    fig.add_trace(go.Scatter(
        x=aligned_data.index,
        y=aligned_data['Close'],
        mode='lines',
        name='Stock Price',
        line=dict(color='gray')
    ))

    # Add buy signals (green triangles) when predicted return > threshold
    buy_mask = predictions > threshold
    buy_signals = aligned_data[buy_mask]
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['Close'],
        mode='markers',
        name=f'Buy Signal (Expected Return > {threshold*100}%)',
        marker=dict(
            symbol='triangle-up',
            size=12,
            color='green'
        )
    ))

    # Add sell signals (red triangles) when predicted return < -threshold
    sell_mask = predictions < -threshold
    sell_signals = aligned_data[sell_mask]
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals['Close'],
        mode='markers',
        name=f'Sell Signal (Expected Return < -{threshold*100}%)',
        marker=dict(
            symbol='triangle-down',
            size=12,
            color='red'
        )
    ))

    # Update layout
    fig.update_layout(
        title='Stock Price with Predicted Return Signals',
        xaxis_title='Date',
        yaxis_title='Price',
        showlegend=True,
        xaxis_rangeslider_visible=True
    )

    fig.show()


if __name__ == "__main__":
    ticker = get_ticker('^GSPC')
    data = ticker.history(period='max')
    model, scaler, history = train_nn_predictor(data, predict_weekly=True)
    plot_results(data, predict_returns(
        model, scaler, data, predict_weekly=True))
