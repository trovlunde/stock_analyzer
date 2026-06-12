import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..helpers import RSI, get_ticker


def prepare_data(data, predict_weekly=False):
    """Prepare data for neural network training."""
    X = pd.DataFrame(index=data.index)

    if predict_weekly:
        X['target'] = data['Close'].pct_change(
            periods=5).shift(-4)
    else:
        X['target'] = data['Close'].pct_change().shift(-1)

    X['return_1d'] = data['Close'].pct_change()
    X['return_2d'] = data['Close'].pct_change(2)
    X['return_5d'] = data['Close'].pct_change(5)

    X['volume_1d'] = data['Volume'].pct_change()
    X['volume_5d'] = data['Volume'].pct_change(5)

    X['volatility_5d'] = data['Close'].pct_change().rolling(5).std()
    X['volatility_21d'] = data['Close'].pct_change().rolling(21).std()

    X['ma_5d'] = data['Close'].rolling(5).mean() / data['Close'] - 1
    X['ma_21d'] = data['Close'].rolling(21).mean() / data['Close'] - 1

    X['rsi_5d'] = RSI(data['Close'], window=5)
    X['rsi_21d'] = RSI(data['Close'], window=21)

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna()

    for col in X.columns:
        mean = X[col].mean()
        std = X[col].std()
        X = X[X[col].between(mean - 5 * std, mean + 5 * std)]

    return X


def train_nn_predictor(data, predict_weekly=False, test_size=0.2, do_tuning=True):
    """Train MLP regressor for stock prediction with optional hyperparameter tuning."""
    prepared_data = prepare_data(data, predict_weekly)

    y = prepared_data['target']
    X = prepared_data.drop('target', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if do_tuning:
        print("Starting hyperparameter tuning...")
        param_dist = {
            'hidden_layer_sizes': [
                (16,), (32,), (64,),
                (16, 8), (32, 16), (64, 32),
                (32, 16, 8), (64, 32, 16),
            ],
            'activation': ['relu', 'tanh'],
            'alpha': [1e-4, 1e-3, 1e-2],
            'learning_rate_init': [1e-4, 1e-3, 1e-2],
        }
        base_model = MLPRegressor(
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            random_state=42,
        )
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_dist,
            n_iter=20,
            cv=3,
            scoring='neg_mean_squared_error',
            random_state=42,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train_scaled, y_train)
        print(f"\nBest params: {search.best_params_}")
        model = search.best_estimator_
    else:
        model = MLPRegressor(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.2,
            n_iter_no_change=10,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

    history = {'loss': model.loss_curve_}

    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('Training Loss', 'Prediction vs Actual (Train)',
                                        'Feature Importance', 'Prediction vs Actual (Test)'))

    fig.add_trace(
        go.Scatter(y=history['loss'], name='Train Loss'),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(x=y_train, y=train_pred, mode='markers',
                   name='Train Predictions', marker=dict(color='blue')),
        row=1, col=2
    )

    feature_importance = {}
    baseline_mse = mean_squared_error(y_test, model.predict(X_test_scaled))
    for i, feature in enumerate(X.columns):
        X_test_permuted = X_test_scaled.copy()
        X_test_permuted[:, i] = np.random.permutation(X_test_permuted[:, i])
        new_mse = mean_squared_error(y_test, model.predict(X_test_permuted))
        feature_importance[feature] = new_mse - baseline_mse

    importance_df = pd.Series(feature_importance).sort_values(ascending=True)
    fig.add_trace(
        go.Bar(y=importance_df.index, x=importance_df.values,
               orientation='h', name='Feature Importance'),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(x=y_test, y=test_pred, mode='markers',
                   name='Test Predictions', marker=dict(color='red')),
        row=2, col=2
    )

    fig.update_layout(height=800, width=1200, showlegend=True,
                      title_text=f'{"Weekly" if predict_weekly else "Daily"} Return Prediction Model Analysis')

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

    print("\nModel Performance Metrics:")
    print(f"{'='*30}")
    print(f"Training MAE: {mean_absolute_error(y_train, train_pred):.4f}")
    print(f"Test MAE: {mean_absolute_error(y_test, test_pred):.4f}")

    return model, scaler, history


def predict_returns(model, scaler, data, predict_weekly=False):
    """Make predictions using the trained model."""
    prepared_data = prepare_data(data, predict_weekly)
    features = prepared_data.drop('target', axis=1)
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)

    return pd.Series(predictions, index=features.index)


def plot_results(data, predictions, threshold=0.01):
    """Plot stock price with buy/sell signals based on predicted returns."""
    fig = go.Figure()

    aligned_data = data.loc[predictions.index]

    fig.add_trace(go.Scatter(
        x=aligned_data.index,
        y=aligned_data['Close'],
        mode='lines',
        name='Stock Price',
        line=dict(color='gray')
    ))

    buy_mask = predictions > threshold
    buy_signals = aligned_data[buy_mask]
    fig.add_trace(go.Scatter(
        x=buy_signals.index,
        y=buy_signals['Close'],
        mode='markers',
        name=f'Buy Signal (Expected Return > {threshold*100}%)',
        marker=dict(symbol='triangle-up', size=12, color='green')
    ))

    sell_mask = predictions < -threshold
    sell_signals = aligned_data[sell_mask]
    fig.add_trace(go.Scatter(
        x=sell_signals.index,
        y=sell_signals['Close'],
        mode='markers',
        name=f'Sell Signal (Expected Return < -{threshold*100}%)',
        marker=dict(symbol='triangle-down', size=12, color='red')
    ))

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
    plot_results(data, predict_returns(model, scaler, data, predict_weekly=True))
