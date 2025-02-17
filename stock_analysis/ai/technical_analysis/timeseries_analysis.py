from prophet import Prophet
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from ..helpers import get_sp500_data


def timeseries_analysis():
    """Perform time series analysis on SP500 data using Prophet."""

    # Get data
    print("Fetching SP500 data...")
    sp500 = get_sp500_data()

    # Prepare time series data for Prophet
    data = pd.DataFrame({
        'ds': sp500.index.tz_localize(None),  # Remove timezone
        'y': sp500['Close']  # target column
    })

    # Initialize and train model
    print("\nTraining Prophet model...")
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(data)

    # Create future predictions
    print("\nGenerating future predictions...")
    last_date = data['ds'].max()
    last_value = data.loc[data['ds'] == last_date, 'y'].iloc[0]

    future = model.make_future_dataframe(
        periods=30, freq='B')  # 30 business days ahead
    full_forecast = model.predict(future)

    # Adjust the forecast to start from the last actual value
    forecast = full_forecast[full_forecast['ds'] > last_date].copy()
    first_forecast = forecast.iloc[0]
    adjustment = last_value - first_forecast['yhat']
    forecast['yhat'] += adjustment
    forecast['yhat_lower'] += adjustment
    forecast['yhat_upper'] += adjustment

    # Create forecast plot with plotly
    fig1 = go.Figure()

    # Add actual values
    fig1.add_trace(go.Scatter(
        x=data['ds'],
        y=data['y'],
        name='Actual',
        mode='lines',
        line=dict(color='blue')
    ))

    # Add predictions for future only
    fig1.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat'],
        name='Forecast',
        mode='lines',
        line=dict(color='red')
    ))

    # Add uncertainty intervals for future only
    fig1.add_trace(go.Scatter(
        x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
        y=forecast['yhat_upper'].tolist(
        ) + forecast['yhat_lower'].tolist()[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        name='Confidence Interval'
    ))

    fig1.update_layout(
        title='Time Series Forecast',
        xaxis_title='Date',
        yaxis_title='Price',
        hovermode='x unified'
    )

    # Show plotly forecast plot
    fig1.show()

    # Create matplotlib components plot
    fig2 = model.plot_components(full_forecast)
    plt.show()

    # Print performance metrics
    print("\nModel Performance Metrics:")

    # Ensure both series have the same index
    forecast_hist = full_forecast.set_index('ds')['yhat'][:len(data)]
    actual = data.set_index('ds')['y']

    # Make sure both indices are timezone naive
    forecast_hist.index = forecast_hist.index.tz_localize(None)
    actual.index = actual.index.tz_localize(None)

    metrics = pd.DataFrame({
        'mae': [abs(forecast_hist - actual).mean()],  # Mean Absolute Error
        # Root Mean Square Error
        'rmse': [((forecast_hist - actual)**2).mean()**0.5]
    })

    print("\nModel Performance Metrics Explanation:")
    print("MAE (Mean Absolute Error):")
    print(
        f"- Average absolute difference between predicted and actual values: ${metrics['mae'].iloc[0]:.2f}")
    print("- Represents the typical prediction error in dollars")
    print("\nRMSE (Root Mean Square Error):")
    print(
        f"- Root mean square difference between predicted and actual values: ${metrics['rmse'].iloc[0]:.2f}")
    print("- Penalizes larger errors more heavily than MAE")
    print("- Also in dollars, but more sensitive to outliers")

    return model, forecast


if __name__ == "__main__":
    model, forecast = timeseries_analysis()
