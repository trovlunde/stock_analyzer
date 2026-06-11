import pandas as pd

from stock_analysis.portfolio_evaluating import analyze_trading_strategies


def test_trading_signal_markers_align_with_price_dates():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    stock_data = pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100, 101, 102, 103, 104],
        },
        index=dates,
    )
    signals = pd.DataFrame(
        {"prediction": ["neutral", "positive", "neutral", "negative", "neutral"]},
        index=dates,
    )

    output = analyze_trading_strategies(
        stock_data,
        signals,
        initial_investment=10000,
        leverage=1,
        signal_type="Test",
    )
    results = output["results"]

    long_rows = results[results["Day_Trading_Position"] == "long"]
    assert len(long_rows) == 1
    assert long_rows.iloc[0]["Date"] == dates[1]
    assert long_rows.iloc[0]["Price"] == stock_data.loc[dates[1], "Close"]

    short_rows = results[results["Day_Trading_Position"] == "short"]
    assert len(short_rows) == 1
    assert short_rows.iloc[0]["Date"] == dates[3]
    assert short_rows.iloc[0]["Price"] == stock_data.loc[dates[3], "Close"]
