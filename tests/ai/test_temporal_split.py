import pandas as pd

from stock_analysis.ai.technical_analysis.movement_classification import (
    temporal_holdout_split,
)


def _sample_prepared_data(n=100):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame(
        {
            "prev_day_return": [0.01] * n,
            "prev_2day_return": [0.01] * n,
            "prev_week_return": [0.01] * n,
            "Target": ["positive"] * n,
        },
        index=dates,
    )


def test_temporal_holdout_split_by_fraction_is_chronological():
    data = _sample_prepared_data(100)
    train, holdout = temporal_holdout_split(data, test_size=0.2)

    assert len(train) == 80
    assert len(holdout) == 20
    assert train.index.max() < holdout.index.min()


def test_temporal_holdout_split_by_months():
    data = _sample_prepared_data(252)
    train, holdout = temporal_holdout_split(data, holdout_months=6)

    assert len(train) > 0
    assert len(holdout) > 0
    assert train.index.max() < holdout.index.min()
