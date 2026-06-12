import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from stock_analysis.ai.technical_analysis.movement_classification import evaluate_all_classifiers


def _minimal_param_grid():
    return {
        "Random Forest": {
            "n_estimators": [10],
            "max_depth": [3],
            "min_samples_split": [2],
            "class_weight": ["balanced"],
            "random_state": [42],
            "min_samples_leaf": [1],
        },
        "LightGBM": {
            "n_estimators": [10],
            "max_depth": [3],
            "learning_rate": [0.1],
            "num_leaves": [15],
        },
    }


@pytest.fixture
def prepared_data():
    n = 80
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    cycle = ["positive", "negative", "neutral", "positive", "negative"]
    targets = [cycle[i % len(cycle)] for i in range(n)]
    returns = [0.02 if t == "positive" else -0.02 if t == "negative" else 0.0 for t in targets]
    return pd.DataFrame(
        {
            "prev_day_return": np.tile([0.01, -0.01, 0.005], n)[:n],
            "prev_2day_return": np.tile([0.008, -0.008, 0.003], n)[:n],
            "prev_week_return": np.tile([0.015, -0.015, 0.002], n)[:n],
            "return": returns,
            "Target": targets,
        },
        index=dates,
    )


@patch(
    "stock_analysis.ai.technical_analysis.classifier_comparison.get_classifier_param_grid",
    side_effect=_minimal_param_grid,
)
def test_lightgbm_in_comparison_results(mock_grid, prepared_data):
    result = evaluate_all_classifiers(
        prepared_data,
        plot=False,
        classifier_names=["LightGBM"],
    )
    assert "LightGBM" in result["comparison_results"]


@patch(
    "stock_analysis.ai.technical_analysis.classifier_comparison.get_classifier_param_grid",
    side_effect=_minimal_param_grid,
)
def test_holdout_size_invariant_to_classifier_selection(mock_grid, prepared_data):
    lgbm_result = evaluate_all_classifiers(
        prepared_data,
        plot=False,
        classifier_names=["LightGBM"],
    )
    sklearn_result = evaluate_all_classifiers(
        prepared_data,
        plot=False,
        classifier_names=["Random Forest"],
    )
    assert lgbm_result["holdout_size"] == sklearn_result["holdout_size"]
