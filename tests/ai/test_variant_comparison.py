import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from stock_analysis.ai.technical_analysis.movement_classification import (
    DEFAULT_VARIANT_SPECS,
    _actual_movement_class,
    _majority_class_accuracy,
    _prediction_accuracy,
    compute_holdout_metrics,
    train_classifier_single_stock,
)


def _sample_prepared_data(n=80, threshold=0.005):
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = [0.02 if index % 3 == 0 else -0.02 if index % 3 == 1 else 0.0 for index in range(n)]
    targets = [
        _actual_movement_class(value, threshold) for value in returns
    ]
    return pd.DataFrame(
        {
            "prev_day_return": [0.01] * n,
            "prev_2day_return": [0.01] * n,
            "prev_week_return": [0.01] * n,
            "return": returns,
            "Target": targets,
        },
        index=dates,
    )


def test_prediction_accuracy_ignores_unknown_actuals():
    accuracy = _prediction_accuracy(
        ["positive", "negative", "neutral"],
        ["positive", None, "neutral"],
    )
    assert accuracy == 1.0


def test_default_variant_specs_have_unique_names():
    names = [variant["name"] for variant in DEFAULT_VARIANT_SPECS]
    assert len(names) == len(set(names))


def test_compute_holdout_metrics_reports_combined_accuracy():
    train_data = _sample_prepared_data(80)
    holdout_data = _sample_prepared_data(20)

    classifier = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3)
    daily_model, daily_scaler, _ = train_classifier_single_stock(
        train_data,
        classifier=classifier,
        eval_split=0,
        verbose=False,
    )
    weekly_model, weekly_scaler, _ = train_classifier_single_stock(
        train_data,
        predict_weekly=True,
        classifier=RandomForestClassifier(n_estimators=10, random_state=42, max_depth=3),
        eval_split=0,
        verbose=False,
    )

    metrics = compute_holdout_metrics(
        holdout_data,
        daily_model,
        daily_scaler,
        holdout_data,
        weekly_model,
        weekly_scaler,
        threshold=0.005,
        use_extra_features=False,
    )

    assert metrics["holdout_samples"] == 20
    assert 0 <= metrics["daily_accuracy"] <= 1
    assert 0 <= metrics["weekly_accuracy"] <= 1
    assert 0 <= metrics["combined_accuracy"] <= 1
    assert 0 <= metrics["combined_signal_rate"] <= 1


def test_majority_baseline_is_share_of_most_common_actual_class():
    train_data = _sample_prepared_data(80)
    # 20 rows cycling positive/negative/neutral -> 7 positive, 7 negative, 6 neutral
    holdout_data = _sample_prepared_data(20)

    models = [
        train_classifier_single_stock(
            train_data,
            predict_weekly=weekly,
            classifier=RandomForestClassifier(n_estimators=5, random_state=0, max_depth=3),
            eval_split=0,
            verbose=False,
        )
        for weekly in (False, True)
    ]
    (daily_model, daily_scaler, _), (weekly_model, weekly_scaler, _) = models

    metrics = compute_holdout_metrics(
        holdout_data, daily_model, daily_scaler,
        holdout_data, weekly_model, weekly_scaler,
        threshold=0.005, use_extra_features=False,
    )

    assert metrics["daily_majority_baseline"] == 7 / 20


def test_majority_class_accuracy_ignores_unknown_and_handles_empty():
    assert _majority_class_accuracy(["neutral", "neutral", None, "positive"]) == 2 / 3
    assert _majority_class_accuracy([None]) is None
