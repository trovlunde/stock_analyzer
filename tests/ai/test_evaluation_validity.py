"""Guards against look-ahead and train/test leakage in the movement-classifier pipeline."""

import numpy as np
import pandas as pd
import pytest

import stock_analysis.ai.technical_analysis.movement_classification as movement_mod
from stock_analysis.ai.fundamental_analysis.fin_statement_classifier import (
    split_by_report_date,
)
from stock_analysis.ai.helpers import get_features
from stock_analysis.ai.technical_analysis.classifier_comparison import (
    purged_time_series_cv,
)
from stock_analysis.ai.technical_analysis.movement_classification import (
    WEEKLY_LABEL_HORIZON,
    compare_variants,
    temporal_holdout_split,
    temporal_three_way_split,
    train_classifier_single_stock,
)
from stock_analysis.ai.technical_analysis.nn_predictor import prepare_data
from stock_analysis.ai.technical_analysis.prepare_classification_data import (
    prepare_classification_data,
    prepare_classification_data_enhanced,
)


def _synthetic_ohlcv(n: int = 120, seed: int = 42) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=dates)


def _labelled_frame(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.DataFrame({"Target": ["positive"] * n}, index=dates)


@pytest.mark.parametrize(
    "prepare", [prepare_classification_data, prepare_classification_data_enhanced]
)
def test_weekly_label_is_return_over_the_next_five_bars(prepare):
    stock_data = _synthetic_ohlcv()
    close = stock_data["Close"]

    data = prepare(stock_data, predict_weekly=True)

    expected = (close.shift(-5) / close - 1).loc[data.index]
    pd.testing.assert_series_equal(data["return"], expected, check_names=False)


def test_weekly_label_excludes_todays_return():
    """Today's return is a feature; a label window starting yesterday would contain it."""
    stock_data = _synthetic_ohlcv()
    bumped = stock_data.copy()
    day = bumped.index[60]
    # Move only the close before `day`: this changes today's return (the
    # prev_day_return feature) but not any price from `day` onwards.
    bumped.loc[bumped.index[59], "Close"] *= 1.05

    original = prepare_classification_data(stock_data, predict_weekly=True)
    changed = prepare_classification_data(bumped, predict_weekly=True)

    assert changed.loc[day, "prev_day_return"] != original.loc[day, "prev_day_return"]
    assert changed.loc[day, "return"] == pytest.approx(original.loc[day, "return"])


def test_daily_label_is_next_bar_return():
    stock_data = _synthetic_ohlcv()
    close = stock_data["Close"]

    data = prepare_classification_data(stock_data, predict_weekly=False)

    expected = (close.shift(-1) / close - 1).loc[data.index]
    pd.testing.assert_series_equal(data["return"], expected, check_names=False)


def test_nn_predictor_weekly_target_is_return_over_the_next_five_bars():
    stock_data = _synthetic_ohlcv(200)
    close = stock_data["Close"]

    prepared = prepare_data(stock_data, predict_weekly=True)

    expected = (close.shift(-5) / close - 1).loc[prepared.index]
    pd.testing.assert_series_equal(prepared["target"], expected, check_names=False)


@pytest.mark.parametrize("use_extra_features", [False, True])
def test_features_do_not_depend_on_later_bars(use_extra_features):
    """Features for a date must be identical whether or not later bars exist."""
    stock_data = _synthetic_ohlcv(120)
    feature_columns = get_features(use_extra_features)

    full = prepare_classification_data(stock_data, use_extra_features=use_extra_features)
    truncated = prepare_classification_data(
        stock_data.iloc[:80], use_extra_features=use_extra_features
    )

    pd.testing.assert_frame_equal(
        full.loc[truncated.index, feature_columns], truncated[feature_columns]
    )


def test_warm_up_rows_are_dropped_not_backfilled():
    stock_data = _synthetic_ohlcv(120)
    feature_columns = get_features(use_extra_features=True)

    data = prepare_classification_data(stock_data, use_extra_features=True)

    # 21-bar rolling windows on returns/volume cannot be computed for the first bars
    assert data.index[0] > stock_data.index[19]
    assert not data[feature_columns].isna().any().any()
    # The latest bar has features but no label yet — it must survive for prediction
    assert data.index[-1] == stock_data.index[-1]
    assert pd.isna(data["Target"].iloc[-1])


def test_holdout_split_purges_train_rows_whose_labels_reach_into_holdout():
    data = _labelled_frame(100)

    train, holdout = temporal_holdout_split(
        data, test_size=0.2, label_horizon=WEEKLY_LABEL_HORIZON
    )

    assert len(holdout) == 20
    assert len(train) == 80 - (WEEKLY_LABEL_HORIZON - 1)
    # A train label looking 5 bars ahead must end no later than the first holdout bar
    last_train_position = data.index.get_loc(train.index[-1])
    first_holdout_position = data.index.get_loc(holdout.index[0])
    assert last_train_position + WEEKLY_LABEL_HORIZON <= first_holdout_position


def test_three_way_split_is_chronological_and_disjoint():
    data = _labelled_frame(780)  # ~3 years of business days

    train, validation, test = temporal_three_way_split(
        data, validation_months=6, test_months=6
    )

    assert len(train) > 0 and len(validation) > 0 and len(test) > 0
    assert train.index.max() < validation.index.min()
    assert validation.index.max() < test.index.min()
    assert test.index.max() == data.index.max()


def test_purged_cv_leaves_gap_between_train_and_test_folds():
    cv = purged_time_series_cv(n_splits=5, label_horizon=WEEKLY_LABEL_HORIZON)

    for train_idx, test_idx in cv.split(np.zeros((120, 1))):
        assert test_idx.min() - train_idx.max() > WEEKLY_LABEL_HORIZON


def test_train_classifier_default_classifier_is_not_shared_between_calls():
    dates = pd.date_range("2020-01-01", periods=60, freq="B")
    rng = np.random.default_rng(0)
    data = pd.DataFrame(
        {
            "prev_day_return": rng.normal(0, 0.01, 60),
            "prev_2day_return": rng.normal(0, 0.01, 60),
            "prev_week_return": rng.normal(0, 0.01, 60),
            "Target": (["positive", "negative", "neutral"] * 20),
        },
        index=dates,
    )

    first_model, _, _ = train_classifier_single_stock(data, eval_split=0, verbose=False)
    second_model, _, _ = train_classifier_single_stock(data, eval_split=0, verbose=False)

    assert first_model is not second_model


def test_split_by_report_date_keeps_test_reports_after_train_reports():
    dates = pd.date_range("2023-03-31", periods=5, freq="QE")
    # Two tickers reporting on the same five dates
    index = dates.append(dates)
    X = pd.DataFrame({"gross_margin": np.arange(10, dtype=float)}, index=index)
    y = pd.DataFrame({"daily_target": ["positive"] * 10}, index=index)

    X_train, X_test, y_train, y_test = split_by_report_date(X, y, test_size=0.2)

    assert len(X_train) + len(X_test) == 10
    assert len(X_test) > 0
    assert X_train.index.max() < X_test.index.min()
    assert y_train.index.equals(X_train.index)
    assert y_test.index.equals(X_test.index)


def test_compare_variants_reports_test_metrics_only_for_validation_winner(monkeypatch):
    stock_data = _synthetic_ohlcv(1100)  # > 4 years of business days
    monkeypatch.setattr(movement_mod, "get_index_data", lambda ticker, period: stock_data)
    variants = [
        {"name": "a", "threshold": 0.005, "extra_features": False,
         "n_estimators": 5, "max_depth": 3},
        {"name": "b", "threshold": 0.01, "extra_features": True,
         "n_estimators": 5, "max_depth": 3},
    ]

    results = compare_variants(
        ticker="TEST",
        holdout_months=6,
        validation_months=6,
        variants=variants,
        verbose=False,
    )

    assert set(results["variant"]) == {"a", "b"}
    assert results.attrs["selected_variant"] == results.iloc[0]["variant"]
    test_metrics = results.attrs["test_metrics"]
    assert test_metrics["holdout_samples"] > 0
    assert 0 <= test_metrics["combined_accuracy"] <= 1
    # Validation rows used for ranking end before the test period starts
    test_start = results.attrs["test_start"]
    assert test_start > stock_data.index[0]
    assert test_start >= stock_data.index[-1] - pd.DateOffset(months=6) - pd.Timedelta(days=7)
