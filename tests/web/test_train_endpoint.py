"""POST /api/v1/train/<ticker>: evaluate on models that never saw the holdout,
then save models fitted on all labelled data."""

import numpy as np
import pandas as pd
import pytest

from stock_analysis.ai.model_manager import ModelManager
from stock_analysis.web import app as app_module


def _prices(n=700) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n)))
    index = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 2_000_000, n),
        },
        index=index,
    )


@pytest.fixture
def trained(tmp_path, monkeypatch):
    """Call the endpoint once and record what every model was fitted on."""
    prices = _prices()
    manager = ModelManager(models_dir=str(tmp_path))
    monkeypatch.setattr(app_module, "get_index_data", lambda ticker, period: prices)
    monkeypatch.setattr(app_module, "model_manager", manager)

    fits = []
    real_train = app_module.train_classifier_single_stock

    def recording_train(rows, **kwargs):
        fits.append({"rows": rows, **kwargs})
        return real_train(rows, **kwargs)

    monkeypatch.setattr(app_module, "train_classifier_single_stock", recording_train)

    def post(body):
        fits.clear()
        response = app_module.app.test_client().post("/api/v1/train/TEST", json=body)
        return response, list(fits)

    return post, manager, prices


def test_train_evaluates_on_holdout_blind_models_then_refits_on_everything(trained):
    post, manager, prices = trained

    response, fits = post({"holdout_months": 6, "threshold": 0.005})

    assert response.status_code == 200
    body = response.get_json()
    holdout_start = pd.Timestamp(body["holdout_metrics"]["holdout_start"])

    # Two models (daily, weekly) for the evaluation, then two for saving
    assert [fit["predict_weekly"] for fit in fits] == [False, True, False, True]
    # Nothing is silently held back from any fit
    assert all(fit["eval_split"] == 0 for fit in fits)
    evaluation, final = fits[:2], fits[2:]
    for fit in evaluation:
        assert fit["rows"].index.max() < holdout_start
    # The weekly evaluation model also loses the rows whose 5-day label reaches
    # into the holdout
    assert evaluation[1]["rows"].index.max() < evaluation[0]["rows"].index.max()
    for fit in final:
        assert fit["rows"].index.max() == prices.index[-1]

    metadata = body["metadata"]
    # The last bar has no label yet (it needs the next close)
    assert metadata["trained_through"] == str(prices.index[-2].date())
    assert metadata["training_samples"] > len(evaluation[0]["rows"])
    assert metadata["holdout_metrics"]["holdout_months"] == 6
    assert 0 <= body["holdout_metrics"]["daily_accuracy"] <= 1


def test_train_saves_models_the_trading_signal_accepts(trained):
    post, manager, _ = trained

    response, _ = post({"holdout_months": 0})

    assert response.status_code == 200
    assert "holdout_metrics" not in response.get_json()
    for name, kind in (
        ("TEST_movement", "daily"),
        ("TEST_movement_scaler", "daily"),
        ("TEST_movement", "weekly"),
        ("TEST_movement_scaler", "weekly"),
    ):
        assert manager.load_model(name, kind, bypass_cache=True) is not None
    metadata = manager.get_model_metadata("TEST_movement", "daily")
    assert metadata["feature_columns"] == ["prev_day_return", "prev_2day_return", "prev_week_return"]
    assert metadata["ticker"] == "TEST"


def test_train_reports_a_data_failure(trained, monkeypatch):
    post, _, _ = trained
    monkeypatch.setattr(app_module, "get_index_data", lambda ticker, period: None)

    response, fits = post({})

    assert response.status_code == 500
    assert response.get_json()["error"] == "Data fetch failed"
    assert fits == []
