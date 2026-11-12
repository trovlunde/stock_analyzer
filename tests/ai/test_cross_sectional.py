"""Pooled cross-sectional model: features, labels, walk-forward windows, threshold,
calibration and the champion/challenger check."""

import numpy as np
import pandas as pd
import pytest

from stock_analysis.ai.cross_sectional import (
    PooledConfig,
    PooledModelScoreProvider,
    build_feature_panel,
    build_labels,
    build_pooled_dataset,
    fit_pooled_model,
)
from stock_analysis.ai.cross_sectional.champion import (
    MODEL_TYPE,
    at_least_as_good,
    retrain_with_champion_check,
)
from stock_analysis.ai.cross_sectional.features import (
    CONTEXT_FEATURES,
    RANKED_FEATURES,
    days_since_earnings,
    member_mask,
    wide_field,
)
from stock_analysis.ai.cross_sectional.model import (
    choose_threshold,
    evaluate_scores,
    reliability_table,
    top_k_picks,
    training_window,
)
from stock_analysis.ai.cross_sectional.report import pooled_report_sections
from stock_analysis.ai.model_manager import ModelManager

SESSIONS = pd.bdate_range("2015-01-02", periods=700)
FAST = {"n_estimators": 20, "min_child_samples": 20, "num_leaves": 7}


def _universe(n_tickers=24, sessions=SESSIONS, seed=0, signal=0.0):
    """Random-walk prices. With `signal > 0` a ticker's recent 21-session return
    carries over into the next weeks, so momentum ranks predict relative returns."""
    rng = np.random.default_rng(seed)
    n = len(sessions)
    data = {}
    drifts = rng.normal(0, 0.0004, n_tickers)
    for i in range(n_tickers):
        noise = rng.normal(0, 0.015, n)
        returns = np.empty(n)
        for t in range(n):
            carry = signal * returns[max(0, t - 21):t].mean() if t > 0 else 0.0
            returns[t] = drifts[i] + noise[t] + carry
        close = 50 * np.exp(np.cumsum(returns))
        data[f"T{i:02d}"] = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": rng.integers(1_000_000, 2_000_000, n).astype(float),
            },
            index=sessions,
        )
    return data


def _market(sessions=SESSIONS, seed=99):
    rng = np.random.default_rng(seed)
    close = 300 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, len(sessions))))
    return pd.DataFrame({"Close": close}, index=sessions)


def _config(**overrides):
    values = dict(
        horizon=5,
        train_sessions=400,
        validation_sessions=100,
        retrain_every=50,
        min_train_sessions=150,
        top_k=4,
        min_picks=20,
        lgbm_params=FAST,
        contribution_sample=200,
    )
    values.update(overrides)
    return PooledConfig(**values)


# --- features ---------------------------------------------------------------


def test_features_of_a_date_do_not_change_when_later_bars_are_removed():
    data = _universe()
    market = _market()
    cut = SESSIONS[400]

    full = build_feature_panel(data, market)
    truncated = build_feature_panel(
        {ticker: df.loc[:cut] for ticker, df in data.items()}, market.loc[:cut]
    )

    pd.testing.assert_frame_equal(full.loc[cut], truncated.loc[cut])


def test_ranked_features_are_percentiles_among_that_dates_members():
    panel = build_feature_panel(_universe(), _market())
    day = panel.loc[SESSIONS[300]]

    for name in RANKED_FEATURES:
        values = day[name].dropna()
        if name == "rs_sector_63":
            assert values.empty  # no sectors given
            continue
        assert values.between(0, 1).all(), name
        assert values.max() == 1.0, name
    # Context features are the same number for every ticker on a date
    for name in CONTEXT_FEATURES:
        assert day[name].nunique(dropna=False) == 1, name


def test_a_ticker_that_is_not_a_member_gets_no_row_and_does_not_count_in_ranks():
    data = _universe(n_tickers=5)

    class OnlyFirstFour:
        def member_matrix(self, sessions, tickers):
            return pd.DataFrame(
                {ticker: ticker != "T04" for ticker in tickers}, index=sessions
            )

    panel = build_feature_panel(data, membership=OnlyFirstFour())

    assert "T04" not in panel.index.get_level_values("ticker")
    # Four members: the highest rank is 1.0 and the lowest 0.25
    assert sorted(panel.loc[SESSIONS[300], "mom_21"]) == [0.25, 0.5, 0.75, 1.0]


def test_start_keeps_full_trailing_windows_for_the_first_rows_kept():
    data = _universe(n_tickers=4)
    full = build_feature_panel(data)
    late = build_feature_panel(data, start=SESSIONS[300])

    assert late.index.get_level_values("date").min() == SESSIONS[300]
    pd.testing.assert_frame_equal(late.loc[SESSIONS[300]], full.loc[SESSIONS[300]])


def test_days_since_earnings_counts_from_the_latest_report_on_or_before_the_session():
    sessions = pd.DatetimeIndex(["2024-01-31", "2024-02-01", "2024-02-15"])
    table = days_since_earnings(sessions, ["AAA", "BBB"], {"AAA": ["2024-02-01"]})

    assert np.isnan(table.loc["2024-01-31", "AAA"])
    assert table.loc["2024-02-01", "AAA"] == 0
    assert table.loc["2024-02-15", "AAA"] == 14
    assert table["BBB"].isna().all()


# --- labels -----------------------------------------------------------------


def test_labels_mark_the_top_fifth_by_forward_return_relative_to_the_universe():
    sessions = pd.bdate_range("2024-01-01", periods=4)
    # Ten tickers growing at different steady rates: T09 fastest
    close = pd.DataFrame(
        {f"T{i:02d}": 100 * (1 + 0.01 * i) ** np.arange(4) for i in range(10)}, index=sessions
    )
    labels = build_labels(close, member_mask(close), horizon=2)

    first = labels.loc[sessions[0]]
    assert first["label"].sum() == 2
    assert set(first.index[first["label"] == 1]) == {"T08", "T09"}
    assert first["rel_return"].mean() == pytest.approx(0, abs=1e-12)
    # The last two sessions have no complete forward window
    assert labels.loc[sessions[2:]].isna().all().all()


def test_label_is_missing_when_the_stock_stops_trading_inside_the_window():
    sessions = pd.bdate_range("2024-01-01", periods=5)
    close = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3, 1.4], "B": [1.0, 1.0, np.nan, np.nan, np.nan]}, index=sessions)
    labels = build_labels(close, member_mask(close), horizon=2)

    assert np.isnan(labels.loc[(sessions[0], "B"), "label"])
    assert labels.loc[(sessions[0], "A"), "label"] == 1.0


# --- walk-forward windows -----------------------------------------------------


def test_training_window_purges_overlapping_labels_and_never_reaches_past_the_decision():
    config = _config()
    window = training_window(600, config)

    # Newest usable label row: its forward window ends exactly at the decision
    assert window.validation_last + config.horizon == 600
    assert window.validation_last - window.validation_first + 1 == config.validation_sessions
    # h rows between train and validation are left out
    assert window.validation_first - window.train_last - 1 == config.horizon
    assert window.validation_last - window.train_first + 1 == config.train_sessions


def test_training_window_is_none_until_there_is_enough_history():
    config = _config()
    assert training_window(100, config) is None
    assert training_window(600, config) is not None


def test_without_a_validation_fold_everything_labelled_is_training_data():
    window = training_window(600, _config(validation_sessions=0))
    assert window.validation_first is None
    assert window.train_last == 595


# --- threshold and scoring ------------------------------------------------------


def test_top_k_picks_per_date():
    picks = top_k_picks(np.array([3, 1, 2, 9, 8, 7.0]), np.array([0, 0, 0, 1, 1, 1]), top_k=2)
    assert picks.tolist() == [True, False, True, True, True, False]


def test_threshold_is_chosen_only_when_it_beats_trading_every_pick():
    rng = np.random.default_rng(0)
    dates = np.repeat(np.arange(100), 10)
    probability = rng.uniform(0, 1, 1000)

    # Picks with a low probability lose money: a threshold should drop them
    rel_return = np.where(probability > 0.9, 0.02, -0.02)
    chosen = choose_threshold(probability, rel_return, dates, top_k=2, round_trip_cost=0.0, min_picks=20)
    assert chosen.threshold > 0
    assert chosen.net_return > chosen.baseline_net_return

    # Every pick earns the same: nothing to gain, no threshold
    flat = choose_threshold(probability, np.full(1000, 0.01), dates, top_k=2, round_trip_cost=0.0, min_picks=20)
    assert flat.threshold == 0.0


def test_evaluate_scores_rank_ic_is_one_for_a_perfect_ranking():
    dates = np.repeat(np.arange(5), 6)
    realised = np.tile(np.linspace(-0.05, 0.05, 6), 5)
    result = evaluate_scores(realised * 10, realised, dates, top_k=2, round_trip_cost=0.001)

    assert result["rank_ic"] == pytest.approx(1.0)
    assert result["top_k_net_return"] == pytest.approx(np.mean([0.05, 0.03]) - 0.001)
    assert result["dates"] == 5


def test_reliability_table_has_equal_sized_bins():
    table = reliability_table(np.linspace(0, 1, 100), np.r_[np.zeros(50), np.ones(50)], bins=4)
    assert table["rows"].tolist() == [25, 25, 25, 25]
    assert table["observed"].tolist() == [0, 0, 1, 1]


# --- fitting and the walk-forward provider -------------------------------------


@pytest.fixture(scope="module")
def dataset():
    return build_pooled_dataset(_universe(signal=0.6), market=_market(), horizon=5)


def test_fitted_model_only_saw_labels_complete_at_the_decision(dataset):
    config = _config()
    fitted = fit_pooled_model(dataset, 600, config)

    assert fitted is not None
    assert fitted.trained_at == dataset.calendar[600]
    assert fitted.trained_through == dataset.calendar[600 - config.horizon]
    assert fitted.calibrator is not None
    assert fitted.importance.sum() == pytest.approx(1.0)
    assert set(fitted.importance.index) == set(dataset.features)
    validation = fitted.validation
    assert validation["rows"] > 0
    assert 0 <= validation["brier_calibrated"] <= validation["brier_raw"] + 0.05


def test_calibrated_probabilities_are_probabilities(dataset):
    fitted = fit_pooled_model(dataset, 600, _config())
    probability = fitted.probability(dataset.X[dataset.rows(600, 600)])
    assert ((probability >= 0) & (probability <= 1)).all()


def test_raw_probability_matches_predict_proba_without_warnings(dataset, recwarn):
    fitted = fit_pooled_model(dataset, 600, _config())
    X = dataset.X[dataset.rows(590, 600)]
    recwarn.clear()

    raw = fitted.raw_probability(X)

    assert not [w for w in recwarn if "feature names" in str(w.message)]
    np.testing.assert_allclose(raw, fitted.model.predict_proba(X)[:, 1])


def test_provider_scores_with_the_latest_model_and_retrains_on_schedule(dataset):
    config = _config(retrain_every=50)
    provider = PooledModelScoreProvider(dataset, config)

    assert provider.scores_on(dataset.calendar[100]) is None  # not enough history
    for position in range(400, 520):
        scores = provider.scores_on(dataset.calendar[position])
        assert scores is not None
        assert len(scores) == 24

    # Fitted at 400, 450 and 500
    assert [fit.trained_at for fit in provider.fits] == [dataset.calendar[p] for p in (400, 450, 500)]
    for fit in provider.fits:
        position = dataset.calendar.get_loc(fit.trained_at)
        assert dataset.calendar.get_loc(fit.trained_through) == position - config.horizon


def test_provider_scores_are_cached_per_session(dataset):
    provider = PooledModelScoreProvider(dataset, _config())
    first = provider.scores_on(dataset.calendar[450])
    assert provider.scores_on(dataset.calendar[450]) is first
    assert len(provider.fits) == 1


def test_model_finds_a_planted_momentum_signal(dataset):
    provider = PooledModelScoreProvider(dataset, _config(use_threshold=False))
    for position in range(400, 690):
        provider.scores_on(dataset.calendar[position])

    sections = "\n".join(pooled_report_sections(provider))
    predictions = pd.concat(provider.history, names=["date", "ticker"])
    realised = pd.Series(dataset.rel_return, index=dataset.index).reindex(predictions.index)
    ic = predictions["score"].groupby(level="date").corr(realised.groupby(level="date").rank())
    assert ic.mean() > 0.05
    assert "## Pooled model (ml_pooled)" in sections
    assert "### Feature importance" in sections
    assert "### Rank IC per year (walk-forward)" in sections
    assert "Walk-forward (out of sample): Brier" in sections


def test_without_threshold_nothing_is_blocked(dataset):
    fitted = fit_pooled_model(dataset, 600, _config(use_threshold=False))
    assert fitted.threshold == 0.0


# --- champion / challenger ------------------------------------------------------


def test_at_least_as_good_needs_every_metric_not_worse():
    champion = {"rank_ic": 0.02, "top_k_net_return": 0.001}
    assert at_least_as_good({"rank_ic": 0.02, "top_k_net_return": 0.002}, champion)
    assert not at_least_as_good({"rank_ic": 0.03, "top_k_net_return": 0.0005}, champion)
    assert at_least_as_good({"rank_ic": None, "top_k_net_return": 0.0}, {"rank_ic": None, "top_k_net_return": None})


def test_first_retrain_saves_the_challenger_and_a_second_compares_against_it(dataset, tmp_path):
    manager = ModelManager(models_dir=str(tmp_path))
    config = _config()

    first = retrain_with_champion_check(dataset, config, manager, "pooled-test")
    assert first.promoted
    assert first.reason == "no champion saved yet"
    metadata = manager.get_model_metadata("pooled-test", MODEL_TYPE)
    assert metadata["horizon"] == 5
    assert metadata["features"] == dataset.features

    # Same data, same seed: an identical challenger is "at least as good"
    second = retrain_with_champion_check(dataset, config, manager, "pooled-test")
    assert second.champion_metrics == second.challenger_metrics
    assert second.promoted
    assert second.champion_seen_sessions == config.validation_sessions


def test_a_worse_challenger_is_not_saved_unless_forced(dataset, tmp_path, monkeypatch):
    import stock_analysis.ai.cross_sectional.champion as champion_mod

    manager = ModelManager(models_dir=str(tmp_path))
    config = _config()
    retrain_with_champion_check(dataset, config, manager, "pooled-test")
    saved_at = (tmp_path / f"pooled-test_{MODEL_TYPE}.joblib").stat().st_mtime_ns

    real_metrics = champion_mod.window_metrics
    calls = []

    def challenger_worse(model, *args):
        metrics = dict(real_metrics(model, *args))
        calls.append(metrics)
        if len(calls) == 1:  # the challenger is scored first
            metrics["rank_ic"] = -1.0
        return metrics

    monkeypatch.setattr(champion_mod, "window_metrics", challenger_worse)
    kept = retrain_with_champion_check(dataset, config, manager, "pooled-test")
    assert not kept.promoted
    assert "kept" in kept.reason
    assert (tmp_path / f"pooled-test_{MODEL_TYPE}.joblib").stat().st_mtime_ns == saved_at

    calls.clear()
    forced = retrain_with_champion_check(dataset, config, manager, "pooled-test", force=True)
    assert forced.promoted
    assert "--force" in forced.reason


def test_a_champion_with_other_features_cannot_be_compared(dataset, tmp_path):
    manager = ModelManager(models_dir=str(tmp_path))
    config = _config()
    retrain_with_champion_check(dataset, config, manager, "pooled-test")

    other_horizon = _config(horizon=3)
    outcome = retrain_with_champion_check(dataset, other_horizon, manager, "pooled-test")
    assert not outcome.promoted
    assert "cannot be compared" in outcome.reason


def test_champion_check_needs_a_validation_fold(dataset, tmp_path):
    with pytest.raises(ValueError, match="validation fold"):
        retrain_with_champion_check(
            dataset, _config(validation_sessions=0), ModelManager(models_dir=str(tmp_path)), "x"
        )


def test_wide_field_aligns_tickers_on_the_union_of_sessions():
    a = pd.DataFrame({"Close": [1.0, 2.0]}, index=pd.DatetimeIndex(["2024-01-01", "2024-01-03"]))
    b = pd.DataFrame({"Close": [5.0]}, index=pd.DatetimeIndex(["2024-01-02"]))
    wide = wide_field({"A": a, "B": b}, "Close")
    assert list(wide.index.strftime("%d")) == ["01", "02", "03"]
    assert np.isnan(wide.loc["2024-01-02", "A"])
