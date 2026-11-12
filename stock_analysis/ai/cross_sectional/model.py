"""One pooled LightGBM model for the whole universe, retrained as history unfolds.

`fit_pooled_model` trains on the rows whose labels were complete at a given
session, holds the most recent part back as a validation fold, calibrates the
model's probabilities on that fold (isotonic) and picks the entry threshold
there by net relative return. `PooledModelScoreProvider` does this every
`retrain_every` sessions and scores each session's universe members with the
latest model; `trading/signals/ranked.RankedUniverseStrategy` turns the scores
into BUY / HOLD / SELL.

Timeline for a decision at session position p (h = horizon):

    train rows            purge      validation rows       not yet labelled
    [p-W+1-h .. v-h-1]    h rows     [v .. p-h]            (p-h .. p]
                                     v = p-h-V+1

The label of row t looks at the close of t+h, so the newest usable row is p-h.
The h rows between train and validation are dropped because their labels
overlap the validation period. The model that scores session p is the one
fitted on the train rows; the validation rows calibrate it and are not added
back, so the calibrator and the model belong together. The price of that is a
model that has not seen the last V+h sessions.

The calibrator and the threshold are fitted on the same validation fold whose
Brier score is reported, so that number is optimistic; the walk-forward
calibration in the backtest report (`report.py`) is the honest one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression

from .features import feature_names

# Shallow, heavily regularised trees: the signal is weak and the rows of one
# date are far from independent, so anything flexible fits noise.
DEFAULT_LGBM_PARAMS: dict = {
    "n_estimators": 300,
    "learning_rate": 0.03,
    "num_leaves": 15,
    "max_depth": 4,
    "min_child_samples": 200,
    "subsample": 0.7,
    "subsample_freq": 1,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "n_jobs": 4,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
DEFAULT_THRESHOLD_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
# Keeps the order of the raw model score among stocks that the (step-shaped)
# isotonic calibrator maps to the same probability
_TIE_BREAK = 1e-6


@dataclass(frozen=True)
class PooledConfig:
    horizon: int = 10
    # Sessions of labelled rows used per retrain, validation fold included (~5 years)
    train_sessions: int = 1260
    # Most recent labelled sessions held out for calibration and the threshold;
    # 0 trains on everything and returns uncalibrated probabilities
    validation_sessions: int = 252
    retrain_every: int = 63
    min_train_sessions: int = 252
    # How many names the strategy buys; the threshold is chosen for these picks
    top_k: int = 10
    # Cost of buying and selling one position, as a fraction of its size
    round_trip_cost: float = 0.0
    # False keeps the calibration but never blocks an entry
    use_threshold: bool = True
    threshold_quantiles: tuple[float, ...] = DEFAULT_THRESHOLD_QUANTILES
    # A threshold must leave at least this many validation picks to be trusted
    min_picks: int = 100
    seed: int = 0
    lgbm_params: dict = field(default_factory=dict)
    contribution_sample: int = 2000


@dataclass
class PooledDataset:
    """Feature panel and labels in arrays sorted by date, for fast row slicing."""

    calendar: pd.DatetimeIndex
    index: pd.MultiIndex
    position: np.ndarray  # calendar position of each row's date
    features: list[str]
    X: np.ndarray
    label: np.ndarray
    rel_return: np.ndarray

    def rows(self, first: int, last: int) -> slice:
        """Rows whose date has calendar position in [first, last]."""
        return slice(
            int(np.searchsorted(self.position, first, side="left")),
            int(np.searchsorted(self.position, last, side="right")),
        )

    def dates(self, rows: slice) -> np.ndarray:
        return self.position[rows]


def make_dataset(
    panel: pd.DataFrame,
    labels: pd.DataFrame,
    calendar: pd.DatetimeIndex,
    features: list[str] | None = None,
) -> PooledDataset:
    """Join the feature panel with its labels. `calendar` is the full session
    index the labels were built on: label windows are counted in its sessions,
    also when the panel starts later. `features` picks the model inputs
    (default: the standard feature set)."""
    panel = panel.sort_index(level="date", sort_remaining=False, kind="stable")
    labels = labels.reindex(panel.index)
    names = list(features) if features is not None else feature_names(panel)
    position = calendar.get_indexer(panel.index.get_level_values("date"))
    if (position < 0).any():
        raise ValueError("The panel has dates that are not in the calendar")
    return PooledDataset(
        calendar=calendar,
        index=panel.index,
        position=position,
        features=names,
        X=panel[names].to_numpy(dtype=np.float32),
        label=labels["label"].to_numpy(dtype=float),
        rel_return=labels["rel_return"].to_numpy(dtype=float),
    )


@dataclass(frozen=True)
class TrainingWindow:
    """Calendar positions (inclusive) of the rows used when deciding at `decided_at`."""

    decided_at: int
    train_first: int
    train_last: int
    validation_first: int | None
    validation_last: int | None

    @property
    def last_label_position(self) -> int:
        return self.train_last if self.validation_last is None else self.validation_last


def training_window(position: int, config: PooledConfig) -> TrainingWindow | None:
    """Rows a model deciding at calendar `position` may use; None while there is
    too little history. No row's label window (its date + horizon) ends after
    `position`."""
    newest = position - config.horizon
    oldest = max(0, newest - config.train_sessions + 1)
    if config.validation_sessions <= 0:
        window = TrainingWindow(position, oldest, newest, None, None)
    else:
        validation_first = newest - config.validation_sessions + 1
        window = TrainingWindow(
            position, oldest, validation_first - config.horizon - 1, validation_first, newest
        )
    if window.train_last - window.train_first + 1 < config.min_train_sessions:
        return None
    return window


def top_k_picks(scores: np.ndarray, dates: np.ndarray, top_k: int) -> np.ndarray:
    """Boolean per row: among the `top_k` best scores of its date."""
    frame = pd.DataFrame({"score": scores, "date": dates})
    rank = frame.groupby("date")["score"].rank(ascending=False, method="first")
    return (rank <= top_k).to_numpy()


def rank_ic_by_date(scores: pd.Series, realised: pd.Series, dates) -> pd.Series:
    """Spearman correlation between score and realised return, per date."""
    frame = pd.DataFrame(
        {"score": np.asarray(scores), "realised": np.asarray(realised), "date": np.asarray(dates)}
    ).dropna()
    if frame.empty:
        return pd.Series(dtype=float)
    ranks = frame.groupby("date")[["score", "realised"]].rank()
    ranks["date"] = frame["date"]

    def correlation(group: pd.DataFrame) -> float:
        if len(group) < 3 or group["score"].nunique() < 2 or group["realised"].nunique() < 2:
            return float("nan")
        return float(group["score"].corr(group["realised"]))

    return ranks.groupby("date")[["score", "realised"]].apply(correlation)


def evaluate_scores(
    scores: np.ndarray,
    rel_return: np.ndarray,
    dates: np.ndarray,
    *,
    top_k: int,
    round_trip_cost: float,
) -> dict:
    """Ranking quality of `scores` against realised relative returns.

    `rank_ic`: mean daily Spearman correlation. `top_k_net_return`: mean
    relative forward return of each date's `top_k` picks minus one round-trip
    cost per pick (one holding period = one label horizon).
    """
    known = ~np.isnan(rel_return) & ~np.isnan(scores)
    scores, rel_return, dates = scores[known], rel_return[known], dates[known]
    if len(scores) == 0:
        return {"rank_ic": None, "top_k_net_return": None, "dates": 0}
    ic = rank_ic_by_date(scores, rel_return, dates).dropna()
    picks = top_k_picks(scores, dates, top_k)
    return {
        "rank_ic": float(ic.mean()) if len(ic) else None,
        "top_k_net_return": float(rel_return[picks].mean() - round_trip_cost),
        "dates": int(len(np.unique(dates))),
    }


@dataclass(frozen=True)
class ThresholdChoice:
    threshold: float  # 0.0 = no threshold
    net_return: float  # objective at the chosen threshold
    baseline_net_return: float  # objective when every top-k pick is traded
    picks: int
    baseline_picks: int


def choose_threshold(
    probability: np.ndarray,
    rel_return: np.ndarray,
    dates: np.ndarray,
    *,
    top_k: int,
    round_trip_cost: float,
    quantiles: tuple[float, ...] = DEFAULT_THRESHOLD_QUANTILES,
    min_picks: int = 100,
) -> ThresholdChoice:
    """Entry threshold on the calibrated probability that would have earned the
    most on the validation fold, net of costs.

    The strategy buys each date's `top_k` names; a threshold leaves a slot
    empty when its candidate's probability is too low. Objective: the sum over
    traded picks of (relative forward return - round-trip cost), divided by all
    slots (dates x top_k), so an empty slot counts as zero - neither the pick's
    relative return nor its cost. Candidates are quantiles of the picks'
    probabilities. No threshold (0.0) is returned unless a candidate beats
    trading every pick and still leaves `min_picks` picks.
    """
    picks = top_k_picks(probability, dates, top_k) & ~np.isnan(rel_return)
    slots = max(1, len(np.unique(dates)) * top_k)
    net = rel_return[picks] - round_trip_cost
    pick_probability = probability[picks]
    baseline = float(net.sum() / slots) if len(net) else 0.0
    best = ThresholdChoice(0.0, baseline, baseline, int(picks.sum()), int(picks.sum()))
    if len(net) == 0:
        return best
    for threshold in sorted(set(np.quantile(pick_probability, quantiles))):
        traded = pick_probability >= threshold
        count = int(traded.sum())
        if count < min_picks or count == len(net):
            continue
        value = float(net[traded].sum() / slots)
        if value > best.net_return:
            best = ThresholdChoice(float(threshold), value, baseline, count, int(picks.sum()))
    return best


def reliability_table(probability: np.ndarray, label: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Predicted probability against observed frequency, in `bins` equal-sized
    groups ordered by predicted probability."""
    frame = pd.DataFrame({"predicted": probability, "observed": label}).dropna()
    if frame.empty:
        return pd.DataFrame(columns=["bin", "predicted", "observed", "rows"])
    order = frame["predicted"].rank(method="first")
    frame["bin"] = np.minimum((order - 1) * bins // len(frame), bins - 1).astype(int) + 1
    table = frame.groupby("bin").agg(
        predicted=("predicted", "mean"), observed=("observed", "mean"), rows=("observed", "size")
    )
    return table.reset_index()


def brier_score(probability: np.ndarray, label: np.ndarray) -> float | None:
    known = ~np.isnan(probability) & ~np.isnan(label)
    if not known.any():
        return None
    return float(np.mean((probability[known] - label[known]) ** 2))


@dataclass
class FittedPooledModel:
    """A trained model with its calibrator, threshold and what it was trained on."""

    model: LGBMClassifier
    calibrator: IsotonicRegression | None
    threshold: float
    features: list[str]
    horizon: int
    window: TrainingWindow
    trained_at: pd.Timestamp  # session the model was fitted for
    trained_through: pd.Timestamp  # date of the newest labelled row it used
    train_rows: int
    importance: pd.Series  # share of total split gain per feature
    contribution: pd.Series  # mean |contribution| to the raw score per feature
    validation: dict = field(default_factory=dict)

    def raw_probability(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def probability(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probability of ending in the top quantile."""
        raw = self.raw_probability(X)
        if self.calibrator is None:
            return raw
        return self.calibrator.predict(raw)

    def score(self, X: np.ndarray) -> np.ndarray:
        """Calibrated probability minus the entry threshold: non-negative means
        "would be bought if ranked high enough". The tiny multiple of the raw
        score only orders stocks the calibrator gives the same probability."""
        raw = self.raw_probability(X)
        calibrated = raw if self.calibrator is None else self.calibrator.predict(raw)
        return calibrated + _TIE_BREAK * raw - self.threshold


def _importances(model: LGBMClassifier, X: np.ndarray, names: list[str], sample: int, seed: int):
    booster = model.booster_
    gain = np.asarray(booster.feature_importance(importance_type="gain"), dtype=float)
    total = gain.sum()
    importance = pd.Series(gain / total if total > 0 else gain, index=names)
    if len(X) > sample:
        X = X[np.random.default_rng(seed).choice(len(X), size=sample, replace=False)]
    # LightGBM's own SHAP values: one column per feature plus the expected value
    contributions = booster.predict(X, pred_contrib=True)[:, :-1]
    return importance, pd.Series(np.abs(contributions).mean(axis=0), index=names)


def fit_pooled_model(
    dataset: PooledDataset, position: int, config: PooledConfig
) -> FittedPooledModel | None:
    """Model for a decision at calendar `position`, or None with too little data."""
    window = training_window(position, config)
    if window is None:
        return None
    train = dataset.rows(window.train_first, window.train_last)
    labelled = ~np.isnan(dataset.label[train])
    X = dataset.X[train][labelled]
    y = dataset.label[train][labelled]
    if len(np.unique(dataset.position[train][labelled])) < config.min_train_sessions:
        return None
    if len(np.unique(y)) < 2:
        return None

    params = {**DEFAULT_LGBM_PARAMS, "random_state": config.seed, **config.lgbm_params}
    model = LGBMClassifier(**params)
    model.fit(X, y)
    importance, contribution = _importances(
        model, X, dataset.features, config.contribution_sample, config.seed
    )

    calibrator = None
    threshold = 0.0
    validation: dict = {}
    if window.validation_first is not None:
        rows = dataset.rows(window.validation_first, window.validation_last)
        known = ~np.isnan(dataset.label[rows])
        label = dataset.label[rows][known]
        if len(label) and len(np.unique(label)) == 2:
            raw = model.predict_proba(dataset.X[rows][known])[:, 1]
            rel_return = dataset.rel_return[rows][known]
            dates = dataset.position[rows][known]
            calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            calibrator.fit(raw, label)
            calibrated = calibrator.predict(raw)
            choice = choose_threshold(
                calibrated + _TIE_BREAK * raw,
                rel_return,
                dates,
                top_k=config.top_k,
                round_trip_cost=config.round_trip_cost,
                quantiles=config.threshold_quantiles,
                min_picks=config.min_picks,
            )
            if config.use_threshold:
                threshold = choice.threshold
            validation = {
                **evaluate_scores(
                    raw, rel_return, dates,
                    top_k=config.top_k, round_trip_cost=config.round_trip_cost,
                ),
                "rows": int(len(label)),
                "base_rate": float(label.mean()),
                "brier_raw": brier_score(raw, label),
                "brier_calibrated": brier_score(calibrated, label),
                "reliability": reliability_table(calibrated, label),
                "threshold": choice.threshold,
                "threshold_applied": bool(config.use_threshold and choice.threshold > 0),
                "threshold_net_return": choice.net_return,
                "all_picks_net_return": choice.baseline_net_return,
                "threshold_picks": choice.picks,
                "all_picks": choice.baseline_picks,
            }

    return FittedPooledModel(
        model=model,
        calibrator=calibrator,
        threshold=threshold,
        features=list(dataset.features),
        horizon=config.horizon,
        window=window,
        trained_at=dataset.calendar[position],
        trained_through=dataset.calendar[window.last_label_position],
        train_rows=int(len(y)),
        importance=importance,
        contribution=contribution,
        validation=validation,
    )


class PooledModelScoreProvider:
    """Walk-forward scores of the pooled model; a `ScoreProvider` for
    `RankedUniverseStrategy`.

    `scores_on(session)` returns, for that session's universe members, the
    calibrated probability of a top-quantile forward return minus the entry
    threshold chosen at the last retrain. Ranking by it is ranking by
    probability; `RankedUniverseStrategy(min_score=0)` then keeps a top-ranked
    name out when its probability is under the threshold. With
    `use_threshold=False`, or when no threshold beat trading every pick on
    validation, the threshold is zero and nothing is blocked.

    The model is refitted when `retrain_every` sessions have passed since the
    last fit, on rows whose label window had ended by the session being
    decided. Every fit is kept in `fits`, and every session's probabilities in
    `history`, for the report.
    """

    def __init__(self, dataset: PooledDataset, config: PooledConfig | None = None):
        self.dataset = dataset
        self.config = config or PooledConfig()
        self.fits: list[FittedPooledModel] = []
        self.history: dict[pd.Timestamp, pd.DataFrame] = {}
        self._fit_position: int | None = None
        self._scores: dict[int, pd.Series | None] = {}

    def _position(self, session: pd.Timestamp) -> int:
        """Calendar position of the last session on or before `session` (-1 if none)."""
        return int(self.dataset.calendar.searchsorted(pd.Timestamp(session), side="right")) - 1

    def _model_for(self, position: int) -> FittedPooledModel | None:
        due = (
            self._fit_position is None
            or position < self._fit_position
            or position - self._fit_position >= self.config.retrain_every
        )
        if due:
            fitted = fit_pooled_model(self.dataset, position, self.config)
            if fitted is None:
                # Too little history: no scores until a model can be trained.
                # A model that already exists stays in use.
                return self.fits[-1] if self.fits and position >= self._fit_position else None
            self.fits.append(fitted)
            self._fit_position = position
        return self.fits[-1]

    def scores_on(self, session: pd.Timestamp) -> pd.Series | None:
        position = self._position(session)
        if position < 0:
            return None
        if position in self._scores:
            return self._scores[position]
        scores = None
        model = self._model_for(position)
        rows = self.dataset.rows(position, position)
        if model is not None and rows.stop > rows.start:
            X = self.dataset.X[rows]
            tickers = self.dataset.index.get_level_values("ticker")[rows]
            scores = pd.Series(model.score(X), index=tickers, name="score")
            self.history[self.dataset.calendar[position]] = pd.DataFrame(
                {"score": scores.to_numpy(), "probability": model.probability(X)}, index=tickers
            )
        self._scores[position] = scores
        return scores
