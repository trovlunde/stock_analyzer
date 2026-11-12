"""Walk-forward versions of the ML movement signals, for backtesting only.

A saved model has already seen the period being backtested. This strategy
instead trains on the bars it is given (the engine never passes bars after the
session being decided) and retrains on a schedule, the way a model retrained in
production would have behaved at each point in history.
"""

import contextlib
import io
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from stock_analysis.ai.technical_analysis.ensemble_classifier import EnsembleClassifier
from stock_analysis.ai.technical_analysis.movement_classification import (
    train_classifier_single_stock,
)
from stock_analysis.ai.technical_analysis.movement_inference import predict_latest
from stock_analysis.ai.technical_analysis.prepare_classification_data import (
    prepare_classification_data,
)
from stock_analysis.trading.signals.base import SignalResult, SignalStrategy

WALK_FORWARD_ML = "ml_walk_forward"
WALK_FORWARD_ENSEMBLE = "ensemble_walk_forward"

_LABEL_TO_ACTION = {"positive": "BUY", "negative": "SELL", "neutral": "HOLD"}


def _default_classifier():
    return RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)


class LabelledEnsembleClassifier:
    """`EnsembleClassifier` behind the string labels the movement pipeline uses.

    The ensemble works on integer classes (0 negative, 1 neutral, 2 positive) and
    prints while fitting; the pipeline trains on and predicts the label names.
    """

    classes_ = np.array(["negative", "neutral", "positive"], dtype=object)

    def __init__(self, **ensemble_kwargs):
        self.ensemble = EnsembleClassifier(**ensemble_kwargs)

    def fit(self, X, y):
        codes = pd.Index(self.classes_).get_indexer(np.asarray(y))
        if (codes < 0).any():
            raise ValueError(f"Labels must be among {list(self.classes_)}")
        with contextlib.redirect_stdout(io.StringIO()):
            self.ensemble.fit(X, codes)
        return self

    def predict(self, X):
        return self.classes_[self.ensemble.predict(X)]

    def predict_proba(self, X):
        return self.ensemble.predict_proba(X)


class WalkForwardMlStrategy:
    def __init__(
        self,
        retrain_every: int = 63,
        min_train_rows: int = 252,
        threshold: float = 0.005,
        classifier_factory: Callable[[], object] = _default_classifier,
    ):
        self.retrain_every = retrain_every
        self.min_train_rows = min_train_rows
        self.threshold = threshold
        self.classifier_factory = classifier_factory
        # ticker -> (model, scaler, last bar the model was trained through)
        self._models: dict[str, tuple[object, object, pd.Timestamp]] = {}

    def _train(self, df: pd.DataFrame, use_extra_features: bool):
        prepared = prepare_classification_data(
            df, threshold=self.threshold, use_extra_features=use_extra_features
        )
        # The latest bar has no label yet (it needs tomorrow's close), so the
        # newest labelled row is always one bar behind the data.
        if prepared["Target"].notna().sum() < self.min_train_rows:
            return None
        model, scaler, _ = train_classifier_single_stock(
            prepared,
            use_extra_features=use_extra_features,
            threshold=self.threshold,
            classifier=self.classifier_factory(),
            eval_split=0,
            verbose=False,
        )
        return model, scaler

    def evaluate(self, df: pd.DataFrame, ticker: str, params: dict) -> SignalResult:
        use_extra_features = bool(params.get("use_extra_features", False))
        confidence_threshold = float(params.get("ml_confidence_threshold", 0.55))

        # Data preparation prints progress on every call; keep backtest output readable
        with contextlib.redirect_stdout(io.StringIO()):
            cached = self._models.get(ticker)
            bars_since_training = (
                int((df.index > cached[2]).sum()) if cached is not None else None
            )
            if cached is None or bars_since_training >= self.retrain_every:
                trained = self._train(df, use_extra_features)
                if trained is not None:
                    cached = (*trained, df.index[-1])
                    self._models[ticker] = cached
            if cached is None:
                return SignalResult(
                    action="HOLD",
                    reason=f"not enough history to train a model for {ticker}",
                    metadata={},
                )
            model, scaler, trained_through = cached
            try:
                label, confidence = predict_latest(df, model, scaler, use_extra_features)
            except ValueError as exc:
                return SignalResult(
                    action="HOLD", reason=f"ML prediction unavailable: {exc}", metadata={}
                )

        metadata = {
            "label": label,
            "confidence": confidence,
            "threshold": confidence_threshold,
            "trained_through": str(trained_through.date()),
        }
        if confidence < confidence_threshold:
            return SignalResult(
                action="HOLD",
                reason=f"ML confidence {confidence:.0%} below threshold {confidence_threshold:.0%}",
                metadata=metadata,
            )
        return SignalResult(
            action=_LABEL_TO_ACTION[label],
            reason=f"ML predicts {label} ({confidence:.0%} confidence)",
            metadata=metadata,
        )


def walk_forward_strategies(retrain_every: int = 63) -> dict[str, SignalStrategy]:
    """The backtest-only strategies by name, each with its own model cache.

    They are kept out of the production registry: retraining inside the daily job
    is not how a live model would be run.
    """
    return {
        WALK_FORWARD_ML: WalkForwardMlStrategy(retrain_every=retrain_every),
        WALK_FORWARD_ENSEMBLE: WalkForwardMlStrategy(
            retrain_every=retrain_every, classifier_factory=LabelledEnsembleClassifier
        ),
    }