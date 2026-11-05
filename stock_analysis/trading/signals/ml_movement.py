import pandas as pd

from stock_analysis.ai.helpers import get_features
from stock_analysis.ai.model_manager import ModelManager
from stock_analysis.ai.technical_analysis.movement_inference import predict_latest

from .base import SignalResult

_LABEL_TO_ACTION = {"positive": "BUY", "negative": "SELL", "neutral": "HOLD"}


class MlMovementStrategy:
    def __init__(self, model_manager: ModelManager | None = None):
        self.model_manager = model_manager or ModelManager()

    def evaluate(self, df: pd.DataFrame, ticker: str, params: dict) -> SignalResult:
        model_name = f"{ticker}_movement"
        use_extra_features = bool(params.get("use_extra_features", False))
        confidence_threshold = float(params.get("ml_confidence_threshold", 0.55))

        model = self.model_manager.load_model(model_name, "daily")
        scaler = self.model_manager.load_model(f"{model_name}_scaler", "daily")
        if model is None and scaler is None:
            return SignalResult(
                action="HOLD",
                reason=f"no trained ML model for {ticker}",
                metadata={},
            )
        if model is None or scaler is None:
            missing = "model" if model is None else "scaler"
            return SignalResult(
                action="HOLD",
                reason=f"ML {missing} missing for {ticker} (partially-saved model?)",
                metadata={},
            )

        # A scaler fitted on a bare array with the same column count would accept
        # today's features and mis-predict silently, so require the training-time
        # feature list to match exactly.
        model_metadata = self.model_manager.get_model_metadata(model_name, "daily") or {}
        trained_features = model_metadata.get("feature_columns")
        if trained_features is None:
            return SignalResult(
                action="HOLD",
                reason=f"ML model for {ticker} predates feature tracking — retrain it",
                metadata={},
            )
        if list(trained_features) != list(get_features(use_extra_features)):
            return SignalResult(
                action="HOLD",
                reason=f"ML model for {ticker} was trained on a different feature set — retrain it",
                metadata={},
            )

        try:
            label, confidence = predict_latest(df, model, scaler, use_extra_features)
        except ValueError as exc:
            return SignalResult(
                action="HOLD",
                reason=f"ML prediction unavailable: {exc}",
                metadata={},
            )

        metadata = {
            "label": label,
            "confidence": confidence,
            "threshold": confidence_threshold,
        }

        if confidence < confidence_threshold:
            return SignalResult(
                action="HOLD",
                reason=f"ML confidence {confidence:.0%} below threshold {confidence_threshold:.0%}",
                metadata=metadata,
            )

        action = _LABEL_TO_ACTION[label]
        return SignalResult(
            action=action,
            reason=f"ML predicts {label} ({confidence:.0%} confidence)",
            metadata=metadata,
        )
