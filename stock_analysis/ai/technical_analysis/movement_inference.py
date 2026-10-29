import pandas as pd

from stock_analysis.ai.helpers import get_features
from stock_analysis.ai.technical_analysis.prepare_classification_data import (
    prepare_classification_data,
)


def predict_latest(
    stock_data: pd.DataFrame,
    model,
    scaler,
    use_extra_features: bool = False,
) -> tuple[str, float]:
    """Predict the movement class for the most recent bar only.

    No plotting, no file writes, no training — safe to call from an unattended
    job, unlike get_recent_predictions()/full_analysis() in movement_classification.py
    which assume an interactive/notebook context.

    Returns (predicted_label, confidence) where confidence is the model's
    predicted probability for that label. Raises ValueError if the latest bar's
    features can't be computed (e.g. not enough history).
    """
    prepared = prepare_classification_data(
        stock_data, predict_weekly=False, use_extra_features=use_extra_features
    )
    feature_columns = get_features(use_extra_features)
    latest_features = prepared[feature_columns].iloc[[-1]]

    if latest_features.isna().any(axis=1).iloc[0]:
        raise ValueError("Insufficient history to compute features for the latest bar")

    scaled = scaler.transform(latest_features)
    label = model.predict(scaled)[0]
    probabilities = model.predict_proba(scaled)[0]
    class_index = list(model.classes_).index(label)
    confidence = float(probabilities[class_index])

    return label, confidence
