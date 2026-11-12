"""Pooled cross-sectional movement model: one LightGBM model for a whole
universe, predicting which stocks will beat it over the next sessions.

- `features.build_feature_panel`: per-date ranked stock features plus market context.
- `labels.build_labels`: top-quintile forward return relative to the universe.
- `model.PooledModelScoreProvider`: walk-forward training, calibration, threshold.
- `report.pooled_report_sections`: importances, rank IC per year, calibration.
- `champion`: champion/challenger comparison for scheduled retraining.
"""

from .features import build_feature_panel
from .labels import build_labels
from .model import (
    FittedPooledModel,
    PooledConfig,
    PooledDataset,
    PooledModelScoreProvider,
    fit_pooled_model,
    make_dataset,
)
from .pipeline import build_pooled_dataset

__all__ = [
    "FittedPooledModel",
    "PooledConfig",
    "PooledDataset",
    "PooledModelScoreProvider",
    "build_feature_panel",
    "build_labels",
    "build_pooled_dataset",
    "fit_pooled_model",
    "make_dataset",
]
