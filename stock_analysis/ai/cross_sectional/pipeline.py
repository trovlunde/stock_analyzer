"""From price history to the arrays the pooled model trains on."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from .features import build_feature_panel, member_mask, wide_field
from .labels import DEFAULT_HORIZON, build_labels
from .model import PooledDataset, make_dataset


def build_pooled_dataset(
    price_data: Mapping[str, pd.DataFrame],
    *,
    market: pd.DataFrame | None = None,
    context: Mapping[str, pd.DataFrame] | None = None,
    sectors: Mapping[str, str] | None = None,
    membership=None,
    horizon: int = DEFAULT_HORIZON,
    start: pd.Timestamp | None = None,
) -> PooledDataset:
    """Feature panel and labels for `price_data`, ready for `fit_pooled_model`.

    Features and labels use the same member mask, so a stock is ranked and
    labelled against exactly the names it competes with on that date. `start`
    drops earlier rows (after the features were computed on the full history)
    to keep the arrays small.
    """
    close = wide_field(price_data, "Close")
    members = member_mask(close, membership)
    panel = build_feature_panel(
        price_data, market, context, sectors, membership=membership, start=start
    )
    labels = build_labels(close, members, horizon)
    return make_dataset(panel, labels, close.index)
