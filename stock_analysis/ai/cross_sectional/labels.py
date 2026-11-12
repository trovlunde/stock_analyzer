"""Labels for the pooled model: did the stock beat its universe over the next sessions.

Next-day direction of one stock is close to noise. Whether a stock ends up in
the best fifth of its universe over the next two weeks is a relative question:
the market's own move, which nobody predicts from price features, drops out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HORIZON = 10
TOP_QUANTILE = 0.2
LABEL_COLUMNS = ("fwd_return", "rel_return", "label")


def forward_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """`Close[t + horizon] / Close[t] - 1` on the shared session calendar.

    NaN for the last `horizon` sessions and wherever either close is missing,
    so a stock that stops trading inside the window has no label rather than a
    made-up one.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least one session")
    return close.shift(-horizon) / close - 1


def build_labels(
    close: pd.DataFrame,
    members: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    top_quantile: float = TOP_QUANTILE,
) -> pd.DataFrame:
    """Labels indexed by (date, ticker) for every member row.

    `close` and `members` are session x ticker (see `features.wide_field` and
    `features.member_mask`). Columns:

    - `fwd_return`: the stock's return over the next `horizon` sessions.
    - `rel_return`: that minus the equal-weight mean of the date's members.
    - `label`: 1.0 when the stock's forward return is in the top `top_quantile`
      of that date's members, 0.0 otherwise.

    All three are NaN while the forward window is not complete. The label of
    date t uses closes up to t + horizon: a model deciding on session s may only
    train on rows with t + horizon <= s (see `model.training_window`).
    """
    forward = forward_returns(close, horizon).where(members)
    relative = forward.sub(forward.mean(axis=1), axis=0)
    rank = forward.rank(axis=1, pct=True)
    label = (rank > 1 - top_quantile).astype(float).where(forward.notna())

    rows, cols = np.nonzero(members.to_numpy())
    index = pd.MultiIndex.from_arrays(
        [members.index[rows], members.columns[cols]], names=["date", "ticker"]
    )
    return pd.DataFrame(
        {
            "fwd_return": forward.to_numpy()[rows, cols],
            "rel_return": relative.to_numpy()[rows, cols],
            "label": label.to_numpy()[rows, cols],
        },
        index=index,
    )
