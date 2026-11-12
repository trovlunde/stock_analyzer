"""Champion/challenger check for retraining the pooled model.

A freshly trained model (the challenger) replaces the saved one (the champion)
only when it ranks the most recent validation window at least as well. Both are
scored on the same rows: the challenger's validation fold, which the
challenger's trees have not seen.

The check leans towards keeping the champion, on purpose and by construction:

- A champion trained earlier than one validation window ago has not seen the
  window either, and the comparison is fair.
- A champion trained more recently (the normal case when retraining monthly
  with a one-year validation window) had part of this window in its own
  training or calibration data, so its numbers on it are flattered. The
  outcome records how many of the window's sessions that applies to.

A wrongly kept champion costs a month of a slightly stale model; a wrongly
promoted challenger can be a broken one. `force=True` skips the check.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pandas as pd

from .model import (
    FittedPooledModel,
    PooledConfig,
    PooledDataset,
    evaluate_scores,
    fit_pooled_model,
)

MODEL_TYPE = "champion"
METRICS = ("rank_ic", "top_k_net_return")


@dataclass
class RetrainOutcome:
    promoted: bool
    reason: str
    challenger: FittedPooledModel | None
    challenger_metrics: dict | None = None
    champion_metrics: dict | None = None
    # Sessions of the comparison window the champion had already trained or
    # calibrated on (0 = a clean comparison)
    champion_seen_sessions: int = 0


def window_metrics(
    model: FittedPooledModel, dataset: PooledDataset, first: int, last: int, config: PooledConfig
) -> dict:
    """Rank IC and net top-k relative return of `model` on calendar positions
    [first, last] of `dataset`."""
    rows = dataset.rows(first, last)
    return evaluate_scores(
        model.score(dataset.X[rows]),
        dataset.rel_return[rows],
        dataset.position[rows],
        top_k=config.top_k,
        round_trip_cost=config.round_trip_cost,
    )


def at_least_as_good(challenger: dict, champion: dict) -> bool:
    """True when the challenger is not worse on any metric. A metric the
    champion has no value for cannot be lost."""
    for name in METRICS:
        ours, theirs = challenger.get(name), champion.get(name)
        if theirs is None:
            continue
        if ours is None or ours < theirs:
            return False
    return True


def model_metadata(
    model: FittedPooledModel, config: PooledConfig, metrics: dict, universe: str
) -> dict:
    validation = {
        key: value for key, value in model.validation.items() if key != "reliability"
    }
    reliability = model.validation.get("reliability")
    if reliability is not None:
        validation["reliability"] = reliability.to_dict("records")
    return {
        "features": list(model.features),
        "horizon": model.horizon,
        "threshold": model.threshold,
        "trained_at": model.trained_at.isoformat(),
        "trained_through": model.trained_through.isoformat(),
        "train_rows": model.train_rows,
        "validation": validation,
        "comparison_metrics": metrics,
        "importance": model.importance.to_dict(),
        "contribution": model.contribution.to_dict(),
        "universe": universe,
        "config": dataclasses.asdict(config),
    }


def retrain_with_champion_check(
    dataset: PooledDataset,
    config: PooledConfig,
    manager,
    model_name: str,
    *,
    universe: str = "",
    force: bool = False,
) -> RetrainOutcome:
    """Train a challenger on everything up to the last session of `dataset` and
    save it through `manager` (a `ModelManager`) if it may replace the champion."""
    if config.validation_sessions <= 0:
        raise ValueError("The champion/challenger check needs a validation fold")
    challenger = fit_pooled_model(dataset, len(dataset.calendar) - 1, config)
    if challenger is None or not challenger.validation:
        return RetrainOutcome(False, "too little data to train a challenger", None)
    window = challenger.window
    challenger_metrics = window_metrics(
        challenger, dataset, window.validation_first, window.validation_last, config
    )

    champion = manager.load_model(model_name, MODEL_TYPE, bypass_cache=True)
    champion_metrics = None
    seen = 0
    if champion is None:
        promoted, reason = True, "no champion saved yet"
    elif list(champion.features) != list(dataset.features) or champion.horizon != config.horizon:
        # Its scores answer a different question; it cannot be compared
        promoted = force
        reason = (
            "the champion was trained with other features or another horizon and "
            "cannot be compared" + ("; replaced because of --force" if force else "; kept")
        )
    else:
        champion_metrics = window_metrics(
            champion, dataset, window.validation_first, window.validation_last, config
        )
        # The champion's newest row is `trained_through`; that row's label used
        # closes up to `horizon` sessions later
        newest_row = int(
            dataset.calendar.searchsorted(pd.Timestamp(champion.trained_through), side="right")
        ) - 1
        seen_last = min(newest_row + champion.horizon, window.validation_last)
        seen = max(0, seen_last - window.validation_first + 1)
        if at_least_as_good(challenger_metrics, champion_metrics):
            promoted, reason = True, "the challenger is at least as good on both metrics"
        elif force:
            promoted, reason = True, "the challenger is worse; replaced because of --force"
        else:
            promoted, reason = False, "the champion is better on at least one metric; kept"

    if promoted:
        saved = manager.save_model(
            challenger,
            model_name,
            MODEL_TYPE,
            metadata=model_metadata(challenger, config, challenger_metrics, universe),
        )
        if not saved:
            return RetrainOutcome(
                False, "saving the challenger failed; the champion is unchanged",
                challenger, challenger_metrics, champion_metrics, seen,
            )
    return RetrainOutcome(
        promoted, reason, challenger, challenger_metrics, champion_metrics, seen
    )
