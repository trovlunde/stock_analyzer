"""Report sections for a walk-forward run of the pooled model.

Everything here is computed after the run from what the provider recorded: the
probabilities it gave on each session, joined with the returns that followed.
Those returns were not known when the scores were made; they are only used to
judge them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import (
    PooledModelScoreProvider,
    brier_score,
    rank_ic_by_date,
    reliability_table,
    top_k_picks,
)


def walk_forward_predictions(provider: PooledModelScoreProvider) -> pd.DataFrame:
    """One row per (date, ticker) scored during the run: score, calibrated
    probability, and the realised label / relative return (NaN when the forward
    window runs past the end of the data)."""
    if not provider.history:
        return pd.DataFrame(columns=["score", "probability", "label", "rel_return"])
    frame = pd.concat(provider.history, names=["date", "ticker"]).sort_index()
    dataset = provider.dataset
    realised = pd.DataFrame(
        {"label": dataset.label, "rel_return": dataset.rel_return}, index=dataset.index
    )
    return frame.join(realised, how="left")


def importance_table(provider: PooledModelScoreProvider) -> pd.DataFrame:
    """Mean over retrains of each feature's gain share and mean |contribution|,
    and how much the gain share moved between retrains."""
    gain = pd.DataFrame([fit.importance for fit in provider.fits])
    contribution = pd.DataFrame([fit.contribution for fit in provider.fits])
    table = pd.DataFrame(
        {
            "gain_share": gain.mean(),
            "gain_share_std": gain.std(ddof=0),
            "mean_abs_contribution": contribution.mean(),
        }
    )
    return table.sort_values("gain_share", ascending=False)


def yearly_rank_ic(predictions: pd.DataFrame, top_k: int) -> pd.DataFrame:
    """Per calendar year: mean daily rank IC, its t-statistic over days, and the
    mean relative forward return of the top-k picks (before costs)."""
    known = predictions.dropna(subset=["rel_return"])
    if known.empty:
        return pd.DataFrame(columns=["year", "days", "rank_ic", "t_stat", "top_k_rel_return"])
    dates = known.index.get_level_values("date")
    ic = rank_ic_by_date(known["score"], known["rel_return"], dates).dropna()
    picks = top_k_picks(known["score"].to_numpy(), dates.to_numpy(), top_k)
    picked = known["rel_return"][picks]
    picked_by_year = picked.groupby(picked.index.get_level_values("date").year).mean()
    rows = []
    groups = [(str(year), values) for year, values in ic.groupby(pd.DatetimeIndex(ic.index).year)]
    groups.append(("All", ic))
    for name, values in groups:
        std = values.std(ddof=1) if len(values) > 1 else float("nan")
        rows.append(
            {
                "year": name,
                "days": len(values),
                "rank_ic": float(values.mean()),
                # Daily ICs of overlapping forward windows are autocorrelated, so
                # this overstates significance; it is a rough guide only
                "t_stat": float(values.mean() / std * np.sqrt(len(values))) if std else float("nan"),
                "top_k_rel_return": float(picked.mean()) if name == "All"
                else float(picked_by_year.get(int(name), float("nan"))),
            }
        )
    return pd.DataFrame(rows)


def _mean(values: list) -> float | None:
    values = [v for v in values if v is not None and not pd.isna(v)]
    return float(np.mean(values)) if values else None


def _fmt(value, spec: str = ".4f") -> str:
    return "—" if value is None or pd.isna(value) else format(value, spec)


def pooled_report_sections(provider: PooledModelScoreProvider) -> list[str]:
    """Markdown lines: setup, feature importance, rank IC per year, calibration."""
    config = provider.config
    lines = ["## Pooled model (ml_pooled)", ""]
    if not provider.fits:
        return lines + ["- The model was never trained (too little history before the run)."]

    first, last = provider.fits[0], provider.fits[-1]
    lines += [
        f"- Label: top quintile of {config.horizon}-session forward return among the "
        f"date's universe members; one LightGBM model for all tickers",
        f"- {len(provider.fits)} fits, every {config.retrain_every} sessions, on up to "
        f"{config.train_sessions} labelled sessions "
        f"({config.validation_sessions} of them held out for calibration and the threshold)",
        f"- First fit for {first.trained_at.date()} used labels through "
        f"{first.trained_through.date()} ({first.train_rows:,} training rows); last fit for "
        f"{last.trained_at.date()} through {last.trained_through.date()} "
        f"({last.train_rows:,} rows)",
        "",
        "### Feature importance (mean over fits)",
        "",
        "| Feature | Gain share | Std over fits | Mean abs. contribution |",
        "|---|---:|---:|---:|",
    ]
    for name, row in importance_table(provider).iterrows():
        lines.append(
            f"| {name} | {row.gain_share:.1%} | {row.gain_share_std:.1%} | "
            f"{row.mean_abs_contribution:.4f} |"
        )

    predictions = walk_forward_predictions(provider)
    lines += [
        "",
        "### Rank IC per year (walk-forward)",
        "",
        "Spearman correlation per day between the model's score and the relative "
        f"return over the following {config.horizon} sessions, averaged. The t-statistic "
        "treats days as independent, which overlapping windows are not: read it as an "
        "upper bound. `Top-k rel. return` is the mean relative forward return of each "
        f"day's {config.top_k} best-scored names, before costs.",
        "",
        "| Year | Days | Rank IC | t-stat | Top-k rel. return |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in yearly_rank_ic(predictions, config.top_k).itertuples(index=False):
        lines.append(
            f"| {row.year} | {row.days} | {_fmt(row.rank_ic)} | {_fmt(row.t_stat, '.1f')} | "
            f"{_fmt(row.top_k_rel_return, '.2%')} |"
        )

    validations = [fit.validation for fit in provider.fits if fit.validation]
    lines += ["", "### Calibration and threshold", ""]
    if validations:
        applied = [v for v in validations if v["threshold_applied"]]
        chosen = [v for v in validations if v["threshold"] > 0]
        lines += [
            f"- Validation folds (calibrator fitted on the same fold, so optimistic): "
            f"Brier {_fmt(_mean([v['brier_calibrated'] for v in validations]))} calibrated vs "
            f"{_fmt(_mean([v['brier_raw'] for v in validations]))} raw; base rate "
            f"{_fmt(_mean([v['base_rate'] for v in validations]), '.1%')}; rank IC "
            f"{_fmt(_mean([v['rank_ic'] for v in validations]))}; top-{config.top_k} net "
            f"relative return per holding period "
            f"{_fmt(_mean([v['top_k_net_return'] for v in validations]), '.2%')}",
            f"- Threshold: a candidate beat trading every pick in {len(chosen)} of "
            f"{len(validations)} folds"
            + (
                f" (median threshold {np.median([v['threshold'] for v in chosen]):.3f}, "
                f"keeping {np.mean([v['threshold_picks'] / v['all_picks'] for v in chosen]):.0%} "
                f"of the picks on validation)"
                if chosen else ""
            )
            + f"; applied in {len(applied)} "
            f"({'thresholds on' if config.use_threshold else 'thresholds off for this run'})",
        ]
    else:
        lines.append("- No validation fold (validation_sessions = 0): probabilities are uncalibrated")

    known = predictions.dropna(subset=["label"])
    if not known.empty:
        probability = known["probability"].to_numpy()
        label = known["label"].to_numpy()
        base = float(label.mean())
        dates = known.index.get_level_values("date").to_numpy()
        picks = top_k_picks(known["score"].to_numpy(), dates, config.top_k)
        blocked = float((known["score"].to_numpy()[picks] < 0).mean()) if picks.any() else 0.0
        lines += [
            f"- Walk-forward (out of sample): Brier {_fmt(brier_score(probability, label))} "
            f"vs {base * (1 - base):.4f} for always predicting the base rate ({base:.1%}); "
            f"{blocked:.0%} of top-{config.top_k} picks were under the threshold",
            "",
            "Reliability by decile of predicted probability (walk-forward):",
            "",
            "| Decile | Predicted | Observed | Rows |",
            "|---|---:|---:|---:|",
        ]
        for row in reliability_table(probability, label).itertuples(index=False):
            lines.append(
                f"| {row.bin} | {row.predicted:.1%} | {row.observed:.1%} | {row.rows:,} |"
            )
    return lines
