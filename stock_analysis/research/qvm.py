"""Quality / Value / Momentum (QVM) sector-relative screener.

Scores stocks 0–100 within each sector (InvestWiser-style). Super stocks
clear a green threshold on all three factors.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

# Higher-better metrics contribute directly; lower-better are inverted after rank.
QUALITY_HIGHER = ("returnOnEquity", "profitMargins", "operatingMargins")
QUALITY_LOWER = ("debtToEquity",)
VALUE_LOWER = ("trailingPE", "forwardPE", "priceToBook")
MOMENTUM_HIGHER = ("fiftyTwoWeekChange",)

# yfinance .info key for 52-week change
_INFO_MOMENTUM_KEY = "52WeekChange"

DEFAULT_MIN_FACTOR_SCORE = 67.0
DEFAULT_MIN_SECTOR_SIZE = 4


def _percentile_rank(series: pd.Series, *, higher_is_better: bool) -> pd.Series:
    """Rank within group as 0–100. NaNs stay NaN.

    Lower-is-better metrics are negated before ranking so the cheapest /
    least-levered names land near 100.
    """
    values = series if higher_is_better else -series
    return values.rank(pct=True, na_option="keep") * 100


def _factor_score(df: pd.DataFrame, columns: list[tuple[str, bool]]) -> pd.Series:
    """Mean of available metric percentile columns; all-NaN → NaN."""
    parts = []
    for col, higher_is_better in columns:
        if col not in df.columns:
            continue
        parts.append(_percentile_rank(df[col], higher_is_better=higher_is_better))
    if not parts:
        return pd.Series(float("nan"), index=df.index)
    return pd.concat(parts, axis=1).mean(axis=1, skipna=True)


def score_qvm(
    df: pd.DataFrame,
    *,
    sector_col: str = "sector",
    min_sector_size: int = DEFAULT_MIN_SECTOR_SIZE,
) -> pd.DataFrame:
    """Add quality_score, value_score, momentum_score, qvm_score (0–100).

    Sectors with fewer than ``min_sector_size`` rows get NaN factor scores.
    Momentum column may be ``fiftyTwoWeekChange`` or ``52WeekChange``.
    """
    if df.empty:
        out = df.copy()
        for col in ("quality_score", "value_score", "momentum_score", "qvm_score"):
            out[col] = pd.Series(dtype=float)
        return out

    out = df.copy()
    if "fiftyTwoWeekChange" not in out.columns and "52WeekChange" in out.columns:
        out["fiftyTwoWeekChange"] = out["52WeekChange"]

    quality_cols = [(c, True) for c in QUALITY_HIGHER] + [
        (c, False) for c in QUALITY_LOWER
    ]
    value_cols = [(c, False) for c in VALUE_LOWER]
    momentum_cols = [(c, True) for c in MOMENTUM_HIGHER]

    out["quality_score"] = float("nan")
    out["value_score"] = float("nan")
    out["momentum_score"] = float("nan")
    out["qvm_score"] = float("nan")

    for _sector, sector_df in out.groupby(sector_col, dropna=False):
        if len(sector_df) < min_sector_size:
            continue
        idx = sector_df.index
        q = _factor_score(sector_df, quality_cols)
        v = _factor_score(sector_df, value_cols)
        m = _factor_score(sector_df, momentum_cols)
        out.loc[idx, "quality_score"] = q
        out.loc[idx, "value_score"] = v
        out.loc[idx, "momentum_score"] = m
        out.loc[idx, "qvm_score"] = pd.concat([q, v, m], axis=1).mean(
            axis=1, skipna=True
        )

    return out


def filter_super_stocks(
    df: pd.DataFrame,
    *,
    min_factor_score: float = DEFAULT_MIN_FACTOR_SCORE,
    min_qvm_score: float | None = None,
) -> list[dict[str, Any]]:
    """Keep rows where all three factors meet the green threshold.

    Sorted by ``qvm_score`` descending.
    """
    if df.empty:
        return []

    required = ("quality_score", "value_score", "momentum_score", "qvm_score")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"score_qvm columns missing: {missing}")

    mask = (
        (df["quality_score"] >= min_factor_score)
        & (df["value_score"] >= min_factor_score)
        & (df["momentum_score"] >= min_factor_score)
    )
    if min_qvm_score is not None:
        mask = mask & (df["qvm_score"] >= min_qvm_score)

    filtered = df.loc[mask].sort_values("qvm_score", ascending=False)
    return filtered.to_dict(orient="records")


def collect_qvm_rows(stocks, *, min_market_cap: float = 1e9) -> pd.DataFrame:
    """Collect QVM input metrics from a yfinance Tickers-like object."""
    import contextlib
    import io
    import logging

    rows: list[dict[str, Any]] = []
    tickers = getattr(stocks, "tickers", {}) or {}
    skipped = 0

    log = logging.getLogger("yfinance")
    prev_level = log.level
    log.setLevel(logging.CRITICAL)
    try:
        for ticker, stock in tickers.items():
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    info = stock.info
                current_price = info.get("currentPrice")
                market_cap = info.get("marketCap")
                sector = info.get("sector")
                if not all([current_price, market_cap, sector]):
                    skipped += 1
                    continue
                if market_cap < min_market_cap:
                    continue
                rows.append(
                    {
                        "ticker": ticker,
                        "company_name": info.get("longName", "N/A"),
                        "sector": sector,
                        "price": current_price,
                        "market_cap": market_cap,
                        "returnOnEquity": info.get("returnOnEquity"),
                        "profitMargins": info.get("profitMargins"),
                        "operatingMargins": info.get("operatingMargins"),
                        "debtToEquity": info.get("debtToEquity"),
                        "trailingPE": info.get("trailingPE"),
                        "forwardPE": info.get("forwardPE"),
                        "priceToBook": info.get("priceToBook"),
                        "fiftyTwoWeekChange": info.get(_INFO_MOMENTUM_KEY),
                    }
                )
            except Exception:
                skipped += 1
                continue
    finally:
        log.setLevel(prev_level)

    if skipped:
        print(f"Skipped {skipped} tickers without usable Yahoo fundamentals")
    return pd.DataFrame(rows)


def find_qvm_stocks(
    stocks,
    *,
    min_market_cap: float = 1e9,
    min_factor_score: float = DEFAULT_MIN_FACTOR_SCORE,
    min_qvm_score: float | None = None,
    super_only: bool = True,
    min_sector_size: int = DEFAULT_MIN_SECTOR_SIZE,
) -> list[dict[str, Any]]:
    """Collect → score → optionally filter to super stocks."""
    raw = collect_qvm_rows(stocks, min_market_cap=min_market_cap)
    if raw.empty:
        return []
    scored = score_qvm(raw, min_sector_size=min_sector_size)
    if super_only:
        return filter_super_stocks(
            scored,
            min_factor_score=min_factor_score,
            min_qvm_score=min_qvm_score,
        )
    return scored.sort_values("qvm_score", ascending=False).to_dict(orient="records")
