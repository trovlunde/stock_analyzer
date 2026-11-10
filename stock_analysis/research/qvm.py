"""Quality / Value / Momentum (QVM) sector-relative screener.

Scores stocks 0–100 within each sector (InvestWiser-style). Super stocks
clear a green threshold on all three factors.

This is the snapshot screen: it reads today's figures from `yfinance.info`, so
it cannot be backtested. The factor definitions live in
`stock_analysis.factors.qvm` and are shared with the point-in-time strategy
(`stock_analysis.factors.qvm_strategy`).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from stock_analysis.factors.prices import momentum_12_1
from stock_analysis.factors.qvm import (
    DEFAULT_MIN_SECTOR_SIZE,
    DEFAULT_WINSOR_LIMITS,
    MOMENTUM_METRICS,
    SCORE_COLUMNS,
    score_factors,
)

# yfinance .info key for 52-week change
_INFO_MOMENTUM_KEY = "52WeekChange"
# Used for momentum only when no price history was supplied: it has no skip
# month, so it includes the short-term reversal that 12-1 momentum leaves out.
FALLBACK_MOMENTUM = "fiftyTwoWeekChange"

DEFAULT_MIN_FACTOR_SCORE = 67.0


def _column(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """numerator / denominator where the denominator is positive, else NaN."""
    return numerator / denominator.where(denominator > 0)


def _inverse(multiple: pd.Series) -> pd.Series:
    """Yield from a multiple (1 / P/E). A negative multiple stays negative, so a
    loss-making stock lands at the bottom instead of looking cheapest."""
    return 1.0 / multiple.where(multiple != 0)


def info_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add the shared factor metrics, derived from `yfinance.info`-style columns.

    A metric column that is already present is left alone. Value metrics are
    yields: from per-share earnings and price where available, otherwise from
    the inverse of the multiple. `.info` has no total assets, so gross
    profitability, accruals and earnings stability are only scored when the
    caller supplies those columns.
    """
    out = df.copy()
    price = _column(out, "price")
    market_cap = _column(out, "market_cap")

    derived = {
        "earnings_yield": _ratio(_column(out, "trailingEps"), price).fillna(
            _inverse(_column(out, "trailingPE"))
        ),
        "forward_earnings_yield": _ratio(_column(out, "forwardEps"), price).fillna(
            _inverse(_column(out, "forwardPE"))
        ),
        "fcf_yield": _ratio(_column(out, "freeCashflow"), market_cap),
        "ebitda_ev": _ratio(_column(out, "ebitda"), _column(out, "enterpriseValue")),
        "book_to_price": _inverse(_column(out, "priceToBook")),
        "roe": _column(out, "returnOnEquity"),
        "profit_margin": _column(out, "profitMargins"),
        "operating_margin": _column(out, "operatingMargins"),
        # Negative equity gives a negative ratio, which would rank as least levered
        "debt_to_equity": _column(out, "debtToEquity").where(lambda s: s >= 0),
    }
    for name, values in derived.items():
        if name not in out.columns:
            out[name] = values

    if FALLBACK_MOMENTUM not in out.columns and _INFO_MOMENTUM_KEY in out.columns:
        out[FALLBACK_MOMENTUM] = out[_INFO_MOMENTUM_KEY]
    return out


def score_qvm(
    df: pd.DataFrame,
    *,
    sector_col: str = "sector",
    min_sector_size: int = DEFAULT_MIN_SECTOR_SIZE,
    winsor_limits: tuple[float, float] | None = DEFAULT_WINSOR_LIMITS,
) -> pd.DataFrame:
    """Add quality_score, value_score, momentum_score, qvm_score (0–100).

    Sectors with fewer than ``min_sector_size`` rows get NaN factor scores.
    Momentum is ``momentum_12_1`` when that column has values (see
    ``collect_qvm_rows(price_data=...)``); otherwise the 52-week change
    (``fiftyTwoWeekChange`` or ``52WeekChange``).
    """
    if df.empty:
        out = df.copy()
        for col in SCORE_COLUMNS:
            out[col] = pd.Series(dtype=float)
        return out

    out = info_metrics(df)
    has_12_1 = "momentum_12_1" in out.columns and out["momentum_12_1"].notna().any()
    return score_factors(
        out,
        sector_col=sector_col,
        min_sector_size=min_sector_size,
        winsor_limits=winsor_limits,
        momentum_metrics=MOMENTUM_METRICS if has_12_1 else (FALLBACK_MOMENTUM,),
    )


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


def collect_qvm_rows(
    stocks,
    *,
    min_market_cap: float = 1e9,
    price_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Collect QVM input metrics from a yfinance Tickers-like object.

    ``price_data`` ({ticker: daily frame with a ``Close`` column}) adds 12-1
    month momentum; without it the screen falls back to Yahoo's 52-week change.
    """
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
                row = {
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
                    "trailingEps": info.get("trailingEps"),
                    "forwardEps": info.get("forwardEps"),
                    "freeCashflow": info.get("freeCashflow"),
                    "ebitda": info.get("ebitda"),
                    "enterpriseValue": info.get("enterpriseValue"),
                    "fiftyTwoWeekChange": info.get(_INFO_MOMENTUM_KEY),
                }
                prices = (price_data or {}).get(ticker)
                if prices is not None and "Close" in prices.columns:
                    row["momentum_12_1"] = momentum_12_1(prices["Close"])
                rows.append(row)
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
    price_data: dict[str, pd.DataFrame] | None = None,
) -> list[dict[str, Any]]:
    """Collect → score → optionally filter to super stocks."""
    raw = collect_qvm_rows(stocks, min_market_cap=min_market_cap, price_data=price_data)
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
