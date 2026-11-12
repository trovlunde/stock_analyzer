"""Feature panel for the pooled cross-sectional model.

One row per (date, ticker). Stock features are turned into a percentile rank
among that date's universe members, so the model learns "this stock is in the
top tenth by six-month momentum today" rather than a level that means different
things in 2009 and 2021. Market context features (VIX, rates, index trend) are
the same for every ticker on a date and are left as levels: they tell the model
what kind of market it is in, not which stock to prefer.

No look-ahead: every feature of date t is computed from bars up to and
including t (trailing windows only), and the ranks of date t use only that
date's row. `tests/ai/test_cross_sectional_features.py` checks this by cutting
the data off after t.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

RANKED_FEATURES = (
    "mom_21",
    "mom_63",
    "mom_126",
    "mom_252",
    "mom_12_1",
    "rev_1",
    "rev_5",
    "vol_21",
    "vol_63",
    "vol_regime",
    "volume_regime",
    "dist_52w_high",
    "beta_126",
    "rs_market_63",
    "rs_market_126",
    "rs_sector_63",
)
CONTEXT_FEATURES = (
    "vix_level",
    "vix_chg_21",
    "tnx_level",
    "tnx_chg_63",
    "mkt_ret_63",
    "mkt_above_200dma",
)
EARNINGS_FEATURE = "days_since_earnings"

# Names under which the builder looks up the context series
VIX = "^VIX"
TNX = "^TNX"


def wide_field(price_data: Mapping[str, pd.DataFrame], column: str) -> pd.DataFrame:
    """One column of every ticker's frame side by side (session x ticker), on the
    union of all sessions. A ticker has NaN where it did not trade."""
    series = {ticker: df[column] for ticker, df in price_data.items() if column in df}
    wide = pd.DataFrame(series).sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    return wide


def member_mask(
    close: pd.DataFrame, membership=None
) -> pd.DataFrame:
    """Boolean (session x ticker): takes part in that session's ranks and labels.

    A ticker must have a close that session, and be an index member then when
    `membership` (an `IndexMembership`) is given.
    """
    mask = close.notna()
    if membership is not None:
        mask &= membership.member_matrix(close.index, close.columns)
    return mask


def rank_among_members(values: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
    """Percentile rank (0-1, highest value = 1) within each session's members.
    Non-members and missing values stay NaN and do not count towards the ranks."""
    return values.where(members).rank(axis=1, pct=True)


def _rolling_beta(returns: pd.DataFrame, market: pd.Series, window: int) -> pd.DataFrame:
    """Trailing regression slope of each stock's daily return on the market's."""
    market_wide = pd.DataFrame(
        np.repeat(market.to_numpy()[:, None], returns.shape[1], axis=1),
        index=returns.index,
        columns=returns.columns,
    ).where(returns.notna())
    minimum = window // 2
    mean_x = returns.rolling(window, min_periods=minimum).mean()
    mean_y = market_wide.rolling(window, min_periods=minimum).mean()
    covariance = (returns * market_wide).rolling(window, min_periods=minimum).mean() - mean_x * mean_y
    variance = (market_wide**2).rolling(window, min_periods=minimum).mean() - mean_y**2
    return covariance / variance.where(variance > 0)


def stock_features(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    market_close: pd.Series | None = None,
    sectors: Mapping[str, str] | None = None,
    members: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Raw (unranked) stock features as {name: session x ticker frame}.

    `rs_market_*` is the stock's return minus beta times the market's return
    over the window. The plain excess return (stock minus market) would rank
    exactly like the stock's own momentum, because the market's return is the
    same number for every stock on a date. `rs_sector_63` is the 63-session
    return minus the mean of the members in the same sector; NaN when the
    sector is unknown.
    """
    members = close.notna() if members is None else members
    daily = close.pct_change(fill_method=None)
    features: dict[str, pd.DataFrame] = {}
    for sessions in (21, 63, 126, 252):
        features[f"mom_{sessions}"] = close / close.shift(sessions) - 1
    # Twelve months skipping the latest one, which tends to reverse
    features["mom_12_1"] = close.shift(21) / close.shift(252) - 1
    features["rev_1"] = daily
    features["rev_5"] = close / close.shift(5) - 1

    annualise = math.sqrt(TRADING_DAYS_PER_YEAR)  # daily std -> annual
    features["vol_21"] = daily.rolling(21).std() * annualise
    features["vol_63"] = daily.rolling(63).std() * annualise
    vol_126 = daily.rolling(126).std() * annualise
    features["vol_regime"] = features["vol_21"] / vol_126.where(vol_126 > 0)

    dollar_volume = close * volume.reindex_like(close)
    long_volume = dollar_volume.rolling(126).mean()
    features["volume_regime"] = dollar_volume.rolling(21).mean() / long_volume.where(long_volume > 0)
    features["dist_52w_high"] = close / close.rolling(252, min_periods=126).max() - 1

    nan = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    if market_close is not None:
        market = market_close.reindex(close.index).ffill()
        beta = _rolling_beta(daily, market.pct_change(fill_method=None), 126)
        features["beta_126"] = beta
        for sessions in (63, 126):
            market_return = market / market.shift(sessions) - 1
            features[f"rs_market_{sessions}"] = features[f"mom_{sessions}"].sub(
                beta.mul(market_return, axis=0)
            )
    else:
        features["beta_126"] = nan
        features["rs_market_63"] = nan
        features["rs_market_126"] = nan

    relative = nan.copy()
    if sectors:
        momentum = features["mom_63"].where(members)
        by_sector: dict[str, list[str]] = {}
        for ticker in close.columns:
            sector = sectors.get(ticker)
            if sector:
                by_sector.setdefault(sector, []).append(ticker)
        for tickers in by_sector.values():
            block = momentum[tickers]
            relative[tickers] = block.sub(block.mean(axis=1), axis=0)
    features["rs_sector_63"] = relative
    return features


def context_features(
    sessions: pd.DatetimeIndex,
    market_close: pd.Series | None,
    context: Mapping[str, pd.DataFrame] | None,
) -> pd.DataFrame:
    """Market-wide features per session (same for every ticker, not ranked).

    Each series is aligned to `sessions` by carrying its last known value
    forward, never backward.
    """

    def aligned(series: pd.Series | None) -> pd.Series:
        if series is None or series.empty:
            return pd.Series(np.nan, index=sessions)
        series = series[~series.index.duplicated(keep="last")].sort_index()
        return series.reindex(series.index.union(sessions)).ffill().reindex(sessions)

    def close_of(name: str) -> pd.Series | None:
        frame = (context or {}).get(name)
        return None if frame is None or frame.empty else frame["Close"]

    vix = aligned(close_of(VIX))
    tnx = aligned(close_of(TNX))
    market = aligned(market_close)
    table = pd.DataFrame(index=sessions)
    table["vix_level"] = vix
    table["vix_chg_21"] = vix / vix.shift(21) - 1
    table["tnx_level"] = tnx
    table["tnx_chg_63"] = tnx - tnx.shift(63)
    table["mkt_ret_63"] = market / market.shift(63) - 1
    average = market.rolling(200).mean()
    table["mkt_above_200dma"] = (market > average).astype(float).where(average.notna())
    return table


def days_since_earnings(
    sessions: pd.DatetimeIndex, tickers: Iterable[str], earnings_dates: Mapping[str, Iterable]
) -> pd.DataFrame:
    """Calendar days since each ticker's latest earnings date on or before the
    session; NaN before the first known date or when none are known."""
    table = pd.DataFrame(np.nan, index=sessions, columns=list(tickers))
    for ticker in table.columns:
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(list(earnings_dates.get(ticker, ())))))
        if dates.empty:
            continue
        if dates.tz is not None:
            dates = dates.tz_localize(None)
        dates = dates.normalize()
        position = dates.searchsorted(sessions, side="right") - 1
        known = position >= 0
        latest = dates[np.where(known, position, 0)]
        table[ticker] = np.where(known, (sessions - latest).days, np.nan)
    return table


def build_feature_panel(
    price_data: Mapping[str, pd.DataFrame],
    market: pd.DataFrame | None = None,
    context: Mapping[str, pd.DataFrame] | None = None,
    sectors: Mapping[str, str] | None = None,
    earnings_dates: Mapping[str, Iterable] | None = None,
    *,
    membership=None,
    start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Features indexed by (date, ticker), one row per universe member per session.

    `price_data` is `{ticker: OHLCV frame}` with a tz-naive daily index.
    `market` is the price history of the market proxy (SPY); `context` maps
    `^VIX` / `^TNX` to their history. With `membership` (an `IndexMembership`)
    only that date's index members get a row and take part in the ranks;
    without it every ticker with a close does. `start` drops rows before that
    date after the features have been computed, so the first rows kept still
    have their full trailing windows.

    `days_since_earnings` is only added when `earnings_dates` is given: free
    earnings calendars cover a few years at most, and a column that is missing
    for most of the history would mostly teach the model which rows are recent.
    """
    close = wide_field(price_data, "Close")
    volume = wide_field(price_data, "Volume")
    members = member_mask(close, membership)
    market_close = None if market is None or market.empty else market["Close"]

    raw = stock_features(close, volume, market_close, sectors, members)
    columns = {name: rank_among_members(raw[name], members) for name in RANKED_FEATURES}
    if earnings_dates is not None:
        columns[EARNINGS_FEATURE] = days_since_earnings(close.index, close.columns, earnings_dates)

    keep = members if start is None else members[members.index >= pd.Timestamp(start)]
    rows, cols = np.nonzero(keep.to_numpy())
    offset = len(members) - len(keep)
    index = pd.MultiIndex.from_arrays(
        [keep.index[rows], keep.columns[cols]], names=["date", "ticker"]
    )
    panel = pd.DataFrame(
        {name: frame.to_numpy()[rows + offset, cols] for name, frame in columns.items()},
        index=index,
    )
    market_table = context_features(close.index, market_close, context)
    for name in CONTEXT_FEATURES:
        panel[name] = market_table[name].to_numpy()[rows + offset]
    return panel


def feature_names(panel: pd.DataFrame) -> list[str]:
    """Model input columns of a panel, in a fixed order."""
    names = [*RANKED_FEATURES, *CONTEXT_FEATURES]
    if EARNINGS_FEATURE in panel.columns:
        names.append(EARNINGS_FEATURE)
    return names
