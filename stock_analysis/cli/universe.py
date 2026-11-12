"""Index-based universes for the CLIs: historical members, cached price history."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from stock_analysis.market_data import MarketDataProvider
from stock_analysis.market_data.index_membership import (
    IndexMembership,
    format_coverage,
    load_sp500_membership,
    membership_coverage,
)

INDEX_UNIVERSES = ("sp500",)
PRICE_CACHE_MAX_AGE = timedelta(days=1)
_PROGRESS_EVERY = 25


def price_cache_key(ticker: str, period: str) -> str:
    return f"prices:{period}:{ticker}"


def load_price_data_cached(
    tickers: list[str],
    provider: MarketDataProvider,
    period: str,
    cache_store=None,
    *,
    max_age: timedelta | None = PRICE_CACHE_MAX_AGE,
    progress: bool = True,
) -> dict[str, pd.DataFrame]:
    """Like `backtest.benchmarks.load_price_data`, but keeps each ticker's history
    in the cache for a day and reports progress.

    Hundreds of tickers take minutes to download one by one; a second run the
    same day reads them from disk. A ticker the provider has nothing for is
    cached as an empty frame, so delisted names are not asked for again either.
    `cache_store=None` downloads everything (used with test providers).
    """
    data: dict[str, pd.DataFrame] = {}
    downloaded = 0
    for number, ticker in enumerate(tickers, start=1):
        key = price_cache_key(ticker, period)
        df = cache_store.get(key, max_age=max_age) if cache_store is not None else None
        if df is None:
            try:
                df = provider.get_history(ticker, period=period)
            except Exception as error:  # one bad ticker must not end a long download
                print(f"  {ticker}: {error}", file=sys.stderr)
                df = None
            if df is None:
                df = pd.DataFrame()
            downloaded += 1
            if cache_store is not None:
                cache_store.put(key, df, metadata={"ticker": ticker, "period": period})
        if progress and (number % _PROGRESS_EVERY == 0 or number == len(tickers)):
            print(
                f"  prices: {number}/{len(tickers)} tickers "
                f"({len(data) + (0 if df.empty else 1)} with data, {downloaded} downloaded)",
                file=sys.stderr,
            )
        if df.empty:
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        data[ticker] = df
    return data


@dataclass
class IndexUniverse:
    membership: IndexMembership
    candidates: list[str]  # the (sampled) all-time members prices were asked for
    price_data: dict[str, pd.DataFrame]

    @property
    def sectors(self) -> dict[str, str]:
        return {t: s for t, s in self.membership.sectors.items() if t in self.price_data}

    def members_with_data(self, when: pd.Timestamp) -> dict[str, pd.DataFrame]:
        """Price data of the tickers that were members on `when`."""
        members = self.membership.members_on(when)
        return {t: df for t, df in self.price_data.items() if t in members}

    def coverage_lines(self, first_year: int, last_year: int) -> list[str]:
        coverage = membership_coverage(
            self.membership,
            self.price_data,
            first_year=first_year,
            last_year=last_year,
            candidates=self.candidates,
        )
        lines = format_coverage(self.membership, coverage)
        lines.insert(
            3,
            f"- Asked prices for {len(self.candidates)} of "
            f"{len(self.membership.all_tickers())} all-time members; "
            f"{len(self.price_data)} have data",
        )
        return lines


def load_index_universe(
    name: str,
    provider: MarketDataProvider,
    period: str,
    *,
    sample: int | None = None,
    seed: int = 0,
    membership: IndexMembership | None = None,
    cache_store=None,
    use_cache: bool = True,
) -> IndexUniverse:
    """Historical members of index `name` with their price history.

    `sample` draws that many tickers at random from the all-time members (not
    from today's, and not from those that happen to have data), so a smaller
    universe is not chosen with hindsight. `membership` and `cache_store` are
    injection points for tests; by default the membership comes from Wikipedia
    and prices are cached in the project's cache store.
    """
    if name not in INDEX_UNIVERSES:
        raise SystemExit(f"Unknown universe '{name}' (available: {', '.join(INDEX_UNIVERSES)})")
    if use_cache and cache_store is None:
        from stock_analysis.storage import get_cache_store

        cache_store = get_cache_store()
    if membership is None:
        membership = load_sp500_membership(cache_store) if use_cache else load_sp500_membership(
            _NoCache()
        )
    candidates = membership.sample(sample, seed)
    price_data = load_price_data_cached(
        candidates, provider, period, cache_store if use_cache else None
    )
    if not price_data:
        raise SystemExit(f"No price data for any member of '{name}'")
    return IndexUniverse(membership=membership, candidates=candidates, price_data=price_data)


class _NoCache:
    """Stands in for the cache store when caching is switched off."""

    def get(self, key, **kwargs):
        return None

    def put(self, key, df, **kwargs) -> None:
        return None
