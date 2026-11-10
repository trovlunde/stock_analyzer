"""Monthly-rebalanced QVM ranking, for the backtester and as a watchlist generator.

`QvmScoreProvider` implements the `ScoreProvider` protocol of
`trading/signals/ranked.py`: it scores the whole universe from prices and
point-in-time fundamentals, and `RankedUniverseStrategy` turns the ranks into
BUY / HOLD / SELL per ticker.
"""

import math
from collections.abc import Mapping
from datetime import date
from typing import Protocol

import pandas as pd

from stock_analysis.trading.signals.ranked import RankedUniverseStrategy

from .prices import momentum_12_1, split_factor, unadjusted_close
from .qvm import (
    DEFAULT_MIN_SECTOR_SIZE,
    DEFAULT_WINSOR_LIMITS,
    MARGIN_OF_SAFETY_METRIC,
    SCORE_COLUMNS,
    VALUE_METRICS,
    fundamental_metrics,
    score_factors,
)

ALL_SECTORS = "All"
UNKNOWN_SECTOR = "Unknown"
WATCHLIST_COLUMNS = ("ticker", "sector", *SCORE_COLUMNS)


class FundamentalsSource(Protocol):
    """What the provider needs from `PointInTimeFundamentals` (or a test fake)."""

    def as_of(self, ticker: str, when: date | pd.Timestamp, lag_days: int | None = None):
        """Snapshot with `.values`, `.period_ends` and `.net_income_quarters` as
        known on `when`, or None when nothing had been filed."""
        ...


class QvmScoreProvider:
    """QVM score per ticker, recomputed on the first session of each month.

    Between rebalances `scores_on` returns the scores of the month's first
    session, so holdings change at most once a month. Everything behind a score
    was available at that session's close: prices up to and including it, and
    filings made at least `lag_days` earlier.

    `price_data` is `{ticker: OHLCV frame}` with a tz-naive daily index, as
    returned by `backtest.benchmarks.load_price_data`. When the frames carry
    yfinance's `Dividends` and `Stock Splits` columns they are used to turn the
    adjusted close back into the traded price for market value; momentum uses
    the adjusted close. `sectors` maps ticker to sector; None ranks the universe
    as one group, and a ticker missing from the mapping is ranked among the
    other unknowns.
    """

    def __init__(
        self,
        price_data: Mapping[str, pd.DataFrame],
        fundamentals: FundamentalsSource,
        sectors: Mapping[str, str] | None = None,
        *,
        lag_days: int = 1,
        min_sector_size: int = DEFAULT_MIN_SECTOR_SIZE,
        winsor_limits: tuple[float, float] | None = DEFAULT_WINSOR_LIMITS,
        include_margin_of_safety: bool = False,
        max_fundamentals_age_days: int = 550,
        max_price_gap_days: int = 7,
    ):
        self.price_data = price_data
        self.fundamentals = fundamentals
        self.sectors = sectors
        self.lag_days = lag_days
        self.min_sector_size = min_sector_size
        self.winsor_limits = winsor_limits
        self.include_margin_of_safety = include_margin_of_safety
        self.max_fundamentals_age_days = max_fundamentals_age_days
        self.max_price_gap_days = max_price_gap_days

        calendar = pd.DatetimeIndex([])
        for df in price_data.values():
            calendar = calendar.union(df.index)
        self.calendar = calendar
        self._traded_close: dict[str, pd.Series] = {}
        self._monthly: dict[tuple[int, int], pd.Series | None] = {}

    def rebalance_session(self, session: pd.Timestamp) -> pd.Timestamp | None:
        """First session of `session`'s month; None if the data starts later than that."""
        session = pd.Timestamp(session)
        position = self.calendar.searchsorted(session.normalize().replace(day=1), side="left")
        if position >= len(self.calendar) or self.calendar[position] > session:
            return None
        return self.calendar[position]

    def scores_on(self, session: pd.Timestamp) -> pd.Series | None:
        session = pd.Timestamp(session)
        rebalance = self.rebalance_session(session)
        if rebalance is None:
            return None
        month = (rebalance.year, rebalance.month)
        if month not in self._monthly:
            scores = self.table_on(rebalance)["qvm_score"].dropna()
            self._monthly[month] = scores if not scores.empty else None
        return self._monthly[month]

    def _market_cap(self, ticker: str, snapshot, position: int) -> float:
        shares = snapshot.values.get("shares_outstanding")
        counted_on = snapshot.period_ends.get("shares_outstanding")
        if shares is None or counted_on is None:
            return float("nan")
        df = self.price_data[ticker]
        if ticker not in self._traded_close:
            self._traded_close[ticker] = unadjusted_close(df)
        bar = df.index[position]
        # A count reported before a split is on the old share basis
        shares = shares * split_factor(df, counted_on, bar)
        return shares * float(self._traded_close[ticker].iloc[position])

    def _is_current(self, snapshot, session: pd.Timestamp) -> bool:
        """False once a company has stopped filing: its last accounts say nothing about today."""
        ends = [end for name, end in snapshot.period_ends.items() if name != "shares_outstanding"]
        if not ends:
            return False
        return (session - max(ends)).days <= self.max_fundamentals_age_days

    def table_on(self, session: pd.Timestamp) -> pd.DataFrame:
        """Metrics and scores for every ticker trading at `session`, indexed by ticker.

        Uses only bars up to and including `session` and fundamentals known then.
        """
        session = pd.Timestamp(session)
        rows: dict[str, dict] = {}
        for ticker, df in self.price_data.items():
            bars = df.index.searchsorted(session, side="right")
            if bars == 0:
                continue
            if (session - df.index[bars - 1]).days > self.max_price_gap_days:
                continue  # stopped trading (delisted, or the data ends)
            if self.sectors is None:
                sector = ALL_SECTORS
            else:
                sector = self.sectors.get(ticker) or UNKNOWN_SECTOR
            row: dict = {
                "sector": sector,
                "momentum_12_1": momentum_12_1(df["Close"].iloc[:bars]),
            }
            snapshot = self.fundamentals.as_of(ticker, session, self.lag_days)
            if snapshot is not None and self._is_current(snapshot, session):
                row.update(
                    fundamental_metrics(
                        snapshot.values,
                        self._market_cap(ticker, snapshot, bars - 1),
                        net_income_quarters=snapshot.net_income_quarters,
                        sector=None if self.sectors is None else self.sectors.get(ticker),
                    )
                )
            rows[ticker] = row

        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "ticker"
        if table.empty:
            return score_factors(pd.DataFrame(columns=["sector"]))
        value_metrics = VALUE_METRICS
        if self.include_margin_of_safety:
            value_metrics = (*VALUE_METRICS, MARGIN_OF_SAFETY_METRIC)
        return score_factors(
            table,
            min_sector_size=self.min_sector_size,
            winsor_limits=self.winsor_limits,
            value_metrics=value_metrics,
            require_all_factors=True,
        )


def build_qvm_strategy(
    price_data: Mapping[str, pd.DataFrame],
    fundamentals: FundamentalsSource,
    sectors: Mapping[str, str] | None = None,
    *,
    top_fraction: float = 0.10,
    exit_buffer: float = 0.5,
    lag_days: int = 1,
    min_sector_size: int = DEFAULT_MIN_SECTOR_SIZE,
    winsor_limits: tuple[float, float] | None = DEFAULT_WINSOR_LIMITS,
    include_margin_of_safety: bool = False,
) -> RankedUniverseStrategy:
    """Strategy that buys the top `top_fraction` of the universe by QVM score, rebalanced monthly.

    A holding is sold only once it falls more than `exit_buffer` x top-k places
    below the top k (0.5 with a top ten: sold below rank 15), so names hovering
    around the cut-off are not traded back and forth every month. The fraction
    is taken of the tickers in `price_data`; tickers that cannot be scored
    (too little history, no fundamentals, sector too small) are left out of the
    ranking, which `RankedUniverseStrategy` treats as a SELL.
    """
    provider = QvmScoreProvider(
        price_data,
        fundamentals,
        sectors,
        lag_days=lag_days,
        min_sector_size=min_sector_size,
        winsor_limits=winsor_limits,
        include_margin_of_safety=include_margin_of_safety,
    )
    top_k = max(1, round(len(price_data) * top_fraction))
    exit_rank = top_k + math.ceil(top_k * exit_buffer)
    return RankedUniverseStrategy(provider, top_k=top_k, exit_rank=exit_rank)


def qvm_watchlist(
    provider: QvmScoreProvider,
    *,
    session: pd.Timestamp | None = None,
    top: int = 20,
) -> pd.DataFrame:
    """The `top` best QVM names at `session` (default: the latest session in the data).

    Same scoring as the backtested strategy, but computed at `session` itself
    rather than at the start of its month, so a watchlist made today reflects today.
    Columns: ticker, sector, quality_score, value_score, momentum_score, qvm_score.
    """
    if session is None:
        if provider.calendar.empty:
            return pd.DataFrame(columns=list(WATCHLIST_COLUMNS))
        session = provider.calendar[-1]
    table = provider.table_on(session).dropna(subset=["qvm_score"])
    table = table.sort_values("qvm_score", ascending=False, kind="stable").head(top)
    return table.reset_index()[list(WATCHLIST_COLUMNS)]
