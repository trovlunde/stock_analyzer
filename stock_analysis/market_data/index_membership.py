"""Point-in-time index membership, so a backtest universe is not picked with hindsight.

Training or testing on today's S&P 500 over the last twenty years only sees the
companies that survived and grew into the index. `IndexMembership` answers "who
was in the index on this date" from Wikipedia's "List of S&P 500 companies"
(and "Historical components of the S&P 500", where the change log now lives):
the table of current constituents (with the date each was added) and the table
of selected changes (date, ticker added, ticker removed). Starting from today's
members the changes are undone one by one, newest first.

What this does not fix, and `membership_coverage` puts a number on:

- Most removed companies were acquired or delisted and have no price history on
  Yahoo, so they are members here but cannot be traded or trained on.
- The change log is "selected changes": it is fairly complete for the last
  fifteen years or so and thin before that. Before `change_log_start` nothing
  is known about removals, so the member count falls well below 500.
- A company that changed ticker appears under its old ticker in the change log
  and its new one in the current table. The "date added" of the current table
  is used to place such a company; its time in the index under the old ticker
  before an earlier removal is lost.
"""

from __future__ import annotations

import bisect
import io
import random
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SP500_CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"
SP500_CACHE_KEY = "index_membership:sp500"
# Wikipedia asks for a descriptive agent; a browser string gets blocked more often
USER_AGENT = "stock-analysis-research/0.1"

CONSTITUENT_COLUMNS = ("ticker", "sector", "date_added")
CHANGE_COLUMNS = ("date", "added", "removed")


def normalise_ticker(raw: object) -> str | None:
    """Wikipedia symbol in Yahoo form (`BRK.B` -> `BRK-B`); None for an empty cell."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip().upper()
    if not text or text in {"NAN", "NONE", "—", "-"}:
        return None
    # Footnote markers such as "GOOG[4]" follow the symbol
    text = text.split("[")[0].split()[0]
    return text.replace(".", "-") or None


@dataclass(frozen=True)
class _Event:
    when: pd.Timestamp
    ticker: str
    joined: bool  # True = added to the index on `when`, False = removed


@dataclass
class IndexMembership:
    """Who was in an index on any date; build with `build_membership`."""

    current: frozenset[str]
    sectors: dict[str, str]
    # Oldest first; undoing every event after a date gives that date's members
    events: list[_Event] = field(default_factory=list)
    # First date in the change log (not counting "date added" of current members)
    change_log_start: pd.Timestamp | None = None
    # Change-log additions of a ticker that is not a member afterwards: usually
    # a later ticker change. They cannot be undone and are counted, not hidden.
    unmatched_additions: int = 0
    name: str = "sp500"

    def __post_init__(self) -> None:
        self.events = sorted(self.events, key=lambda event: event.when)
        self._event_dates = [event.when for event in self.events]
        # Membership just before each event date, built lazily newest first
        self._snapshots: dict[int, frozenset[str]] = {len(self.events): self.current}

    def _members_before_event(self, position: int) -> frozenset[str]:
        """Members while only `events[:position]` have happened."""
        if position in self._snapshots:
            return self._snapshots[position]
        known = min(p for p in self._snapshots if p > position)
        members = set(self._snapshots[known])
        for index in range(known - 1, position - 1, -1):
            event = self.events[index]
            if event.joined:
                members.discard(event.ticker)
            else:
                members.add(event.ticker)
            self._snapshots[index] = frozenset(members)
        return self._snapshots[position]

    def members_on(self, when: date | pd.Timestamp) -> set[str]:
        """Members at the close of `when`. A change takes effect on its own date:
        the added ticker is a member from that day, the removed one no longer."""
        when = pd.Timestamp(when).normalize()
        happened = bisect.bisect_right(self._event_dates, when)
        return set(self._members_before_event(happened))

    def all_tickers(self) -> list[str]:
        """Every ticker that was a member at some point, sorted."""
        return sorted(set(self.current) | {event.ticker for event in self.events})

    def sample(self, size: int | None, seed: int = 0) -> list[str]:
        """Random sample of all-time members (everything when `size` is None or
        larger than the list). Sampling from all-time members rather than from
        today's keeps a smaller universe free of hindsight."""
        tickers = self.all_tickers()
        if size is None or size >= len(tickers):
            return tickers
        return sorted(random.Random(seed).sample(tickers, size))

    def member_matrix(self, sessions: pd.DatetimeIndex, tickers: Iterable[str]) -> pd.DataFrame:
        """Boolean table (session x ticker): was the ticker a member that session."""
        tickers = list(tickers)
        matrix = pd.DataFrame(False, index=sessions, columns=tickers)
        if sessions.empty or not tickers:
            return matrix
        positions = [bisect.bisect_right(self._event_dates, s.normalize()) for s in sessions]
        values = matrix.to_numpy()
        column = {ticker: i for i, ticker in enumerate(tickers)}
        start = 0
        # Sessions between two events share one member set
        for row in range(1, len(positions) + 1):
            if row == len(positions) or positions[row] != positions[start]:
                members = self._members_before_event(positions[start])
                columns = [column[t] for t in members if t in column]
                values[start:row, columns] = True
                start = row
        return pd.DataFrame(values, index=sessions, columns=tickers)

    def dense_log_from(self, min_removals: int = 5) -> int | None:
        """First year from which every year has at least `min_removals` logged
        removals. The index loses 20-30 members in a normal year, so earlier
        years with only a handful logged have incomplete membership."""
        removals: dict[int, int] = {}
        for event in self.events:
            if not event.joined:
                removals[event.when.year] = removals.get(event.when.year, 0) + 1
        if not removals:
            return None
        dense = None
        for year in range(max(removals), min(removals) - 1, -1):
            if removals.get(year, 0) < min_removals:
                break
            dense = year
        return dense

    def describe(self) -> str:
        start = self.change_log_start.date() if self.change_log_start is not None else "n/a"
        return (
            f"{self.name}: {len(self.current)} current members, "
            f"{len(self.all_tickers())} all-time, change log from {start} "
            f"(at least 5 removals logged every year only from {self.dense_log_from()}), "
            f"{self.unmatched_additions} logged additions that could not be matched "
            f"(ticker changes)"
        )


def build_membership(
    constituents: pd.DataFrame, changes: pd.DataFrame, *, name: str = "sp500"
) -> IndexMembership:
    """Membership history from the two Wikipedia tables.

    `constituents`: one row per current member with `ticker`, `sector` and
    `date_added` (may be missing). `changes`: one row per logged change with
    `date`, `added` and `removed` (either ticker may be missing). Tickers are
    normalised to Yahoo form here.
    """
    current: set[str] = set()
    sectors: dict[str, str] = {}
    added_on: dict[str, pd.Timestamp] = {}
    for row in constituents.itertuples(index=False):
        ticker = normalise_ticker(row.ticker)
        if ticker is None:
            continue
        current.add(ticker)
        sector = getattr(row, "sector", None)
        if isinstance(sector, str) and sector.strip():
            sectors[ticker] = sector.strip()
        when = pd.to_datetime(getattr(row, "date_added", None), errors="coerce")
        if not pd.isna(when):
            added_on[ticker] = pd.Timestamp(when).normalize()

    events: list[_Event] = []
    logged_additions: set[str] = set()
    for row in changes.itertuples(index=False):
        when = pd.to_datetime(row.date, errors="coerce")
        if pd.isna(when):
            continue
        when = pd.Timestamp(when).normalize()
        added = normalise_ticker(row.added)
        removed = normalise_ticker(row.removed)
        if added is not None:
            events.append(_Event(when, added, True))
            logged_additions.add(added)
        if removed is not None:
            events.append(_Event(when, removed, False))
    change_log_start = min((event.when for event in events), default=None)

    # A current member whose addition is not in the change log (it joined before
    # the log starts, or under another ticker) still has a "date added"
    for ticker, when in added_on.items():
        if ticker not in logged_additions:
            events.append(_Event(when, ticker, True))

    # Count additions that the walk back cannot undo
    members = set(current)
    unmatched = 0
    for event in sorted(events, key=lambda e: e.when, reverse=True):
        if event.joined:
            if event.ticker in members:
                members.discard(event.ticker)
            else:
                unmatched += 1
        else:
            members.add(event.ticker)

    return IndexMembership(
        current=frozenset(current),
        sectors=sectors,
        events=events,
        change_log_start=change_log_start,
        unmatched_additions=unmatched,
        name=name,
    )


def _flat_columns(table: pd.DataFrame) -> list[str]:
    if isinstance(table.columns, pd.MultiIndex):
        return [" ".join(dict.fromkeys(str(p) for p in parts)).lower() for parts in table.columns]
    return [str(column).lower() for column in table.columns]


def parse_wikipedia_tables(html: str) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(constituents, changes) found in the page HTML, with the columns
    `build_membership` expects; None for a table the page does not have."""
    constituents = changes = None
    for table in pd.read_html(io.StringIO(html)):
        columns = _flat_columns(table)
        table = table.copy()
        table.columns = columns
        if constituents is None and "symbol" in columns and any("sector" in c for c in columns):
            sector = next(c for c in columns if "sector" in c)
            added = next((c for c in columns if "date added" in c), None)
            constituents = pd.DataFrame(
                {
                    "ticker": table["symbol"],
                    "sector": table[sector],
                    "date_added": table[added] if added else None,
                }
            )
        elif changes is None and any(c.startswith("added") for c in columns):
            when = next(c for c in columns if "date" in c)
            added = next(c for c in columns if c.startswith("added") and "ticker" in c)
            removed = next(c for c in columns if c.startswith("removed") and "ticker" in c)
            changes = pd.DataFrame(
                {"date": table[when], "added": table[added], "removed": table[removed]}
            )
    return constituents, changes


def _get(url: str) -> str:
    import requests

    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    return response.text


def fetch_sp500_tables(get=_get) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Current constituents and the change log.

    Until August 2026 both tables were on the list page; the change log then
    moved to its own article. The list page is read first and the second page
    only when the log is not there, so either layout works.
    """
    constituents, changes = parse_wikipedia_tables(get(SP500_URL))
    if changes is None:
        _, changes = parse_wikipedia_tables(get(SP500_CHANGES_URL))
    if constituents is None or changes is None:
        raise ValueError(
            "Could not find the S&P 500 constituents and changes tables on Wikipedia; "
            "the page layout may have changed"
        )
    return constituents, changes

def _to_cache_frame(constituents: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    """Both tables in one frame of strings (the cache stores one DataFrame per key)."""
    left = pd.DataFrame(
        {
            "kind": "constituent",
            "ticker": constituents["ticker"],
            "sector": constituents["sector"],
            "date": constituents["date_added"],
        }
    )
    right = pd.DataFrame(
        {"kind": "change", "ticker": changes["added"], "removed": changes["removed"],
         "date": changes["date"]}
    )
    frame = pd.concat([left, right], ignore_index=True)
    return frame.astype("string")


def _from_cache_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = frame.astype(object).where(frame.notna(), None)
    rows = frame[frame["kind"] == "constituent"]
    constituents = pd.DataFrame(
        {"ticker": rows["ticker"], "sector": rows["sector"], "date_added": rows["date"]}
    )
    rows = frame[frame["kind"] == "change"]
    changes = pd.DataFrame(
        {"date": rows["date"], "added": rows["ticker"], "removed": rows["removed"]}
    )
    return constituents, changes


def load_sp500_membership(
    cache_store=None, max_age: timedelta | None = timedelta(days=7), fetch=fetch_sp500_tables
) -> IndexMembership:
    """S&P 500 membership history, fetched from Wikipedia at most once per `max_age`."""
    if cache_store is None:
        from stock_analysis.storage import get_cache_store

        cache_store = get_cache_store()
    cached = cache_store.get(SP500_CACHE_KEY, max_age=max_age)
    if cached is not None and not cached.empty:
        return build_membership(*_from_cache_frame(cached))
    constituents, changes = fetch()
    cache_store.put(
        SP500_CACHE_KEY, _to_cache_frame(constituents, changes), metadata={"source": SP500_URL}
    )
    return build_membership(constituents, changes)


def membership_coverage(
    membership: IndexMembership,
    price_data: Mapping[str, pd.DataFrame],
    *,
    first_year: int,
    last_year: int,
    candidates: Iterable[str] | None = None,
    today: date | None = None,
) -> pd.DataFrame:
    """Per year: index members, how many were candidates, how many have prices.

    Members are counted at the end of each year (today for the current year).
    `candidates` are the tickers price data was asked for (a sample of all-time
    members); None means every member was a candidate. `with_prices` counts
    candidates with at least one bar in that year. The gap between `candidates`
    and `with_prices` is the survivorship bias that remains: members that the
    data source no longer knows.
    """
    today = today or date.today()
    candidate_set = None if candidates is None else set(candidates)
    years_with_bars = {
        ticker: set(df.index.year.unique()) for ticker, df in price_data.items() if len(df)
    }
    rows = []
    for year in range(first_year, last_year + 1):
        members = membership.members_on(min(date(year, 12, 31), today))
        asked = members if candidate_set is None else members & candidate_set
        with_prices = {t for t in asked if year in years_with_bars.get(t, ())}
        rows.append(
            {
                "year": year,
                "members": len(members),
                "candidates": len(asked),
                "with_prices": len(with_prices),
                "coverage": len(with_prices) / len(asked) if asked else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def format_coverage(membership: IndexMembership, coverage: pd.DataFrame) -> list[str]:
    """Markdown lines for the backtest report."""
    lines = [
        "## Universe: historical index membership",
        "",
        f"- {membership.describe()}",
        "- Members are counted at each year end. `Candidates` are the members among "
        "the tickers this run asked prices for; `With prices` those that have bars "
        "that year. Members without prices (mostly acquired or delisted companies) "
        "are missing from both the strategy and the benchmark: that is the "
        "survivorship bias left in this run.",
        "",
        "| Year | Members | Candidates | With prices | Coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in coverage.itertuples(index=False):
        share = "—" if pd.isna(row.coverage) else f"{row.coverage:.0%}"
        lines.append(
            f"| {row.year} | {row.members} | {row.candidates} | {row.with_prices} | {share} |"
        )
    return lines
