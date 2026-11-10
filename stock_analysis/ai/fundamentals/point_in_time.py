"""Point-in-time fundamentals: what was known about a company on a given date.

`yfinance.info` and the statement frames only show today's view of the accounts,
so a screen built on them cannot be backtested: it would use numbers that were
published (or restated) after the day being simulated. EDGAR's XBRL company
facts carry the date each value was *filed*, which is what makes an honest
"as of" answer possible.

The data handling works on a plain DataFrame of facts (`FACT_COLUMNS`) and has
no EDGAR dependency; only `fetch_edgar_facts` talks to the network.
"""

import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv

FACT_COLUMNS = ["concept", "tag", "period_start", "period_end", "filed", "value", "form"]

# Alternatives in order of preference: when one filing reports the same period
# under several tags, the first one listed here wins.
CONCEPT_TAGS: dict[str, tuple[str, ...]] = {
    "revenue": (
        "us-gaap:Revenues",
        "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
        "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax",
        "us-gaap:SalesRevenueNet",
        "us-gaap:SalesRevenueGoodsNet",
    ),
    "gross_profit": ("us-gaap:GrossProfit",),
    "cost_of_revenue": (
        "us-gaap:CostOfRevenue",
        "us-gaap:CostOfGoodsAndServicesSold",
        "us-gaap:CostOfGoodsSold",
    ),
    "operating_income": ("us-gaap:OperatingIncomeLoss",),
    "net_income": (
        "us-gaap:NetIncomeLoss",
        "us-gaap:NetIncomeLossAvailableToCommonStockholdersBasic",
        "us-gaap:ProfitLoss",
    ),
    "depreciation_amortization": (
        "us-gaap:DepreciationDepletionAndAmortization",
        "us-gaap:DepreciationAndAmortization",
        "us-gaap:DepreciationAmortizationAndAccretionNet",
        "us-gaap:Depreciation",
    ),
    "operating_cash_flow": (
        "us-gaap:NetCashProvidedByUsedInOperatingActivities",
        "us-gaap:NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ),
    "capex": (
        "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
        "us-gaap:PaymentsToAcquireProductiveAssets",
    ),
    "total_assets": ("us-gaap:Assets",),
    "total_equity": (
        "us-gaap:StockholdersEquity",
        "us-gaap:StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ),
    "total_debt_reported": ("us-gaap:DebtLongtermAndShorttermCombinedAmount",),
    "long_term_debt": (
        "us-gaap:LongTermDebtNoncurrent",
        "us-gaap:LongTermDebtAndCapitalLeaseObligations",
        "us-gaap:LongTermDebt",
    ),
    "short_term_debt": (
        "us-gaap:DebtCurrent",
        "us-gaap:LongTermDebtAndCapitalLeaseObligationsCurrent",
        "us-gaap:LongTermDebtCurrent",
        "us-gaap:ShortTermBorrowings",
    ),
    "cash": (
        "us-gaap:CashAndCashEquivalentsAtCarryingValue",
        "us-gaap:CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ),
    "shares_outstanding": (
        "dei:EntityCommonStockSharesOutstanding",
        "us-gaap:CommonStockSharesOutstanding",
    ),
    # Fallback share count for companies whose cover-page count is split by
    # share class and therefore missing from the company-facts feed.
    "weighted_shares": (
        "us-gaap:WeightedAverageNumberOfSharesOutstandingBasic",
        "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding",
    ),
}

FLOW_CONCEPTS = (
    "revenue",
    "gross_profit",
    "cost_of_revenue",
    "operating_income",
    "net_income",
    "depreciation_amortization",
    "operating_cash_flow",
    "capex",
)
BALANCE_CONCEPTS = (
    "total_assets",
    "total_equity",
    "total_debt_reported",
    "long_term_debt",
    "short_term_debt",
    "cash",
)
_SHARE_CONCEPTS = ("shares_outstanding", "weighted_shares")

_TAG_TO_CONCEPT = {tag: concept for concept, tags in CONCEPT_TAGS.items() for tag in tags}
_TAG_PRIORITY = {tag: rank for tags in CONCEPT_TAGS.values() for rank, tag in enumerate(tags)}

_QUARTER_DAYS = (75, 105)
_ANNUAL_DAYS = (350, 380)
# 52/53-week fiscal calendars move period boundaries by a few days
_BOUNDARY_TOLERANCE_DAYS = 7
# Periods older than this cannot contribute to a trailing year or to the
# earnings-stability history, so they are not worth deriving quarters from.
_HISTORY_YEARS = 5

DEFAULT_LAG_DAYS = 1
DEFAULT_HISTORY_QUARTERS = 12
CACHE_KEY_PREFIX = "edgar_pit:"


class EdgarIdentityMissing(RuntimeError):
    """EDGAR requires requests to identify the caller; none has been configured."""


@dataclass(frozen=True)
class FundamentalSnapshot:
    """The accounts of one company as they were known on `as_of`.

    `values` holds trailing-twelve-month sums for flow items (revenue,
    net_income, ...), the latest balance-sheet value for instant items, and the
    derived `ebitda`, `free_cash_flow` and `total_debt`. A value that was not
    known is absent, never zero.
    """

    as_of: pd.Timestamp
    values: dict[str, float]
    # Fiscal period end each value belongs to (the last quarter of a trailing year)
    period_ends: dict[str, pd.Timestamp] = field(default_factory=dict)
    # Most recent discrete quarters of net income, oldest first, indexed by quarter end
    net_income_quarters: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    latest_filed: pd.Timestamp | None = None

    def get(self, name: str) -> float:
        return self.values.get(name, float("nan"))


def empty_facts() -> pd.DataFrame:
    return normalize_facts(pd.DataFrame(columns=FACT_COLUMNS))


def normalize_facts(facts: pd.DataFrame) -> pd.DataFrame:
    """Coerce a facts frame to the column set and dtypes the functions below expect."""
    out = facts.reindex(columns=FACT_COLUMNS).copy()
    for column in ("period_start", "period_end", "filed"):
        out[column] = pd.to_datetime(out[column]).astype("datetime64[ns]")
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["concept", "period_end", "filed", "value"])
    return out.reset_index(drop=True)


def known_facts(
    facts: pd.DataFrame, as_of: date | pd.Timestamp, lag_days: int = DEFAULT_LAG_DAYS
) -> pd.DataFrame:
    """Facts that were public on `as_of`, one row per concept and period.

    A fact counts as known `lag_days` after its filing date (a filing made after
    the close cannot be traded on the same day). When a period was reported more
    than once, the latest filing known *by then* wins, so a restatement changes
    the answer only from its own filing date onwards.
    """
    cutoff = pd.Timestamp(as_of).normalize() - pd.Timedelta(lag_days, unit="D")
    known = facts[facts["filed"] <= cutoff]
    if known.empty:
        return known
    priority = known["tag"].map(_TAG_PRIORITY).fillna(len(_TAG_PRIORITY))
    known = known.assign(_priority=priority).sort_values(
        ["filed", "_priority"], ascending=[True, False], kind="stable"
    )
    known = known.drop_duplicates(["concept", "period_start", "period_end"], keep="last")
    return known.drop(columns="_priority")


Period = tuple[int, int, float]  # (first day, last day, value); days since 1970-01-01


def _day_numbers(column: pd.Series) -> np.ndarray:
    return column.to_numpy().astype("datetime64[D]").astype("int64")


def _timestamp(day: int) -> pd.Timestamp:
    return pd.Timestamp(np.datetime64(day, "D"))


def _is_length(start: int, end: int, bounds: tuple[int, int]) -> bool:
    return bounds[0] <= end - start + 1 <= bounds[1]


def _periods(durations: pd.DataFrame) -> list[Period]:
    """Rows with a start date (instants are not periods) as plain tuples."""
    rows = durations[durations["period_start"].notna()]
    return list(
        zip(
            _day_numbers(rows["period_start"]).tolist(),
            _day_numbers(rows["period_end"]).tolist(),
            rows["value"].astype(float).tolist(),
        )
    )


def _quarters(periods: list[Period]) -> list[Period]:
    """Single quarters, reported or derived, oldest first. See `discrete_quarters`."""
    quarters: dict[int, tuple[int, float]] = {}  # end -> (start, value)
    longer: list[Period] = []
    for start, end, value in periods:
        if _is_length(start, end, _QUARTER_DAYS):
            quarters[end] = (start, value)
        else:
            longer.append((start, end, value))

    # Shorter periods first, so a nine-month figure can build on the quarter
    # derived from the six-month one
    for start, end, value in sorted(longer, key=lambda item: (item[1], item[1] - item[0])):
        if end in quarters:
            continue
        cursor = start
        remainder = value
        while True:
            match = next(
                (
                    (q_end, q_value)
                    for q_end, (q_start, q_value) in quarters.items()
                    if abs(q_start - cursor) <= _BOUNDARY_TOLERANCE_DAYS and q_end < end
                ),
                None,
            )
            if match is None:
                break
            remainder -= match[1]
            cursor = match[0] + 1
        if cursor > start and _is_length(cursor, end, _QUARTER_DAYS):
            quarters[end] = (cursor, remainder)

    return sorted((start, end, value) for end, (start, value) in quarters.items())


def discrete_quarters(durations: pd.DataFrame) -> pd.DataFrame:
    """Single-quarter values from the mix of periods companies actually report.

    Income statements give three-month figures for Q1–Q3 but only the full year
    in the 10-K; cash-flow statements give year-to-date figures only. A quarter
    that was not reported on its own is therefore the longer period minus the
    quarters that came before it within that period (Q4 = year − Q1 − Q2 − Q3,
    Q2 = six months − Q1, and so on).

    `durations` has `period_start`, `period_end` and `value` for one concept,
    one row per period. Returns a frame indexed by quarter end with `start` and
    `value`, oldest first.
    """
    return _quarters_frame(_quarters(_periods(durations)))


def _quarters_frame(quarters: list[Period]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "start": [_timestamp(start) for start, _, _ in quarters],
            "value": [value for _, _, value in quarters],
        },
        index=pd.DatetimeIndex([_timestamp(end) for _, end, _ in quarters], name="period_end"),
    ).astype({"value": float})


def _last_four_contiguous(quarters: list[Period]) -> list[Period] | None:
    """The four latest quarters if they tile one year without gaps, else None."""
    if len(quarters) < 4:
        return None
    last = quarters[-4:]
    for (_, previous_end, _), (start, _, _) in zip(last[:-1], last[1:]):
        if abs(start - (previous_end + 1)) > _BOUNDARY_TOLERANCE_DAYS:
            return None
    if not _is_length(last[0][0], last[-1][1], _ANNUAL_DAYS):
        return None
    return last


def _trailing_twelve_months(
    periods: list[Period], quarters: list[Period]
) -> tuple[float, pd.Timestamp] | None:
    annual = [(end, value) for start, end, value in periods if _is_length(start, end, _ANNUAL_DAYS)]
    latest_annual = max(annual, default=None)
    last_four = _last_four_contiguous(quarters)
    if last_four is not None and (latest_annual is None or last_four[-1][1] > latest_annual[0]):
        return sum(value for _, _, value in last_four), _timestamp(last_four[-1][1])
    if latest_annual is not None:
        return latest_annual[1], _timestamp(latest_annual[0])
    return None


def trailing_twelve_months(durations: pd.DataFrame) -> tuple[float, pd.Timestamp] | None:
    """Sum over the latest twelve months and the date that year ends.

    Uses the four latest quarters when they are complete; otherwise the latest
    full fiscal year, which is older but never wrong.
    """
    periods = _periods(durations)
    return _trailing_twelve_months(periods, _quarters(periods))


def _latest_instant(rows: pd.DataFrame) -> tuple[float, pd.Timestamp] | None:
    if rows.empty:
        return None
    row = rows.loc[rows["period_end"].idxmax()]
    return float(row["value"]), row["period_end"]


def snapshot_as_of(
    facts: pd.DataFrame,
    as_of: date | pd.Timestamp,
    *,
    lag_days: int = DEFAULT_LAG_DAYS,
    history_quarters: int = DEFAULT_HISTORY_QUARTERS,
) -> FundamentalSnapshot | None:
    """Build the snapshot of one company's facts as known on `as_of`.

    None when nothing had been filed by then.
    """
    as_of = pd.Timestamp(as_of).normalize()
    known = known_facts(facts, as_of, lag_days)
    if known.empty:
        return None
    latest_filed = known["filed"].max()
    recent = known[known["period_end"] >= as_of - pd.DateOffset(years=_HISTORY_YEARS)]
    by_concept = {concept: rows for concept, rows in recent.groupby("concept", sort=False)}
    nothing = recent.iloc[0:0]

    values: dict[str, float] = {}
    period_ends: dict[str, pd.Timestamp] = {}

    net_income_quarters: list[Period] = []
    for concept in FLOW_CONCEPTS:
        if concept not in by_concept:
            continue
        periods = _periods(by_concept[concept])
        quarters = _quarters(periods)
        if concept == "net_income":
            net_income_quarters = quarters[-history_quarters:]
        ttm = _trailing_twelve_months(periods, quarters)
        if ttm is not None:
            values[concept], period_ends[concept] = ttm

    # Balance-sheet items must come from the same balance sheet: a line a company
    # stopped reporting (say, short-term borrowings once repaid) would otherwise
    # linger at its last reported value forever.
    assets = _latest_instant(by_concept.get("total_assets", nothing))
    instants = recent[recent["concept"].isin(BALANCE_CONCEPTS) & recent["period_start"].isna()]
    balance_date = assets[1] if assets is not None else (
        instants["period_end"].max() if not instants.empty else None
    )
    if balance_date is not None:
        on_balance_date = instants[instants["period_end"] == balance_date]
        for concept, value in zip(on_balance_date["concept"], on_balance_date["value"]):
            values[concept] = float(value)
            period_ends[concept] = balance_date

    shares = _latest_instant(by_concept.get("shares_outstanding", nothing))
    if shares is None:
        # Weighted-average count of the most recent period, shortest period first
        weighted = by_concept.get("weighted_shares", nothing)
        if not weighted.empty:
            weighted = weighted.assign(_length=weighted["period_end"] - weighted["period_start"])
            row = weighted.sort_values(["period_end", "_length"], ascending=[False, True]).iloc[0]
            shares = float(row["value"]), row["period_end"]
    if shares is not None:
        values["shares_outstanding"], period_ends["shares_outstanding"] = shares

    _add_derived(values, period_ends)

    return FundamentalSnapshot(
        as_of=as_of,
        values=values,
        period_ends=period_ends,
        net_income_quarters=_quarters_frame(net_income_quarters)["value"],
        latest_filed=latest_filed,
    )


def _add_derived(values: dict[str, float], period_ends: dict[str, pd.Timestamp]) -> None:
    same_period = period_ends.get("revenue") == period_ends.get("cost_of_revenue")
    if "gross_profit" not in values and same_period and "revenue" in values:
        values["gross_profit"] = values["revenue"] - values["cost_of_revenue"]
        period_ends["gross_profit"] = period_ends["revenue"]

    if "operating_income" in values and "depreciation_amortization" in values:
        values["ebitda"] = values["operating_income"] + values["depreciation_amortization"]
    if "operating_cash_flow" in values:
        # Capex is reported as a positive payment; none reported means none made
        values["free_cash_flow"] = values["operating_cash_flow"] - values.get("capex", 0.0)

    # Long + short is preferred over the combined tag because few companies use
    # the combined one. Approximate: a company reporting both a current portion
    # of long-term debt and separate short-term borrowings is undercounted.
    parts = [values[name] for name in ("long_term_debt", "short_term_debt") if name in values]
    if parts:
        values["total_debt"] = float(sum(parts))
    elif "total_debt_reported" in values:
        values["total_debt"] = values["total_debt_reported"]


def facts_from_edgar(entity_facts: Iterable) -> pd.DataFrame:
    """Flatten edgartools `EntityFacts` (an iterable of `FinancialFact`) into a facts frame.

    Only the tags in `CONCEPT_TAGS` are kept, in USD (or shares for share counts).
    """
    rows = []
    for fact in entity_facts:
        concept = _TAG_TO_CONCEPT.get(fact.concept)
        if concept is None or fact.numeric_value is None:
            continue
        if fact.filing_date is None or fact.period_end is None:
            continue
        expected_unit = "shares" if concept in _SHARE_CONCEPTS else "USD"
        if fact.unit != expected_unit:
            continue
        rows.append(
            {
                "concept": concept,
                "tag": fact.concept,
                "period_start": fact.period_start,
                "period_end": fact.period_end,
                "filed": fact.filing_date,
                "value": fact.numeric_value,
                "form": fact.form_type,
            }
        )
    return normalize_facts(pd.DataFrame(rows, columns=FACT_COLUMNS))


def require_edgar_identity() -> str:
    """The `EDGAR_IDENTITY` ("Name email@domain") the SEC asks every client to send."""
    load_dotenv()
    identity = os.environ.get("EDGAR_IDENTITY", "").strip()
    if not identity:
        raise EdgarIdentityMissing(
            "EDGAR_IDENTITY is not set. The SEC requires a name and e-mail address "
            "with every request; set the environment variable (or add it to .env) as "
            "'Your Name you@example.com' before fetching point-in-time fundamentals."
        )
    return identity


def fetch_edgar_facts(ticker: str) -> pd.DataFrame:
    """Download a company's XBRL facts from EDGAR; empty frame for an unknown ticker."""
    require_edgar_identity()  # edgartools reads the same variable itself
    import edgar

    try:
        entity_facts = edgar.Company(ticker).get_facts()
    except edgar.CompanyNotFoundError:
        return empty_facts()
    if entity_facts is None:
        return empty_facts()
    return facts_from_edgar(entity_facts)


class PointInTimeFundamentals:
    """Per-ticker fact store answering "what was known on this date?".

    Facts come from `fetcher` (EDGAR by default) and are cached in the SQLite
    `CacheStore` under `edgar_pit:{ticker}`. Because every fact keeps its filing
    date, a cached frame answers any past date; `max_age` only decides how soon
    new filings are picked up.

        fundamentals = PointInTimeFundamentals()          # needs EDGAR_IDENTITY
        fundamentals.preload(["AAPL", "MSFT"])            # optional; fetches up front
        snapshot = fundamentals.as_of("AAPL", date(2020, 3, 2))
    """

    def __init__(
        self,
        cache_store=None,
        *,
        fetcher: Callable[[str], pd.DataFrame] = fetch_edgar_facts,
        lag_days: int = DEFAULT_LAG_DAYS,
        max_age: timedelta | None = timedelta(days=7),
    ):
        self._cache_store = cache_store
        self._fetcher = fetcher
        self.lag_days = lag_days
        self.max_age = max_age
        self._frames: dict[str, pd.DataFrame] = {}

    @classmethod
    def from_frames(
        cls, frames: dict[str, pd.DataFrame], *, lag_days: int = DEFAULT_LAG_DAYS
    ) -> "PointInTimeFundamentals":
        """A store over facts already in memory: no cache, no network."""
        store = cls(fetcher=lambda ticker: empty_facts(), lag_days=lag_days)
        store._frames = {ticker: normalize_facts(frame) for ticker, frame in frames.items()}
        return store

    def _store(self):
        if self._cache_store is not None:
            return self._cache_store
        from stock_analysis.storage import get_cache_store

        return get_cache_store()

    def facts(self, ticker: str) -> pd.DataFrame:
        """All facts ever filed for `ticker` (memory, then cache, then `fetcher`)."""
        if ticker in self._frames:
            return self._frames[ticker]
        key = f"{CACHE_KEY_PREFIX}{ticker}"
        store = self._store()
        frame = store.get(key, max_age=self.max_age)
        if frame is None:
            frame = self._fetcher(ticker)
            if not frame.empty:
                store.put(key, frame)
        frame = normalize_facts(frame)
        self._frames[ticker] = frame
        return frame

    def preload(self, tickers: Iterable[str]) -> list[str]:
        """Load facts for `tickers`; returns the ones with no facts at all."""
        return [ticker for ticker in tickers if self.facts(ticker).empty]

    def as_of(
        self,
        ticker: str,
        when: date | pd.Timestamp,
        lag_days: int | None = None,
    ) -> FundamentalSnapshot | None:
        """Snapshot of `ticker` as known on `when`; None if nothing was filed by then."""
        return snapshot_as_of(
            self.facts(ticker),
            when,
            lag_days=self.lag_days if lag_days is None else lag_days,
        )
