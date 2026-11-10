"""Point-in-time fundamentals: synthetic facts only, no EDGAR."""

from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from stock_analysis.ai.fundamentals import point_in_time as pit
from stock_analysis.ai.fundamentals.point_in_time import (
    EdgarIdentityMissing,
    PointInTimeFundamentals,
    discrete_quarters,
    facts_from_edgar,
    known_facts,
    normalize_facts,
    snapshot_as_of,
)
from stock_analysis.storage.cache_store import SqlCacheStore
from stock_analysis.storage.schema import create_engine_for_url, init_db


def _fact(concept, start, end, filed, value, tag=None, form="10-Q"):
    return {
        "concept": concept,
        "tag": tag or pit.CONCEPT_TAGS[concept][0],
        "period_start": start,
        "period_end": end,
        "filed": filed,
        "value": value,
        "form": form,
    }


def _frame(rows):
    return normalize_facts(pd.DataFrame(rows))


def _income_facts():
    """Net income the way an income statement reports it: three-month figures
    for Q1–Q3 and only the full year in the 10-K."""
    return [
        _fact("net_income", "2022-01-01", "2022-03-31", "2022-05-02", 90),
        _fact("net_income", "2022-04-01", "2022-06-30", "2022-08-01", 100),
        _fact("net_income", "2022-07-01", "2022-09-30", "2022-11-01", 105),
        _fact("net_income", "2022-01-01", "2022-12-31", "2023-02-15", 400, form="10-K"),
        _fact("net_income", "2023-01-01", "2023-03-31", "2023-05-01", 110),
        _fact("net_income", "2023-04-01", "2023-06-30", "2023-08-01", 120),
        _fact("net_income", "2023-07-01", "2023-09-30", "2023-11-01", 130),
    ]


def test_ttm_sums_four_quarters_with_q4_derived_from_the_annual_figure():
    snapshot = snapshot_as_of(_frame(_income_facts()), date(2023, 11, 10))

    # Q4 2022 = 400 - (90 + 100 + 105) = 105; then Q1-Q3 2023
    assert snapshot.values["net_income"] == pytest.approx(105 + 110 + 120 + 130)
    assert snapshot.period_ends["net_income"] == pd.Timestamp("2023-09-30")


def test_ttm_is_the_annual_figure_right_after_the_10k():
    snapshot = snapshot_as_of(_frame(_income_facts()), date(2023, 3, 1))

    assert snapshot.values["net_income"] == pytest.approx(400)
    assert snapshot.period_ends["net_income"] == pd.Timestamp("2022-12-31")


def test_quarters_are_derived_from_year_to_date_cash_flow_figures():
    """Cash-flow statements report three, six and nine months and the year — never a single quarter after Q1."""
    rows = [
        _fact("operating_cash_flow", "2022-01-01", "2022-03-31", "2022-05-02", 80),
        _fact("operating_cash_flow", "2022-01-01", "2022-06-30", "2022-08-01", 180),
        _fact("operating_cash_flow", "2022-01-01", "2022-09-30", "2022-11-01", 290),
        _fact("operating_cash_flow", "2022-01-01", "2022-12-31", "2023-02-15", 400),
        _fact("operating_cash_flow", "2023-01-01", "2023-03-31", "2023-05-01", 100),
        _fact("operating_cash_flow", "2023-01-01", "2023-06-30", "2023-08-01", 220),
    ]
    frame = _frame(rows)

    quarters = discrete_quarters(frame)
    assert list(quarters["value"]) == [80, 100, 110, 110, 100, 120]

    snapshot = snapshot_as_of(frame, date(2023, 8, 10))
    assert snapshot.values["operating_cash_flow"] == pytest.approx(110 + 110 + 100 + 120)
    assert snapshot.period_ends["operating_cash_flow"] == pd.Timestamp("2023-06-30")


def test_a_gap_in_the_quarters_falls_back_to_the_last_full_year():
    rows = [row for row in _income_facts() if row["period_end"] != "2023-06-30"]

    snapshot = snapshot_as_of(_frame(rows), date(2023, 11, 10))

    assert snapshot.values["net_income"] == pytest.approx(400)


def test_filing_lag_decides_the_first_day_a_value_is_known():
    frame = _frame([_fact("total_assets", None, "2023-03-31", "2023-05-01", 1000)])

    assert snapshot_as_of(frame, date(2023, 5, 1)) is None
    assert snapshot_as_of(frame, date(2023, 5, 2)).values["total_assets"] == 1000
    assert snapshot_as_of(frame, date(2023, 5, 1), lag_days=0).values["total_assets"] == 1000
    assert snapshot_as_of(frame, date(2023, 5, 5), lag_days=5) is None


def test_a_restatement_does_not_reach_back_before_its_own_filing():
    original = _income_facts()
    restated = original + [
        # The 2024 10-Q restates Q1 2023 as a comparative, and reports Q1 2024
        _fact("net_income", "2023-01-01", "2023-03-31", "2024-05-01", 60),
        _fact("net_income", "2023-01-01", "2023-12-31", "2024-02-15", 500, form="10-K"),
        _fact("net_income", "2024-01-01", "2024-03-31", "2024-05-01", 140),
    ]

    before = snapshot_as_of(_frame(original), date(2023, 11, 10))
    with_later_filings = snapshot_as_of(_frame(restated), date(2023, 11, 10))

    assert with_later_filings.values == before.values
    assert with_later_filings.period_ends == before.period_ends
    pd.testing.assert_series_equal(
        with_later_filings.net_income_quarters, before.net_income_quarters
    )

    # Once the restatement is public it is the value in use
    known = known_facts(_frame(restated), date(2024, 5, 10))
    q1 = known[known["period_end"] == pd.Timestamp("2023-03-31")]
    assert list(q1["value"]) == [60]
    after = snapshot_as_of(_frame(restated), date(2024, 5, 10))
    # Q4 2023 = 500 - (60 + 120 + 130) = 190, using the restated Q1
    assert list(after.net_income_quarters.iloc[-2:]) == [190, 140]


def test_balance_sheet_uses_the_latest_date_and_drops_lines_no_longer_reported():
    rows = [
        _fact("total_assets", None, "2022-12-31", "2023-02-15", 1000),
        _fact("short_term_debt", None, "2022-12-31", "2023-02-15", 50),
        _fact("long_term_debt", None, "2022-12-31", "2023-02-15", 200),
        _fact("total_assets", None, "2023-03-31", "2023-05-01", 1100),
        _fact("long_term_debt", None, "2023-03-31", "2023-05-01", 180),
        _fact("cash", None, "2023-03-31", "2023-05-01", 70),
    ]
    frame = _frame(rows)

    earlier = snapshot_as_of(frame, date(2023, 3, 1))
    assert earlier.values["total_debt"] == 250

    later = snapshot_as_of(frame, date(2023, 6, 1))
    assert later.values["total_assets"] == 1100
    assert later.values["cash"] == 70
    # Short-term debt was repaid: the old 50 must not linger
    assert "short_term_debt" not in later.values
    assert later.values["total_debt"] == 180


def test_derived_values_and_net_income_history():
    rows = _income_facts() + [
        _fact("revenue", "2022-01-01", "2022-12-31", "2023-02-15", 2000),
        _fact("cost_of_revenue", "2022-01-01", "2022-12-31", "2023-02-15", 1200),
        _fact("operating_income", "2022-01-01", "2022-12-31", "2023-02-15", 500),
        _fact("depreciation_amortization", "2022-01-01", "2022-12-31", "2023-02-15", 80),
        _fact("operating_cash_flow", "2022-01-01", "2022-12-31", "2023-02-15", 450),
        _fact("capex", "2022-01-01", "2022-12-31", "2023-02-15", 150),
        _fact("shares_outstanding", None, "2023-01-31", "2023-02-15", 10_000),
    ]

    snapshot = snapshot_as_of(_frame(rows), date(2023, 3, 1))

    assert snapshot.values["gross_profit"] == 800
    assert snapshot.values["ebitda"] == 580
    assert snapshot.values["free_cash_flow"] == 300
    assert snapshot.values["shares_outstanding"] == 10_000
    assert snapshot.period_ends["shares_outstanding"] == pd.Timestamp("2023-01-31")
    assert list(snapshot.net_income_quarters) == [90, 100, 105, 105]
    assert snapshot.latest_filed == pd.Timestamp("2023-02-15")
    assert pd.isna(snapshot.get("total_assets"))


def test_preferred_tag_wins_when_one_filing_reports_a_period_twice():
    rows = [
        _fact("revenue", "2022-01-01", "2022-12-31", "2023-02-15", 900, tag="us-gaap:SalesRevenueNet"),
        _fact("revenue", "2022-01-01", "2022-12-31", "2023-02-15", 1000, tag="us-gaap:Revenues"),
    ]

    assert snapshot_as_of(_frame(rows), date(2023, 3, 1)).values["revenue"] == 1000
    assert snapshot_as_of(_frame(rows[::-1]), date(2023, 3, 1)).values["revenue"] == 1000


def _edgar_fact(concept, value, *, unit="USD", start=None, end=date(2023, 3, 31),
                filed=date(2023, 5, 1)):
    return SimpleNamespace(
        concept=concept, numeric_value=value, unit=unit, period_start=start,
        period_end=end, filing_date=filed, form_type="10-Q",
    )


def test_facts_from_edgar_keeps_mapped_tags_in_the_right_unit():
    facts = facts_from_edgar(
        [
            _edgar_fact("us-gaap:Assets", 1000.0),
            _edgar_fact("us-gaap:Assets", 900.0, unit="EUR"),
            _edgar_fact("us-gaap:Goodwill", 5.0),
            _edgar_fact("us-gaap:NetIncomeLoss", 10.0, start=date(2023, 1, 1)),
            _edgar_fact("us-gaap:NetIncomeLoss", None, start=date(2022, 1, 1)),
            _edgar_fact("dei:EntityCommonStockSharesOutstanding", 500.0, unit="shares"),
            _edgar_fact("us-gaap:Assets", 1.0, filed=None),
        ]
    )

    assert sorted(facts["concept"]) == ["net_income", "shares_outstanding", "total_assets"]
    assets = facts[facts["concept"] == "total_assets"].iloc[0]
    assert assets["value"] == 1000.0
    assert pd.isna(assets["period_start"])
    assert assets["filed"] == pd.Timestamp("2023-05-01")
    assert assets["tag"] == "us-gaap:Assets"


def test_fetching_from_edgar_without_an_identity_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("EDGAR_IDENTITY", raising=False)
    monkeypatch.setattr(pit, "load_dotenv", lambda: None)

    def no_network(*args, **kwargs):
        raise AssertionError("EDGAR must not be contacted without an identity")

    monkeypatch.setattr("edgar.Company", no_network)

    with pytest.raises(EdgarIdentityMissing, match="EDGAR_IDENTITY"):
        pit.fetch_edgar_facts("AAPL")


@pytest.fixture
def cache_store():
    engine = create_engine_for_url("sqlite://")
    init_db(engine)
    return SqlCacheStore(engine)


def test_facts_are_fetched_once_and_cached_under_a_namespaced_key(cache_store):
    calls = []

    def fetcher(ticker):
        calls.append(ticker)
        return _frame(_income_facts())

    store = PointInTimeFundamentals(cache_store, fetcher=fetcher)
    first = store.as_of("AAA", date(2023, 11, 10))
    store.as_of("AAA", date(2023, 3, 1))
    assert calls == ["AAA"]
    assert cache_store.get("edgar_pit:AAA") is not None

    # A new instance reads the SQLite cache and gets the same answer
    again = PointInTimeFundamentals(cache_store, fetcher=fetcher).as_of("AAA", date(2023, 11, 10))
    assert calls == ["AAA"]
    assert again.values == first.values


def test_ticker_without_facts_is_not_cached_and_has_no_snapshot(cache_store):
    store = PointInTimeFundamentals(cache_store, fetcher=lambda ticker: pit.empty_facts())

    assert store.as_of("NONE", date(2023, 1, 1)) is None
    assert cache_store.get("edgar_pit:NONE") is None
    assert store.preload(["NONE"]) == ["NONE"]


def test_from_frames_needs_neither_cache_nor_network():
    store = PointInTimeFundamentals.from_frames({"AAA": pd.DataFrame(_income_facts())})

    assert store.as_of("AAA", date(2023, 3, 1)).values["net_income"] == 400
    assert "net_income" not in store.as_of("AAA", date(2023, 2, 15)).values


def test_facts_from_edgar_reads_what_the_installed_edgartools_parser_produces():
    """SEC company-facts JSON through edgartools' own parser (no network), so a
    rename of `FinancialFact` fields in the library shows up here."""
    from edgar.entity.parser import EntityFactsParser

    def entry(end, value, filed, start=None, form="10-Q"):
        row = {"end": end, "val": value, "accn": "0000000000-23-000001", "fy": 2023,
               "fp": "Q1", "form": form, "filed": filed}
        if start:
            row["start"] = start
        return row

    company_facts = {
        "cik": 1,
        "entityName": "Example Corp",
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {"label": "Net income", "description": "", "units": {"USD": [
                    entry("2023-03-31", 110, "2023-05-01", start="2023-01-01"),
                    entry("2023-03-31", 95, "2024-05-01", start="2023-01-01"),
                ]}},
                "Assets": {"label": "Assets", "description": "", "units": {"USD": [
                    entry("2023-03-31", 5000, "2023-05-01"),
                ]}},
                "Goodwill": {"label": "Goodwill", "description": "", "units": {"USD": [
                    entry("2023-03-31", 7, "2023-05-01"),
                ]}},
            },
            "dei": {
                "EntityCommonStockSharesOutstanding": {"label": "Shares", "description": "", "units": {
                    "shares": [entry("2023-04-20", 1000, "2023-05-01")],
                }},
            },
        },
    }

    facts = facts_from_edgar(EntityFactsParser.parse_company_facts(company_facts))

    assert sorted(facts["concept"]) == ["net_income", "net_income", "shares_outstanding", "total_assets"]
    snapshot = snapshot_as_of(facts, date(2023, 6, 1))
    assert snapshot.values["total_assets"] == 5000
    assert snapshot.values["shares_outstanding"] == 1000
    assert snapshot.period_ends["shares_outstanding"] == pd.Timestamp("2023-04-20")
    assert list(snapshot.net_income_quarters) == [110]
    assert list(snapshot_as_of(facts, date(2024, 6, 1)).net_income_quarters) == [95]
