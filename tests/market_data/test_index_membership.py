from datetime import date, timedelta

import pandas as pd
import pytest

from stock_analysis.market_data.index_membership import (
    SP500_CACHE_KEY,
    build_membership,
    fetch_sp500_tables,
    format_coverage,
    load_sp500_membership,
    membership_coverage,
    normalise_ticker,
    parse_wikipedia_tables,
)
from stock_analysis.storage.cache_store import SqlCacheStore
from stock_analysis.storage.schema import create_engine_for_url, init_db


def _constituents():
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BRK.B", "CCC", "NEW"],
            "sector": ["Tech", "Financials", "Energy", "Tech"],
            "date_added": ["1990-01-02", "2010-02-16", "2015-06-01", "2020-03-02"],
        }
    )


def _changes():
    # Newest first, as on Wikipedia. CCC joined in 2015 replacing OLD; NEW joined
    # in 2020 replacing BF.B; GONE left in 2012 with no replacement logged.
    return pd.DataFrame(
        {
            "date": ["March 2, 2020", "June 1, 2015", "July 2, 2012"],
            "added": ["NEW", "CCC", None],
            "removed": ["BF.B", "OLD", "GONE"],
        }
    )


@pytest.fixture
def membership():
    return build_membership(_constituents(), _changes())


def test_tickers_are_normalised_to_yahoo_form():
    assert normalise_ticker("BRK.B") == "BRK-B"
    assert normalise_ticker(" goog[4] ") == "GOOG"
    assert normalise_ticker(float("nan")) is None
    assert normalise_ticker("") is None


def test_current_members_and_sectors(membership):
    assert membership.members_on("2026-01-01") == {"AAA", "BRK-B", "CCC", "NEW"}
    assert membership.sectors["BRK-B"] == "Financials"
    assert "OLD" not in membership.sectors


def test_changes_are_walked_back(membership):
    assert membership.members_on("2019-12-31") == {"AAA", "BRK-B", "CCC", "BF-B"}
    assert membership.members_on("2014-01-01") == {"AAA", "BRK-B", "OLD", "BF-B"}
    assert membership.members_on("2011-01-01") == {"AAA", "BRK-B", "OLD", "BF-B", "GONE"}
    # BRK-B's addition is not in the change log; its "date added" places it
    assert membership.members_on("2009-01-01") == {"AAA", "OLD", "BF-B", "GONE"}
    assert membership.members_on("1989-01-01") == {"OLD", "BF-B", "GONE"}


def test_a_change_takes_effect_on_its_own_date(membership):
    assert "NEW" not in membership.members_on("2020-03-01")
    assert "BF-B" in membership.members_on(date(2020, 3, 1))
    assert "NEW" in membership.members_on(pd.Timestamp("2020-03-02 16:00"))
    assert "BF-B" not in membership.members_on("2020-03-02")


def test_all_tickers_and_sample(membership):
    everyone = ["AAA", "BF-B", "BRK-B", "CCC", "GONE", "NEW", "OLD"]
    assert membership.all_tickers() == everyone
    assert membership.sample(None) == everyone
    assert membership.sample(100) == everyone
    sample = membership.sample(3, seed=1)
    assert len(sample) == 3 and set(sample) <= set(everyone)
    assert sample == membership.sample(3, seed=1)
    assert any(membership.sample(3, seed=s) != sample for s in range(2, 8))


def test_member_matrix_agrees_with_members_on(membership):
    sessions = pd.bdate_range("2011-06-01", "2021-01-01", freq="50B")
    tickers = membership.all_tickers()
    matrix = membership.member_matrix(sessions, tickers)
    for session in sessions:
        assert set(matrix.columns[matrix.loc[session]]) == membership.members_on(session)


def test_change_log_extent_and_unmatched_additions():
    changes = pd.concat(
        [_changes(), pd.DataFrame({"date": ["2013-12-23"], "added": ["FB"], "removed": [None]})]
    )
    membership = build_membership(_constituents(), changes)

    assert membership.change_log_start == pd.Timestamp("2012-07-02")
    # FB was added under a ticker that is not a member today (renamed since)
    assert membership.unmatched_additions == 1
    assert "change log from 2012-07-02" in membership.describe()


def test_dense_log_from_is_the_start_of_the_unbroken_run():
    rows = [(f"{year}-03-01", None, f"R{year}{i}") for year in (2015, 2016) for i in range(5)]
    rows += [("2013-03-01", None, "LONE"), ("2012-03-01", None, f"X{0}")]
    changes = pd.DataFrame(rows, columns=["date", "added", "removed"])
    membership = build_membership(_constituents(), changes)

    assert membership.dense_log_from(min_removals=5) == 2015


def test_coverage_counts_members_candidates_and_prices(membership):
    bars_2011_2021 = pd.DataFrame(
        {"Close": 1.0}, index=pd.bdate_range("2011-01-03", "2021-12-31")
    )
    bars_2020_only = pd.DataFrame({"Close": 1.0}, index=pd.bdate_range("2020-01-02", "2020-12-31"))
    price_data = {"AAA": bars_2011_2021, "CCC": bars_2011_2021, "NEW": bars_2020_only}

    coverage = membership_coverage(
        membership, price_data, first_year=2011, last_year=2021,
        candidates=["AAA", "CCC", "NEW", "GONE", "OLD"], today=date(2026, 1, 1),
    ).set_index("year")

    # End of 2011: AAA, BRK-B, OLD, BF-B, GONE are members; AAA, OLD, GONE were asked for
    assert coverage.loc[2011, ["members", "candidates", "with_prices"]].tolist() == [5, 3, 1]
    assert coverage.loc[2011, "coverage"] == pytest.approx(1 / 3)
    # End of 2015: CCC has replaced OLD, GONE has left
    assert coverage.loc[2015, ["members", "candidates", "with_prices"]].tolist() == [4, 2, 2]
    assert coverage.loc[2020, ["members", "candidates", "with_prices"]].tolist() == [4, 3, 3]
    # NEW has no bars in 2021
    assert coverage.loc[2021, "with_prices"] == 2

    lines = format_coverage(membership, coverage.reset_index())
    assert "| 2011 | 5 | 3 | 1 | 33% |" in lines
    assert any("survivorship" in line for line in lines)


def test_coverage_without_candidates_uses_every_member(membership):
    coverage = membership_coverage(
        membership, {}, first_year=2021, last_year=2021, today=date(2026, 1, 1)
    )
    assert coverage.iloc[0][["members", "candidates", "with_prices"]].tolist() == [4, 4, 0]


LIST_PAGE = """
<table><tr><th>Symbol</th><th>Security</th><th>GICS Sector</th><th>Date added</th></tr>
<tr><td>AAA</td><td>Aaa Inc</td><td>Tech</td><td>1990-01-02</td></tr>
<tr><td>BRK.B</td><td>Berkshire</td><td>Financials</td><td>2010-02-16</td></tr></table>
"""
CHANGES_TABLE = """
<table>
<tr><th rowspan="2">Effective Date</th><th colspan="2">Added</th><th colspan="2">Removed</th>
<th rowspan="2">Reason</th></tr>
<tr><th>Ticker</th><th>Security</th><th>Ticker</th><th>Security</th></tr>
<tr><td>February 16, 2010</td><td>BRK.B</td><td>Berkshire</td><td>OLD</td><td>Old Co</td>
<td>Acquired</td></tr>
</table>
"""


def test_tables_are_parsed_from_one_page():
    constituents, changes = parse_wikipedia_tables(LIST_PAGE + CHANGES_TABLE)

    assert constituents["ticker"].tolist() == ["AAA", "BRK.B"]
    assert constituents["sector"].tolist() == ["Tech", "Financials"]
    assert changes.iloc[0].tolist() == ["February 16, 2010", "BRK.B", "OLD"]


def test_change_log_is_read_from_its_own_page_when_the_list_page_lacks_it():
    pages = []

    def get(url):
        pages.append(url)
        return LIST_PAGE if len(pages) == 1 else "<p>intro</p>" + CHANGES_TABLE

    constituents, changes = fetch_sp500_tables(get)

    assert len(pages) == 2 and "Historical_components" in pages[1]
    membership = build_membership(constituents, changes)
    assert membership.members_on("2009-01-01") == {"AAA", "OLD"}


def test_missing_tables_raise():
    with pytest.raises(ValueError, match="layout"):
        fetch_sp500_tables(lambda url: "<table><tr><th>x</th></tr><tr><td>1</td></tr></table>")


def test_loader_caches_the_tables():
    engine = create_engine_for_url("sqlite://")
    init_db(engine)
    store = SqlCacheStore(engine)
    calls = []

    def fetch():
        calls.append(1)
        return _constituents(), _changes()

    first = load_sp500_membership(store, fetch=fetch)
    second = load_sp500_membership(store, fetch=fetch)

    assert len(calls) == 1
    assert store.get(SP500_CACHE_KEY) is not None
    for when in ("2009-01-01", "2014-01-01", "2026-01-01"):
        assert first.members_on(when) == second.members_on(when)
    assert second.sectors == first.sectors

    load_sp500_membership(store, max_age=timedelta(seconds=-1), fetch=fetch)
    assert len(calls) == 2
