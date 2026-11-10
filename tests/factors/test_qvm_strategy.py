"""QvmScoreProvider: monthly rebalance, no look-ahead, and the strategy / watchlist built on it."""

import numpy as np
import pandas as pd
import pytest

from stock_analysis.ai.fundamentals.point_in_time import PointInTimeFundamentals
from stock_analysis.factors.qvm_strategy import (
    QvmScoreProvider,
    build_qvm_strategy,
    qvm_watchlist,
)
from tests.factors.synthetic import (
    LOSS_MAKER,
    SECTORS,
    TICKERS,
    make_fact_frames,
    make_price_data,
)

SESSION = pd.Timestamp("2023-05-17")
MONTH_START = pd.Timestamp("2023-05-01")


@pytest.fixture(scope="module")
def price_data():
    return make_price_data()


@pytest.fixture(scope="module")
def fact_frames():
    return make_fact_frames()


def _provider(price_data, fact_frames, sectors=SECTORS, **options):
    return QvmScoreProvider(
        price_data, PointInTimeFundamentals.from_frames(fact_frames), sectors, **options
    )


def test_every_stock_with_prices_and_filings_is_scored(price_data, fact_frames):
    scores = _provider(price_data, fact_frames).scores_on(SESSION)

    assert sorted(scores.index) == sorted(TICKERS)
    assert scores.between(0, 100).all()


def test_scores_are_those_of_the_first_session_of_the_month(price_data, fact_frames):
    provider = _provider(price_data, fact_frames)

    assert provider.rebalance_session(SESSION) == MONTH_START
    expected = provider.table_on(MONTH_START)["qvm_score"]
    for session in pd.bdate_range("2023-05-01", "2023-05-31"):
        pd.testing.assert_series_equal(provider.scores_on(session), expected)


def test_scores_change_at_the_next_rebalance(price_data, fact_frames):
    provider = _provider(price_data, fact_frames)

    may = provider.scores_on(pd.Timestamp("2023-05-31"))
    june = provider.scores_on(pd.Timestamp("2023-06-01"))

    assert not may.equals(june)


def test_no_look_ahead_in_prices_or_filings(price_data, fact_frames):
    """Cutting everything after the session away, or rewriting it, leaves the session's scores unchanged."""
    full = _provider(price_data, fact_frames)

    truncated_prices = {ticker: df.loc[:SESSION] for ticker, df in price_data.items()}
    truncated_facts = {
        ticker: frame[pd.to_datetime(frame["filed"]) < SESSION]
        for ticker, frame in fact_frames.items()
    }
    truncated = _provider(truncated_prices, truncated_facts)

    rewritten_prices = {}
    for ticker, df in price_data.items():
        changed = df.copy()
        changed.loc[changed.index > SESSION, ["Open", "High", "Low", "Close"]] *= 3.0
        rewritten_prices[ticker] = changed
    rewritten_facts = {}
    for ticker, frame in fact_frames.items():
        changed = frame.copy()
        later = pd.to_datetime(changed["filed"]) >= SESSION
        changed.loc[later, "value"] *= -5.0
        # A later filing that restates every earlier period
        restated = frame.assign(filed=SESSION + pd.Timedelta(30, unit="D"), value=frame["value"] * 10)
        rewritten_facts[ticker] = pd.concat([changed, restated], ignore_index=True)
    rewritten = _provider(rewritten_prices, rewritten_facts)

    expected = full.scores_on(SESSION)
    pd.testing.assert_series_equal(truncated.scores_on(SESSION), expected)
    pd.testing.assert_series_equal(rewritten.scores_on(SESSION), expected)
    pd.testing.assert_frame_equal(truncated.table_on(SESSION), full.table_on(SESSION))
    pd.testing.assert_frame_equal(rewritten.table_on(SESSION), full.table_on(SESSION))


def test_a_filing_counts_only_after_the_lag(price_data, fact_frames):
    """Q1 2023 was filed on 2023-05-05; the scores of that day still rest on the 10-K."""
    provider = _provider(price_data, fact_frames)
    filing_day = pd.Timestamp("2023-05-05")

    on_filing_day = provider.table_on(filing_day)
    day_after = provider.table_on(filing_day + pd.Timedelta(3, unit="D"))  # next session (Monday)

    without_q1 = _provider(
        price_data,
        {t: f[pd.to_datetime(f["filed"]) < filing_day] for t, f in fact_frames.items()},
    ).table_on(filing_day)
    pd.testing.assert_frame_equal(on_filing_day, without_q1)
    assert not np.allclose(on_filing_day["earnings_yield"], day_after["earnings_yield"])


def test_the_loss_maker_is_the_most_expensive_in_its_sector(price_data, fact_frames):
    table = _provider(price_data, fact_frames).table_on(SESSION)

    assert table.loc[LOSS_MAKER, "earnings_yield"] < 0
    tech = table[table["sector"] == "Technology"]
    assert tech["earnings_yield"].idxmin() == LOSS_MAKER
    assert tech["value_score"].idxmin() == LOSS_MAKER


def test_no_ranking_before_there_is_a_year_of_prices(price_data, fact_frames):
    provider = _provider(price_data, fact_frames)

    assert provider.scores_on(pd.Timestamp("2022-06-15")) is None
    assert provider.scores_on(pd.Timestamp("2021-06-15")) is None  # before the data starts


def test_stock_without_filings_or_no_longer_trading_is_left_out(price_data, fact_frames):
    facts = {t: f for t, f in fact_frames.items() if t != "T1"}
    prices = dict(price_data)
    prices["E1"] = prices["E1"].loc[: pd.Timestamp("2023-03-31")]  # delisted

    scores = _provider(prices, facts, min_sector_size=3).scores_on(SESSION)

    assert "T1" not in scores.index
    assert "E1" not in scores.index
    assert {"T0", "T2", "T3", "E0", "E2", "E3"} == set(scores.index)


def test_stale_filings_are_not_used(price_data, fact_frames):
    stale = dict(fact_frames)
    stale["T0"] = stale["T0"][pd.to_datetime(stale["T0"]["period_end"]) < "2021-07-01"]

    scores = _provider(price_data, stale, min_sector_size=3).scores_on(SESSION)

    assert "T0" not in scores.index


def test_sectors_none_ranks_the_universe_as_one_group(price_data, fact_frames):
    table = _provider(price_data, fact_frames, sectors=None).table_on(SESSION)

    assert set(table["sector"]) == {"All"}
    assert table["momentum_score"].max() == 100.0
    assert table["momentum_score"].min() == pytest.approx(100 / 8)


def test_market_value_uses_the_traded_price_and_the_share_basis_of_the_day(fact_frames):
    """A 2-for-1 split after the share count was reported doubles the count; the
    adjusted close before the split is half the price that traded."""
    index = pd.bdate_range("2022-01-03", periods=420)
    close = np.full(420, 50.0)  # adjusted: flat
    splits = np.zeros(420)
    split_day = index.get_loc(pd.Timestamp("2023-06-01"))
    splits[split_day] = 2.0
    frame = pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6,
         "Dividends": 0.0, "Stock Splits": splits},
        index=index,
    )
    provider = QvmScoreProvider(
        {"T0": frame}, PointInTimeFundamentals.from_frames({"T0": fact_frames["T0"]}), None,
        min_sector_size=1,
    )

    before = provider.table_on(pd.Timestamp("2023-05-17")).loc["T0", "market_cap"]
    after = provider.table_on(pd.Timestamp("2023-06-05")).loc["T0", "market_cap"]

    # 1e6 shares at 100 traded; then 2e6 shares at 50 (the 10-Q with the new count is not out yet)
    assert before == pytest.approx(1e6 * 100.0)
    assert after == pytest.approx(2e6 * 50.0)


def test_margin_of_safety_is_off_by_default_and_changes_value_when_on(price_data, fact_frames):
    default = _provider(price_data, fact_frames).table_on(SESSION)
    with_models = _provider(price_data, fact_frames, include_margin_of_safety=True).table_on(SESSION)

    assert "epv_to_price" in default.columns  # computed for inspection, not scored
    pd.testing.assert_series_equal(default["quality_score"], with_models["quality_score"])
    assert not default["value_score"].equals(with_models["value_score"])


def test_build_qvm_strategy_buys_the_top_fraction_and_keeps_a_buffer(price_data, fact_frames):
    fundamentals = PointInTimeFundamentals.from_frames(fact_frames)
    strategy = build_qvm_strategy(
        price_data, fundamentals, SECTORS, top_fraction=0.25, exit_buffer=0.5
    )
    assert strategy.top_k == 2
    assert strategy.exit_rank == 3

    scores = strategy.provider.scores_on(SESSION)
    order = list(scores.rank(ascending=False, method="first").sort_values().index)
    actions = {
        ticker: strategy.evaluate(price_data[ticker].loc[:SESSION], ticker, {}).action
        for ticker in TICKERS
    }

    assert [actions[t] for t in order[:2]] == ["BUY", "BUY"]
    assert actions[order[2]] == "HOLD"
    assert all(actions[t] == "SELL" for t in order[3:])


def test_build_qvm_strategy_always_holds_at_least_one_name(price_data, fact_frames):
    strategy = build_qvm_strategy(
        price_data, PointInTimeFundamentals.from_frames(fact_frames), SECTORS, top_fraction=0.01
    )

    assert strategy.top_k == 1
    assert strategy.exit_rank == 2


def test_watchlist_lists_the_best_names_of_the_latest_session(price_data, fact_frames):
    provider = _provider(price_data, fact_frames)

    watchlist = qvm_watchlist(provider, top=3)

    assert list(watchlist.columns) == [
        "ticker", "sector", "quality_score", "value_score", "momentum_score", "qvm_score",
    ]
    assert len(watchlist) == 3
    assert watchlist["qvm_score"].is_monotonic_decreasing
    latest = provider.table_on(provider.calendar[-1])["qvm_score"]
    assert watchlist["qvm_score"].iloc[0] == latest.max()
    assert set(watchlist["sector"]) <= {"Technology", "Energy"}
