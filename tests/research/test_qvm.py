"""Fixture-only tests for QVM sector-relative scoring."""

import pandas as pd
import pytest

from stock_analysis.research.qvm import (
    collect_qvm_rows,
    filter_super_stocks,
    find_qvm_stocks,
    score_qvm,
)


def _tech_sector_frame() -> pd.DataFrame:
    """Four tech names with clear Q/V/M ordering."""
    return pd.DataFrame(
        [
            {
                "ticker": "CHEAP_HQ_MOM",
                "sector": "Technology",
                "returnOnEquity": 0.40,
                "profitMargins": 0.30,
                "operatingMargins": 0.35,
                "debtToEquity": 10.0,
                "trailingPE": 8.0,
                "forwardPE": 7.0,
                "priceToBook": 1.0,
                "fiftyTwoWeekChange": 0.80,
            },
            {
                "ticker": "MID",
                "sector": "Technology",
                "returnOnEquity": 0.20,
                "profitMargins": 0.15,
                "operatingMargins": 0.18,
                "debtToEquity": 50.0,
                "trailingPE": 20.0,
                "forwardPE": 18.0,
                "priceToBook": 3.0,
                "fiftyTwoWeekChange": 0.20,
            },
            {
                "ticker": "MID2",
                "sector": "Technology",
                "returnOnEquity": 0.18,
                "profitMargins": 0.12,
                "operatingMargins": 0.16,
                "debtToEquity": 60.0,
                "trailingPE": 22.0,
                "forwardPE": 20.0,
                "priceToBook": 3.5,
                "fiftyTwoWeekChange": 0.10,
            },
            {
                "ticker": "EXP_LQ_WEAK",
                "sector": "Technology",
                "returnOnEquity": 0.05,
                "profitMargins": 0.02,
                "operatingMargins": 0.03,
                "debtToEquity": 200.0,
                "trailingPE": 50.0,
                "forwardPE": 45.0,
                "priceToBook": 10.0,
                "fiftyTwoWeekChange": -0.20,
            },
        ]
    )


def test_score_qvm_ranks_best_name_highest_within_sector():
    scored = score_qvm(_tech_sector_frame())
    by_ticker = scored.set_index("ticker")

    assert by_ticker.loc["CHEAP_HQ_MOM", "quality_score"] > by_ticker.loc[
        "EXP_LQ_WEAK", "quality_score"
    ]
    assert by_ticker.loc["CHEAP_HQ_MOM", "value_score"] > by_ticker.loc[
        "EXP_LQ_WEAK", "value_score"
    ]
    assert by_ticker.loc["CHEAP_HQ_MOM", "momentum_score"] > by_ticker.loc[
        "EXP_LQ_WEAK", "momentum_score"
    ]
    assert by_ticker.loc["CHEAP_HQ_MOM", "qvm_score"] == pytest.approx(
        (
            by_ticker.loc["CHEAP_HQ_MOM", "quality_score"]
            + by_ticker.loc["CHEAP_HQ_MOM", "value_score"]
            + by_ticker.loc["CHEAP_HQ_MOM", "momentum_score"]
        )
        / 3
    )


def test_inverted_metrics_penalize_high_debt_and_high_pe():
    scored = score_qvm(_tech_sector_frame())
    by_ticker = scored.set_index("ticker")

    # High D/E → lower quality; high multiples → lower value
    assert by_ticker.loc["CHEAP_HQ_MOM", "quality_score"] > by_ticker.loc[
        "EXP_LQ_WEAK", "quality_score"
    ]
    assert by_ticker.loc["CHEAP_HQ_MOM", "value_score"] > by_ticker.loc[
        "EXP_LQ_WEAK", "value_score"
    ]


def test_tiny_sectors_excluded_from_scoring():
    df = pd.DataFrame(
        [
            {
                "ticker": "A",
                "sector": "Tiny",
                "returnOnEquity": 0.5,
                "profitMargins": 0.4,
                "operatingMargins": 0.4,
                "debtToEquity": 5.0,
                "trailingPE": 5.0,
                "forwardPE": 5.0,
                "priceToBook": 1.0,
                "fiftyTwoWeekChange": 0.5,
            },
            {
                "ticker": "B",
                "sector": "Tiny",
                "returnOnEquity": 0.1,
                "profitMargins": 0.1,
                "operatingMargins": 0.1,
                "debtToEquity": 100.0,
                "trailingPE": 40.0,
                "forwardPE": 40.0,
                "priceToBook": 8.0,
                "fiftyTwoWeekChange": -0.1,
            },
        ]
    )
    scored = score_qvm(df, min_sector_size=4)
    assert scored["quality_score"].isna().all()
    assert scored["qvm_score"].isna().all()


def test_filter_super_stocks_threshold_and_sort_order():
    scored = score_qvm(_tech_sector_frame())
    # With 4 names, top rank is 100 and bottom is 25 (pct ranks).
    # Only CHEAP_HQ_MOM should clear a high threshold on all three.
    supers = filter_super_stocks(scored, min_factor_score=90.0)
    assert len(supers) == 1
    assert supers[0]["ticker"] == "CHEAP_HQ_MOM"

    # Lower threshold may include more; still sorted by qvm_score desc
    more = filter_super_stocks(scored, min_factor_score=50.0)
    assert len(more) >= 1
    scores = [row["qvm_score"] for row in more]
    assert scores == sorted(scores, reverse=True)


def test_filter_super_stocks_empty_and_optional_qvm_floor():
    scored = score_qvm(_tech_sector_frame())
    assert filter_super_stocks(scored, min_factor_score=101.0) == []

    with_floor = filter_super_stocks(
        scored, min_factor_score=50.0, min_qvm_score=99.0
    )
    assert all(row["qvm_score"] >= 99.0 for row in with_floor)


def test_negative_earnings_do_not_rank_as_cheap():
    """A negative P/E is the lowest multiple in the sector; as a yield it is the worst."""
    df = _tech_sector_frame()
    loss_maker = df["ticker"] == "EXP_LQ_WEAK"
    df.loc[loss_maker, "trailingPE"] = float("nan")  # Yahoo gives no trailing P/E for a loss
    df.loc[loss_maker, "forwardPE"] = -4.0
    df.loc[loss_maker, "priceToBook"] = 3.2
    df["trailingEps"] = [5.0, 2.0, 1.8, -6.0]
    df["price"] = [40.0, 40.0, 40.0, 40.0]

    by_ticker = score_qvm(df).set_index("ticker")

    assert by_ticker.loc["EXP_LQ_WEAK", "earnings_yield"] == pytest.approx(-0.15)
    assert by_ticker.loc["EXP_LQ_WEAK", "forward_earnings_yield"] == pytest.approx(-0.25)
    assert by_ticker.loc["EXP_LQ_WEAK", "value_score"] == by_ticker["value_score"].min()
    assert by_ticker.loc["EXP_LQ_WEAK", "value_score"] < by_ticker.loc["MID2", "value_score"]


def test_value_uses_yields_from_cash_flow_and_enterprise_value_when_available():
    df = _tech_sector_frame()[["ticker", "sector"]].copy()
    df["market_cap"] = 1000.0
    df["freeCashflow"] = [10.0, 80.0, 40.0, -20.0]
    df["ebitda"] = [50.0, 200.0, 100.0, 10.0]
    df["enterpriseValue"] = [1000.0, 1000.0, 1000.0, -50.0]

    by_ticker = score_qvm(df).set_index("ticker")

    assert by_ticker.loc["MID", "fcf_yield"] == pytest.approx(0.08)
    assert by_ticker.loc["MID", "value_score"] == 100.0
    # Negative free cash flow is the worst yield; a negative enterprise value gives no EBITDA yield
    assert by_ticker.loc["EXP_LQ_WEAK", "fcf_yield"] == pytest.approx(-0.02)
    assert pd.isna(by_ticker.loc["EXP_LQ_WEAK", "ebitda_ev"])
    assert by_ticker.loc["EXP_LQ_WEAK", "value_score"] == 25.0


def test_negative_debt_to_equity_is_not_scored_as_low_leverage():
    df = _tech_sector_frame()
    df.loc[df["ticker"] == "EXP_LQ_WEAK", "debtToEquity"] = -300.0  # negative equity

    scored = score_qvm(df)

    assert pd.isna(scored.set_index("ticker").loc["EXP_LQ_WEAK", "debt_to_equity"])
    assert scored.set_index("ticker").loc["CHEAP_HQ_MOM", "quality_score"] == 100.0


def test_momentum_is_12_1_when_prices_are_supplied_and_52_week_change_otherwise():
    df = _tech_sector_frame()
    # 52-week change says CHEAP_HQ_MOM is strongest; 12-1 momentum says the opposite
    without_prices = score_qvm(df).set_index("ticker")
    assert without_prices.loc["CHEAP_HQ_MOM", "momentum_score"] == 100.0

    df["momentum_12_1"] = [-0.30, 0.10, 0.20, 0.50]
    with_prices = score_qvm(df).set_index("ticker")
    assert with_prices.loc["CHEAP_HQ_MOM", "momentum_score"] == 25.0
    assert with_prices.loc["EXP_LQ_WEAK", "momentum_score"] == 100.0


def test_52_week_change_under_its_yahoo_key_still_works():
    df = _tech_sector_frame().rename(columns={"fiftyTwoWeekChange": "52WeekChange"})

    scored = score_qvm(df).set_index("ticker")

    assert scored.loc["CHEAP_HQ_MOM", "momentum_score"] == 100.0


class _FakeStock:
    def __init__(self, info):
        self.info = info


class _FakeTickers:
    def __init__(self, infos):
        self.tickers = {ticker: _FakeStock(info) for ticker, info in infos.items()}


def _info(sector="Technology", **overrides):
    info = {
        "currentPrice": 50.0,
        "marketCap": 5e9,
        "sector": sector,
        "longName": "Example",
        "returnOnEquity": 0.2,
        "trailingEps": 2.5,
        "forwardEps": 3.0,
        "freeCashflow": 2e8,
        "ebitda": 5e8,
        "enterpriseValue": 6e9,
        "priceToBook": 4.0,
        "52WeekChange": 0.1,
    }
    info.update(overrides)
    return info


def test_collect_rows_carries_yield_inputs_and_12_1_momentum_from_prices():
    index = pd.bdate_range("2023-01-02", periods=300)
    prices = pd.DataFrame({"Close": [100.0] * 47 + [100.0] + [150.0] * 252}, index=index)
    stocks = _FakeTickers({"AAA": _info(), "BBB": _info(), "TINY": _info(marketCap=1e6)})

    rows = collect_qvm_rows(stocks, price_data={"AAA": prices}).set_index("ticker")

    assert list(rows.index) == ["AAA", "BBB"]
    assert rows.loc["AAA", "trailingEps"] == 2.5
    assert rows.loc["AAA", "enterpriseValue"] == 6e9
    assert rows.loc["AAA", "momentum_12_1"] == pytest.approx(0.5)
    assert pd.isna(rows.loc["BBB", "momentum_12_1"])
    assert rows.loc["BBB", "fiftyTwoWeekChange"] == 0.1


def test_find_qvm_stocks_scores_a_loss_maker_below_its_profitable_peers():
    stocks = _FakeTickers(
        {
            "GOOD": _info(trailingEps=5.0, forwardEps=5.5, freeCashflow=5e8),
            "OK1": _info(trailingEps=2.5),
            "OK2": _info(trailingEps=2.0),
            "LOSS": _info(trailingEps=-8.0, forwardEps=-6.0, freeCashflow=-3e8, ebitda=-1e8),
        }
    )

    ranked = find_qvm_stocks(stocks, super_only=False)

    by_ticker = {row["ticker"]: row for row in ranked}
    assert by_ticker["LOSS"]["value_score"] == min(row["value_score"] for row in ranked)
    assert by_ticker["GOOD"]["value_score"] == max(row["value_score"] for row in ranked)
