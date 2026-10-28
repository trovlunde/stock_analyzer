"""Fixture-only tests for QVM sector-relative scoring."""

import pandas as pd
import pytest

from stock_analysis.research.qvm import filter_super_stocks, score_qvm


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
