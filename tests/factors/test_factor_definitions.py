"""Factor maths: momentum window, yields, winsorising, sector-neutral ranks, valuation."""

import numpy as np
import pandas as pd
import pytest

from stock_analysis.factors.prices import momentum_12_1, split_factor, unadjusted_close
from stock_analysis.factors.qvm import (
    earnings_variability,
    fundamental_metrics,
    score_factors,
    winsorise,
)
from stock_analysis.factors.valuation import (
    SECTOR_DISCOUNT_RATES,
    dcf_value,
    discount_rate_for,
    epv_to_price,
    implied_growth,
    margin_of_safety,
)


# --- momentum -----------------------------------------------------------------

def _close(values):
    return pd.Series(values, index=pd.bdate_range("2022-01-03", periods=len(values)), dtype=float)


def test_momentum_runs_from_252_sessions_ago_to_21_sessions_ago():
    values = np.full(300, 80.0)
    values[-253] = 100.0  # twelve months before the last bar
    values[-22] = 150.0  # one month before the last bar

    assert momentum_12_1(_close(values)) == pytest.approx(0.5)


def test_momentum_ignores_the_most_recent_month():
    values = np.linspace(100, 200, 300)
    crashed = values.copy()
    crashed[-21:] = 1.0

    assert momentum_12_1(_close(crashed)) == pytest.approx(momentum_12_1(_close(values)))


def test_momentum_needs_a_full_year_of_bars():
    assert np.isnan(momentum_12_1(_close(np.linspace(100, 200, 252))))
    assert not np.isnan(momentum_12_1(_close(np.linspace(100, 200, 253))))


# --- traded price from adjusted price -------------------------------------------

def _adjusted_frame():
    """Ten sessions with a dividend on day 3 and a 2-for-1 split on day 6, adjusted the way yfinance does it."""
    traded = np.array([100, 102, 104, 103, 105, 106, 54, 55, 56, 57], dtype=float)
    split_adjusted = traded.copy()
    split_adjusted[:6] /= 2
    dividend = 1.0  # per share on today's share basis, as yfinance reports it
    adjusted = split_adjusted.copy()
    adjusted[:3] *= 1 - dividend / split_adjusted[2]
    dividends = np.zeros(10)
    dividends[3] = dividend
    splits = np.zeros(10)
    splits[6] = 2.0
    index = pd.bdate_range("2023-01-02", periods=10)
    frame = pd.DataFrame({"Close": adjusted, "Dividends": dividends, "Stock Splits": splits}, index=index)
    return frame, traded


def test_unadjusted_close_recovers_the_traded_price():
    frame, traded = _adjusted_frame()

    np.testing.assert_allclose(unadjusted_close(frame).to_numpy(), traded)


def test_unadjusted_close_without_corporate_action_columns_is_the_close():
    frame, _ = _adjusted_frame()

    pd.testing.assert_series_equal(unadjusted_close(frame[["Close"]]), frame["Close"])


def test_split_factor_counts_splits_after_the_share_count_up_to_the_session():
    frame, _ = _adjusted_frame()
    index = frame.index

    assert split_factor(frame, index[2], index[9]) == 2.0
    assert split_factor(frame, index[2], index[5]) == 1.0  # split not yet happened
    assert split_factor(frame, index[6], index[9]) == 1.0  # counted after the split
    assert split_factor(frame[["Close"]], index[0], index[9]) == 1.0


# --- metrics from fundamentals ----------------------------------------------------

VALUES = {
    "net_income": 100.0,
    "revenue": 1000.0,
    "operating_income": 150.0,
    "gross_profit": 400.0,
    "ebitda": 200.0,
    "free_cash_flow": 80.0,
    "operating_cash_flow": 130.0,
    "total_assets": 2000.0,
    "total_equity": 500.0,
    "total_debt": 300.0,
    "cash": 100.0,
}


def test_value_metrics_are_yields_on_market_value():
    metrics = fundamental_metrics(VALUES, 1000.0)

    assert metrics["earnings_yield"] == pytest.approx(0.10)
    assert metrics["fcf_yield"] == pytest.approx(0.08)
    assert metrics["ebitda_ev"] == pytest.approx(200 / 1200)
    assert metrics["book_to_price"] == pytest.approx(0.5)


def test_quality_metrics():
    metrics = fundamental_metrics(VALUES, 1000.0)

    assert metrics["roe"] == pytest.approx(0.2)
    assert metrics["profit_margin"] == pytest.approx(0.1)
    assert metrics["operating_margin"] == pytest.approx(0.15)
    assert metrics["gross_profitability"] == pytest.approx(0.2)
    assert metrics["leverage"] == pytest.approx(0.15)
    # Earnings below cash flow: negative accruals, the good direction
    assert metrics["accruals"] == pytest.approx((100 - 130) / 2000)


def test_a_loss_gives_a_negative_yield_not_a_missing_or_flattering_one():
    metrics = fundamental_metrics({**VALUES, "net_income": -100.0, "free_cash_flow": -50.0}, 1000.0)

    assert metrics["earnings_yield"] == pytest.approx(-0.10)
    assert metrics["fcf_yield"] == pytest.approx(-0.05)


def test_ratios_with_meaningless_denominators_are_missing():
    negative_equity = fundamental_metrics({**VALUES, "total_equity": -200.0, "net_income": -50.0}, 1000.0)
    assert np.isnan(negative_equity["roe"])  # loss / negative equity would look like a good ROE
    assert negative_equity["book_to_price"] == pytest.approx(-0.2)

    cash_rich = fundamental_metrics({**VALUES, "cash": 5000.0}, 1000.0)
    assert np.isnan(cash_rich["ebitda_ev"])  # enterprise value below zero

    no_price = fundamental_metrics(VALUES, float("nan"))
    assert np.isnan(no_price["earnings_yield"])
    assert no_price["roe"] == pytest.approx(0.2)


def test_missing_debt_and_cash_count_as_zero():
    values = {k: v for k, v in VALUES.items() if k not in ("total_debt", "cash")}

    metrics = fundamental_metrics(values, 1000.0)

    assert metrics["ebitda_ev"] == pytest.approx(0.2)
    assert metrics["leverage"] == 0.0


def test_earnings_variability_ignores_seasonality_and_needs_eight_quarters():
    seasonal = pd.Series([10.0, 20.0, 30.0, 80.0] * 3)
    assert earnings_variability(seasonal, 1000.0) == pytest.approx(0.0)

    erratic = pd.Series([10.0, 20.0, 30.0, 80.0, 60.0, -40.0, 90.0, 10.0, 5.0, 70.0, -20.0, 95.0])
    assert earnings_variability(erratic, 1000.0) > 0.01

    assert np.isnan(earnings_variability(seasonal.iloc[:7], 1000.0))
    assert np.isnan(earnings_variability(seasonal, float("nan")))


# --- winsorising and scoring ----------------------------------------------------

def test_winsorise_clips_the_tails_and_keeps_missing_values_missing():
    series = pd.Series([*range(1, 20), 10_000.0, np.nan])

    clipped = winsorise(series, 0.05, 0.95)

    # One value in twenty is pulled in at each end, to the nearest value kept
    assert clipped.max() == 19.0
    assert clipped.min() == 2.0
    assert clipped.iloc[9] == 10.0
    assert np.isnan(clipped.iloc[-1])
    # Too few values for 5 % to be a whole observation: nothing changes
    pd.testing.assert_series_equal(winsorise(pd.Series([1.0, 2.0, 900.0])), pd.Series([1.0, 2.0, 900.0]))


def _two_sector_frame():
    """Energy is cheap across the board; technology is expensive across the board."""
    rows = []
    for position in range(4):
        rows.append({"ticker": f"E{position}", "sector": "Energy", "earnings_yield": 0.10 + 0.01 * position,
                     "roe": 0.10 + 0.01 * position, "momentum_12_1": 0.05 * position})
        rows.append({"ticker": f"T{position}", "sector": "Technology", "earnings_yield": 0.01 + 0.005 * position,
                     "roe": 0.30 + 0.01 * position, "momentum_12_1": 0.20 * position})
    return pd.DataFrame(rows).set_index("ticker")


def test_ranks_are_within_sector():
    scored = score_factors(_two_sector_frame())

    # The cheapest technology stock scores as high as the cheapest energy stock,
    # although every energy stock has a higher earnings yield than every technology stock
    assert scored.loc["T3", "value_score"] == scored.loc["E3", "value_score"] == 100.0
    assert scored.loc["T0", "value_score"] == scored.loc["E0", "value_score"] == 25.0
    assert scored.loc["T3", "qvm_score"] == pytest.approx(100.0)


def test_ranking_as_one_group_lets_the_cheap_sector_dominate():
    scored = score_factors(_two_sector_frame(), sector_col=None)

    assert scored.loc[["E0", "E1", "E2", "E3"], "value_score"].min() > scored.loc[
        ["T0", "T1", "T2", "T3"], "value_score"
    ].max()


def test_small_sectors_are_not_scored():
    frame = _two_sector_frame().drop(index=["T2", "T3"])

    scored = score_factors(frame, min_sector_size=4)

    assert scored.loc[["T0", "T1"], "qvm_score"].isna().all()
    assert scored.loc[["E0", "E1", "E2", "E3"], "qvm_score"].notna().all()


def test_loss_maker_ranks_as_the_most_expensive():
    frame = _two_sector_frame()
    frame.loc["E3", "earnings_yield"] = -0.50

    scored = score_factors(frame)

    assert scored.loc["E3", "value_score"] == 25.0


def test_lower_is_better_metrics_are_inverted():
    frame = _two_sector_frame()
    frame["accruals"] = [0.0, 0.0, 0.01, 0.01, 0.02, 0.02, 0.03, 0.03]  # E0/T0 lowest
    frame = frame.drop(columns="roe")

    scored = score_factors(frame)

    assert scored.loc["E0", "quality_score"] == 100.0
    assert scored.loc["E3", "quality_score"] == 25.0


def test_require_all_factors_leaves_partly_scored_stocks_unranked():
    frame = _two_sector_frame()
    frame.loc["E0", "earnings_yield"] = np.nan

    lenient = score_factors(frame)
    strict = score_factors(frame, require_all_factors=True)

    assert not np.isnan(lenient.loc["E0", "qvm_score"])
    assert np.isnan(strict.loc["E0", "qvm_score"])
    assert not np.isnan(strict.loc["E1", "qvm_score"])


def test_winsorising_ties_extreme_outliers_before_ranking():
    frame = pd.DataFrame(
        {"sector": "All", "earnings_yield": [*np.linspace(0.01, 0.20, 38), 5.0, 9.0]},
        index=[f"S{i}" for i in range(40)],
    )

    raw = score_factors(frame, winsor_limits=None)
    clipped = score_factors(frame, winsor_limits=(0.05, 0.95))

    assert raw.loc["S39", "value_score"] > raw.loc["S38", "value_score"]
    assert clipped.loc["S39", "value_score"] == clipped.loc["S38", "value_score"]


# --- valuation models -------------------------------------------------------------

def test_sector_discount_rates_replace_the_flat_ten_percent():
    assert discount_rate_for("Utilities") < discount_rate_for("Technology")
    assert discount_rate_for("Somewhere Else") == discount_rate_for(None)
    assert all(0.05 < rate <= 0.12 for rate in SECTOR_DISCOUNT_RATES.values())


def test_epv_to_price():
    ratio = epv_to_price(
        operating_income=100.0, market_cap=500.0, discount_rate=0.10,
        total_debt=200.0, cash=50.0, tax_rate=0.21,
    )
    # EPV = 100 * 0.79 / 0.10 = 790; equity = 790 - 200 + 50 = 640
    assert ratio == pytest.approx(640 / 500)

    assert epv_to_price(operating_income=-100.0, market_cap=500.0, discount_rate=0.10) < 0
    assert np.isnan(epv_to_price(operating_income=100.0, market_cap=0.0, discount_rate=0.10))


def test_implied_growth_inverts_the_dcf():
    price = dcf_value(100.0, 0.07, 0.09)

    assert implied_growth(free_cash_flow=100.0, market_cap=price, discount_rate=0.09) == pytest.approx(0.07, abs=1e-6)
    # A higher price needs more growth to justify it
    assert implied_growth(free_cash_flow=100.0, market_cap=price * 2, discount_rate=0.09) > 0.07


def test_implied_growth_is_missing_without_positive_cash_flow_and_clamped_at_the_bounds():
    assert np.isnan(implied_growth(free_cash_flow=-10.0, market_cap=1000.0, discount_rate=0.09))
    assert implied_growth(free_cash_flow=100.0, market_cap=1e12, discount_rate=0.09) == 1.0
    assert implied_growth(free_cash_flow=100.0, market_cap=1.0, discount_rate=0.09) == -0.5


def test_margin_of_safety_uses_the_sector_rate():
    utility = margin_of_safety(VALUES, 1000.0, "Utilities")
    tech = margin_of_safety(VALUES, 1000.0, "Technology")

    assert utility["epv_to_price"] > tech["epv_to_price"]
    assert utility["implied_growth"] < tech["implied_growth"]
    assert fundamental_metrics(VALUES, 1000.0, sector="Utilities")["epv_to_price"] == pytest.approx(
        utility["epv_to_price"]
    )
