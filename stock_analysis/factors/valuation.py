"""Valuation models as a margin-of-safety input to the value factor.

Pure versions of the EPV and reverse-DCF calculations in
`ai/fundamental_analysis/valuation_methods.py`, taking plain numbers so they
can run on point-in-time fundamentals, and using a discount rate that depends
on the sector instead of a flat 10 %.
"""

from collections.abc import Mapping

# Rough required returns by sector (yfinance sector names): steadier cash flows
# get a lower rate. Judgement values, not estimates — good enough to stop a
# utility and a software company being discounted alike.
SECTOR_DISCOUNT_RATES: dict[str, float] = {
    "Utilities": 0.070,
    "Consumer Defensive": 0.075,
    "Real Estate": 0.080,
    "Healthcare": 0.085,
    "Financial Services": 0.090,
    "Industrials": 0.090,
    "Communication Services": 0.090,
    "Basic Materials": 0.095,
    "Consumer Cyclical": 0.095,
    "Energy": 0.100,
    "Technology": 0.100,
}
DEFAULT_DISCOUNT_RATE = 0.09
DEFAULT_TAX_RATE = 0.21
TERMINAL_GROWTH = 0.03
_NAN = float("nan")


def discount_rate_for(sector: str | None) -> float:
    return SECTOR_DISCOUNT_RATES.get(sector or "", DEFAULT_DISCOUNT_RATE)


def _known(value: float | None) -> bool:
    return value is not None and value == value  # NaN != NaN


def epv_to_price(
    *,
    operating_income: float,
    market_cap: float,
    discount_rate: float,
    total_debt: float = 0.0,
    cash: float = 0.0,
    tax_rate: float = DEFAULT_TAX_RATE,
) -> float:
    """Earnings power value of the equity divided by its market value.

    EPV values the business as if today's operating profit continued for ever
    with no growth: EBIT x (1 - tax) / discount rate, less debt, plus cash.
    Above 1 the market pays less than the no-growth value (a margin of safety);
    a loss-making company comes out negative and so ranks as expensive.
    """
    if not (_known(operating_income) and _known(market_cap)) or market_cap <= 0:
        return _NAN
    enterprise_value = operating_income * (1 - tax_rate) / discount_rate
    equity_value = enterprise_value - (total_debt if _known(total_debt) else 0.0)
    equity_value += cash if _known(cash) else 0.0
    return equity_value / market_cap


def dcf_value(
    free_cash_flow: float,
    growth: float,
    discount_rate: float,
    *,
    years: int = 10,
    terminal_growth: float = TERMINAL_GROWTH,
) -> float:
    """Present value of `free_cash_flow` growing at `growth` for `years`, then at `terminal_growth`."""
    value = 0.0
    for year in range(1, years + 1):
        value += free_cash_flow * (1 + growth) ** year / (1 + discount_rate) ** year
    terminal = (
        free_cash_flow * (1 + growth) ** years * (1 + terminal_growth)
        / (discount_rate - terminal_growth)
    )
    return value + terminal / (1 + discount_rate) ** years


def implied_growth(
    *,
    free_cash_flow: float,
    market_cap: float,
    discount_rate: float,
    years: int = 10,
    terminal_growth: float = TERMINAL_GROWTH,
    bounds: tuple[float, float] = (-0.5, 1.0),
) -> float:
    """Reverse DCF: the yearly cash-flow growth the market value already assumes.

    Lower means less has to go right. NaN without positive free cash flow (no
    growth rate turns a negative cash flow into a positive value); clamped to
    `bounds` when the price is outside what they can explain.
    """
    if not (_known(free_cash_flow) and _known(market_cap)):
        return _NAN
    if free_cash_flow <= 0 or market_cap <= 0 or discount_rate <= terminal_growth:
        return _NAN
    low, high = bounds
    if dcf_value(free_cash_flow, low, discount_rate, years=years, terminal_growth=terminal_growth) >= market_cap:
        return low
    if dcf_value(free_cash_flow, high, discount_rate, years=years, terminal_growth=terminal_growth) <= market_cap:
        return high
    for _ in range(60):
        mid = (low + high) / 2
        value = dcf_value(
            free_cash_flow, mid, discount_rate, years=years, terminal_growth=terminal_growth
        )
        if value < market_cap:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def margin_of_safety(
    values: Mapping[str, float], market_cap: float, sector: str | None = None
) -> dict[str, float]:
    """Price against EPV and the growth implied by the price, from point-in-time values.

    `values` uses the names of `FundamentalSnapshot.values` (`operating_income`,
    `free_cash_flow`, `total_debt`, `cash`).
    """
    rate = discount_rate_for(sector)
    return {
        "epv_to_price": epv_to_price(
            operating_income=values.get("operating_income", _NAN),
            market_cap=market_cap,
            discount_rate=rate,
            total_debt=values.get("total_debt", 0.0),
            cash=values.get("cash", 0.0),
        ),
        "implied_growth": implied_growth(
            free_cash_flow=values.get("free_cash_flow", _NAN),
            market_cap=market_cap,
            discount_rate=rate,
        ),
    }
