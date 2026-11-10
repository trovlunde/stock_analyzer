"""Deterministic prices and filings for an eight-stock, two-sector universe."""

import numpy as np
import pandas as pd

TICKERS = ["T0", "T1", "T2", "T3", "E0", "E1", "E2", "E3"]
SECTORS = {ticker: ("Technology" if ticker.startswith("T") else "Energy") for ticker in TICKERS}
# T3 loses money in every quarter
LOSS_MAKER = "T3"


def make_prices(position: int, *, bars: int = 420, start: str = "2022-01-03") -> pd.DataFrame:
    index = pd.bdate_range(start, periods=bars)
    steps = np.arange(bars)
    drift = (position - 3) * 0.0004
    close = 50 * (1 + position) * np.exp(drift * steps) * (1 + 0.03 * np.sin(steps / 9 + position))
    return pd.DataFrame(
        {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
        index=index,
    )


def make_price_data() -> dict[str, pd.DataFrame]:
    return {ticker: make_prices(position) for position, ticker in enumerate(TICKERS)}


def _row(concept, start, end, filed, value):
    return {
        "concept": concept,
        "tag": concept,
        "period_start": start,
        "period_end": end,
        "filed": filed,
        "value": float(value),
        "form": "10-Q",
    }


def make_facts(position: int, *, years=(2020, 2021, 2022, 2023)) -> pd.DataFrame:
    """Quarterly filings 35 days after each quarter, the 10-K 50 days after year end.

    Income items come as single quarters plus the annual total; cash-flow items
    as year-to-date figures only — the shapes EDGAR actually has.
    """
    ticker = TICKERS[position]
    sign = -1.0 if ticker == LOSS_MAKER else 1.0
    scale = 1e6 * (1 + position)
    rows = []
    quarter_number = 0
    for year in years:
        year_start = pd.Timestamp(year=year, month=1, day=1)
        income_ytd = revenue_ytd = operating_ytd = cash_flow_ytd = 0.0
        for quarter in range(4):
            start = year_start + pd.DateOffset(months=3 * quarter)
            end = start + pd.DateOffset(months=3) - pd.Timedelta(1, unit="D")
            filed = end + pd.Timedelta(50 if quarter == 3 else 35, unit="D")
            growth = 1 + 0.02 * quarter_number + 0.05 * ((quarter_number * (position + 1)) % 3)
            revenue = 10 * scale * growth
            income = sign * (0.5 + 0.2 * (position % 4)) * scale * growth
            operating = income * 1.3
            cash_flow = income * (1.1 + 0.1 * (position % 3)) + 0.2 * scale
            income_ytd += income
            revenue_ytd += revenue
            operating_ytd += operating
            cash_flow_ytd += cash_flow
            if quarter < 3:
                rows += [
                    _row("net_income", start, end, filed, income),
                    _row("revenue", start, end, filed, revenue),
                    _row("operating_income", start, end, filed, operating),
                ]
            else:
                rows += [
                    _row("net_income", year_start, end, filed, income_ytd),
                    _row("revenue", year_start, end, filed, revenue_ytd),
                    _row("operating_income", year_start, end, filed, operating_ytd),
                ]
            rows += [
                _row("operating_cash_flow", year_start, end, filed, cash_flow_ytd),
                _row("capex", year_start, end, filed, 0.1 * revenue_ytd / 10),
                _row("depreciation_amortization", year_start, end, filed, 0.08 * revenue_ytd / 10),
                _row("gross_profit", year_start, end, filed, (0.3 + 0.05 * (position % 5)) * revenue_ytd),
                _row("total_assets", None, end, filed, 40 * scale),
                _row("total_equity", None, end, filed, (10 + 2 * position) * scale),
                _row("long_term_debt", None, end, filed, (2 + position % 4) * scale),
                _row("cash", None, end, filed, 3 * scale),
                _row("shares_outstanding", None, end, filed, 1e6),
            ]
            quarter_number += 1
    return pd.DataFrame(rows)


def make_fact_frames() -> dict[str, pd.DataFrame]:
    return {ticker: make_facts(position) for position, ticker in enumerate(TICKERS)}
