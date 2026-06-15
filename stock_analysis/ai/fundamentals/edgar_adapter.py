import edgar
import pandas as pd

_CONCEPT_TO_YFINANCE: dict[str, str] = {
    "Revenue": "Total Revenue",
    "NetIncome": "Net Income",
    "GrossProfit": "Gross Profit",
    "OperatingIncomeLoss": "Operating Income",
    "CostOfGoodsAndServicesSold": "Cost Of Revenue",
    "CostOfRevenue": "Cost Of Revenue",
    "DepreciationExpense": "Depreciation",
    "DepreciationAmortizationCF": "Depreciation",
}

_N_ANNUAL = 4
_N_QUARTERLY = 8
_METADATA_COLS = frozenset({"label", "concept", "standard_concept", "preferred_sign"})


def _raw_to_financials_df(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert StitchedStatement.to_dataframe() output to yfinance-compatible format.

    Expects columns: label, concept, standard_concept, preferred_sign, <date-strings...>
    Returns: index=yfinance line-item names, columns=pd.Timestamp periods.
    """
    period_cols = [c for c in raw.columns if c not in _METADATA_COLS]
    if not period_cols:
        return pd.DataFrame()

    rows: dict[str, pd.Series] = {}
    for _, row in raw.iterrows():
        sc = row.get("standard_concept")
        if sc in _CONCEPT_TO_YFINANCE:
            yf_label = _CONCEPT_TO_YFINANCE[sc]
            if yf_label not in rows:
                rows[yf_label] = row[period_cols]

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows).T
    result.columns = pd.to_datetime(result.columns)

    if "Operating Income" in result.index and "Depreciation" in result.index:
        result.loc["EBIT"] = result.loc["Operating Income"]
        result.loc["EBITDA"] = result.loc["Operating Income"] + result.loc["Depreciation"]

    return result


class EdgarAdapter:
    def get_annual_financials(self, ticker: str) -> pd.DataFrame:
        try:
            company = edgar.Company(ticker)
            filings = company.get_filings(form="10-K")
        except edgar.CompanyNotFoundError:
            return pd.DataFrame()
        if filings.empty:
            return pd.DataFrame()
        multi = edgar.MultiFinancials.extract(filings.head(_N_ANNUAL))
        stmt = multi.income_statement()
        if stmt is None:
            return pd.DataFrame()
        return _raw_to_financials_df(stmt.to_dataframe())

    def get_quarterly_financials(self, ticker: str) -> pd.DataFrame:
        try:
            company = edgar.Company(ticker)
            filings = company.get_filings(form="10-Q")
        except edgar.CompanyNotFoundError:
            return pd.DataFrame()
        if filings.empty:
            return pd.DataFrame()
        multi = edgar.MultiFinancials.extract(filings.head(_N_QUARTERLY))
        stmt = multi.income_statement()
        if stmt is None:
            return pd.DataFrame()
        return _raw_to_financials_df(stmt.to_dataframe())
