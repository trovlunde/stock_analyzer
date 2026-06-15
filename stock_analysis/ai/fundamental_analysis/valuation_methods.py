import pandas as pd
import numpy as np
from typing import Dict, Optional, List


class StockValuation:
    """A class containing various methods for stock valuation."""

    def __init__(self, ticker: str, provider=None):
        """Initialize with ticker symbol and optional MarketDataProvider."""
        from stock_analysis.market_data import YFinanceProvider
        self.ticker = (provider or YFinanceProvider()).get_raw_ticker(ticker)
        self._load_financial_data()

    def _load_financial_data(self):
        """Load all necessary financial data for valuation calculations."""
        try:
            self.info = self.ticker.info
            self.financials = self.ticker.financials
            self.quarterly_financials = self.ticker.quarterly_financials
            self.balance_sheet = self.ticker.balance_sheet
            self.quarterly_balance = self.ticker.quarterly_balance_sheet
            self.cashflow = self.ticker.cashflow
            self.quarterly_cashflow = self.ticker.quarterly_cashflow

            # Get current market price
            self.current_price = self.info.get('currentPrice', 0)
            self.shares_outstanding = self.info.get('sharesOutstanding', 0)
            self.market_cap = self.info.get('marketCap', 0)
        except Exception as e:
            print(f"Error loading financial data: {str(e)}")
            raise

    def asset_based_valuation(self) -> Dict[str, float]:
        """
        Calculate asset-based valuation.
        Current Value = (Asset Value) / (1 – Debt Ratio)

        Best suited for:
        - Asset-heavy companies (manufacturing, real estate, commodities)
        - Companies with significant tangible assets
        - Companies trading below book value
        - Companies that may be liquidation candidates
        """
        try:
            total_assets = float(
                self.balance_sheet.loc['Total Assets'].iloc[0])
            total_liabilities = float(
                self.balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0])
            debt_ratio = total_liabilities / total_assets
            current_value = total_assets / (1 - debt_ratio)

            return {
                'current_value': current_value,
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'debt_ratio': debt_ratio
            }
        except Exception as e:
            print(f"Error in asset-based valuation: {str(e)}")
            return {}

    def income_based_valuation(self, discount_rate: float = 0.10, projection_years: int = 5) -> Dict[str, float]:
        """
        Calculate income-based valuation.
        Present Value = (Annual Income / (1 + Discount Rate)^(1/number of years))

        Best suited for:
        - Companies with stable, predictable earnings
        - Mature companies with consistent income streams
        - Companies with strong operating history
        - Service-based businesses with recurring revenue
        """
        try:
            annual_income = float(self.financials.loc['Net Income'].iloc[0])
            present_value = annual_income / \
                (1 + discount_rate)**(1/projection_years)

            # Calculate projected values for each year
            projected_values = []
            for year in range(1, projection_years + 1):
                projected_value = annual_income / (1 + discount_rate)**(1/year)
                projected_values.append(projected_value)

            return {
                'present_value': present_value,
                'annual_income': annual_income,
                'projected_values': projected_values,
                'discount_rate': discount_rate,
                'projection_years': projection_years
            }
        except Exception as e:
            print(f"Error in income-based valuation: {str(e)}")
            return {}

    def market_based_valuation(self) -> Dict[str, float]:
        """
        Calculate market-based valuation.
        For service companies: CV = (EBITDA x 1.5) - (current liabilities x 0.5)
        For retail/trading: V = (EBITDA * 1.3) / (Revenue - COGS)

        Best suited for:
        - Companies with comparable peers in the market
        - Industries with standardized business models
        - Companies in mature markets with established multiples
        - Both service and retail/trading companies
        """
        try:
            # Calculate EBITDA
            ebitda = float(self.financials.loc['EBITDA'].iloc[0])
            current_liabilities = float(
                self.balance_sheet.loc['Current Liabilities'].iloc[0])
            revenue = float(self.financials.loc['Total Revenue'].iloc[0])
            try:
                cogs = float(self.financials.loc['Cost Of Revenue'].iloc[0])
            except:
                cogs = 0  # Some companies might not report COGS

            # Calculate both service and retail valuations
            service_valuation = (ebitda * 1.5) - (current_liabilities * 0.5)
            retail_valuation = (ebitda * 1.3) / (revenue -
                                                 cogs) if (revenue - cogs) != 0 else 0

            return {
                'service_valuation': service_valuation,
                'retail_valuation': retail_valuation,
                'ebitda': ebitda,
                'current_liabilities': current_liabilities,
                'revenue': revenue,
                'cogs': cogs
            }
        except Exception as e:
            print(f"Error in market-based valuation: {str(e)}")
            return {}

    def dcf_valuation(self, discount_rate: float = 0.10, growth_rate: float = 0.03, projection_years: int = 5) -> Dict[str, float]:
        """
        Calculate Discounted Cash Flow valuation.
        Value = Future Cash Flow x Discount Rate / (1 + Discount Rate)^n

        Best suited for:
        - Companies with predictable future cash flows
        - Growth companies with positive free cash flow
        - Companies with stable or steadily growing operations
        - Businesses not heavily dependent on cyclical factors
        Reference: Old School Value's DCF method
        """
        try:
            # Get the latest free cash flow
            free_cash_flow = float(self.cashflow.loc['Free Cash Flow'].iloc[0])

            # Project future cash flows
            projected_cash_flows = []
            dcf_value = 0

            for year in range(1, projection_years + 1):
                # Calculate projected cash flow with growth
                projected_cf = free_cash_flow * (1 + growth_rate)**year
                # Calculate present value of this cash flow
                present_value = projected_cf / (1 + discount_rate)**year
                projected_cash_flows.append(present_value)
                dcf_value += present_value

            # Calculate terminal value
            terminal_value = (free_cash_flow * (1 + growth_rate)**projection_years *
                              (1 + growth_rate)) / (discount_rate - growth_rate)
            terminal_value_pv = terminal_value / \
                (1 + discount_rate)**projection_years

            # Add terminal value to DCF
            total_value = dcf_value + terminal_value_pv

            return {
                'dcf_value': dcf_value,
                'terminal_value': terminal_value_pv,
                'total_value': total_value,
                'projected_cash_flows': projected_cash_flows,
                'initial_free_cash_flow': free_cash_flow
            }
        except Exception as e:
            print(f"Error in DCF valuation: {str(e)}")
            return {}

    def equity_multiplier_valuation(self) -> Dict[str, float]:
        """
        Calculate Equity Multiplier valuation.
        Equity multiplier = current value / EBITDA

        Best suited for:
        - Companies with significant operational leverage
        - Financial institutions and banks
        - Companies where financial leverage is a key metric
        - Businesses with stable EBITDA margins
        """
        try:
            market_cap = float(self.info.get('marketCap', 0))
            ebitda = float(self.financials.loc['EBITDA'].iloc[0])
            equity_multiplier = market_cap / ebitda if ebitda != 0 else 0

            return {
                'equity_multiplier': equity_multiplier,
                'market_cap': market_cap,
                'ebitda': ebitda
            }
        except Exception as e:
            print(f"Error in equity multiplier valuation: {str(e)}")
            return {}

    def book_value_valuation(self) -> Dict[str, float]:
        """
        Calculate Book Value.
        Book Value = Total Assets - Total Liabilities

        Best suited for:
        - Financial institutions and banks
        - Asset-heavy businesses
        - Companies with significant tangible assets
        - Value stocks trading near book value
        """
        try:
            total_assets = float(
                self.balance_sheet.loc['Total Assets'].iloc[0])
            total_liabilities = float(
                self.balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0])
            book_value = total_assets - total_liabilities

            # Calculate book value per share
            shares_outstanding = float(self.info.get('sharesOutstanding', 0))
            book_value_per_share = book_value / \
                shares_outstanding if shares_outstanding != 0 else 0

            return {
                'book_value': book_value,
                'book_value_per_share': book_value_per_share,
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'shares_outstanding': shares_outstanding
            }
        except Exception as e:
            print(f"Error in book value valuation: {str(e)}")
            return {}

    def liquidation_value(self) -> Dict[str, float]:
        """
        Calculate Liquidation Value.
        Liquidation value = Current Assets - Current Liabilities

        Best suited for:
        - Distressed companies
        - Companies with significant tangible current assets
        - Businesses considering bankruptcy or restructuring
        - Value investments with strong asset protection
        """
        try:
            current_assets = float(
                self.balance_sheet.loc['Current Assets'].iloc[0])
            current_liabilities = float(
                self.balance_sheet.loc['Current Liabilities'].iloc[0])
            liquidation_value = current_assets - current_liabilities

            return {
                'liquidation_value': liquidation_value,
                'current_assets': current_assets,
                'current_liabilities': current_liabilities
            }
        except Exception as e:
            print(f"Error in liquidation value calculation: {str(e)}")
            return {}

    def breakup_value(self) -> Dict[str, float]:
        """
        Calculate Break-up Value.
        Break-up value = (Asset Value + Liability Value) - Total Debt

        Best suited for:
        - Conglomerates with diverse business units
        - Companies with distinct divisional operations
        - Businesses that might be worth more in parts than as a whole
        - Companies with undervalued subsidiaries or divisions
        """
        try:
            total_assets = float(
                self.balance_sheet.loc['Total Assets'].iloc[0])
            total_liabilities = float(
                self.balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0])
            total_debt = float(self.balance_sheet.loc['Total Debt'].iloc[0])

            breakup_value = (total_assets + total_liabilities) - total_debt

            return {
                'breakup_value': breakup_value,
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'total_debt': total_debt
            }
        except Exception as e:
            print(f"Error in break-up value calculation: {str(e)}")
            return {}

    def epv_valuation(self, cost_of_capital: float = 0.10) -> Dict[str, float]:
        """
        Calculate Earnings Power Value (EPV).
        EPV = Adjusted EBIT * (1 - tax_rate) / cost_of_capital

        Best suited for:
        - Companies with stable earnings
        - Businesses with competitive advantages
        - Companies with consistent margins
        - Mature companies in stable industries
        Reference: Based on Bruce Greenwald's EPV methodology
        """
        try:
            # Get EBIT and calculate normalized earnings
            ebit = float(self.financials.loc['EBIT'].iloc[0])
            tax_rate = 0.21  # Standard corporate tax rate, could be made configurable

            # Calculate maintenance capex (simplified as depreciation)
            try:
                depreciation = float(
                    self.financials.loc['Depreciation'].iloc[0])
            except:
                depreciation = 0

            # Calculate EPV
            adjusted_earnings = ebit * (1 - tax_rate)
            epv = adjusted_earnings / cost_of_capital

            return {
                'epv': epv,
                'adjusted_earnings': adjusted_earnings,
                'ebit': ebit,
                'cost_of_capital': cost_of_capital,
                'maintenance_capex': depreciation
            }
        except Exception as e:
            print(f"Error in EPV calculation: {str(e)}")
            return {}

    def reverse_dcf(self, current_price: Optional[float] = None) -> Dict[str, float]:
        """
        Calculate Reverse DCF to find implied growth rate.
        Solves for growth rate that would justify current stock price.

        Best suited for:
        - Growth companies where you want to validate market expectations
        - Companies where you want to check if current price is reasonable
        - Businesses with positive free cash flow
        - Companies where market might be over/under estimating growth
        Reference: Old School Value's Reverse DCF method
        """
        try:
            if current_price is None:
                current_price = self.current_price

            fcf = float(self.cashflow.loc['Free Cash Flow'].iloc[0])
            shares_outstanding = float(self.info.get('sharesOutstanding', 0))
            fcf_per_share = fcf / shares_outstanding if shares_outstanding != 0 else 0

            # Use binary search to find implied growth rate
            discount_rate = 0.10  # Standard 10% discount rate
            years = 10  # Standard 10-year projection

            def calculate_dcf(growth_rate):
                total_value = 0
                terminal_growth = 0.03  # Terminal growth rate

                # Calculate DCF with given growth rate
                for year in range(1, years + 1):
                    projected_fcf = fcf_per_share * (1 + growth_rate) ** year
                    total_value += projected_fcf / (1 + discount_rate) ** year

                # Add terminal value
                terminal_value = (fcf_per_share * (1 + growth_rate) ** years *
                                  (1 + terminal_growth)) / (discount_rate - terminal_growth)
                terminal_value_pv = terminal_value / \
                    (1 + discount_rate) ** years

                return total_value + terminal_value_pv

            # Binary search for implied growth rate
            low, high = -0.5, 0.5  # Search between -50% and 50% growth
            implied_growth = 0

            for _ in range(20):  # 20 iterations should give good precision
                mid = (low + high) / 2
                value = calculate_dcf(mid)

                if abs(value - current_price) < 0.01:
                    implied_growth = mid
                    break
                elif value < current_price:
                    low = mid
                else:
                    high = mid

                implied_growth = mid

            return {
                'implied_growth_rate': implied_growth * 100,  # Convert to percentage
                'current_price': current_price,
                'fcf_per_share': fcf_per_share,
                'discount_rate': discount_rate * 100,  # Convert to percentage
                'projection_years': years
            }
        except Exception as e:
            print(f"Error in reverse DCF calculation: {str(e)}")
            return {}

    def graham_valuation(self) -> Dict[str, float]:
        """
        Calculate Benjamin Graham's intrinsic value using both original and revised formulas.
        Original Formula: V = EPS × (8.5 + 2g) × 4.4 / Y
        Revised Formula: V = EPS × (7 + 1.5g) × 4.4 / Y

        Where:
        - V is the intrinsic value
        - EPS is the trailing 12-month earnings per share
        - 8.5 is the P/E ratio for a no-growth company
        - 7 is the "modern" P/E ratio for a no-growth company
        - g is the expected growth rate (estimated from historical data)
        - 4.4 was the average yield of high-grade corporate bonds in 1962
        - Y is the current yield on AAA corporate bonds

        Best suited for:
        - Value stocks with consistent earnings
        - Companies with moderate but steady growth
        - Companies with strong balance sheets
        - Businesses with long operating history
        Reference: Benjamin Graham's "The Intelligent Investor"
        """
        try:
            # Get required financial metrics
            net_income = float(self.financials.loc['Net Income'].iloc[0])
            eps = net_income / self.shares_outstanding if self.shares_outstanding != 0 else 0

            # Calculate historical growth rate (simplified)
            try:
                previous_year_income = float(
                    self.financials.loc['Net Income'].iloc[1])
                growth_rate = (net_income - previous_year_income) / \
                    abs(previous_year_income)
                # Cap the growth rate as Graham suggested
                growth_rate = min(max(growth_rate, 0),
                                  0.15)  # Between 0% and 15%
            except:
                growth_rate = 0.0

            # Constants from Graham's formula
            PE_BASE = 8.5  # P/E ratio for a no-growth company
            PE_BASE_MODERN = 7  # "Modern" P/E ratio for a no-growth company
            ORIGINAL_BOND_YIELD = 4.4  # Average AAA corporate bond yield in 1962
            # Current AAA corporate bond yield (this should be updated regularly)
            CURRENT_BOND_YIELD = 4.0

            # Calculate original Graham formula
            original_value = eps * \
                (PE_BASE + 2 * growth_rate * 100) * \
                (ORIGINAL_BOND_YIELD / CURRENT_BOND_YIELD)

            # Calculate revised Graham formula
            revised_value = eps * \
                (PE_BASE_MODERN + 1.5 * growth_rate * 100) * \
                (ORIGINAL_BOND_YIELD / CURRENT_BOND_YIELD)

            # Calculate Graham's margin of safety (using revised value)
            margin_of_safety = (revised_value - self.current_price) / \
                revised_value * 100 if revised_value != 0 else 0

            # Additional Graham criteria
            current_ratio = float(self.balance_sheet.loc['Current Assets'].iloc[0]) / \
                float(self.balance_sheet.loc['Current Liabilities'].iloc[0]) \
                if float(self.balance_sheet.loc['Current Liabilities'].iloc[0]) != 0 else 0

            try:
                debt_to_equity = float(self.balance_sheet.loc['Total Debt'].iloc[0]) / \
                    float(self.balance_sheet.loc['Stockholders Equity'].iloc[0]) \
                    if float(self.balance_sheet.loc['Stockholders Equity'].iloc[0]) != 0 else float('inf')
            except:
                debt_to_equity = float('inf')

            return {
                'original_graham_value': original_value,
                'revised_graham_value': revised_value,
                'margin_of_safety_percentage': margin_of_safety,
                'eps': eps,
                'growth_rate': growth_rate * 100,  # Convert to percentage
                'current_ratio': current_ratio,
                'debt_to_equity': debt_to_equity,
                'meets_graham_criteria': {
                    'adequate_size': self.market_cap > 100e6,  # Market cap > $100M
                    'strong_financial_condition': current_ratio >= 2.0,
                    'earnings_stability': True,  # Simplified, should check multiple years
                    'dividend_record': self.info.get('dividendYield', 0) > 0,
                    'reasonable_pe': (self.current_price / eps if eps != 0 else float('inf')) <= 15,
                    'reasonable_price_to_book': float(self.info.get('priceToBook', float('inf'))) <= 1.5
                }
            }
        except Exception as e:
            print(f"Error in Graham valuation calculation: {str(e)}")
            return {}

    def calculate_multiples(self) -> Dict[str, float]:
        """
        Calculate common valuation multiples.

        Returns a dictionary containing:
        1. P/E (Price to Earnings)
        2. Forward P/E
        3. PEG (Price/Earnings to Growth)
        4. P/B (Price to Book)
        5. P/S (Price to Sales)
        6. EV/EBITDA (Enterprise Value to EBITDA)
        7. EV/Sales
        8. EV/FCF (Enterprise Value to Free Cash Flow)
        9. Dividend Yield
        10. ROE (Return on Equity)
        11. ROA (Return on Assets)
        12. ROIC (Return on Invested Capital)
        13. Gross Margin
        14. Operating Margin
        15. Net Margin
        """
        try:
            # Get required financial metrics

            net_income = float(self.financials.loc['Net Income'].iloc[0])
            total_revenue = float(self.financials.loc['Total Revenue'].iloc[0])
            total_assets = float(
                self.balance_sheet.loc['Total Assets'].iloc[0])
            stockholders_equity = float(
                self.balance_sheet.loc['Stockholders Equity'].iloc[0])
            ebitda = float(self.financials.loc['EBITDA'].iloc[0])
            free_cash_flow = float(self.cashflow.loc['Free Cash Flow'].iloc[0])

            # Calculate Enterprise Value
            total_debt = float(self.balance_sheet.loc['Total Debt'].iloc[0])
            cash_and_equivalents = float(
                self.balance_sheet.loc['Cash And Cash Equivalents'].iloc[0])
            enterprise_value = self.market_cap + total_debt - cash_and_equivalents

            # Get growth and forward metrics from info
            forward_eps = self.info.get('forwardEps', 0)
            earnings_growth = self.info.get('earningsGrowth', 0)

            # Calculate basic earnings metrics
            eps = net_income / self.shares_outstanding if self.shares_outstanding != 0 else 0

            # Calculate margins
            gross_profit = float(self.financials.loc['Gross Profit'].iloc[0])
            operating_income = float(
                self.financials.loc['Operating Income'].iloc[0])

            multiples = {
                # Price multiples
                'pe_ratio': self.current_price / eps if eps != 0 else 0,
                'forward_pe': self.current_price / forward_eps if forward_eps != 0 else 0,
                'peg_ratio': (self.current_price / eps) / (earnings_growth * 100) if eps != 0 and earnings_growth != 0 else 0,
                'price_to_book': self.market_cap / stockholders_equity if stockholders_equity != 0 else 0,
                'price_to_sales': self.market_cap / total_revenue if total_revenue != 0 else 0,

                # Enterprise Value multiples
                'ev_to_ebitda': enterprise_value / ebitda if ebitda != 0 else 0,
                'ev_to_sales': enterprise_value / total_revenue if total_revenue != 0 else 0,
                'ev_to_fcf': enterprise_value / free_cash_flow if free_cash_flow != 0 else 0,

                # Dividend metrics
                # Convert to percentage
                'dividend_yield': self.info.get('dividendYield', 0) * 100,

                # Return metrics
                'return_on_equity': (net_income / stockholders_equity * 100) if stockholders_equity != 0 else 0,
                'return_on_assets': (net_income / total_assets * 100) if total_assets != 0 else 0,
                'return_on_invested_capital': (operating_income * (1 - 0.21)) / (total_assets - cash_and_equivalents) * 100 if (total_assets - cash_and_equivalents) != 0 else 0,

                # Margin metrics
                'gross_margin': (gross_profit / total_revenue * 100) if total_revenue != 0 else 0,
                'operating_margin': (operating_income / total_revenue * 100) if total_revenue != 0 else 0,
                'net_margin': (net_income / total_revenue * 100) if total_revenue != 0 else 0
            }

            # Add additional context
            multiples.update({
                'current_price': self.current_price,
                'market_cap': self.market_cap,
                'enterprise_value': enterprise_value,
                'shares_outstanding': self.shares_outstanding
            })

            return multiples

        except Exception as e:
            print(f"Error calculating multiples: {str(e)}")
            return {}

    def get_all_valuations(self, discount_rate: float = 0.10, growth_rate: float = 0.03,
                           projection_years: int = 5) -> Dict[str, Dict[str, float]]:
        """Get all available valuations for the stock."""
        return {
            'asset_based': self.asset_based_valuation(),
            'income_based': self.income_based_valuation(discount_rate, projection_years),
            'market_based': self.market_based_valuation(),
            'dcf': self.dcf_valuation(discount_rate, growth_rate, projection_years),
            'equity_multiplier': self.equity_multiplier_valuation(),
            'book_value': self.book_value_valuation(),
            'liquidation': self.liquidation_value(),
            'breakup': self.breakup_value(),
            'epv': self.epv_valuation(),
            'reverse_dcf': self.reverse_dcf(),
            'graham': self.graham_valuation(),
            'multiples': self.calculate_multiples()
        }


def format_valuation_results(results: Dict[str, Dict[str, float]], currency: str = 'USD') -> pd.DataFrame:
    """Format valuation results into a readable DataFrame."""
    formatted_data = []

    for method, values in results.items():
        if not values:  # Skip empty results
            continue

        # Format main valuation metrics
        main_metrics = {
            'asset_based': 'current_value',
            'income_based': 'present_value',
            'market_based': 'service_valuation',
            'dcf': 'total_value',
            'equity_multiplier': 'equity_multiplier',
            'book_value': 'book_value',
            'liquidation': 'liquidation_value',
            'breakup': 'breakup_value'
        }

        if main_metric := main_metrics.get(method):
            if value := values.get(main_metric):
                formatted_data.append({
                    'Valuation Method': method.replace('_', ' ').title(),
                    'Value': f'{currency} {value:,.2f}',
                    'Details': ', '.join([f'{k}: {v:,.2f}' for k, v in values.items()
                                          if k != main_metric and isinstance(v, (int, float))])
                })

    return pd.DataFrame(formatted_data)


def compare_stocks_valuation(tickers: List[str],
                             methods: Optional[List[str]] = None,
                             currency: str = 'USD') -> pd.DataFrame:
    """Compare valuation metrics across multiple stocks."""
    if methods is None:
        methods = ['asset_based', 'market_based', 'dcf', 'book_value']

    results = {}
    for ticker in tickers:
        try:
            valuation = StockValuation(ticker)
            all_values = valuation.get_all_valuations()

            # Extract main metrics for each method
            stock_results = {}
            for method in methods:
                if method_results := all_values.get(method):
                    main_metrics = {
                        'asset_based': 'current_value',
                        'income_based': 'present_value',
                        'market_based': 'service_valuation',
                        'dcf': 'total_value',
                        'equity_multiplier': 'equity_multiplier',
                        'book_value': 'book_value',
                        'liquidation': 'liquidation_value',
                        'breakup': 'breakup_value'
                    }
                    if main_metric := main_metrics.get(method):
                        stock_results[method] = method_results.get(
                            main_metric, 0)

            results[ticker] = stock_results

        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")
            continue

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index')

    # Format column names and values
    df.columns = [method.replace('_', ' ').title() for method in df.columns]
    df = df.applymap(lambda x: f'{currency} {x:,.2f}' if isinstance(
        x, (int, float)) else x)

    return df
