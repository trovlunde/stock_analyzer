"""Backward-compatible re-exports; prefer stock_analysis.research."""

from stock_analysis.research.stock_finder import (
    display_stocks,
    find_growth_stocks,
    find_high_dividend_stocks,
    find_stocks_by_method,
    find_undervalued_stocks,
)
from stock_analysis.research.qvm import find_qvm_stocks

__all__ = [
    "display_stocks",
    "find_growth_stocks",
    "find_high_dividend_stocks",
    "find_qvm_stocks",
    "find_stocks_by_method",
    "find_undervalued_stocks",
]
