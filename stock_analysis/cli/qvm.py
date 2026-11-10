"""`stock-analysis-evaluate qvm-watchlist` — today's top QVM names as a `positions:` snippet."""

from collections.abc import Callable
from pathlib import Path

import pandas as pd

from stock_analysis.backtest.benchmarks import load_price_data
from stock_analysis.factors.qvm import DEFAULT_MIN_SECTOR_SIZE
from stock_analysis.factors.qvm_strategy import (
    FundamentalsSource,
    QvmScoreProvider,
    qvm_watchlist,
)
from stock_analysis.market_data import MarketDataProvider, YFinanceProvider

UNIVERSES = ("sp500",)

SectorLookup = Callable[[list[str]], dict[str, str]]


def _universe_tickers(name: str) -> list[str]:
    from stock_analysis.research.market_indices import MarketIndices

    if name == "sp500":
        # Wikipedia writes share classes with a dot (BRK.B); Yahoo uses a dash
        return [ticker.replace(".", "-") for ticker in MarketIndices.get_sp500_tickers()]
    raise SystemExit(f"Unknown universe '{name}' (available: {', '.join(UNIVERSES)})")


def _yahoo_sectors(provider) -> SectorLookup:
    def lookup(tickers: list[str]) -> dict[str, str]:
        sectors: dict[str, str] = {}
        for ticker in tickers:
            try:
                sector = provider.get_raw_ticker(ticker).info.get("sector")
            except Exception:
                sector = None  # ranked among the unknowns rather than aborting the run
            if sector:
                sectors[ticker] = sector
        return sectors

    return lookup


def format_watchlist(watchlist: pd.DataFrame, *, session: pd.Timestamp, ranked: int) -> str:
    """YAML that can be pasted into a portfolio config; the scores ride along as comments."""
    lines = [
        f"# QVM watchlist as of {session.date()} "
        f"(top {len(watchlist)} of {ranked} ranked; scores are 0-100 within sector)",
    ]
    if watchlist.empty:
        return "\n".join(lines + ["positions: []"]) + "\n"
    lines.append("positions:")
    for row in watchlist.itertuples(index=False):
        lines.append(
            f"  - ticker: {row.ticker}  # {row.sector}: QVM {row.qvm_score:.0f} "
            f"(Q {row.quality_score:.0f}, V {row.value_score:.0f}, M {row.momentum_score:.0f})"
        )
    return "\n".join(lines) + "\n"


def run_qvm_watchlist_command(
    args,
    provider: MarketDataProvider | None = None,
    fundamentals: FundamentalsSource | None = None,
    sector_lookup: SectorLookup | None = None,
) -> str:
    """Build the watchlist described by `args` and return the YAML text."""
    if args.tickers:
        tickers = [ticker.upper() for ticker in args.tickers]
    elif args.universe:
        tickers = _universe_tickers(args.universe)
    else:
        raise SystemExit("Give --tickers or --universe")

    provider = provider or YFinanceProvider()
    if fundamentals is None:
        from stock_analysis.ai.fundamentals.point_in_time import (
            PointInTimeFundamentals,
            require_edgar_identity,
        )

        require_edgar_identity()  # fail before the slow price download, not after it
        fundamentals = PointInTimeFundamentals()
    sector_lookup = sector_lookup or _yahoo_sectors(provider)

    price_data = load_price_data(tickers, provider, period=args.period)
    if not price_data:
        raise SystemExit(f"No price data for any of: {tickers}")
    sectors = None if args.no_sector_neutral else sector_lookup(list(price_data))

    scores = QvmScoreProvider(
        price_data,
        fundamentals,
        sectors,
        lag_days=args.lag_days,
        min_sector_size=args.min_sector_size,
        include_margin_of_safety=args.margin_of_safety,
    )
    session = scores.calendar[-1]
    ranked = int(scores.table_on(session)["qvm_score"].notna().sum())
    watchlist = qvm_watchlist(scores, session=session, top=args.top)
    text = format_watchlist(watchlist, session=session, ranked=ranked)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    return text


def cmd_qvm_watchlist(args) -> None:
    print(run_qvm_watchlist_command(args), end="")


def add_qvm_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "qvm-watchlist",
        help="Today's top Quality/Value/Momentum names as a positions: snippet",
        description=(
            "Ranks a universe on the same QVM scores as the backtested 'qvm' strategy "
            "(12-1 momentum from prices, value and quality from EDGAR filings known "
            "today) and prints the best names as YAML for a portfolio config. "
            "Needs EDGAR_IDENTITY ('Name email') in the environment."
        ),
    )
    universe = parser.add_mutually_exclusive_group(required=True)
    universe.add_argument("--tickers", nargs="+", default=None, help="Universe to rank")
    universe.add_argument("--universe", choices=list(UNIVERSES), default=None)
    parser.add_argument("--top", type=int, default=20, help="How many names to list")
    parser.add_argument("--output", default=None, help="Also write the snippet to this file")
    parser.add_argument(
        "--period", default="2y",
        help="Price history to fetch (12-1 momentum needs a little over one year)",
    )
    parser.add_argument(
        "--min-sector-size", type=int, default=DEFAULT_MIN_SECTOR_SIZE,
        help="Sectors with fewer names than this are not ranked",
    )
    parser.add_argument(
        "--no-sector-neutral", action="store_true",
        help="Rank the whole universe as one group instead of within sectors",
    )
    parser.add_argument(
        "--lag-days", type=int, default=1,
        help="Days after its filing date before a filing counts as known",
    )
    parser.add_argument(
        "--margin-of-safety", action="store_true",
        help="Add price against earnings power value to the value factor",
    )
    parser.set_defaults(func=cmd_qvm_watchlist)
