"""`stock-analysis-evaluate retrain-pooled` — retrain the pooled model, keep the better one.

Trains a challenger on data up to today, scores it and the saved champion on
the same most recent validation window (rank IC, and net relative return of
the top-k picks), and saves the challenger only when it is at least as good on
both, when there is no champion yet, or with `--force`. See
`ai/cross_sectional/champion.py` for why the check leans towards the champion:
a champion trained recently has already seen part of the window it is judged on.

Running it on a schedule (it is a one-shot command, not a daemon):

- cron, first Saturday of the month at 06:00:

      0 6 1-7 * 6  cd /path/to/stock-analysis && uv run stock-analysis-evaluate \
          retrain-pooled --universe sp500 >> data/retrain.log 2>&1

- Windows Task Scheduler:

      schtasks /Create /TN "stock-analysis retrain" /SC MONTHLY /MO FIRST /D SAT /ST 06:00 ^
          /TR "cmd /c cd /d C:\\path\\to\\stock-analysis && uv run stock-analysis-evaluate retrain-pooled --universe sp500"

- Docker compose, as a one-off container started by the host's scheduler:

      docker compose run --rm trader stock-analysis-evaluate retrain-pooled \
          --universe sp500 --models-dir /app/models

Nothing reads the saved model yet. The pooled model is not a strategy of the
daily job: a strategy is promoted to paper trading only after it has beaten
buy-and-hold out of sample in the backtester (see docs/strategy-improvement-plan.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_analysis.backtest.benchmarks import load_price_data
from stock_analysis.market_data import MarketDataProvider, YFinanceProvider

DEFAULT_MODEL_NAME = "pooled-sp500"
CONTEXT_TICKERS = ("SPY", "^VIX", "^TNX")


def _metric_line(name: str, metrics: dict | None) -> str:
    if metrics is None:
        return f"- {name}: not scored"
    ic = metrics.get("rank_ic")
    net = metrics.get("top_k_net_return")
    return (
        f"- {name}: rank IC {'—' if ic is None else f'{ic:.4f}'}, "
        f"top-k net relative return per holding period "
        f"{'—' if net is None else f'{net:.2%}'} ({metrics.get('dates', 0)} sessions)"
    )


def run_retrain_command(
    args,
    provider: MarketDataProvider | None = None,
    *,
    membership=None,
    cache_store=None,
    manager=None,
) -> str:
    """Retrain as described by `args` and return a short report."""
    from stock_analysis.ai.cross_sectional import PooledConfig, build_pooled_dataset
    from stock_analysis.ai.cross_sectional.champion import retrain_with_champion_check
    from stock_analysis.ai.model_manager import ModelManager
    from stock_analysis.cli.universe import load_index_universe

    provider = provider or YFinanceProvider()
    sectors: dict[str, str] = {}
    if args.tickers:
        tickers = [ticker.upper() for ticker in args.tickers]
        price_data = load_price_data(tickers, provider, period=args.period)
        if not price_data:
            raise SystemExit(f"No price data for any of: {tickers}")
        universe_text = f"tickers: {', '.join(sorted(price_data))}"
    else:
        universe = load_index_universe(
            args.universe,
            provider,
            args.period,
            sample=args.universe_sample,
            seed=args.universe_seed,
            membership=membership,
            cache_store=cache_store,
            use_cache=not args.no_price_cache,
        )
        price_data = universe.price_data
        membership = universe.membership
        sectors = universe.sectors
        universe_text = (
            f"{args.universe} historical members, sample "
            f"{args.universe_sample or 'all'} (seed {args.universe_seed}): "
            f"{len(price_data)} tickers with data"
        )

    series = load_price_data(list(CONTEXT_TICKERS), provider, period=args.period)
    config = PooledConfig(
        horizon=args.horizon,
        train_sessions=int(args.train_years * 252),
        validation_sessions=args.validation_sessions,
        top_k=args.top_k,
        round_trip_cost=args.round_trip_cost,
        use_threshold=not args.no_threshold,
        seed=args.seed,
    )
    dataset = build_pooled_dataset(
        price_data,
        market=series.get("SPY"),
        context={name: frame for name, frame in series.items() if name != "SPY"},
        sectors=sectors or None,
        membership=membership if not args.tickers else None,
        horizon=config.horizon,
    )
    manager = manager or ModelManager(models_dir=args.models_dir)
    outcome = retrain_with_champion_check(
        dataset, config, manager, args.model_name, universe=universe_text, force=args.force
    )

    lines = [f"# Pooled model retrain — {pd.Timestamp(dataset.calendar[-1]).date()}", ""]
    lines.append(f"- Universe: {universe_text}")
    challenger = outcome.challenger
    if challenger is not None:
        window = challenger.window
        lines += [
            f"- Challenger: {challenger.train_rows:,} training rows, labels through "
            f"{challenger.trained_through.date()}, entry threshold {challenger.threshold:.3f}",
            f"- Compared on {dataset.calendar[window.validation_first].date()} to "
            f"{dataset.calendar[window.validation_last].date()}",
            _metric_line("Challenger", outcome.challenger_metrics),
            _metric_line("Champion", outcome.champion_metrics),
        ]
        if outcome.champion_seen_sessions:
            lines.append(
                f"- The champion had already trained or calibrated on "
                f"{outcome.champion_seen_sessions} of those sessions, which flatters it"
            )
    lines.append(
        f"- Result: {'challenger saved as the new champion' if outcome.promoted else 'champion unchanged'}"
        f" — {outcome.reason}"
    )
    report = "\n".join(lines) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    return report


def cmd_retrain(args) -> None:
    print(run_retrain_command(args), end="")


def add_retrain_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "retrain-pooled",
        help="Retrain the pooled cross-sectional model; keep it only if it beats the saved one",
        description=(
            "Trains a challenger on data up to today and compares it with the saved "
            "champion on the most recent validation window. Meant to be run on a "
            "schedule (cron / Task Scheduler); see the module docstring."
        ),
    )
    universe = parser.add_mutually_exclusive_group(required=True)
    universe.add_argument("--tickers", nargs="+", default=None, help="Universe to train on")
    universe.add_argument("--universe", choices=["sp500"], default=None)
    parser.add_argument("--universe-sample", type=int, default=None, metavar="N")
    parser.add_argument("--universe-seed", type=int, default=0)
    parser.add_argument("--no-price-cache", action="store_true")
    parser.add_argument(
        "--period", default="10y",
        help="Price history to fetch (training window plus a year of feature warm-up)",
    )
    parser.add_argument("--horizon", type=int, default=10, help="Label horizon in sessions")
    parser.add_argument("--train-years", type=float, default=5.0)
    parser.add_argument("--validation-sessions", type=int, default=252)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--round-trip-cost", type=float, default=0.0014,
        help="Cost of one buy plus one sell as a fraction of the position "
        "(default: 2 x 5 bp slippage + 2 x $1 on a $5,000 position)",
    )
    parser.add_argument("--no-threshold", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--force", action="store_true", help="Save the challenger even if the champion is better"
    )
    parser.add_argument("--output", default=None, help="Also write the report to this file")
    parser.set_defaults(func=cmd_retrain)
