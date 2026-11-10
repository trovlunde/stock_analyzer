"""`stock-analysis-evaluate backtest` — replay a portfolio config over history."""

import dataclasses
import json
from datetime import date, datetime, timezone
from pathlib import Path

from stock_analysis.backtest.benchmarks import buy_and_hold_equity, load_price_data
from stock_analysis.backtest.engine import run_backtest
from stock_analysis.backtest.metrics import (
    daily_sharpe,
    deflated_sharpe_ratio,
    paired_block_bootstrap,
    performance_metrics,
    probability_of_backtest_overfitting,
    trade_statistics,
)
from stock_analysis.backtest.report import (
    DEFAULT_RUN_LOG,
    append_run,
    format_comparison,
    logged_runs,
)
from stock_analysis.market_data import MarketDataProvider, YFinanceProvider
from stock_analysis.trading.config import (
    PortfolioSettings,
    PositionEntry,
    load_portfolio_config,
    strategy_names_for_position,
)
from stock_analysis.trading.signals import get_signal_strategy

WALK_FORWARD_ML = "ml_walk_forward"
SAVED_MODEL_ML = "ml_movement"


def _override_settings(settings: PortfolioSettings, args) -> PortfolioSettings:
    positions = settings.positions
    if args.tickers:
        positions = [PositionEntry(ticker=ticker.upper()) for ticker in args.tickers]
    defaults = dict(settings.defaults)
    if args.strategies:
        defaults["strategies"] = list(args.strategies)
        # Per-position strategy overrides would otherwise win over the CLI choice
        positions = [
            dataclasses.replace(position, strategy=None, strategies=None)
            for position in positions
        ]
    if args.signal_mode:
        defaults["signal_mode"] = args.signal_mode
    return dataclasses.replace(settings, positions=positions, defaults=defaults)


def _strategies_in_use(settings: PortfolioSettings) -> list[str]:
    names: list[str] = []
    for position in settings.positions:
        if not position.enabled:
            continue
        for name in strategy_names_for_position(settings, position):
            if name not in names:
                names.append(name)
    return names


def run_backtest_command(args, provider: MarketDataProvider | None = None) -> str:
    """Run the backtest described by `args` and return the report text."""
    from stock_analysis.backtest.walk_forward import WalkForwardMlStrategy

    provider = provider or YFinanceProvider()
    settings = _override_settings(load_portfolio_config(args.config), args)
    strategies = _strategies_in_use(settings)
    if SAVED_MODEL_ML in strategies and not args.allow_in_sample_ml:
        raise SystemExit(
            f"'{SAVED_MODEL_ML}' uses a saved model that was trained on the period "
            f"being backtested, so its results would be in-sample. Use "
            f"'--strategies {WALK_FORWARD_ML} ...' to retrain as history unfolds, "
            f"or pass --allow-in-sample-ml to run it anyway."
        )

    tickers = [p.ticker for p in settings.positions if p.enabled]
    price_data = load_price_data(tickers, provider, period=args.period)
    missing = [ticker for ticker in tickers if ticker not in price_data]
    if not price_data:
        raise SystemExit(f"No price data for any of: {tickers}")

    walk_forward = WalkForwardMlStrategy(retrain_every=args.retrain_every)

    def resolve_strategy(name: str):
        return walk_forward if name == WALK_FORWARD_ML else get_signal_strategy(name)

    result = run_backtest(
        settings,
        price_data,
        start=date.fromisoformat(args.start) if args.start else None,
        end=date.fromisoformat(args.end) if args.end else None,
        resolve_strategy=resolve_strategy,
        apply_kill_switch=not args.no_kill_switch,
    )
    sessions = result.equity.index
    initial_cash = result.initial_cash

    columns = {
        "Strategy": {**performance_metrics(result.equity), **trade_statistics(result)},
    }
    universe_equity = buy_and_hold_equity(price_data, sessions, initial_cash)
    columns["Buy & hold (universe)"] = performance_metrics(universe_equity)
    if args.benchmark:
        benchmark_data = load_price_data([args.benchmark], provider, period=args.period)
        if benchmark_data:
            columns[f"Buy & hold ({args.benchmark})"] = performance_metrics(
                buy_and_hold_equity(benchmark_data, sessions, initial_cash)
            )

    strategy_returns = result.equity.pct_change().dropna()
    bootstrap = paired_block_bootstrap(
        strategy_returns, universe_equity.pct_change().dropna()
    )

    experiment = "|".join(
        [",".join(sorted(price_data)), args.start or "", args.end or "", args.period]
    )
    config_key = json.dumps(
        {
            "strategies": strategies,
            "defaults": settings.defaults,
            "costs": settings.costs,
            "portfolio": dataclasses.asdict(settings.portfolio),
            "risk": dataclasses.asdict(settings.risk) if not args.no_kill_switch else None,
            "retrain_every": args.retrain_every if WALK_FORWARD_ML in strategies else None,
        },
        sort_keys=True,
        default=str,
    )
    log_path = Path(args.run_log)
    sharpe = daily_sharpe(strategy_returns)
    if not args.no_log:
        append_run(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "experiment": experiment,
                "config_key": config_key,
                "daily_sharpe": sharpe,
                "sharpe": columns["Strategy"]["sharpe"],
                "total_return": columns["Strategy"]["total_return"],
                # Kept so later runs can test whether the best configuration's
                # lead survives on data it was not chosen on
                "returns": [round(float(value), 8) for value in strategy_returns],
            },
            log_path,
        )
    # One trial per distinct configuration: re-running the same one is not a new try
    earlier_runs = logged_runs(experiment, log_path)
    trials = {
        run["config_key"]: run["daily_sharpe"]
        for run in earlier_runs
        if run.get("daily_sharpe") is not None
    }
    if sharpe is not None:
        trials[config_key] = sharpe
    deflated = deflated_sharpe_ratio(strategy_returns, list(trials.values()))

    # Same experiment means same sessions; a different length means the price
    # history has moved on since that run, so it cannot be lined up day for day
    trial_returns = {
        run["config_key"]: run["returns"]
        for run in earlier_runs
        if len(run.get("returns") or []) == len(strategy_returns)
    }
    trial_returns[config_key] = list(strategy_returns)
    overfitting = probability_of_backtest_overfitting(list(trial_returns.values()))

    lines = [
        f"# Backtest — {', '.join(strategies)}",
        "",
        f"- Universe: {', '.join(sorted(price_data))}",
        f"- Period: {sessions[0].date()} to {sessions[-1].date()} ({len(sessions)} sessions)",
        f"- Initial cash: ${initial_cash:,.0f}",
        "- Fills at next session's Open; costs from the portfolio config",
    ]
    if missing:
        lines.append(f"- No data (left out): {', '.join(missing)}")
    if result.kill_switch_tripped_on is not None:
        lines.append(
            f"- Kill switch tripped on {result.kill_switch_tripped_on} "
            f"({result.kill_switch_reason}) — no new BUYs after that date"
        )
    if SAVED_MODEL_ML in strategies:
        lines.append(
            f"- **In-sample:** '{SAVED_MODEL_ML}' used a saved model trained on this period"
        )
    lines += ["", format_comparison(columns), "", "## Is it better than chance?", ""]

    if bootstrap["sharpe_ci_low"] is None:
        lines.append("- Too little data (or no trading) for a bootstrap estimate")
    else:
        lines.append(
            f"- Sharpe 90% interval (block bootstrap): "
            f"{bootstrap['sharpe_ci_low']:.2f} to {bootstrap['sharpe_ci_high']:.2f}"
        )
        if bootstrap["prob_beats_benchmark"] is not None:
            lines.append(
                f"- Resamples where the strategy's Sharpe beats universe buy & hold: "
                f"{bootstrap['prob_beats_benchmark']:.0%}"
            )
    lines.append(f"- Configurations tried on this universe and period: {len(trials)}")
    if deflated is not None:
        lines.append(
            f"- Deflated Sharpe (chance the Sharpe is above zero after allowing for "
            f"{len(trials)} tries): {deflated:.0%}"
        )
    else:
        lines.append("- Deflated Sharpe: needs at least two different configurations")
    if overfitting is not None:
        lines.append(
            f"- Probability of backtest overfitting ({len(trial_returns)} configurations; "
            f"how often the best one on half the data falls in the bottom half on the "
            f"other half): {overfitting:.0%}"
        )
    else:
        lines.append(
            "- Probability of backtest overfitting: needs at least two different "
            "configurations logged over the same sessions"
        )

    report = "\n".join(lines) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    return report


def cmd_backtest(args) -> None:
    print(run_backtest_command(args))


def add_backtest_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "backtest",
        help="Replay a portfolio config over history with the daily job's trading rules",
        description=(
            "Runs the same signal, sizing, fill and cost logic as the daily paper-trading "
            "job over historical prices and compares the result with buy-and-hold."
        ),
    )
    parser.add_argument("--config", default="config/portfolio.yaml")
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Universe to trade (default: positions in the config)",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=None, metavar="NAME",
        help=(
            "Strategies to run (default: from the config). "
            f"'{WALK_FORWARD_ML}' retrains the ML model as history unfolds."
        ),
    )
    parser.add_argument("--signal-mode", choices=["consensus", "priority", "any"], default=None)
    parser.add_argument(
        "--period", default="10y",
        help="How much price history to fetch (bars before --start are warm-up history)",
    )
    parser.add_argument("--start", default=None, help="First session to trade (YYYY-MM-DD)")
    parser.add_argument("--end", default=None, help="Last session to trade (YYYY-MM-DD)")
    parser.add_argument(
        "--benchmark", default="SPY", help="Extra buy-and-hold benchmark ('' to skip)"
    )
    parser.add_argument(
        "--retrain-every", type=int, default=63,
        help=f"Sessions between retrains for '{WALK_FORWARD_ML}'",
    )
    parser.add_argument(
        "--no-kill-switch", action="store_true",
        help="Ignore the config's risk limits (they stop new BUYs for the rest of the run)",
    )
    parser.add_argument(
        "--allow-in-sample-ml", action="store_true",
        help=f"Allow '{SAVED_MODEL_ML}' even though its saved model has seen the test period",
    )
    parser.add_argument("--output", default=None, help="Also write the report to this file")
    parser.add_argument("--run-log", default=str(DEFAULT_RUN_LOG))
    parser.add_argument(
        "--no-log", action="store_true", help="Do not record this run as a trial"
    )
    parser.set_defaults(func=cmd_backtest)
