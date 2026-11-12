"""`stock-analysis-evaluate backtest` — replay a portfolio config over history."""

import argparse
import dataclasses
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd
import yaml

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
    format_yearly_returns,
    logged_runs,
)
from stock_analysis.market_data import MarketDataProvider, YFinanceProvider
from stock_analysis.trading.config import (
    PortfolioConfig,
    PortfolioSettings,
    PositionEntry,
    load_portfolio_config,
    strategy_names_for_position,
)
from stock_analysis.trading.regime import RegimeFilter
from stock_analysis.trading.signals import SignalStrategy, get_signal_strategy
from stock_analysis.trading.signals.combine import SIGNAL_MODES
from stock_analysis.trading.sizing import SIZING_MODES

WALK_FORWARD_ML = "ml_walk_forward"
WALK_FORWARD_ENSEMBLE = "ensemble_walk_forward"
SAVED_MODEL_ML = "ml_movement"


@dataclasses.dataclass
class StrategyBuildContext:
    """What a data-dependent strategy is built from in a backtest."""

    settings: PortfolioSettings
    price_data: dict[str, pd.DataFrame]
    provider: MarketDataProvider
    args: argparse.Namespace
    # Filled by whatever loaded the universe (index membership, sectors, ...)
    extras: dict = dataclasses.field(default_factory=dict)


StrategyBuilder = Callable[[StrategyBuildContext], SignalStrategy]

# Strategies that rank the whole universe need its price history to be built, so
# they cannot sit in the production registry as ready-made objects. They register
# a builder here instead and are available to `backtest --strategies <name>`.
_DATA_STRATEGY_BUILDERS: dict[str, StrategyBuilder] = {}


def register_backtest_strategy(name: str, builder: StrategyBuilder) -> None:
    _DATA_STRATEGY_BUILDERS[name] = builder


def _build_qvm(context: StrategyBuildContext) -> SignalStrategy:
    """Monthly-rebalanced quality/value/momentum ranking on point-in-time EDGAR
    fundamentals. Needs EDGAR_IDENTITY set (the SEC asks for a contact address)."""
    from stock_analysis.ai.fundamentals import PointInTimeFundamentals
    from stock_analysis.cli.qvm import _yahoo_sectors
    from stock_analysis.factors.qvm_strategy import build_qvm_strategy

    params = context.settings.defaults
    tickers = sorted(context.price_data)
    sectors = dict(context.extras.get("sectors") or {})
    sectors.update({p.ticker: p.sector for p in context.settings.positions if p.sector})
    unknown = [ticker for ticker in tickers if ticker not in sectors]
    if unknown and params.get("qvm_sector_neutral", True):
        sectors.update(_yahoo_sectors(context.provider)(unknown))
    fundamentals = context.extras.get("fundamentals")
    if fundamentals is None:
        fundamentals = PointInTimeFundamentals()
        fundamentals.preload(tickers)
    return build_qvm_strategy(
        context.price_data,
        fundamentals,
        sectors if params.get("qvm_sector_neutral", True) else None,
        top_fraction=float(params.get("qvm_top_fraction", 0.10)),
        min_sector_size=int(params.get("qvm_min_sector_size", 4)),
        include_margin_of_safety=bool(params.get("qvm_margin_of_safety", False)),
    )


register_backtest_strategy("qvm", _build_qvm)

POOLED_ML = "ml_pooled"
POOLED_CONTEXT_TICKERS = ("SPY", "^VIX", "^TNX")


def _build_ml_pooled(context: StrategyBuildContext) -> SignalStrategy:
    """One LightGBM model for the whole universe, retrained as history unfolds.

    Parameters (strategy params, so `--set pooled_horizon=5` works):
    `pooled_horizon` (10 sessions), `pooled_top_k` (10), `pooled_exit_rank`
    (2 x top_k), `pooled_retrain_every` (63), `pooled_train_years` (5),
    `pooled_validation_sessions` (252), `pooled_use_threshold` (true),
    `pooled_min_picks` (100), `pooled_min_train_sessions` (252), `pooled_seed` (0).
    """
    from stock_analysis.ai.cross_sectional import (
        PooledConfig,
        PooledModelScoreProvider,
        build_pooled_dataset,
    )
    from stock_analysis.ai.cross_sectional.report import pooled_report_sections
    from stock_analysis.trading.cost_models import build_cost_model
    from stock_analysis.trading.signals.ranked import RankedUniverseStrategy

    settings = context.settings
    params = settings.defaults
    top_k = int(params.get("pooled_top_k", 10))
    # What one buy plus one sell of a typical position costs, as a fraction of it
    notional = settings.portfolio.initial_cash * settings.portfolio.position_size_pct
    cost_model = build_cost_model(settings.costs)
    round_trip = (
        cost_model.entry_cost(notional).total + cost_model.exit_cost(notional).total
    ) / notional
    config = PooledConfig(
        horizon=int(params.get("pooled_horizon", 10)),
        train_sessions=int(float(params.get("pooled_train_years", 5)) * 252),
        validation_sessions=int(params.get("pooled_validation_sessions", 252)),
        retrain_every=int(params.get("pooled_retrain_every", 63)),
        min_train_sessions=int(params.get("pooled_min_train_sessions", 252)),
        top_k=top_k,
        round_trip_cost=round_trip,
        use_threshold=bool(params.get("pooled_use_threshold", True)),
        min_picks=int(params.get("pooled_min_picks", 100)),
        seed=int(params.get("pooled_seed", 0)),
    )

    series = load_price_data(
        list(POOLED_CONTEXT_TICKERS), context.provider, period=context.args.period
    )
    sectors = dict(context.extras.get("sectors") or {})
    sectors.update({p.ticker: p.sector for p in settings.positions if p.sector})
    start = None
    if context.args.start:
        # Rows older than one training window before the first decision are never used
        calendar_days = int(config.train_sessions * 365 / 252 * 1.05) + 60
        start = pd.Timestamp(context.args.start) - pd.Timedelta(days=calendar_days)
    dataset = build_pooled_dataset(
        context.price_data,
        market=series.get("SPY"),
        context={name: frame for name, frame in series.items() if name != "SPY"},
        sectors=sectors or None,
        membership=context.extras.get("membership"),
        horizon=config.horizon,
        start=start,
    )
    provider = PooledModelScoreProvider(dataset, config)
    context.extras["pooled_provider"] = provider
    context.extras.setdefault("report_sections", []).append(
        lambda: pooled_report_sections(provider)
    )
    return RankedUniverseStrategy(
        provider,
        top_k=top_k,
        exit_rank=int(params.get("pooled_exit_rank", 2 * top_k)),
        # Scores are calibrated probability minus the validated entry threshold
        min_score=0.0 if config.use_threshold else None,
    )


register_backtest_strategy(POOLED_ML, _build_ml_pooled)


def _report_sections(extras: dict) -> list[str]:
    """Extra report lines from strategy builders and universe loaders.

    `extras["report_sections"]` holds markdown strings, lists of them, or
    callables returning either; callables run after the backtest, so a section
    can report what the strategy did during it.
    """
    lines: list[str] = []
    sections = extras.get("report_sections") or []
    if callable(sections) or isinstance(sections, str):
        sections = [sections]
    for section in sections:
        if callable(section):
            section = section()
        if isinstance(section, str):
            section = [section]
        if section:
            lines += ["", *section]
    return lines


def _parse_overrides(pairs: list[str] | None) -> tuple[dict, dict]:
    """`--set` values as (defaults overrides, portfolio overrides).

    `KEY=VALUE` or `defaults.KEY=VALUE` sets a strategy parameter;
    `portfolio.KEY=VALUE` sets a portfolio field. Values are parsed as YAML, so
    numbers, booleans, lists and `null` work as they do in the config file.
    """
    defaults: dict = {}
    portfolio: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise SystemExit(f"--set expects KEY=VALUE (got {pair!r})")
        key, raw = pair.split("=", 1)
        value = yaml.safe_load(raw)
        section, _, name = key.rpartition(".")
        if section in ("", "defaults"):
            defaults[name] = value
        elif section == "portfolio":
            portfolio[name] = value
        else:
            raise SystemExit(f"--set key must start with defaults. or portfolio. (got {key!r})")
    return defaults, portfolio


def _override_settings(settings: PortfolioSettings, args) -> PortfolioSettings:
    positions = settings.positions
    if args.tickers:
        positions = [PositionEntry(ticker=ticker.upper()) for ticker in args.tickers]
    default_overrides, portfolio_overrides = _parse_overrides(getattr(args, "set", None))
    unknown = set(portfolio_overrides) - {f.name for f in dataclasses.fields(settings.portfolio)}
    if unknown:
        raise SystemExit(f"Unknown portfolio field(s) in --set: {sorted(unknown)}")
    settings = dataclasses.replace(
        settings,
        portfolio=dataclasses.replace(settings.portfolio, **portfolio_overrides),
        defaults={**settings.defaults, **default_overrides},
    )
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
    portfolio = settings.portfolio
    if args.sizing:
        portfolio = dataclasses.replace(portfolio, sizing=args.sizing)
    regime_filter = settings.regime_filter
    if args.no_regime_filter:
        regime_filter = None
    elif args.regime_filter:
        regime_filter = RegimeFilter(ticker=args.regime_filter.upper())
    return dataclasses.replace(
        settings,
        portfolio=portfolio,
        positions=positions,
        defaults=defaults,
        regime_filter=regime_filter,
    )


# Portfolio fields that were part of the run-log key from the start. Fields added
# later are only written to the key when they differ from their default, so a
# configuration logged before a field existed is still recognised as the same
# trial (and not counted twice) when it is run again.
_ORIGINAL_KEY_FIELDS = frozenset(
    {
        "name", "enabled", "initial_cash", "max_positions", "position_size_pct",
        "long_only", "stop_loss_pct", "sizing", "risk_per_position",
        "max_position_pct", "vol_lookback",
    }
)


def _portfolio_key(portfolio: PortfolioConfig) -> dict:
    key = {}
    for field in dataclasses.fields(portfolio):
        value = getattr(portfolio, field.name)
        if field.name in _ORIGINAL_KEY_FIELDS or value != field.default:
            key[field.name] = value
    return key


def _strategies_in_use(settings: PortfolioSettings) -> list[str]:
    names: list[str] = []
    for position in settings.positions:
        if not position.enabled:
            continue
        for name in strategy_names_for_position(settings, position):
            if name not in names:
                names.append(name)
    return names


def run_backtest_command(
    args, provider: MarketDataProvider | None = None, *, membership=None, cache_store=None
) -> str:
    """Run the backtest described by `args` and return the report text.

    `membership` (an `IndexMembership`) and `cache_store` replace the Wikipedia
    lookup and the on-disk price cache of `--universe`; tests pass them in.
    """
    from stock_analysis.backtest.walk_forward import walk_forward_strategies

    index_universe = getattr(args, "universe", None)
    if index_universe and args.tickers:
        raise SystemExit("Give either --tickers or --universe, not both")
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

    extras: dict = {}
    if index_universe:
        from stock_analysis.cli.universe import load_index_universe

        universe = load_index_universe(
            index_universe,
            provider,
            args.period,
            sample=args.universe_sample,
            seed=args.universe_seed,
            membership=membership,
            cache_store=cache_store,
            use_cache=not args.no_price_cache,
        )
        price_data = universe.price_data
        tickers = list(universe.candidates)
        # Every sampled all-time member with data is a position; the strategy
        # decides from the membership which of them may be held on a date
        settings = dataclasses.replace(
            settings,
            positions=[
                PositionEntry(ticker=ticker, sector=universe.sectors.get(ticker))
                for ticker in sorted(price_data)
            ],
        )
        strategies = _strategies_in_use(settings)
        first_bar = min(df.index[0] for df in price_data.values())
        last_bar = max(df.index[-1] for df in price_data.values())
        first_session = max(pd.Timestamp(args.start), first_bar) if args.start else first_bar
        last_session = min(pd.Timestamp(args.end), last_bar) if args.end else last_bar
        extras["membership"] = universe.membership
        extras["sectors"] = universe.sectors
        # Buy-and-hold of the index as it was on the first session, not of
        # everything that was ever in it
        extras["benchmark_universe"] = universe.members_with_data(first_session)
        extras["report_sections"] = [
            lambda: universe.coverage_lines(first_session.year - 5, last_session.year)
        ]
    else:
        tickers = [p.ticker for p in settings.positions if p.enabled]
        price_data = load_price_data(tickers, provider, period=args.period)
    missing = [ticker for ticker in tickers if ticker not in price_data]
    if not price_data:
        raise SystemExit(f"No price data for any of: {tickers}")

    walk_forward = walk_forward_strategies(args.retrain_every)
    build_context = StrategyBuildContext(
        settings=settings, price_data=price_data, provider=provider, args=args, extras=extras
    )
    built = {
        name: _DATA_STRATEGY_BUILDERS[name](build_context)
        for name in strategies
        if name in _DATA_STRATEGY_BUILDERS
    }

    def resolve_strategy(name: str):
        return built.get(name) or walk_forward.get(name) or get_signal_strategy(name)

    market_data = None
    if settings.regime_filter is not None:
        regime_ticker = settings.regime_filter.ticker
        market_data = load_price_data([regime_ticker], provider, period=args.period).get(
            regime_ticker
        )
        if market_data is None:
            raise SystemExit(f"No price data for the regime filter ticker {regime_ticker}")

    result = run_backtest(
        settings,
        price_data,
        start=date.fromisoformat(args.start) if args.start else None,
        end=date.fromisoformat(args.end) if args.end else None,
        resolve_strategy=resolve_strategy,
        apply_kill_switch=not args.no_kill_switch,
        market_data=market_data,
    )
    sessions = result.equity.index
    initial_cash = result.initial_cash

    columns = {
        "Strategy": {**performance_metrics(result.equity), **trade_statistics(result)},
    }
    universe_equity = buy_and_hold_equity(
        build_context.extras.get("benchmark_universe", price_data), sessions, initial_cash
    )
    columns["Buy & hold (universe)"] = performance_metrics(universe_equity)
    equity_curves = {"Strategy": result.equity, "Buy & hold (universe)": universe_equity}
    if args.benchmark:
        benchmark_data = load_price_data([args.benchmark], provider, period=args.period)
        if benchmark_data:
            benchmark_equity = buy_and_hold_equity(benchmark_data, sessions, initial_cash)
            columns[f"Buy & hold ({args.benchmark})"] = performance_metrics(benchmark_equity)
            equity_curves[f"Buy & hold ({args.benchmark})"] = benchmark_equity

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
            "portfolio": _portfolio_key(settings.portfolio),
            "risk": dataclasses.asdict(settings.risk) if not args.no_kill_switch else None,
            "regime_filter": (
                dataclasses.asdict(settings.regime_filter)
                if settings.regime_filter is not None else None
            ),
            "retrain_every": (
                args.retrain_every
                if any(name in walk_forward for name in strategies) else None
            ),
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
    lines += [
        "",
        format_comparison(columns),
        "",
        "## Return per calendar year",
        "",
        format_yearly_returns(equity_curves),
        "",
        "## Is it better than chance?",
        "",
    ]

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

    lines += _report_sections(build_context.extras)

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
        "--universe", choices=["sp500"], default=None,
        help=(
            "Trade the historical members of an index instead of --tickers / the "
            "config's positions (membership per date from Wikipedia's change log)"
        ),
    )
    parser.add_argument(
        "--universe-sample", type=int, default=None, metavar="N",
        help="Random sample of N all-time index members (default: all of them)",
    )
    parser.add_argument("--universe-seed", type=int, default=0, help="Seed of the sample")
    parser.add_argument(
        "--no-price-cache", action="store_true",
        help="With --universe: download every price history again instead of using the cache",
    )
    parser.add_argument(
        "--strategies", nargs="+", default=None, metavar="NAME",
        help=(
            "Strategies to run (default: from the config). "
            f"'{WALK_FORWARD_ML}' (random forest) and '{WALK_FORWARD_ENSEMBLE}' "
            "(ensemble classifier) retrain their model as history unfolds."
        ),
    )
    parser.add_argument("--signal-mode", choices=list(SIGNAL_MODES), default=None)
    parser.add_argument(
        "--set", action="append", default=None, metavar="KEY=VALUE",
        help=(
            "Override a config value for this run; repeatable. KEY is a strategy "
            "parameter (e.g. trend_entry_threshold=0.02) or portfolio.FIELD "
            "(e.g. portfolio.stop_type=atr_trailing). VALUE is parsed as YAML."
        ),
    )
    parser.add_argument(
        "--sizing", choices=list(SIZING_MODES), default=None,
        help="Position sizing (default: from the config)",
    )
    parser.add_argument(
        "--regime-filter", default=None, metavar="TICKER",
        help="Block new BUYs while TICKER is below its 200-day average (default: from the config)",
    )
    parser.add_argument(
        "--no-regime-filter", action="store_true",
        help="Ignore the config's regime filter",
    )
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
        help=f"Sessions between retrains for '{WALK_FORWARD_ML}' and '{WALK_FORWARD_ENSEMBLE}'",
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
