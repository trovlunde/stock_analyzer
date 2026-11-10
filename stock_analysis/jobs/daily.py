import argparse
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from stock_analysis.market_data import MarketDataProvider, YFinanceProvider
from stock_analysis.notifications import EmailNotifier, load_smtp_config_from_env
from stock_analysis.storage import get_db_engine
from stock_analysis.storage.portfolio_store import PortfolioState, PortfolioStore
from stock_analysis.trading.config import (
    PortfolioSettings,
    load_portfolio_config,
    stop_loss_for_position,
    strategy_names_for_position,
    strategy_params_for_position,
)
from stock_analysis.trading.broker import Broker
from stock_analysis.trading.broker_config import load_broker_from_env
from stock_analysis.trading.cost_models import build_cost_model
from stock_analysis.trading.decision import DecisionResult, decide, portfolio_value
from stock_analysis.trading.live_gate import (
    live_notional_ceiling_violation,
    live_portfolio_ceiling_violation,
)
from stock_analysis.trading.market_calendar import (
    is_trading_day,
    next_trading_day_after,
)
from stock_analysis.trading.portfolio_metrics import (
    benchmark_relative_return,
    compute_portfolio_metrics,
)
from stock_analysis.trading.report import write_daily_report
from stock_analysis.trading.risk import check_kill_switch
from stock_analysis.trading.signals import combine_signals, get_signal_strategy
from stock_analysis.trading.signals.base import SignalResult

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "config/portfolio.yaml"
DATA_PERIOD = "2y"
# Buy-and-hold yardstick in the daily report; override with `defaults.benchmark`
# in the portfolio config (empty disables it).
DEFAULT_BENCHMARK = "SPY"
# A run still "running" after this long is treated as crashed, not just slow or
# legitimately still in flight (e.g. an overlapping deploy) — this run takes seconds
# to minutes normally, so hours of margin avoids false-positive trips.
ORPHANED_RUN_STALE_AFTER = timedelta(hours=6)
_default_provider = YFinanceProvider()


def _fetch_price_data(
    tickers: list[str], provider: MarketDataProvider
) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = provider.get_history(ticker, period=DATA_PERIOD)
        if df is None or df.empty:
            logger.warning("No data for %s", ticker)
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        data[ticker] = df
    return data


def _benchmark_comparison(
    equity_curve: list[tuple[str, float]],
    benchmark: str | None,
    provider: MarketDataProvider,
) -> dict | None:
    """Portfolio return against buy-and-hold of `benchmark` since the first recorded
    session. Reporting only — any failure is logged and the run carries on."""
    if not benchmark or len(equity_curve) < 2:
        return None
    try:
        history = provider.get_history(benchmark, start=equity_curve[0][0])
        if history is None or history.empty:
            logger.warning("No benchmark data for %s", benchmark)
            return None
        close = history["Close"]
        if close.index.tz is not None:
            close = close.tz_localize(None)
        return benchmark_relative_return(equity_curve, close, benchmark)
    except Exception:
        logger.exception("Benchmark comparison against %s failed", benchmark)
        return None


def _session_date(now: datetime | None = None) -> datetime.date:
    return (now or datetime.now(timezone.utc)).date()


def _latest_close(df: pd.DataFrame) -> float:
    return float(df.iloc[-1]["Close"])


RECONCILIATION_CASH_TOLERANCE = 1.0  # USD — avoid alerting on float/rounding noise


def _check_broker_reconciliation(
    broker: Broker,
    state: PortfolioState,
    store: PortfolioStore,
    notifier: EmailNotifier | None,
) -> bool:
    """Once a real broker is in the loop, it's the source of truth and the local
    SQLite ledger can drift (manual trades via the broker's own UI, corporate
    actions, a bug in fill handling). A divergence trips the kill switch — exactly
    the scenario `stock-analysis-risk reconcile --accept-broker-truth` exists to
    fix — rather than silently trusting local state for position-sizing math.

    Returns True if the broker is PaperBroker (nothing to compare) or the
    comparison found no mismatch; False otherwise. Never raises — a failure to
    even fetch the snapshot is logged and treated as "not clean" (fail safe)."""
    try:
        snapshot = broker.get_account_snapshot(state)
    except Exception:
        logger.exception("Failed to fetch broker account snapshot for reconciliation check")
        return False

    if snapshot.source == "local":
        return True  # PaperBroker — local state IS the source of truth, nothing to compare

    mismatches: list[str] = []
    cash_diff = abs(snapshot.cash - state.cash)
    if cash_diff > RECONCILIATION_CASH_TOLERANCE:
        mismatches.append(
            f"cash: broker=${snapshot.cash:,.2f} local=${state.cash:,.2f} (diff ${cash_diff:,.2f})"
        )

    local_positions = {p.ticker: p.shares for p in state.positions}
    if local_positions != snapshot.positions:
        mismatches.append(f"positions: broker={snapshot.positions} local={local_positions}")

    if not mismatches:
        return True

    message = "; ".join(mismatches)
    logger.warning("Broker/local state mismatch: %s", message)
    tripped_at, _reason = store.get_kill_switch_state(state.id)
    if tripped_at is None:
        store.trip_kill_switch(state.id, f"broker/local state mismatch: {message}")
        if notifier is not None:
            notifier.send_error_alert("Broker/local state mismatch — kill switch tripped", message)
    return False


def run_daily(
    config_path: str,
    dry_run: bool = False,
    *,
    provider: MarketDataProvider | None = None,
    engine: Engine | None = None,
    notifier: EmailNotifier | None = None,
    broker: Broker | None = None,
) -> int:
    provider = provider or _default_provider
    settings = load_portfolio_config(config_path)
    portfolio_cfg = settings.portfolio

    if not portfolio_cfg.enabled:
        logger.info("Portfolio disabled in config — exiting")
        return 0

    today = _session_date()
    if not is_trading_day(today):
        logger.info("NYSE closed today (%s) — skipping", today)
        report_path = write_daily_report(
            Path("data/reports"),
            today,
            portfolio_name=portfolio_cfg.name,
            cash=0,
            portfolio_value=0,
            fills=[],
            decisions=[],
            dry_run=dry_run,
            skipped="NYSE closed",
        )
        logger.info("Wrote report %s", report_path)
        return 0

    engine = engine or get_db_engine()
    store = PortfolioStore(engine)
    unclean_shutdown_reason: str | None = None
    if dry_run:
        state = store.get_portfolio_by_name(portfolio_cfg.name)
        if state is None:
            state = PortfolioState(
                id=0,
                name=portfolio_cfg.name,
                cash=portfolio_cfg.initial_cash,
                positions=[],
            )
        run_id = 0
    else:
        state = store.get_or_create_portfolio(
            portfolio_cfg.name, portfolio_cfg.initial_cash
        )
        orphaned_run_ids = store.get_running_run_ids(
            state.id, older_than=datetime.now(timezone.utc) - ORPHANED_RUN_STALE_AFTER
        )
        if orphaned_run_ids:
            unclean_shutdown_reason = (
                f"unclean shutdown detected — run(s) {orphaned_run_ids} never finished"
            )
            for orphaned_id in orphaned_run_ids:
                store.abandon_run(orphaned_id, "orphaned by process restart")
        run_id = store.start_run(state.id, dry_run=False)

    try:
        return _run_session(
            settings,
            store,
            state,
            run_id,
            today,
            dry_run,
            unclean_shutdown_reason,
            provider=provider,
            notifier=notifier,
            broker=broker,
        )
    except Exception as exc:
        # A handled failure (data outage, broker API error, LiveTradingBlocked) isn't
        # a crash — finish the row as "failed" so the next run's orphan detection
        # doesn't mistake it for an unclean shutdown and trip the kill switch.
        # "failed" still breaks the live gate's consecutive-clean-run streak.
        if not dry_run and run_id in store.get_running_run_ids(state.id):
            store.finish_run(run_id, "failed", {"error": repr(exc)})
        raise


def _run_session(
    settings: PortfolioSettings,
    store: PortfolioStore,
    state: PortfolioState,
    run_id: int,
    today: date,
    dry_run: bool,
    unclean_shutdown_reason: str | None,
    *,
    provider: MarketDataProvider,
    notifier: EmailNotifier | None,
    broker: Broker | None,
) -> int:
    portfolio_cfg = settings.portfolio
    cost_model = build_cost_model(settings.costs)
    broker = broker or load_broker_from_env(
        store, cost_model, settings=settings, portfolio_id=state.id
    )

    enabled_positions = [p for p in settings.positions if p.enabled]
    tickers = [p.ticker for p in enabled_positions]
    price_data = _fetch_price_data(tickers, provider)

    fills = broker.sync_fills(
        state, price_data, session_date=today, dry_run=dry_run
    )
    if fills:
        logger.info("Filled %d pending order(s)", len(fills))

    if not dry_run:
        reconciliation_clean = _check_broker_reconciliation(broker, state, store, notifier)
        if reconciliation_clean and getattr(broker, "read_only", False):
            store.mark_live_confirmed(state.id)
            logger.warning(
                "First live run reconciled cleanly — live trading is now fully "
                "enabled for this portfolio from the next run onward"
            )

    prices = {
        ticker: _latest_close(df) for ticker, df in price_data.items()
    }
    next_fill_day = next_trading_day_after(today)

    # Read-only — reflected in both dry-run previews and real runs, so `--dry-run`
    # accurately shows a BUY being blocked if the switch is already tripped.
    tripped_at, tripped_reason = store.get_kill_switch_state(state.id)
    kill_switch_active = tripped_at is not None
    if kill_switch_active:
        logger.warning(
            "Kill switch already tripped (%s) — blocking new BUY orders this run",
            tripped_reason,
        )

    # Evaluating and persisting a *new* trip requires a real session/equity history
    # and must never happen as a side effect of a dry-run preview.
    equity_curve_history: list[tuple[str, float]] = []
    if not dry_run:
        equity_curve_history = store.get_equity_curve(state.id)
        if not kill_switch_active:
            session_start_value = (
                equity_curve_history[-1][1] if equity_curve_history else portfolio_cfg.initial_cash
            )
            value_before_decisions = portfolio_value(state, prices)

            if unclean_shutdown_reason is not None:
                store.trip_kill_switch(state.id, unclean_shutdown_reason)
                kill_switch_active = True
                logger.error("Kill switch tripped: %s", unclean_shutdown_reason)
                if notifier is not None:
                    notifier.send_error_alert(
                        "Kill switch tripped: unclean shutdown", unclean_shutdown_reason
                    )
            else:
                kill_switch_result = check_kill_switch(
                    config=settings.risk,
                    session_start_value=session_start_value,
                    current_value=value_before_decisions,
                    equity_curve=equity_curve_history,
                )
                if kill_switch_result.tripped:
                    store.trip_kill_switch(state.id, kill_switch_result.reason)
                    kill_switch_active = True
                    logger.error("Kill switch tripped: %s", kill_switch_result.reason)
                    if notifier is not None:
                        notifier.send_error_alert(
                            "Kill switch tripped", kill_switch_result.reason
                        )

    decisions: list[tuple[str, object]] = []
    queued = 0
    failed_tickers: list[str] = []

    for position_entry in enabled_positions:
        ticker = position_entry.ticker
        df = price_data.get(ticker)
        if df is None:
            logger.warning("Skipping %s — no price data", ticker)
            continue

        try:
            strategy_names = strategy_names_for_position(settings, position_entry)
            params = strategy_params_for_position(settings, position_entry)

            signal_results: dict[str, SignalResult] = {}
            for strategy_name in strategy_names:
                strategy = get_signal_strategy(strategy_name)
                signal_results[strategy_name] = strategy.evaluate(df, ticker, params)

            if len(signal_results) == 1:
                signal = next(iter(signal_results.values()))
                if not dry_run:
                    store.record_signal(
                        run_id, ticker, signal.action, signal.reason, signal.metadata
                    )
            else:
                if not dry_run:
                    for strategy_name, result in signal_results.items():
                        store.record_signal(
                            run_id,
                            ticker,
                            result.action,
                            f"[{strategy_name}] {result.reason}",
                            result.metadata,
                        )
                signal_mode = settings.defaults.get("signal_mode", "consensus")
                signal = combine_signals(signal_results, mode=signal_mode)
                if not dry_run:
                    store.record_signal(
                        run_id,
                        ticker,
                        signal.action,
                        f"[combined:{signal_mode}] {signal.reason}",
                        signal.metadata,
                    )

            current_price = prices[ticker]
            decision = decide(
                ticker=ticker,
                signal=signal,
                state=state,
                config=portfolio_cfg,
                current_price=current_price,
                prices=prices,
                stop_loss_pct=stop_loss_for_position(settings, position_entry),
            )
            if kill_switch_active and decision.intent is not None and decision.intent.side == "BUY":
                decision = DecisionResult(
                    signal=decision.signal,
                    intent=None,
                    skipped_reason="kill switch tripped — new BUY orders blocked",
                )
            elif (
                decision.intent is not None
                and decision.intent.side == "BUY"
                and getattr(broker, "live", False)
                and not getattr(broker, "read_only", False)
            ):
                # Defense in depth, independent of decide()'s position sizing — the
                # only place in the orchestration that has both a price and the
                # live/read-only state needed to apply these.
                violation = live_notional_ceiling_violation(
                    ticker, decision.intent.shares, current_price
                ) or live_portfolio_ceiling_violation(
                    portfolio_value(state, prices) + decision.intent.shares * current_price
                )
                if violation is not None:
                    logger.error("Live order blocked: %s", violation)
                    if notifier is not None:
                        notifier.send_error_alert("Live order blocked by notional ceiling", violation)
                    decision = DecisionResult(
                        signal=decision.signal, intent=None, skipped_reason=violation
                    )
            decisions.append((ticker, decision))

            if decision.intent and next_fill_day is not None:
                order_id = broker.submit_order(
                    state, run_id, decision.intent, dry_run=dry_run, fill_date=next_fill_day
                )
                if order_id is not None:
                    queued += 1
                    logger.info(
                        "Queued %s %s %.0f shares for %s — %s",
                        decision.intent.side,
                        ticker,
                        decision.intent.shares,
                        next_fill_day,
                        decision.intent.reason,
                    )
                elif dry_run and decision.intent:
                    logger.info(
                        "[dry-run] Would queue %s %s %.0f shares for %s",
                        decision.intent.side,
                        ticker,
                        decision.intent.shares,
                        next_fill_day,
                    )
        except Exception:
            logger.exception("Skipping %s — signal/decision/order error", ticker)
            failed_tickers.append(ticker)
            continue

    if failed_tickers and len(failed_tickers) == len(enabled_positions):
        logger.error("Every ticker failed this run: %s", failed_tickers)
        if notifier is not None:
            notifier.send_error_alert(
                "Daily job: all tickers failed",
                f"Every configured ticker raised an error this run: {failed_tickers}",
            )

    total_value = portfolio_value(state, prices)
    summary = {
        "session_date": today.isoformat(),
        "fills": len(fills),
        "queued_orders": queued,
        "failed_tickers": failed_tickers,
        "kill_switch_active": kill_switch_active,
        "cash": state.cash,
        "portfolio_value": total_value,
        "dry_run": dry_run,
    }
    if not dry_run:
        store.finish_run(run_id, "completed", summary)

    portfolio_metrics = None
    benchmark_comparison = None
    if not dry_run:
        today_str = today.isoformat()
        if not equity_curve_history or equity_curve_history[-1][0] != today_str:
            equity_curve_history.append((today_str, total_value))
        if len(equity_curve_history) >= 2:
            values = pd.Series([v for _, v in equity_curve_history], dtype=float)
            returns = values.pct_change().dropna()
            portfolio_metrics = compute_portfolio_metrics(returns)
        benchmark_comparison = _benchmark_comparison(
            equity_curve_history,
            settings.defaults.get("benchmark", DEFAULT_BENCHMARK),
            provider,
        )

    report_path = write_daily_report(
        Path("data/reports"),
        today,
        portfolio_name=portfolio_cfg.name,
        cash=state.cash,
        portfolio_value=total_value,
        fills=fills,
        decisions=decisions,
        dry_run=dry_run,
        portfolio_metrics=portfolio_metrics,
        benchmark_comparison=benchmark_comparison,
    )

    logger.info(
        "Run complete — cash $%.2f, value $%.2f, fills %d, queued %d",
        state.cash,
        total_value,
        len(fills),
        queued,
    )
    logger.info("Report: %s", report_path)

    if not dry_run and notifier is not None:
        notifier.send_daily_summary(report_path, summary)

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Daily paper trading job")
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to portfolio YAML config",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute signals and log decisions without changing state",
    )
    args = parser.parse_args(argv)
    notifier = EmailNotifier(load_smtp_config_from_env())
    try:
        return run_daily(args.config, dry_run=args.dry_run, notifier=notifier)
    except Exception as exc:
        logger.exception("Daily job failed")
        notifier.send_error_alert("Daily job failed", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
