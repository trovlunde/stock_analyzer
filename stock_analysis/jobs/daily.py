import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_analysis.fetch_stock_data import fetch_stock_data
from stock_analysis.storage import get_db_engine
from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.trading.config import (
    load_portfolio_config,
    stop_loss_for_position,
    strategy_name_for_position,
    strategy_params_for_position,
)
from stock_analysis.trading.cost_models import build_cost_model
from stock_analysis.trading.decision import decide, portfolio_value
from stock_analysis.trading.market_calendar import (
    is_trading_day,
    next_trading_day_after,
)
from stock_analysis.trading.paper_broker import PaperBroker
from stock_analysis.trading.portfolio_metrics import compute_portfolio_metrics
from stock_analysis.trading.report import write_daily_report
from stock_analysis.trading.signals import get_signal_strategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "config/portfolio.yaml"
DATA_PERIOD = "2y"


def _session_date(now: datetime | None = None) -> datetime.date:
    return (now or datetime.now(timezone.utc)).date()


def _latest_close(df: pd.DataFrame) -> float:
    return float(df.iloc[-1]["Close"])


def _fetch_price_data(tickers: list[str]) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df, _ = fetch_stock_data(ticker, period=DATA_PERIOD)
        if df is None or df.empty:
            logger.warning("No data for %s", ticker)
            continue
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        data[ticker] = df
    return data


def run_daily(config_path: str, dry_run: bool = False) -> int:
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

    engine = get_db_engine()
    store = PortfolioStore(engine)
    if dry_run:
        state = store.get_portfolio_by_name(portfolio_cfg.name)
        if state is None:
            from stock_analysis.storage.portfolio_store import PortfolioState

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
        run_id = store.start_run(state.id, dry_run=False)

    cost_model = build_cost_model(settings.costs)
    broker = PaperBroker(store, cost_model)

    enabled_positions = [p for p in settings.positions if p.enabled]
    tickers = [p.ticker for p in enabled_positions]
    price_data = _fetch_price_data(tickers)

    fills = broker.fill_pending_orders(
        state, price_data, session_date=today, dry_run=dry_run
    )
    if fills:
        logger.info("Filled %d pending order(s)", len(fills))

    prices = {
        ticker: _latest_close(df) for ticker, df in price_data.items()
    }
    next_fill_day = next_trading_day_after(today)

    decisions: list[tuple[str, object]] = []
    queued = 0

    for position_entry in enabled_positions:
        ticker = position_entry.ticker
        df = price_data.get(ticker)
        if df is None:
            logger.warning("Skipping %s — no price data", ticker)
            continue

        strategy_name = strategy_name_for_position(settings, position_entry)
        params = strategy_params_for_position(settings, position_entry)
        strategy = get_signal_strategy(strategy_name)
        signal = strategy.evaluate(df, params)

        if not dry_run:
            store.record_signal(
                run_id, ticker, signal.action, signal.reason, signal.metadata
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
        decisions.append((ticker, decision))

        if decision.intent and next_fill_day is not None:
            order_id = broker.queue_order(
                state, run_id, decision.intent, fill_date=next_fill_day, dry_run=dry_run
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

    total_value = portfolio_value(state, prices)
    summary = {
        "session_date": today.isoformat(),
        "fills": len(fills),
        "queued_orders": queued,
        "cash": state.cash,
        "portfolio_value": total_value,
        "dry_run": dry_run,
    }
    if not dry_run:
        store.finish_run(run_id, "completed", summary)

    portfolio_metrics = None
    if not dry_run:
        equity_curve = store.get_equity_curve(state.id)
        today_str = today.isoformat()
        if not equity_curve or equity_curve[-1][0] != today_str:
            equity_curve.append((today_str, total_value))
        if len(equity_curve) >= 2:
            values = pd.Series([v for _, v in equity_curve], dtype=float)
            returns = values.pct_change().dropna()
            portfolio_metrics = compute_portfolio_metrics(returns)

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
    )

    logger.info(
        "Run complete — cash $%.2f, value $%.2f, fills %d, queued %d",
        state.cash,
        total_value,
        len(fills),
        queued,
    )
    logger.info("Report: %s", report_path)
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
    try:
        return run_daily(args.config, dry_run=args.dry_run)
    except Exception:
        logger.exception("Daily job failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
