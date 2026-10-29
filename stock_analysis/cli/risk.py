"""Operator CLI for inspecting/clearing the kill switch and reconciling local
portfolio state against a real broker.

This is the only supported way to resume trading after a kill-switch trip, and
the only way to accept a broker/local divergence — there is no automated or
web-UI path, by design.
"""

import argparse
import sys

from stock_analysis.storage import get_db_engine
from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.trading.config import load_portfolio_config
from stock_analysis.trading.cost_models import build_cost_model

DEFAULT_CONFIG = "config/portfolio.yaml"


def _portfolio_name(config_path: str) -> str:
    return load_portfolio_config(config_path).portfolio.name


def status(config_path: str) -> int:
    store = PortfolioStore(get_db_engine())
    name = _portfolio_name(config_path)
    state = store.get_portfolio_by_name(name)
    if state is None:
        print(f"No portfolio named '{name}' found yet.")
        return 0

    tripped_at, reason = store.get_kill_switch_state(state.id)
    if tripped_at is None:
        print(f"Kill switch for '{name}': clear.")
    else:
        print(f"Kill switch for '{name}': TRIPPED at {tripped_at} — {reason}")
    return 0


def clear(config_path: str, reason: str) -> int:
    store = PortfolioStore(get_db_engine())
    name = _portfolio_name(config_path)
    state = store.get_portfolio_by_name(name)
    if state is None:
        print(f"No portfolio named '{name}' found yet.")
        return 1

    tripped_at, tripped_reason = store.get_kill_switch_state(state.id)
    if tripped_at is None:
        print(f"Kill switch for '{name}' is already clear — nothing to do.")
        return 0

    store.clear_kill_switch(state.id)
    print(f"Cleared kill switch for '{name}' (was: {tripped_reason}). Clearance reason: {reason}")
    return 0


def reconcile(config_path: str, accept_broker_truth: bool) -> int:
    from stock_analysis.trading.broker_config import connect_broker_for_inspection

    store = PortfolioStore(get_db_engine())
    settings = load_portfolio_config(config_path)
    name = settings.portfolio.name
    state = store.get_portfolio_by_name(name)
    if state is None:
        print(f"No portfolio named '{name}' found yet.")
        return 1

    broker = connect_broker_for_inspection(store, build_cost_model(settings.costs))
    snapshot = broker.get_account_snapshot(state)
    if snapshot.source == "local":
        print(f"'{name}' is running against the paper simulator — nothing to reconcile.")
        return 1

    local_positions = {p.ticker: p for p in state.positions}
    cash_diff = snapshot.cash - state.cash
    print(f"Broker cash: ${snapshot.cash:,.2f}  Local cash: ${state.cash:,.2f}  (diff ${cash_diff:,.2f})")
    print(f"Broker positions: {snapshot.positions}")
    print(f"Local positions:  {{{', '.join(f'{t!r}: {p.shares}' for t, p in local_positions.items())}}}")

    if not accept_broker_truth:
        print("\nPass --accept-broker-truth to overwrite local state with the above broker truth.")
        return 0

    for ticker in list(local_positions):
        if ticker not in snapshot.positions:
            store.remove_position(state.id, ticker)
            print(f"Removed local position {ticker} (broker shows none)")

    for ticker, shares in snapshot.positions.items():
        existing = local_positions.get(ticker)
        broker_avg_cost = snapshot.avg_costs.get(ticker)
        if broker_avg_cost is not None:
            avg_cost = broker_avg_cost
        elif existing is not None:
            avg_cost = existing.avg_cost
        else:
            avg_cost = 0.0
            # A stop-loss check treats avg_cost<=0 as "never trigger" — an unknown
            # cost basis silently disables stop-loss for this position until fixed.
            print(
                f"WARNING: {ticker} has no broker-reported cost basis and no local "
                f"record — avg_cost set to 0.0, which DISABLES its stop-loss check "
                f"until corrected"
            )
        store.upsert_position(state.id, ticker, shares, avg_cost)
        if existing is None:
            print(f"Added local position {ticker}: {shares} shares @ avg_cost ${avg_cost:,.2f}")
        elif existing.shares != shares:
            print(f"Updated {ticker}: {existing.shares} -> {shares} shares (avg_cost ${avg_cost:,.2f})")

    store.update_cash(state.id, snapshot.cash)
    print(f"\nReconciled '{name}' to broker truth. Local cash is now ${snapshot.cash:,.2f}.")
    print("Kill switch is unchanged — run 'stock-analysis-risk clear --reason ...' to resume trading.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or clear the trading kill switch")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to portfolio YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show the current kill-switch state")

    clear_parser = subparsers.add_parser("clear", help="Clear a tripped kill switch")
    clear_parser.add_argument(
        "--reason", required=True, help="Why it's safe to resume trading (recorded in logs)"
    )

    reconcile_parser = subparsers.add_parser(
        "reconcile", help="Compare local state to the broker, optionally accepting broker truth"
    )
    reconcile_parser.add_argument(
        "--accept-broker-truth",
        action="store_true",
        help="Overwrite local cash/positions with the broker's — without this flag, only prints the diff",
    )

    args = parser.parse_args(argv)
    if args.command == "status":
        return status(args.config)
    if args.command == "clear":
        return clear(args.config, args.reason)
    return reconcile(args.config, args.accept_broker_truth)


if __name__ == "__main__":
    sys.exit(main())
