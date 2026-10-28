"""Operator CLI for inspecting and clearing the portfolio-level kill switch.

This is the only supported way to resume trading after a kill-switch trip —
there is no automated or web-UI path, by design.
"""

import argparse
import sys

from stock_analysis.storage import get_db_engine
from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.trading.config import load_portfolio_config

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect or clear the trading kill switch")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Path to portfolio YAML config")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("status", help="Show the current kill-switch state")

    clear_parser = subparsers.add_parser("clear", help="Clear a tripped kill switch")
    clear_parser.add_argument(
        "--reason", required=True, help="Why it's safe to resume trading (recorded in logs)"
    )

    args = parser.parse_args(argv)
    if args.command == "status":
        return status(args.config)
    return clear(args.config, args.reason)


if __name__ == "__main__":
    sys.exit(main())
