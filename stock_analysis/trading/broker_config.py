import logging
import os

from dotenv import load_dotenv

from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.trading.broker import Broker
from stock_analysis.trading.config import PortfolioSettings
from stock_analysis.trading.cost_models import CostModel

logger = logging.getLogger(__name__)


def load_broker_from_env(
    store: PortfolioStore,
    cost_model: CostModel,
    *,
    settings: PortfolioSettings | None = None,
    portfolio_id: int | None = None,
) -> Broker:
    """Single switch point between the paper simulator and Alpaca.

    BROKER_MODE=paper_sim (default, zero risk) vs BROKER_MODE=alpaca (Alpaca's
    paper endpoint unless LIVE_TRADING_ENABLED=true AND the full live_gate check
    passes — `settings`/`portfolio_id` are required in that case, to evaluate it).

    Raises `LiveTradingBlocked` (not a silent fallback to paper) if live trading
    was requested but the gate refuses — the caller should treat that exactly
    like any other run_daily() failure.
    """
    load_dotenv()
    mode = os.environ.get("BROKER_MODE", "paper_sim")

    if mode == "alpaca":
        from alpaca.trading.client import TradingClient

        from stock_analysis.trading.alpaca_broker import AlpacaBroker
        from stock_analysis.trading.live_gate import (
            LiveTradingBlocked,
            check_live_trading_gate,
            is_first_live_run,
            live_trading_requested,
        )

        live = False
        read_only = False
        if live_trading_requested():
            if settings is None or portfolio_id is None:
                raise ValueError(
                    "settings and portfolio_id are required to evaluate live trading"
                )
            gate = check_live_trading_gate(settings, store, portfolio_id)
            if not gate.allowed:
                raise LiveTradingBlocked(gate.reason)
            live = True
            read_only = is_first_live_run(store, portfolio_id)
            if read_only:
                logger.warning(
                    "First live run for this portfolio — read-only (connect/reconcile/"
                    "report only, no orders will be submitted) until a clean "
                    "reconciliation is observed"
                )

        client = TradingClient(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_SECRET_KEY"],
            paper=not live,
        )
        return AlpacaBroker(client, store, live=live, read_only=read_only)

    if mode != "paper_sim":
        raise ValueError(f"Unknown BROKER_MODE '{mode}' — expected 'paper_sim' or 'alpaca'")

    from stock_analysis.trading.paper_broker import PaperBroker

    return PaperBroker(store, cost_model)


def connect_broker_for_inspection(store: PortfolioStore, cost_model: CostModel) -> Broker:
    """Build a broker connection for read-only inspection (the `stock-analysis-risk
    status`/`reconcile` CLI) — unlike `load_broker_from_env`, this never runs the
    live-trading authorization gate.

    That's deliberate: the gate requires the kill switch to be clear, but checking
    broker truth via `reconcile` is exactly how an operator recovers from a trip in
    the first place (e.g. the near-certain reconciliation mismatch on a portfolio's
    very first live run, before local state has ever matched the real account).
    Gating a read-only snapshot fetch behind "the kill switch must already be
    clear" would make that recovery path unusable.

    Connects to whichever Alpaca endpoint ALPACA_PAPER actually names — it mirrors
    the real configured endpoint rather than defaulting to paper, so the operator
    sees the account that actually diverged. The returned broker is always
    read_only=True; this path never submits orders."""
    load_dotenv()
    mode = os.environ.get("BROKER_MODE", "paper_sim")

    if mode == "alpaca":
        from alpaca.trading.client import TradingClient

        from stock_analysis.trading.alpaca_broker import AlpacaBroker

        paper = os.environ.get("ALPACA_PAPER", "true").lower() != "false"
        client = TradingClient(
            os.environ["ALPACA_API_KEY"], os.environ["ALPACA_SECRET_KEY"], paper=paper
        )
        return AlpacaBroker(client, store, live=not paper, read_only=True)

    if mode != "paper_sim":
        raise ValueError(f"Unknown BROKER_MODE '{mode}' — expected 'paper_sim' or 'alpaca'")

    from stock_analysis.trading.paper_broker import PaperBroker

    return PaperBroker(store, cost_model)
