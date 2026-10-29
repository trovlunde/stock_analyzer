import logging
import os

from dotenv import load_dotenv

from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.trading.broker import Broker
from stock_analysis.trading.cost_models import CostModel

logger = logging.getLogger(__name__)


def load_broker_from_env(store: PortfolioStore, cost_model: CostModel) -> Broker:
    """Single switch point between the paper simulator and Alpaca.

    BROKER_MODE=paper_sim (default, zero risk) vs BROKER_MODE=alpaca (Alpaca's
    *paper* endpoint, real API calls, fake money).

    There is no live-trading path yet — that's gated behind a not-yet-written
    live_gate.py (a future phase). `paper=True` is hard-coded below rather than
    read from an env var on purpose: a loose string comparison here (e.g.
    `ALPACA_PAPER=true`) would fail open to Alpaca's live endpoint on any typo
    or unexpected value, with nothing else in the codebase to catch it.
    """
    load_dotenv()
    mode = os.environ.get("BROKER_MODE", "paper_sim")

    if mode == "alpaca":
        from alpaca.trading.client import TradingClient

        from stock_analysis.trading.alpaca_broker import AlpacaBroker

        if os.environ.get("ALPACA_PAPER", "true").lower() != "true":
            logger.warning(
                "ALPACA_PAPER is set but ignored — live trading isn't wired up yet, "
                "forcing Alpaca's paper endpoint"
            )

        client = TradingClient(
            os.environ["ALPACA_API_KEY"],
            os.environ["ALPACA_SECRET_KEY"],
            paper=True,
        )
        return AlpacaBroker(client, store)

    if mode != "paper_sim":
        raise ValueError(f"Unknown BROKER_MODE '{mode}' — expected 'paper_sim' or 'alpaca'")

    from stock_analysis.trading.paper_broker import PaperBroker

    return PaperBroker(store, cost_model)
