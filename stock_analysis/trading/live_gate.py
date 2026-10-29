import os
from dataclasses import dataclass

from stock_analysis.storage.portfolio_store import PortfolioStore
from stock_analysis.trading.config import PortfolioSettings

# The exact phrase LIVE_TRADING_CONFIRM must equal — a separate, deliberate
# confirmation so a single fat-fingered LIVE_TRADING_ENABLED=true can't flip real
# money trading on by itself.
LIVE_CONFIRM_PHRASE = "I understand this trades real money"

# Trading days with no trip before live trading is allowed. Matches the plan's
# ~1-month default — a DB-backed check (count_consecutive_completed_runs), not a
# human promise.
MIN_CONSECUTIVE_CLEAN_RUNS = 20


class LiveTradingBlocked(Exception):
    """Raised by broker_config.load_broker_from_env when LIVE_TRADING_ENABLED is set
    but the gate refuses — the caller should treat this exactly like any other
    run_daily() failure (log, alert, exit non-zero), not silently fall back to paper."""


@dataclass(frozen=True)
class LiveGateResult:
    allowed: bool
    reasons: tuple[str, ...] = ()

    @property
    def reason(self) -> str | None:
        return "; ".join(self.reasons) if self.reasons else None


def live_trading_requested() -> bool:
    """Has the operator asked for live trading at all, independent of whether the
    gate actually allows it?"""
    return os.environ.get("LIVE_TRADING_ENABLED", "false").lower() == "true"


def check_live_trading_gate(
    settings: PortfolioSettings, store: PortfolioStore, portfolio_id: int
) -> LiveGateResult:
    """All checks are required (AND) — re-run this every time live trading is about
    to be used, not just once at startup. Pure w.r.t. env/DB reads — makes no writes."""
    reasons: list[str] = []

    if not live_trading_requested():
        reasons.append("LIVE_TRADING_ENABLED is not 'true'")

    if os.environ.get("LIVE_TRADING_CONFIRM") != LIVE_CONFIRM_PHRASE:
        reasons.append(f"LIVE_TRADING_CONFIRM must exactly equal '{LIVE_CONFIRM_PHRASE}'")

    if os.environ.get("ALPACA_PAPER", "true").lower() != "false":
        reasons.append(
            "ALPACA_PAPER must be explicitly 'false' to go live (config-drift guard)"
        )

    if settings.risk.max_daily_loss_pct is None or settings.risk.max_drawdown_pct is None:
        reasons.append("risk.max_daily_loss_pct and risk.max_drawdown_pct must both be set")

    tripped_at, tripped_reason = store.get_kill_switch_state(portfolio_id)
    if tripped_at is not None:
        reasons.append(f"kill switch is tripped: {tripped_reason}")

    clean_runs = store.count_consecutive_completed_runs(portfolio_id)
    if clean_runs < MIN_CONSECUTIVE_CLEAN_RUNS:
        reasons.append(
            f"only {clean_runs} consecutive clean run(s), need {MIN_CONSECUTIVE_CLEAN_RUNS}"
        )

    for var in ("LIVE_MAX_NOTIONAL_PER_ORDER", "LIVE_MAX_PORTFOLIO_NOTIONAL"):
        raw = os.environ.get(var)
        if not raw:
            reasons.append(f"{var} is not set")
            continue
        try:
            if float(raw) <= 0:
                reasons.append(f"{var} must be a positive number")
        except ValueError:
            reasons.append(f"{var} is not a valid number: {raw!r}")

    return LiveGateResult(allowed=not reasons, reasons=tuple(reasons))


def is_first_live_run(store: PortfolioStore, portfolio_id: int) -> bool:
    """True until a clean live reconciliation has ever been observed for this
    portfolio — forces the first (and only the first) live run to be read-only."""
    return store.get_live_confirmed_at(portfolio_id) is None


def live_notional_ceiling_violation(ticker: str, shares: float, price: float) -> str | None:
    """Defense in depth, independent of decide()'s position sizing — enforced at the
    orchestration level (jobs/daily.py), which is the only place a pre-fill price
    estimate is available. Returns a rejection reason, or None if within limits."""
    notional = shares * price
    per_order = os.environ.get("LIVE_MAX_NOTIONAL_PER_ORDER")
    if per_order:
        try:
            if notional > float(per_order):
                return (
                    f"{ticker} order notional ${notional:,.2f} exceeds "
                    f"LIVE_MAX_NOTIONAL_PER_ORDER ${float(per_order):,.2f}"
                )
        except ValueError:
            return f"LIVE_MAX_NOTIONAL_PER_ORDER is not a valid number: {per_order!r}"
    return None


def live_portfolio_ceiling_violation(total_portfolio_value: float) -> str | None:
    """Blocks any new BUY once total portfolio value is already at/over the ceiling
    — a hard cap independent of per-order sizing or how many positions are open."""
    ceiling = os.environ.get("LIVE_MAX_PORTFOLIO_NOTIONAL")
    if ceiling:
        try:
            if total_portfolio_value >= float(ceiling):
                return (
                    f"portfolio value ${total_portfolio_value:,.2f} is at/over "
                    f"LIVE_MAX_PORTFOLIO_NOTIONAL ${float(ceiling):,.2f}"
                )
        except ValueError:
            return f"LIVE_MAX_PORTFOLIO_NOTIONAL is not a valid number: {ceiling!r}"
    return None
