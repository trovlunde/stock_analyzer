"""Inputs to `decide()` that come from price history rather than from a signal.

The daily job and the backtester both call these, so sizing, stops, caps and
rebalancing behave the same in a replay as they do in paper trading.
"""

from dataclasses import dataclass, field

import pandas as pd

from stock_analysis.trading.allocation import universe_weights
from stock_analysis.trading.config import (
    PortfolioConfig,
    PortfolioSettings,
    PositionEntry,
    stop_loss_for_position,
)
from stock_analysis.trading.sizing import (
    ALLOCATION_MODES,
    annualised_volatility,
    exposure_scale,
)
from stock_analysis.trading.stops import trailing_stop


@dataclass(frozen=True)
class RunContext:
    """What every ticker's decision in one run shares."""

    universe_weights: dict[str, float] | None = None
    exposure_scale: float = 1.0
    correlations: pd.DataFrame | None = None
    sectors: dict[str, str] = field(default_factory=dict)
    rebalance_due: bool = False


def needs_universe_closes(config: PortfolioConfig) -> bool:
    """Whether the config has a rule that looks across tickers (so the caller has
    to supply a table of closes for the whole universe)."""
    return config.sizing in ALLOCATION_MODES or config.max_correlation is not None


def is_rebalance_session(
    frequency: str, session: pd.Timestamp, previous_session: pd.Timestamp | None
) -> bool:
    """True on the first session of each week / month (or every session for `daily`)."""
    if frequency == "daily":
        return True
    if previous_session is None:
        return False
    if frequency == "weekly":
        return session.isocalendar()[:2] != previous_session.isocalendar()[:2]
    return (session.year, session.month) != (previous_session.year, previous_session.month)


def allocation_weights(config: PortfolioConfig, closes: pd.DataFrame) -> dict[str, float] | None:
    """Universe weights for `min_variance` / `hrp` sizing from the closes up to now."""
    if config.sizing not in ALLOCATION_MODES:
        return None
    returns = closes.iloc[-(config.allocation_lookback + 1):].pct_change(fill_method=None).iloc[1:]
    return universe_weights(returns, config.sizing, max_weight=config.max_position_pct)


def build_run_context(
    settings: PortfolioSettings,
    *,
    closes: pd.DataFrame | None,
    equity: list[float],
    rebalance_due: bool = False,
    weights: dict[str, float] | None = None,
) -> RunContext:
    """`closes` is one column of closes per ticker, ending at the session being
    decided (only needed when `needs_universe_closes`). `equity` is the portfolio
    value per session so far, including this one. `weights` lets a caller that
    caches the allocation pass it in instead of having it recomputed."""
    config = settings.portfolio
    if weights is None and closes is not None:
        weights = allocation_weights(config, closes)
    correlations = None
    if config.max_correlation is not None and closes is not None:
        returns = closes.iloc[-(config.correlation_lookback + 1):].pct_change(fill_method=None)
        correlations = returns.corr(min_periods=config.correlation_lookback // 2)
    return RunContext(
        universe_weights=weights,
        exposure_scale=exposure_scale(equity, config.target_volatility, config.vol_lookback),
        correlations=correlations,
        sectors={p.ticker: p.sector for p in settings.positions if p.sector},
        rebalance_due=rebalance_due and config.rebalance_band is not None,
    )


def decision_inputs(
    settings: PortfolioSettings,
    position_entry: PositionEntry,
    df: pd.DataFrame,
    context: RunContext,
    *,
    sizing_needed: bool,
) -> dict:
    """Keyword arguments for `decide()` for one ticker.

    `df` is the ticker's bars up to the session being decided. `sizing_needed` is
    False when no order can be sized (a HOLD or SELL signal on a ticker that will
    not be rebalanced), which skips the volatility estimate.
    """
    config = settings.portfolio
    ticker = position_entry.ticker
    stop_loss_pct = stop_loss_for_position(settings, position_entry)
    stop = trailing_stop(
        df,
        stop_type=config.stop_type,
        stop_loss_pct=stop_loss_pct,
        atr_multiple=config.stop_atr_multiple,
        atr_period=config.stop_atr_period,
        lookback=config.stop_lookback,
    )
    correlations = None
    if context.correlations is not None and ticker in context.correlations.columns:
        correlations = context.correlations[ticker].drop(ticker).dropna().to_dict()
    return {
        # With a trailing stop the percentage applies to the recent high, not to cost
        "stop_loss_pct": stop_loss_pct if config.stop_type == "fixed" else None,
        "stop_price": stop[0] if stop else None,
        "stop_reason": stop[1] if stop else None,
        "volatility": (
            annualised_volatility(df["Close"], config.vol_lookback)
            if sizing_needed and config.sizing == "vol_target"
            else None
        ),
        "universe_weight": (
            context.universe_weights.get(ticker) if context.universe_weights else None
        ),
        "exposure_scale": context.exposure_scale,
        "sectors": context.sectors or None,
        "correlations": correlations,
        "rebalance_due": context.rebalance_due,
    }
