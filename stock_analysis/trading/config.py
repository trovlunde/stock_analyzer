from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from stock_analysis.trading.regime import RegimeFilter
from stock_analysis.trading.risk import RiskConfig
from stock_analysis.trading.sizing import SIZING_MODES


@dataclass
class PortfolioConfig:
    name: str
    enabled: bool
    initial_cash: float
    max_positions: int
    position_size_pct: float
    long_only: bool
    stop_loss_pct: float | None
    # Sizing of new positions — see trading.sizing.position_weight
    sizing: str = "fixed"
    risk_per_position: float = 0.02
    max_position_pct: float = 0.20
    vol_lookback: int = 60


@dataclass
class PositionEntry:
    ticker: str
    enabled: bool = True
    strategy: str | None = None
    strategies: list[str] | None = None
    ma_short: int | None = None
    ma_long: int | None = None
    stop_loss_pct: float | None = None


@dataclass
class PortfolioSettings:
    portfolio: PortfolioConfig
    costs: dict[str, Any]
    defaults: dict[str, Any]
    positions: list[PositionEntry]
    schedule: dict[str, Any] = field(default_factory=dict)
    risk: RiskConfig = field(default_factory=RiskConfig)
    regime_filter: RegimeFilter | None = None


def _coerce_strategy_list(raw: Any, where: str) -> list[str] | None:
    """Reject the common YAML typo `strategies: ml_movement` (a bare string) instead
    of silently exploding into per-character strategy names via list("ml_movement")."""
    if raw is None:
        return None
    if isinstance(raw, str) or not isinstance(raw, list):
        raise ValueError(
            f"{where} must be a list of strategy names, e.g. [ma_crossover, ml_movement] "
            f"(got {raw!r})"
        )
    return list(raw)


def load_portfolio_config(path: str | Path) -> PortfolioSettings:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Config root must be a mapping")

    portfolio_raw = raw.get("portfolio", {})
    positions_raw = raw.get("positions", [])
    if not positions_raw:
        raise ValueError("Config must include at least one position")

    portfolio = PortfolioConfig(
        name=str(portfolio_raw.get("name", "paper-us")),
        enabled=bool(portfolio_raw.get("enabled", True)),
        initial_cash=float(portfolio_raw.get("initial_cash", 100_000)),
        max_positions=int(portfolio_raw.get("max_positions", 10)),
        position_size_pct=float(portfolio_raw.get("position_size_pct", 0.10)),
        long_only=bool(portfolio_raw.get("long_only", True)),
        stop_loss_pct=portfolio_raw.get("stop_loss_pct"),
        sizing=str(portfolio_raw.get("sizing", "fixed")),
        risk_per_position=float(portfolio_raw.get("risk_per_position", 0.02)),
        max_position_pct=float(portfolio_raw.get("max_position_pct", 0.20)),
        vol_lookback=int(portfolio_raw.get("vol_lookback", 60)),
    )
    if portfolio.sizing not in SIZING_MODES:
        raise ValueError(
            f"portfolio.sizing must be one of {list(SIZING_MODES)} (got {portfolio.sizing!r})"
        )

    regime_raw = raw.get("regime_filter")
    regime_filter = None
    if regime_raw:
        if not isinstance(regime_raw, dict) or not regime_raw.get("ticker"):
            raise ValueError("regime_filter needs a ticker, e.g. {ticker: SPY, ma: 200}")
        regime_filter = RegimeFilter(
            ticker=str(regime_raw["ticker"]).upper(), ma=int(regime_raw.get("ma", 200))
        )

    positions = []
    for entry in positions_raw:
        if isinstance(entry, str):
            positions.append(PositionEntry(ticker=entry.upper()))
            continue
        if not isinstance(entry, dict) or "ticker" not in entry:
            raise ValueError(f"Invalid position entry: {entry}")
        ticker = str(entry["ticker"]).upper()
        positions.append(
            PositionEntry(
                ticker=ticker,
                enabled=bool(entry.get("enabled", True)),
                strategy=entry.get("strategy"),
                strategies=_coerce_strategy_list(
                    entry.get("strategies"), f"positions[{ticker}].strategies"
                ),
                ma_short=entry.get("ma_short"),
                ma_long=entry.get("ma_long"),
                stop_loss_pct=entry.get("stop_loss_pct"),
            )
        )

    risk_raw = raw.get("risk", {}) or {}
    liquidate_on_trip = bool(risk_raw.get("liquidate_on_trip", False))
    if liquidate_on_trip:
        # The kill switch only halts new BUYs today — it never liquidates existing
        # positions. Fail loudly rather than silently ignore the setting.
        raise ValueError(
            "risk.liquidate_on_trip is not yet implemented — leave it false or unset"
        )
    risk = RiskConfig(
        max_daily_loss_pct=risk_raw.get("max_daily_loss_pct"),
        max_drawdown_pct=risk_raw.get("max_drawdown_pct"),
        liquidate_on_trip=liquidate_on_trip,
    )

    defaults = dict(raw.get("defaults", {}))
    if "strategies" in defaults:
        defaults["strategies"] = _coerce_strategy_list(
            defaults["strategies"], "defaults.strategies"
        )

    return PortfolioSettings(
        portfolio=portfolio,
        costs=dict(raw.get("costs", {})),
        defaults=defaults,
        positions=positions,
        schedule=dict(raw.get("schedule", {})),
        risk=risk,
        regime_filter=regime_filter,
    )


def strategy_params_for_position(
    settings: PortfolioSettings, position: PositionEntry
) -> dict[str, Any]:
    params = dict(settings.defaults)
    if position.strategy:
        params["strategy"] = position.strategy
    if position.ma_short is not None:
        params["ma_short"] = position.ma_short
    if position.ma_long is not None:
        params["ma_long"] = position.ma_long
    return params


def strategy_names_for_position(
    settings: PortfolioSettings, position: PositionEntry
) -> list[str]:
    if position.strategies:
        return list(position.strategies)
    if position.strategy:
        return [position.strategy]
    default_strategies = settings.defaults.get("strategies")
    if default_strategies:
        return list(default_strategies)
    return [settings.defaults.get("strategy", "ma_crossover")]


def stop_loss_for_position(
    settings: PortfolioSettings, position: PositionEntry
) -> float | None:
    if position.stop_loss_pct is not None:
        return float(position.stop_loss_pct)
    return settings.portfolio.stop_loss_pct
