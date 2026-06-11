from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class PortfolioConfig:
    name: str
    enabled: bool
    initial_cash: float
    max_positions: int
    position_size_pct: float
    long_only: bool
    stop_loss_pct: float | None


@dataclass
class PositionEntry:
    ticker: str
    enabled: bool = True
    strategy: str | None = None
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
    )

    positions = []
    for entry in positions_raw:
        if isinstance(entry, str):
            positions.append(PositionEntry(ticker=entry.upper()))
            continue
        if not isinstance(entry, dict) or "ticker" not in entry:
            raise ValueError(f"Invalid position entry: {entry}")
        positions.append(
            PositionEntry(
                ticker=str(entry["ticker"]).upper(),
                enabled=bool(entry.get("enabled", True)),
                strategy=entry.get("strategy"),
                ma_short=entry.get("ma_short"),
                ma_long=entry.get("ma_long"),
                stop_loss_pct=entry.get("stop_loss_pct"),
            )
        )

    return PortfolioSettings(
        portfolio=portfolio,
        costs=dict(raw.get("costs", {})),
        defaults=dict(raw.get("defaults", {})),
        positions=positions,
        schedule=dict(raw.get("schedule", {})),
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


def strategy_name_for_position(
    settings: PortfolioSettings, position: PositionEntry
) -> str:
    return position.strategy or settings.defaults.get("strategy", "ma_crossover")


def stop_loss_for_position(
    settings: PortfolioSettings, position: PositionEntry
) -> float | None:
    if position.stop_loss_pct is not None:
        return float(position.stop_loss_pct)
    return settings.portfolio.stop_loss_pct
