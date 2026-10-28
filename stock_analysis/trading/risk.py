from dataclasses import dataclass


@dataclass(frozen=True)
class RiskConfig:
    max_daily_loss_pct: float | None = None
    max_drawdown_pct: float | None = None
    liquidate_on_trip: bool = False


@dataclass(frozen=True)
class KillSwitchResult:
    tripped: bool
    reason: str | None = None


def check_kill_switch(
    *,
    config: RiskConfig,
    session_start_value: float,
    current_value: float,
    equity_curve: list[tuple[str, float]],
) -> KillSwitchResult:
    """Pure portfolio-level risk check — no I/O, no persistence. The caller is
    responsible for persisting a trip and deciding what to do about it."""
    if config.max_daily_loss_pct is not None and session_start_value > 0:
        daily_loss = (session_start_value - current_value) / session_start_value
        if daily_loss >= config.max_daily_loss_pct:
            return KillSwitchResult(
                True,
                f"daily loss {daily_loss:.1%} >= limit {config.max_daily_loss_pct:.1%}",
            )

    if config.max_drawdown_pct is not None and equity_curve:
        peak = max(v for _, v in equity_curve + [("today", current_value)])
        if peak > 0:
            drawdown = (peak - current_value) / peak
            if drawdown >= config.max_drawdown_pct:
                return KillSwitchResult(
                    True,
                    f"drawdown {drawdown:.1%} >= limit {config.max_drawdown_pct:.1%}",
                )

    return KillSwitchResult(False, None)
