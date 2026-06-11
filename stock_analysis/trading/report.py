from datetime import date
from pathlib import Path

from stock_analysis.trading.decision import DecisionResult
from stock_analysis.trading.paper_broker import FillResult


def write_daily_report(
    report_dir: Path,
    session_date: date,
    *,
    portfolio_name: str,
    cash: float,
    portfolio_value: float,
    fills: list[FillResult],
    decisions: list[tuple[str, DecisionResult]],
    dry_run: bool,
    skipped: str | None = None,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"{session_date.isoformat()}.md"

    lines = [
        f"# Paper trading report — {session_date.isoformat()}",
        "",
        f"- Portfolio: `{portfolio_name}`",
        f"- Mode: `{'dry-run' if dry_run else 'paper'}`",
        f"- Cash: `${cash:,.2f}`",
        f"- Portfolio value: `${portfolio_value:,.2f}`",
        "",
    ]

    if skipped:
        lines.extend([f"**Skipped:** {skipped}", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    lines.append("## Fills")
    if fills:
        for fill in fills:
            lines.append(
                f"- {fill.side} {fill.shares:.0f} {fill.ticker} @ ${fill.fill_price:.2f} "
                f"(costs ${fill.costs.total:.2f})"
            )
    else:
        lines.append("- None")

    lines.extend(["", "## Signals & decisions"])
    for ticker, decision in decisions:
        signal = decision.signal
        lines.append(f"### {ticker}")
        lines.append(f"- Signal: **{signal.action}** — {signal.reason}")
        if decision.intent:
            lines.append(
                f"- Order: **{decision.intent.side}** "
                f"{decision.intent.shares:.0f} shares — {decision.intent.reason}"
            )
        elif decision.skipped_reason:
            lines.append(f"- No order: {decision.skipped_reason}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
