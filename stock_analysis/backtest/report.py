"""Backtest report formatting and the log of every configuration tried."""

import json
from pathlib import Path

DEFAULT_RUN_LOG = Path("data/backtests/runs.jsonl")

_ROWS = [
    ("Total return", "total_return", "pct"),
    ("CAGR", "cagr", "pct"),
    ("Volatility (ann.)", "volatility", "pct"),
    ("Sharpe", "sharpe", "num"),
    ("Sortino", "sortino", "num"),
    ("Max drawdown", "max_drawdown", "pct"),
    ("Calmar", "calmar", "num"),
    ("Trades", "trades", "int"),
    ("Round trips", "round_trips", "int"),
    ("Hit rate", "hit_rate", "pct"),
    ("Turnover (x/yr)", "annual_turnover", "num"),
    ("Cost drag", "cost_drag", "pct"),
    ("Avg exposure", "avg_exposure", "pct"),
]


def _format(value, kind: str) -> str:
    if value is None:
        return "—"
    if kind == "pct":
        return f"{value:.2%}"
    if kind == "int":
        return f"{value:d}"
    return f"{value:.2f}"


def format_comparison(columns: dict[str, dict]) -> str:
    """Markdown table with one column per named result ({name: metrics dict}).

    Metrics a column does not have (e.g. trade counts for a benchmark) show as —.
    """
    names = list(columns)
    lines = [
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---:|" * len(names),
    ]
    for label, key, kind in _ROWS:
        if not any(key in columns[name] for name in names):
            continue
        cells = [_format(columns[name].get(key), kind) for name in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def format_yearly_returns(equity_curves: dict) -> str:
    """Markdown table of each curve's return per calendar year ({name: equity Series}).

    One good or bad year can carry a whole backtest; this shows whether a result
    depends on a particular stretch of the market. The first and last year are
    usually part-years.
    """
    names = list(equity_curves)
    yearly = {}
    for name, equity in equity_curves.items():
        year_end = equity.groupby(equity.index.year).last()
        previous = year_end.shift(1)
        previous.iloc[0] = equity.iloc[0]
        yearly[name] = year_end / previous - 1
    years = sorted(set().union(*(series.index for series in yearly.values())))
    lines = ["| Year | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for year in years:
        cells = [_format(yearly[name].get(year), "pct") for name in names]
        lines.append(f"| {year} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def append_run(record: dict, path: Path = DEFAULT_RUN_LOG) -> None:
    """Append one backtest run to the log. Every run counts as a trial when judging
    whether the best one is better than chance, so nothing is ever overwritten."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def logged_runs(experiment: str, path: Path = DEFAULT_RUN_LOG) -> list[dict]:
    """Earlier runs of the same experiment (same tickers and dates)."""
    if not path.exists():
        return []
    runs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("experiment") == experiment:
            runs.append(record)
    return runs
