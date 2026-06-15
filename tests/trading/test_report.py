from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from stock_analysis.trading.report import write_daily_report


def _base_kwargs(tmp_path: Path) -> dict:
    return dict(
        report_dir=tmp_path,
        session_date=date(2024, 6, 3),
        portfolio_name="paper-us",
        cash=5000.0,
        portfolio_value=10000.0,
        fills=[],
        decisions=[],
        dry_run=False,
    )


def test_write_daily_report_no_metrics_section(tmp_path):
    path = write_daily_report(**_base_kwargs(tmp_path))
    content = path.read_text(encoding="utf-8")
    assert "## Portfolio metrics" not in content


def test_write_daily_report_with_metrics(tmp_path):
    rolling = pd.Series([0.8, 0.9, 1.1, 1.2])
    metrics = {"calmar": 1.5, "sortino": 0.75, "rolling_sharpe": rolling}
    path = write_daily_report(**_base_kwargs(tmp_path), portfolio_metrics=metrics)
    content = path.read_text(encoding="utf-8")
    assert "## Portfolio metrics" in content
    assert "Calmar ratio: 1.5000" in content
    assert "Sortino ratio: 0.7500" in content
    assert "Rolling Sharpe (latest): 1.2000" in content


def test_write_daily_report_insufficient_history(tmp_path):
    metrics = {"calmar": None, "sortino": None, "rolling_sharpe": pd.Series(dtype=float)}
    path = write_daily_report(**_base_kwargs(tmp_path), portfolio_metrics=metrics)
    content = path.read_text(encoding="utf-8")
    assert "## Portfolio metrics" in content
    assert "Insufficient history" in content


def test_write_daily_report_partial_metrics(tmp_path):
    metrics = {"calmar": 0.5, "sortino": None, "rolling_sharpe": pd.Series(dtype=float)}
    path = write_daily_report(**_base_kwargs(tmp_path), portfolio_metrics=metrics)
    content = path.read_text(encoding="utf-8")
    assert "Calmar ratio: 0.5000" in content
    assert "Sortino ratio: N/A" in content
    assert "Rolling Sharpe (latest): N/A" in content


def test_write_daily_report_fills_signals_unchanged(tmp_path):
    path = write_daily_report(**_base_kwargs(tmp_path))
    content = path.read_text(encoding="utf-8")
    assert "## Fills" in content
    assert "## Signals & decisions" in content
