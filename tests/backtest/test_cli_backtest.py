import numpy as np
import pandas as pd
import pytest

from stock_analysis.cli.backtest import run_backtest_command
from stock_analysis.cli.evaluate_models import build_parser

CONFIG = """
portfolio:
  name: bt
  initial_cash: 100000
  max_positions: 10
  position_size_pct: 0.10
costs:
  model: equity
  commission_per_trade: 1.00
  slippage_pct: 0.0005
defaults:
  strategies: [ma_crossover, ml_movement]
  ma_short: 5
  ma_long: 20
positions:
  - ticker: AAA
  - ticker: BBB
"""


class FakeProvider:
    """Deterministic trending-and-cycling prices so MA crossovers actually occur."""

    def __init__(self, unknown=()):
        self.unknown = set(unknown)
        self.requested: list[str] = []

    def get_history(self, ticker, period="1y", start=None, end=None):
        self.requested.append(ticker)
        if ticker in self.unknown:
            return pd.DataFrame()
        n = 400
        dates = pd.date_range("2022-01-03", periods=n, freq="B", tz="America/New_York")
        phase = sum(ord(c) for c in ticker)
        close = 100 + 0.05 * np.arange(n) + 10 * np.sin((np.arange(n) + phase) / 15)
        return pd.DataFrame(
            {"Open": close, "High": close, "Low": close, "Close": close, "Volume": 1e6},
            index=dates,
        )


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "portfolio.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def _args(config_path, tmp_path, *extra):
    return build_parser().parse_args(
        [
            "backtest",
            "--config", str(config_path),
            "--run-log", str(tmp_path / "runs.jsonl"),
            *extra,
        ]
    )


def test_backtest_report_compares_strategy_with_benchmarks(config_path, tmp_path):
    provider = FakeProvider()
    output = tmp_path / "report.md"
    args = _args(
        config_path, tmp_path, "--strategies", "ma_crossover", "--output", str(output)
    )

    report = run_backtest_command(args, provider)

    assert "# Backtest — ma_crossover" in report
    assert "| Metric | Strategy | Buy & hold (universe) | Buy & hold (SPY) |" in report
    assert "Universe: AAA, BBB" in report
    assert "400 sessions" in report
    assert "Configurations tried on this universe and period: 1" in report
    assert output.read_text(encoding="utf-8") == report
    assert provider.requested == ["AAA", "BBB", "SPY"]
    # The strategy must actually have traded for the comparison to mean anything
    trades_row = next(line for line in report.splitlines() if line.startswith("| Trades"))
    assert int(trades_row.split("|")[2]) > 0


def test_each_distinct_configuration_counts_as_one_trial(config_path, tmp_path):
    provider = FakeProvider()
    base = ["--strategies", "ma_crossover", "--benchmark", ""]

    run_backtest_command(_args(config_path, tmp_path, *base), provider)
    run_backtest_command(_args(config_path, tmp_path, *base), provider)  # same config again
    report = run_backtest_command(
        _args(config_path, tmp_path, *base, "--no-kill-switch"), provider
    )

    assert "Configurations tried on this universe and period: 2" in report
    assert len((tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()) == 3


def test_no_log_does_not_record_a_trial(config_path, tmp_path):
    args = _args(config_path, tmp_path, "--strategies", "ma_crossover", "--no-log")

    run_backtest_command(args, FakeProvider())

    assert not (tmp_path / "runs.jsonl").exists()


def test_saved_model_strategy_is_refused_as_in_sample(config_path, tmp_path):
    # The config's default strategies include ml_movement
    with pytest.raises(SystemExit, match="in-sample"):
        run_backtest_command(_args(config_path, tmp_path), FakeProvider())


def test_tickers_override_and_missing_data_are_reported(config_path, tmp_path):
    provider = FakeProvider(unknown={"NOPE"})
    args = _args(
        config_path, tmp_path,
        "--strategies", "ma_crossover", "--tickers", "ccc", "NOPE", "--benchmark", "",
    )

    report = run_backtest_command(args, provider)

    assert "Universe: CCC" in report
    assert "No data (left out): NOPE" in report
