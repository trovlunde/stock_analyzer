import json

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


def _logged_records(tmp_path):
    lines = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_overfitting_probability_appears_once_two_configurations_are_logged(
    config_path, tmp_path
):
    provider = FakeProvider()
    base = ["--strategies", "ma_crossover", "--benchmark", ""]

    first = run_backtest_command(_args(config_path, tmp_path, *base), provider)
    second = run_backtest_command(
        _args(config_path, tmp_path, *base, "--signal-mode", "any"), provider
    )

    assert "Probability of backtest overfitting: needs at least two" in first
    assert "Probability of backtest overfitting (2 configurations" in second
    assert [len(record["returns"]) for record in _logged_records(tmp_path)] == [399, 399]


def test_logged_runs_without_daily_returns_are_left_out_of_the_overfitting_check(
    config_path, tmp_path
):
    provider = FakeProvider()
    base = ["--strategies", "ma_crossover", "--benchmark", ""]
    run_backtest_command(_args(config_path, tmp_path, *base), provider)
    # Rewrite the log as a line from before daily returns were recorded
    (record,) = _logged_records(tmp_path)
    record.pop("returns")
    (tmp_path / "runs.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    report = run_backtest_command(
        _args(config_path, tmp_path, *base, "--signal-mode", "any", "--no-log"), provider
    )

    assert "Configurations tried on this universe and period: 2" in report
    assert "Probability of backtest overfitting: needs at least two" in report


def test_legacy_strategies_command_is_gone(capsys):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["strategies"])

    assert "invalid choice: 'strategies'" in capsys.readouterr().err


@pytest.mark.parametrize("name", ["ml_walk_forward", "ensemble_walk_forward"])
def test_walk_forward_strategies_resolve_by_name(config_path, tmp_path, monkeypatch, name):
    import stock_analysis.backtest.walk_forward as walk_forward_mod
    from stock_analysis.trading.signals import SignalResult

    class Recorder:
        def __init__(self):
            self.calls = 0

        def evaluate(self, df, ticker, params):
            self.calls += 1
            return SignalResult(action="HOLD", reason="stub")

    real_names = set(walk_forward_mod.walk_forward_strategies())
    recorders = {key: Recorder() for key in real_names}
    retrain_every = []

    def fake_factory(every):
        retrain_every.append(every)
        return recorders

    monkeypatch.setattr(walk_forward_mod, "walk_forward_strategies", fake_factory)
    args = _args(
        config_path, tmp_path,
        "--strategies", name, "--retrain-every", "21", "--benchmark", "", "--no-log",
    )

    report = run_backtest_command(args, FakeProvider())

    assert name in real_names
    assert f"# Backtest — {name}" in report
    assert retrain_every == [21]
    assert recorders[name].calls > 0
    assert all(rec.calls == 0 for key, rec in recorders.items() if key != name)


def test_walk_forward_strategies_are_not_in_the_production_registry():
    from stock_analysis.trading.signals import get_signal_strategy

    with pytest.raises(ValueError, match="Unknown strategy"):
        get_signal_strategy("ensemble_walk_forward")


def test_set_overrides_strategy_params_and_portfolio_fields(config_path, tmp_path):
    base = ["--strategies", "ma_crossover", "--benchmark", ""]

    run_backtest_command(_args(config_path, tmp_path, *base), FakeProvider())
    report = run_backtest_command(
        _args(config_path, tmp_path, *base, "--set", "ma_short=10",
              "--set", "portfolio.position_size_pct=0.2"),
        FakeProvider(),
    )

    # A different configuration, so a second trial
    assert "Configurations tried on this universe and period: 2" in report
    logged = [
        json.loads(line)
        for line in (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    key = json.loads(logged[-1]["config_key"])
    assert key["defaults"]["ma_short"] == 10
    assert key["portfolio"]["position_size_pct"] == 0.2


def test_set_rejects_unknown_portfolio_fields_and_bad_syntax(config_path, tmp_path):
    with pytest.raises(SystemExit, match="Unknown portfolio field"):
        run_backtest_command(
            _args(config_path, tmp_path, "--strategies", "ma_crossover",
                  "--set", "portfolio.nope=1"),
            FakeProvider(),
        )
    with pytest.raises(SystemExit, match="KEY=VALUE"):
        run_backtest_command(
            _args(config_path, tmp_path, "--strategies", "ma_crossover", "--set", "ma_short"),
            FakeProvider(),
        )


def test_run_log_key_leaves_out_newer_portfolio_fields_at_their_default(config_path, tmp_path):
    """A configuration logged before a portfolio field existed must still count as
    the same trial when it is run again."""
    run_backtest_command(
        _args(config_path, tmp_path, "--strategies", "ma_crossover", "--benchmark", ""),
        FakeProvider(),
    )

    line = (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()[0]
    portfolio = json.loads(json.loads(line)["config_key"])["portfolio"]
    assert "stop_type" not in portfolio and "cash_buffer_pct" not in portfolio
    assert portfolio["sizing"] == "fixed"  # original fields are always written


def test_report_has_returns_per_calendar_year(config_path, tmp_path):
    report = run_backtest_command(
        _args(config_path, tmp_path, "--strategies", "ma_crossover", "--no-log"),
        FakeProvider(),
    )

    assert "## Return per calendar year" in report
    assert "| Year | Strategy | Buy & hold (universe) | Buy & hold (SPY) |" in report
    assert "| 2022 |" in report and "| 2023 |" in report
