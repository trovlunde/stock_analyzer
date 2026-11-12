"""End to end: `backtest --universe sp500 --strategies ml_pooled` and `retrain-pooled`."""

import numpy as np
import pandas as pd
import pytest

from stock_analysis.ai.cross_sectional.champion import MODEL_TYPE
from stock_analysis.ai.model_manager import ModelManager
from stock_analysis.cli.backtest import run_backtest_command
from stock_analysis.cli.evaluate_models import build_parser
from stock_analysis.cli.retrain import run_retrain_command
from stock_analysis.cli.universe import load_price_data_cached, price_cache_key
from stock_analysis.market_data.index_membership import build_membership

SESSIONS = pd.bdate_range("2016-01-04", periods=650)
TICKERS = [f"T{i:02d}" for i in range(16)]
REMOVED = "OLD"  # an index member until 2017-06-01, then replaced by T15
REMOVED_ON = "2017-06-01"

CONFIG = """
portfolio:
  name: pooled-test
  initial_cash: 100000
  max_positions: 4
  position_size_pct: 0.20
costs:
  model: equity
  commission_per_trade: 1.00
  slippage_pct: 0.0005
defaults:
  strategies: [ml_pooled]
positions:
  - ticker: AAA
"""

POOLED_SETTINGS = [
    "--set", "pooled_horizon=5",
    "--set", "pooled_top_k=3",
    "--set", "pooled_train_years=1.5",
    "--set", "pooled_validation_sessions=60",
    "--set", "pooled_min_train_sessions=120",
    "--set", "pooled_retrain_every=60",
    "--set", "pooled_min_picks=10",
]


class UniverseProvider:
    """Random-walk prices for every ticker; OLD stops trading when it leaves the index."""

    def __init__(self):
        self.requested: list[str] = []

    def get_history(self, ticker, period="1y", start=None, end=None):
        self.requested.append(ticker)
        if ticker == "NODATA":
            return pd.DataFrame()
        seed = sum(ord(c) for c in ticker)
        rng = np.random.default_rng(seed)
        close = 50 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, len(SESSIONS))))
        frame = pd.DataFrame(
            {
                "Open": close,
                "High": close,
                "Low": close,
                "Close": close,
                "Volume": rng.integers(1_000_000, 2_000_000, len(SESSIONS)).astype(float),
            },
            index=SESSIONS.tz_localize("America/New_York"),
        )
        if ticker == REMOVED:
            frame = frame[frame.index < pd.Timestamp(REMOVED_ON, tz="America/New_York")]
        return frame


def _membership():
    constituents = pd.DataFrame(
        {
            "ticker": TICKERS,
            "sector": ["Tech" if i % 2 else "Energy" for i in range(len(TICKERS))],
            "date_added": ["1990-01-02"] * (len(TICKERS) - 1) + [REMOVED_ON],
        }
    )
    changes = pd.DataFrame({"date": [REMOVED_ON], "added": ["T15"], "removed": [REMOVED]})
    return build_membership(constituents, changes)


class MemoryCache:
    def __init__(self):
        self.store: dict = {}

    def get(self, key, max_age=None):
        return self.store.get(key)

    def put(self, key, df, metadata=None):
        self.store[key] = df


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "portfolio.yaml"
    path.write_text(CONFIG, encoding="utf-8")
    return path


def test_pooled_backtest_on_historical_index_members(config_path, tmp_path):
    args = build_parser().parse_args(
        [
            "backtest",
            "--config", str(config_path),
            "--universe", "sp500",
            "--start", "2017-03-01",
            "--benchmark", "",
            "--no-log",
            *POOLED_SETTINGS,
        ]
    )

    report = run_backtest_command(args, UniverseProvider(), membership=_membership())

    assert "# Backtest — ml_pooled" in report
    assert "## Pooled model (ml_pooled)" in report
    assert "### Rank IC per year (walk-forward)" in report
    trades = next(line for line in report.splitlines() if line.startswith("| Trades"))
    assert int(trades.split("|")[2]) > 0


def test_pooled_backtest_never_buys_a_ticker_after_it_left_the_index(config_path, tmp_path, monkeypatch):
    import stock_analysis.backtest.engine as engine_mod

    captured = {}
    real_run = engine_mod.run_backtest

    def recording_run(*args, **kwargs):
        result = real_run(*args, **kwargs)
        captured["result"] = result
        return result

    monkeypatch.setattr("stock_analysis.cli.backtest.run_backtest", recording_run)
    args = build_parser().parse_args(
        [
            "backtest", "--config", str(config_path), "--universe", "sp500",
            "--start", "2017-03-01", "--benchmark", "", "--no-log", *POOLED_SETTINGS,
        ]
    )

    run_backtest_command(args, UniverseProvider(), membership=_membership())

    for trade in captured["result"].trades:
        if trade.side == "BUY" and trade.ticker == REMOVED:
            assert pd.Timestamp(trade.date) < pd.Timestamp(REMOVED_ON)
        if trade.side == "BUY" and trade.ticker == "T15":
            assert pd.Timestamp(trade.date) >= pd.Timestamp(REMOVED_ON)


def test_universe_and_tickers_cannot_be_combined(config_path):
    args = build_parser().parse_args(
        ["backtest", "--config", str(config_path), "--universe", "sp500", "--tickers", "AAA"]
    )
    with pytest.raises(SystemExit, match="either --tickers or --universe"):
        run_backtest_command(args, UniverseProvider(), membership=_membership())


def _retrain_args(tmp_path, *extra):
    return build_parser().parse_args(
        [
            "retrain-pooled",
            "--tickers", *TICKERS,
            "--horizon", "5",
            "--train-years", "1.5",
            "--validation-sessions", "60",
            "--top-k", "3",
            "--models-dir", str(tmp_path / "models"),
            "--model-name", "pooled-test",
            *extra,
        ]
    )


def test_retrain_saves_the_first_model_then_compares_with_it(tmp_path):
    manager = ModelManager(models_dir=str(tmp_path / "models"))

    first = run_retrain_command(_retrain_args(tmp_path), UniverseProvider(), manager=manager)
    assert "- Result: challenger saved as the new champion — no champion saved yet" in first
    assert (tmp_path / "models" / f"pooled-test_{MODEL_TYPE}.joblib").exists()

    output = tmp_path / "report.md"
    second = run_retrain_command(
        _retrain_args(tmp_path, "--output", str(output)), UniverseProvider(), manager=manager
    )
    assert "- Champion: rank IC" in second
    assert "had already trained or calibrated on" in second
    assert output.read_text(encoding="utf-8") == second


def test_retrain_with_no_data_stops():
    args = build_parser().parse_args(["retrain-pooled", "--tickers", "NODATA"])
    with pytest.raises(SystemExit, match="No price data"):
        run_retrain_command(args, UniverseProvider())


def test_price_cache_serves_the_second_load_and_remembers_missing_tickers():
    provider = UniverseProvider()
    cache = MemoryCache()

    first = load_price_data_cached(["T00", "NODATA"], provider, "5y", cache, progress=False)
    second = load_price_data_cached(["T00", "NODATA"], provider, "5y", cache, progress=False)

    assert list(first) == ["T00"] and list(second) == ["T00"]
    assert provider.requested == ["T00", "NODATA"]  # nothing downloaded the second time
    assert cache.store[price_cache_key("NODATA", "5y")].empty
    assert second["T00"].index.tz is None
