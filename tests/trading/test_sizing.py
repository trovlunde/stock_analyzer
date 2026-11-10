import math

import numpy as np
import pandas as pd
import pytest

from stock_analysis.storage.portfolio_store import PortfolioState
from stock_analysis.trading.config import PortfolioConfig, load_portfolio_config
from stock_analysis.trading.decision import decide
from stock_analysis.trading.regime import RegimeFilter, risk_off_reason
from stock_analysis.trading.signals.base import SignalResult
from stock_analysis.trading.sizing import annualised_volatility, position_weight


def _config(**kwargs) -> PortfolioConfig:
    values = dict(
        name="test",
        enabled=True,
        initial_cash=100_000,
        max_positions=10,
        position_size_pct=0.10,
        long_only=True,
        stop_loss_pct=None,
    )
    values.update(kwargs)
    return PortfolioConfig(**values)


def _buy(config, volatility=None):
    return decide(
        ticker="AAA",
        signal=SignalResult(action="BUY", reason="test"),
        state=PortfolioState(id=1, name="test", cash=100_000, positions=[]),
        config=config,
        current_price=100.0,
        prices={"AAA": 100.0},
        stop_loss_pct=None,
        volatility=volatility,
    )


def test_annualised_volatility_matches_definition():
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))))
    expected = close.pct_change().dropna().iloc[-60:].std(ddof=1) * math.sqrt(252)

    assert annualised_volatility(close, 60) == pytest.approx(expected)


def test_annualised_volatility_is_unknown_without_enough_history_or_movement():
    assert annualised_volatility(pd.Series([100.0, 101.0, 102.0]), 60) is None
    assert annualised_volatility(pd.Series([100.0] * 100), 60) is None


def test_fixed_sizing_ignores_volatility():
    assert position_weight(_config(), 0.50) == 0.10
    assert _buy(_config(), volatility=0.50).intent.shares == 100


def test_vol_target_gives_calmer_assets_larger_positions():
    config = _config(sizing="vol_target", risk_per_position=0.02, max_position_pct=0.20)

    assert position_weight(config, 0.40) == pytest.approx(0.05)
    assert position_weight(config, 0.20) == pytest.approx(0.10)
    assert _buy(config, volatility=0.40).intent.shares == 50
    assert _buy(config, volatility=0.20).intent.shares == 100


def test_vol_target_is_capped_for_very_calm_assets():
    config = _config(sizing="vol_target", risk_per_position=0.02, max_position_pct=0.20)

    assert position_weight(config, 0.05) == 0.20
    assert _buy(config, volatility=0.05).intent.shares == 200


def test_vol_target_falls_back_to_fixed_size_when_volatility_is_unknown():
    config = _config(sizing="vol_target")

    assert _buy(config, volatility=None).intent.shares == 100


def _write(tmp_path, body: str):
    path = tmp_path / "portfolio.yaml"
    path.write_text(body + "positions:\n  - ticker: AAA\n", encoding="utf-8")
    return path


def test_config_defaults_to_fixed_sizing_and_no_regime_filter(tmp_path):
    settings = load_portfolio_config(_write(tmp_path, "portfolio:\n  name: x\n"))

    assert settings.portfolio.sizing == "fixed"
    assert settings.regime_filter is None


def test_config_reads_sizing_and_regime_filter(tmp_path):
    settings = load_portfolio_config(
        _write(
            tmp_path,
            "portfolio:\n  name: x\n  sizing: vol_target\n  risk_per_position: 0.01\n"
            "  max_position_pct: 0.15\n  vol_lookback: 40\n"
            "regime_filter:\n  ticker: spy\n",
        )
    )

    assert settings.portfolio.sizing == "vol_target"
    assert settings.portfolio.risk_per_position == 0.01
    assert settings.portfolio.max_position_pct == 0.15
    assert settings.portfolio.vol_lookback == 40
    assert settings.regime_filter == RegimeFilter(ticker="SPY", ma=200)


def test_config_rejects_unknown_sizing_and_regime_filter_without_ticker(tmp_path):
    with pytest.raises(ValueError, match="portfolio.sizing"):
        load_portfolio_config(_write(tmp_path, "portfolio:\n  sizing: kelly\n"))
    with pytest.raises(ValueError, match="regime_filter needs a ticker"):
        load_portfolio_config(_write(tmp_path, "regime_filter:\n  ma: 200\n"))


def test_shipped_etf_trend_config_loads():
    settings = load_portfolio_config("config/portfolio-etf-trend.yaml")

    assert settings.portfolio.name == "paper-etf-trend"
    assert settings.portfolio.sizing == "vol_target"
    assert settings.regime_filter == RegimeFilter(ticker="SPY", ma=200)
    assert settings.defaults["strategies"] == ["trend_state"]
    assert len(settings.positions) == 20


def test_regime_is_risk_off_only_below_the_moving_average():
    regime = RegimeFilter(ticker="SPY", ma=5)

    assert risk_off_reason(regime, pd.Series([100, 100, 100, 100, 100, 101.0])) is None
    reason = risk_off_reason(regime, pd.Series([100, 100, 100, 100, 100, 90.0]))
    assert "SPY is below its 5-day average" in reason


def test_regime_blocks_nothing_without_enough_history():
    regime = RegimeFilter(ticker="SPY", ma=200)

    assert risk_off_reason(regime, pd.Series([100.0, 50.0])) is None
