from pathlib import Path

import pytest

from stock_analysis.trading.config import load_portfolio_config
from stock_analysis.trading.risk import RiskConfig


def test_load_portfolio_config():
    config_path = Path("config/portfolio.yaml")
    settings = load_portfolio_config(config_path)
    assert settings.portfolio.name == "paper-us"
    assert settings.portfolio.initial_cash == 100_000
    assert len(settings.positions) >= 3
    assert settings.defaults["strategy"] == "ma_crossover"
    assert settings.risk.max_daily_loss_pct == 0.05
    assert settings.risk.max_drawdown_pct == 0.20
    assert settings.risk.liquidate_on_trip is False


def test_load_portfolio_config_defaults_risk_when_section_missing(tmp_path):
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        "portfolio:\n  name: x\npositions:\n  - ticker: AAPL\n", encoding="utf-8"
    )

    settings = load_portfolio_config(config_path)

    assert settings.risk == RiskConfig()


def test_load_portfolio_config_rejects_unimplemented_liquidate_on_trip(tmp_path):
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        "portfolio:\n  name: x\npositions:\n  - ticker: AAPL\n"
        "risk:\n  liquidate_on_trip: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="liquidate_on_trip"):
        load_portfolio_config(config_path)
