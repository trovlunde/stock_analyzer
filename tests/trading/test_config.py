from pathlib import Path

from stock_analysis.trading.config import load_portfolio_config


def test_load_portfolio_config():
    config_path = Path("config/portfolio.yaml")
    settings = load_portfolio_config(config_path)
    assert settings.portfolio.name == "paper-us"
    assert settings.portfolio.initial_cash == 100_000
    assert len(settings.positions) >= 3
    assert settings.defaults["strategy"] == "ma_crossover"
