from pathlib import Path

import pytest

from stock_analysis.trading.config import load_portfolio_config, strategy_names_for_position
from stock_analysis.trading.risk import RiskConfig


def test_load_portfolio_config():
    config_path = Path("config/portfolio.yaml")
    settings = load_portfolio_config(config_path)
    assert settings.portfolio.name == "paper-us"
    assert settings.portfolio.initial_cash == 100_000
    assert len(settings.positions) >= 3
    assert settings.defaults["strategies"] == ["ma_crossover"]
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


def _load(tmp_path, positions_yaml: str, defaults_yaml: str = ""):
    config_path = tmp_path / "portfolio.yaml"
    config_path.write_text(
        f"portfolio:\n  name: x\n{defaults_yaml}positions:\n{positions_yaml}",
        encoding="utf-8",
    )
    return load_portfolio_config(config_path)


def test_strategy_names_for_position_uses_position_strategies_list(tmp_path):
    settings = _load(tmp_path, "  - ticker: AAPL\n    strategies: [ma_crossover, ml_movement]\n")
    assert strategy_names_for_position(settings, settings.positions[0]) == [
        "ma_crossover",
        "ml_movement",
    ]


def test_strategy_names_for_position_falls_back_to_position_strategy(tmp_path):
    settings = _load(tmp_path, "  - ticker: AAPL\n    strategy: ml_movement\n")
    assert strategy_names_for_position(settings, settings.positions[0]) == ["ml_movement"]


def test_strategy_names_for_position_falls_back_to_defaults_strategies(tmp_path):
    settings = _load(
        tmp_path,
        "  - ticker: AAPL\n",
        defaults_yaml="defaults:\n  strategies: [ma_crossover, ml_movement]\n",
    )
    assert strategy_names_for_position(settings, settings.positions[0]) == [
        "ma_crossover",
        "ml_movement",
    ]


def test_strategy_names_for_position_falls_back_to_defaults_strategy(tmp_path):
    settings = _load(
        tmp_path, "  - ticker: AAPL\n", defaults_yaml="defaults:\n  strategy: ml_movement\n"
    )
    assert strategy_names_for_position(settings, settings.positions[0]) == ["ml_movement"]


def test_strategy_names_for_position_defaults_to_ma_crossover(tmp_path):
    settings = _load(tmp_path, "  - ticker: AAPL\n")
    assert strategy_names_for_position(settings, settings.positions[0]) == ["ma_crossover"]


def test_strategy_names_for_position_position_overrides_defaults(tmp_path):
    settings = _load(
        tmp_path,
        "  - ticker: AAPL\n    strategy: ma_crossover\n",
        defaults_yaml="defaults:\n  strategies: [ma_crossover, ml_movement]\n",
    )
    assert strategy_names_for_position(settings, settings.positions[0]) == ["ma_crossover"]


def test_rejects_bare_string_strategies_on_a_position(tmp_path):
    """`strategies: ml_movement` (missing brackets) must not silently become
    list("ml_movement") == ['m','l','_',...] — a real YAML typo risk."""
    with pytest.raises(ValueError, match="strategies"):
        _load(tmp_path, "  - ticker: AAPL\n    strategies: ml_movement\n")


def test_rejects_bare_string_strategies_in_defaults(tmp_path):
    with pytest.raises(ValueError, match="strategies"):
        _load(
            tmp_path,
            "  - ticker: AAPL\n",
            defaults_yaml="defaults:\n  strategies: ml_movement\n",
        )
