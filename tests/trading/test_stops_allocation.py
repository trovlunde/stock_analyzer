"""Trailing stop levels, universe allocation and the shared run context."""

import numpy as np
import pandas as pd
import pytest

from stock_analysis.trading.allocation import universe_weights
from stock_analysis.trading.config import PortfolioConfig, PortfolioSettings, PositionEntry
from stock_analysis.trading.portfolio_rules import (
    build_run_context,
    decision_inputs,
    is_rebalance_session,
    needs_universe_closes,
)
from stock_analysis.trading.sizing import exposure_scale, position_weight
from stock_analysis.trading.stops import average_true_range, trailing_stop

DATES = pd.date_range("2023-01-02", periods=300, freq="B")


def _bars(closes, spread=1.0) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    return pd.DataFrame(
        {"Open": closes, "High": closes + spread, "Low": closes - spread, "Close": closes},
        index=DATES[: len(closes)],
    )


def _config(**kwargs) -> PortfolioConfig:
    values = dict(
        name="test", enabled=True, initial_cash=100_000, max_positions=10,
        position_size_pct=0.10, long_only=True, stop_loss_pct=None,
    )
    values.update(kwargs)
    return PortfolioConfig(**values)


def _settings(tickers=("AAA",), sectors=None, **portfolio) -> PortfolioSettings:
    return PortfolioSettings(
        portfolio=_config(**portfolio),
        costs={},
        defaults={},
        positions=[
            PositionEntry(ticker=t, sector=(sectors or {}).get(t)) for t in tickers
        ],
    )


# --- stops -------------------------------------------------------------------------


def test_average_true_range_includes_gaps():
    bars = _bars([100, 100, 110], spread=1.0)  # a 10-point gap on the last day

    # Day 2: range 2. Day 3: High 111 against the previous Close of 100 = 11.
    assert average_true_range(bars, 2) == pytest.approx(6.5)
    assert average_true_range(bars, 3) is None


def _stop(bars, **kwargs):
    values = dict(stop_type="atr_trailing", stop_loss_pct=None, atr_multiple=3.0,
                  atr_period=5, lookback=10)
    values.update(kwargs)
    return trailing_stop(bars, **values)


def test_atr_trailing_stop_sits_below_the_recent_high():
    bars = _bars(np.linspace(100, 119, 20), spread=1.0)  # +1 a day, range 2

    level, reason = _stop(bars)

    assert level == pytest.approx(119 - 3 * 2)
    assert "3 ATR" in reason


def test_trailing_stop_does_not_fall_with_the_price():
    rising = list(np.linspace(100, 119, 20))
    falling = rising + [117, 115, 113]

    assert _stop(_bars(falling))[0] > _stop(_bars(rising))[0] - 3  # still anchored at 119
    assert _stop(_bars(falling))[0] == pytest.approx(119 - 3 * _atr(falling))


def _atr(closes):
    return average_true_range(_bars(closes), 5)


def test_trailing_stop_forgets_highs_older_than_the_lookback():
    closes = [150] + [100] * 20

    assert _stop(_bars(closes))[0] < 100


def test_pct_trailing_stop():
    bars = _bars(np.linspace(100, 119, 20))

    level, _ = _stop(bars, stop_type="pct_trailing", stop_loss_pct=0.10)

    assert level == pytest.approx(119 * 0.9)


def test_no_trailing_level_for_fixed_stops_or_short_history():
    bars = _bars(np.linspace(100, 119, 20))

    assert _stop(bars, stop_type="fixed") is None
    assert _stop(bars.iloc[:5]) is None
    with pytest.raises(ValueError):
        _stop(bars, stop_type="nope")


def test_trailing_stop_uses_only_the_bars_it_is_given():
    closes = list(np.linspace(100, 119, 20)) + [300, 300]

    assert _stop(_bars(closes).iloc[:20]) == _stop(_bars(closes[:20]))


# --- allocation --------------------------------------------------------------------


def _returns(seed=0, n=252) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    market = rng.normal(0, 0.01, n)
    return pd.DataFrame(
        {
            "CALM": rng.normal(0, 0.003, n),
            "EQ1": market + rng.normal(0, 0.002, n),
            "EQ2": market + rng.normal(0, 0.002, n),
            "WILD": rng.normal(0, 0.03, n),
        }
    )


@pytest.mark.parametrize("method", ["min_variance", "hrp"])
def test_weights_are_long_only_sum_to_one_and_favour_the_calm_asset(method):
    weights = universe_weights(_returns(), method)

    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-4)
    assert all(weight >= 0 for weight in weights.values())
    assert weights["CALM"] > weights["EQ1"] > weights["WILD"]


def test_min_variance_respects_the_weight_cap():
    weights = universe_weights(_returns(), "min_variance", max_weight=0.4)

    assert max(weights.values()) <= 0.4 + 1e-6


def test_weights_need_enough_assets_and_history():
    returns = _returns()

    assert universe_weights(returns[["CALM"]], "hrp") is None
    assert universe_weights(returns.iloc[:3], "hrp") is None
    with pytest.raises(ValueError):
        universe_weights(returns, "max_sharpe")


def test_assets_with_gaps_are_left_out():
    returns = _returns()
    returns.loc[5, "WILD"] = np.nan

    assert set(universe_weights(returns, "hrp")) == {"CALM", "EQ1", "EQ2"}


# --- sizing ------------------------------------------------------------------------


def test_allocation_sizing_uses_the_universe_weight_capped():
    config = _config(sizing="hrp", max_position_pct=0.15)

    assert position_weight(config, None, 0.08) == 0.08
    assert position_weight(config, None, 0.40) == 0.15
    assert position_weight(config, None, None) == 0.10  # fixed size until weights exist


def test_exposure_scale_sizes_down_when_the_portfolio_is_too_volatile():
    rng = np.random.default_rng(0)
    equity = list(100_000 * np.exp(np.cumsum(rng.normal(0, 0.02, 61))))  # about 32% a year

    assert exposure_scale(equity, 0.10, 60) == pytest.approx(0.10 / 0.32, rel=0.2)
    assert exposure_scale(equity, 0.90, 60) == 1.0  # never sizes up
    assert exposure_scale(equity, None, 60) == 1.0
    assert exposure_scale(equity[:30], 0.10, 60) == 1.0  # too little history


# --- run context -------------------------------------------------------------------


def test_is_rebalance_session():
    friday, monday = pd.Timestamp("2024-05-31"), pd.Timestamp("2024-06-03")
    tuesday = pd.Timestamp("2024-06-04")

    assert is_rebalance_session("monthly", monday, friday)
    assert not is_rebalance_session("monthly", tuesday, monday)
    assert is_rebalance_session("weekly", monday, friday)
    assert not is_rebalance_session("weekly", tuesday, monday)
    assert is_rebalance_session("daily", tuesday, monday)
    assert not is_rebalance_session("monthly", monday, None)


def test_run_context_only_rebalances_when_a_band_is_set():
    without = build_run_context(_settings(), closes=None, equity=[1.0], rebalance_due=True)
    with_band = build_run_context(
        _settings(rebalance_band=0.2), closes=None, equity=[1.0], rebalance_due=True
    )

    assert not without.rebalance_due
    assert with_band.rebalance_due


def test_run_context_carries_weights_correlations_and_sectors():
    closes = (1 + _returns()).cumprod()
    settings = _settings(
        tuple(closes.columns), sectors={"EQ1": "equity", "EQ2": "equity"},
        sizing="hrp", max_correlation=0.8,
    )
    assert needs_universe_closes(settings.portfolio)

    context = build_run_context(settings, closes=closes, equity=[1.0])

    assert sum(context.universe_weights.values()) == pytest.approx(1.0, abs=1e-4)
    assert context.correlations.loc["EQ1", "EQ2"] > 0.9
    assert context.sectors == {"EQ1": "equity", "EQ2": "equity"}

    inputs = decision_inputs(
        settings, settings.positions[1], _bars(closes["EQ1"] * 100), context, sizing_needed=True
    )
    assert inputs["universe_weight"] == context.universe_weights["EQ1"]
    assert inputs["correlations"]["EQ2"] > 0.9
    assert "EQ1" not in inputs["correlations"]


def test_decision_inputs_swap_the_fixed_stop_for_the_trailing_one():
    bars = _bars(np.linspace(100, 199, 100))
    fixed = _settings(stop_loss_pct=0.08)
    trailing = _settings(stop_loss_pct=0.08, stop_type="pct_trailing", stop_lookback=20)
    context = build_run_context(fixed, closes=None, equity=[1.0])

    fixed_inputs = decision_inputs(fixed, fixed.positions[0], bars, context, sizing_needed=False)
    trailing_inputs = decision_inputs(
        trailing, trailing.positions[0], bars, context, sizing_needed=False
    )

    assert fixed_inputs["stop_loss_pct"] == 0.08 and fixed_inputs["stop_price"] is None
    assert trailing_inputs["stop_loss_pct"] is None
    assert trailing_inputs["stop_price"] == pytest.approx(199 * 0.92)


def test_volatility_is_only_estimated_when_an_order_may_be_sized():
    bars = _bars(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 100)))
    settings = _settings(sizing="vol_target")
    context = build_run_context(settings, closes=None, equity=[1.0])

    needed = decision_inputs(settings, settings.positions[0], bars, context, sizing_needed=True)
    skipped = decision_inputs(settings, settings.positions[0], bars, context, sizing_needed=False)

    assert needed["volatility"] > 0
    assert skipped["volatility"] is None
