from stock_analysis.trading.cost_models import EquityCostModel, MiniFutureCostModel


def test_equity_entry_cost():
    model = EquityCostModel(commission_per_trade=1.0, slippage_pct=0.001)
    cost = model.entry_cost(10_000)
    assert cost.commission == 1.0
    assert cost.slippage == 10.0
    assert cost.total == 11.0
    assert model.daily_holding_cost(10_000) == 0.0


def test_mini_future_has_spread_and_holding_cost():
    model = MiniFutureCostModel(leverage=5, spread_pct=0.5)
    cost = model.entry_cost(1_000)
    assert cost.slippage == 5.0
    assert model.daily_holding_cost(1_000) > 0
