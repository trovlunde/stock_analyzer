from dataclasses import dataclass
from typing import Protocol

from stock_analysis.trading_cost_analysis import MiniFutureAnalyzer


@dataclass(frozen=True)
class TradeCost:
    commission: float
    slippage: float

    @property
    def total(self) -> float:
        return self.commission + self.slippage


class CostModel(Protocol):
    def entry_cost(self, notional: float) -> TradeCost: ...

    def exit_cost(self, notional: float) -> TradeCost: ...

    def daily_holding_cost(self, position_value: float) -> float: ...


@dataclass
class EquityCostModel:
    """US cash equity: commission + slippage per side, no financing."""

    commission_per_trade: float = 1.0
    slippage_pct: float = 0.0005

    def entry_cost(self, notional: float) -> TradeCost:
        return self._side_cost(notional)

    def exit_cost(self, notional: float) -> TradeCost:
        return self._side_cost(notional)

    def daily_holding_cost(self, position_value: float) -> float:
        return 0.0

    def _side_cost(self, notional: float) -> TradeCost:
        slippage = abs(notional) * self.slippage_pct
        return TradeCost(commission=self.commission_per_trade, slippage=slippage)


@dataclass
class MiniFutureCostModel:
    """Norwegian mini futures / leveraged certs: spread + daily financing."""

    initial_investment: float = 100.0
    leverage: int = 15
    spread_pct: float = 0.5
    variable_rate: float = 0.04
    fixed_rate: float = 0.03

    def __post_init__(self) -> None:
        self._analyzer = MiniFutureAnalyzer(
            initial_investment=self.initial_investment,
            leverage=self.leverage,
            spread_pct=self.spread_pct,
            variable_rate=self.variable_rate,
            fixed_rate=self.fixed_rate,
        )

    def entry_cost(self, notional: float) -> TradeCost:
        spread = abs(notional) * self._analyzer.spread_pct
        return TradeCost(commission=0.0, slippage=spread)

    def exit_cost(self, notional: float) -> TradeCost:
        spread = abs(notional) * self._analyzer.spread_pct
        return TradeCost(commission=0.0, slippage=spread)

    def daily_holding_cost(self, position_value: float) -> float:
        return self._analyzer.simple_calculate_holding_cost(position_value)


def build_cost_model(config: dict) -> CostModel:
    model = config.get("model", "equity")
    if model == "mini_future":
        return MiniFutureCostModel(
            leverage=config.get("leverage", 15),
            spread_pct=config.get("spread_pct", 0.5),
            variable_rate=config.get("variable_rate", 0.04),
            fixed_rate=config.get("fixed_rate", 0.03),
        )
    return EquityCostModel(
        commission_per_trade=config.get("commission_per_trade", 1.0),
        slippage_pct=config.get("slippage_pct", 0.0005),
    )
