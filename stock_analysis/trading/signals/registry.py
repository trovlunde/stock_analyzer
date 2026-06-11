from .base import SignalStrategy
from .ma_crossover import MaCrossoverStrategy

_STRATEGIES: dict[str, SignalStrategy] = {
    "ma_crossover": MaCrossoverStrategy(),
}


def get_signal_strategy(name: str) -> SignalStrategy:
    try:
        return _STRATEGIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_STRATEGIES))
        raise ValueError(f"Unknown strategy '{name}'. Known: {known}") from exc


def register_signal_strategy(name: str, strategy: SignalStrategy) -> None:
    _STRATEGIES[name] = strategy
