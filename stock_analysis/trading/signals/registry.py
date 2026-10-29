from .base import SignalStrategy
from .ma_crossover import MaCrossoverStrategy
from .ml_movement import MlMovementStrategy

_STRATEGIES: dict[str, SignalStrategy] = {
    "ma_crossover": MaCrossoverStrategy(),
    "ml_movement": MlMovementStrategy(),
}


def get_signal_strategy(name: str) -> SignalStrategy:
    try:
        return _STRATEGIES[name]
    except KeyError as exc:
        known = ", ".join(sorted(_STRATEGIES))
        raise ValueError(f"Unknown strategy '{name}'. Known: {known}") from exc


def register_signal_strategy(name: str, strategy: SignalStrategy) -> None:
    _STRATEGIES[name] = strategy


def unregister_signal_strategy(name: str) -> None:
    _STRATEGIES.pop(name, None)
