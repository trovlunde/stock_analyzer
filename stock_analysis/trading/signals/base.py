from dataclasses import dataclass, field
from typing import Literal, Protocol

import pandas as pd

SignalAction = Literal["BUY", "SELL", "HOLD"]


@dataclass(frozen=True)
class SignalResult:
    action: SignalAction
    reason: str
    metadata: dict = field(default_factory=dict)


class SignalStrategy(Protocol):
    def evaluate(self, df: pd.DataFrame, params: dict) -> SignalResult: ...
