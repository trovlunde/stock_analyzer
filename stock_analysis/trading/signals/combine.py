from typing import Literal

from .base import SignalResult

SignalMode = Literal["consensus", "priority", "any"]


def combine_signals(
    results: dict[str, SignalResult], mode: SignalMode = "consensus"
) -> SignalResult:
    """Reconcile multiple named strategies' SignalResults into one.

    Pure function — no I/O. `results` is {strategy_name: SignalResult}, in the
    order the strategies were evaluated (insertion order matters for "priority").
    """
    if not results:
        raise ValueError("combine_signals requires at least one result")

    actionable = {name: r for name, r in results.items() if r.action != "HOLD"}
    if not actionable:
        return next(iter(results.values()))

    if mode == "any":
        return next(iter(actionable.values()))

    if mode == "priority":
        return next(iter(actionable.values()))

    # consensus (default): every actionable strategy must agree on direction.
    actions = {r.action for r in actionable.values()}
    if len(actions) > 1:
        names = ", ".join(f"{name}={r.action}" for name, r in actionable.items())
        return SignalResult(
            action="HOLD",
            reason=f"signals conflict: {names}",
            metadata={"conflict": True},
        )

    agreed_action = actions.pop()
    reasons = "; ".join(f"{name}: {r.reason}" for name, r in actionable.items())
    return SignalResult(
        action=agreed_action,
        reason=reasons,
        metadata={"agreed_strategies": list(actionable)},
    )
