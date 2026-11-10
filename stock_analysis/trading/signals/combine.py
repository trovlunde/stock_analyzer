from typing import Literal

from .base import SignalResult

SignalMode = Literal["consensus", "priority", "any", "quorum", "score"]
SIGNAL_MODES: tuple[str, ...] = ("consensus", "priority", "any", "quorum", "score")

_ACTION_SCORE = {"BUY": 1.0, "HOLD": 0.0, "SELL": -1.0}


def signal_score(result: SignalResult) -> float:
    """A strategy's view as a number from -1 (sell) to +1 (buy).

    Strategies that have a graded view put it in `metadata["score"]`; for the
    rest the action itself is the score.
    """
    score = result.metadata.get("score") if result.metadata else None
    if score is None:
        return _ACTION_SCORE[result.action]
    return max(-1.0, min(1.0, float(score)))


def _quorum(results: dict[str, SignalResult], params: dict) -> SignalResult:
    needed_buy = int(params.get("signal_quorum", len(results)))
    needed_sell = int(params.get("signal_exit_quorum", needed_buy))
    buyers = [name for name, r in results.items() if r.action == "BUY"]
    sellers = [name for name, r in results.items() if r.action == "SELL"]
    metadata = {"buy": buyers, "sell": sellers, "strategies": len(results)}

    if buyers and sellers:
        return SignalResult(
            action="HOLD",
            reason=f"signals conflict: BUY from {', '.join(buyers)}; SELL from {', '.join(sellers)}",
            metadata={**metadata, "conflict": True},
        )
    for action, names, needed in (("BUY", buyers, needed_buy), ("SELL", sellers, needed_sell)):
        if names and len(names) >= needed:
            reasons = "; ".join(f"{name}: {results[name].reason}" for name in names)
            return SignalResult(
                action=action,
                reason=f"{len(names)} of {len(results)} agree — {reasons}",
                metadata={**metadata, "agreed_strategies": names},
            )
    if buyers or sellers:
        action, names, needed = (
            ("BUY", buyers, needed_buy) if buyers else ("SELL", sellers, needed_sell)
        )
        return SignalResult(
            action="HOLD",
            reason=f"only {len(names)} of {len(results)} say {action} (need {needed})",
            metadata=metadata,
        )
    return next(iter(results.values()))


def _score(results: dict[str, SignalResult], params: dict) -> SignalResult:
    weights = params.get("signal_weights") or {}
    buy_at = float(params.get("signal_score_buy", 0.5))
    sell_at = float(params.get("signal_score_sell", -0.5))
    scores = {name: signal_score(result) for name, result in results.items()}
    total_weight = sum(float(weights.get(name, 1.0)) for name in results)
    if total_weight <= 0:
        raise ValueError("signal_weights must sum to more than zero")
    blended = sum(float(weights.get(name, 1.0)) * scores[name] for name in results) / total_weight
    metadata: dict = {"score": blended, "scores": scores}
    detail = ", ".join(f"{name} {score:+.2f}" for name, score in scores.items())

    if blended >= buy_at:
        if params.get("signal_score_sizing", True):
            # A stronger blended view takes a larger share of the normal position size
            metadata["size_factor"] = min(1.0, blended)
        return SignalResult(
            action="BUY", reason=f"blended score {blended:+.2f} ({detail})", metadata=metadata
        )
    if blended <= sell_at:
        return SignalResult(
            action="SELL", reason=f"blended score {blended:+.2f} ({detail})", metadata=metadata
        )
    return SignalResult(
        action="HOLD",
        reason=f"blended score {blended:+.2f} between {sell_at:+.2f} and {buy_at:+.2f} ({detail})",
        metadata=metadata,
    )


def combine_signals(
    results: dict[str, SignalResult],
    mode: SignalMode = "consensus",
    params: dict | None = None,
) -> SignalResult:
    """Reconcile multiple named strategies' SignalResults into one.

    Pure function — no I/O. `results` is {strategy_name: SignalResult}, in the
    order the strategies were evaluated (insertion order matters for "priority").

    - `consensus`: HOLDs are ignored; the strategies that do say BUY or SELL must
      all say the same. One BUY among HOLDs is therefore enough to act.
    - `priority` / `any`: the first strategy that says BUY or SELL wins.
    - `quorum`: at least `signal_quorum` strategies must say BUY (default: all of
      them) and none SELL; likewise `signal_exit_quorum` for SELL (default: same).
    - `score`: each strategy's view is a number from -1 to +1 (`signal_score`);
      the weighted average (`signal_weights`, default equal) is a BUY at or above
      `signal_score_buy` (0.5) and a SELL at or below `signal_score_sell` (-0.5).
      A BUY carries `size_factor` so a weaker blend takes a smaller position.

    `params` is the strategy parameter dict (config `defaults`) holding those keys.
    """
    if not results:
        raise ValueError("combine_signals requires at least one result")
    if mode not in SIGNAL_MODES:
        raise ValueError(f"Unknown signal_mode {mode!r}; known: {list(SIGNAL_MODES)}")
    params = params or {}

    if mode == "quorum":
        return _quorum(results, params)
    if mode == "score":
        return _score(results, params)

    actionable = {name: r for name, r in results.items() if r.action != "HOLD"}
    if not actionable:
        return next(iter(results.values()))

    if mode in ("any", "priority"):
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
