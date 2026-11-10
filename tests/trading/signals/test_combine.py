import pytest

from stock_analysis.trading.signals.base import SignalResult
from stock_analysis.trading.signals.combine import combine_signals


def _signal(action, reason="r"):
    return SignalResult(action=action, reason=reason, metadata={})


def test_raises_on_empty_results():
    with pytest.raises(ValueError):
        combine_signals({})


def test_all_hold_returns_hold():
    results = {"ma": _signal("HOLD", "no crossover"), "ml": _signal("HOLD", "low confidence")}

    result = combine_signals(results)

    assert result.action == "HOLD"


def test_single_actionable_signal_passes_through_in_consensus_mode():
    results = {"ma": _signal("BUY", "golden cross"), "ml": _signal("HOLD", "low confidence")}

    result = combine_signals(results, mode="consensus")

    assert result.action == "BUY"
    assert result.metadata["agreed_strategies"] == ["ma"]


def test_consensus_agreement_combines_reasons():
    results = {"ma": _signal("BUY", "golden cross"), "ml": _signal("BUY", "ML predicts positive")}

    result = combine_signals(results, mode="consensus")

    assert result.action == "BUY"
    assert "golden cross" in result.reason
    assert "ML predicts positive" in result.reason
    assert set(result.metadata["agreed_strategies"]) == {"ma", "ml"}


def test_consensus_conflict_results_in_hold():
    results = {"ma": _signal("BUY", "golden cross"), "ml": _signal("SELL", "ML predicts negative")}

    result = combine_signals(results, mode="consensus")

    assert result.action == "HOLD"
    assert result.metadata["conflict"] is True
    assert "ma=BUY" in result.reason
    assert "ml=SELL" in result.reason


def test_any_mode_returns_first_actionable():
    results = {"ma": _signal("HOLD"), "ml": _signal("SELL", "ML predicts negative")}

    result = combine_signals(results, mode="any")

    assert result.action == "SELL"


def test_priority_mode_returns_first_actionable_in_insertion_order():
    results = {"ma": _signal("BUY", "golden cross"), "ml": _signal("SELL", "ML predicts negative")}

    result = combine_signals(results, mode="priority")

    assert result.action == "BUY"


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="signal_mode"):
        combine_signals({"ma": _signal("BUY")}, mode="majority")


def _three(a, b, c):
    return {"ma": _signal(a, "ma reason"), "ml": _signal(b, "ml reason"), "trend": _signal(c)}


def test_quorum_needs_every_strategy_by_default():
    assert combine_signals(_three("BUY", "BUY", "HOLD"), mode="quorum").action == "HOLD"
    assert combine_signals(_three("BUY", "BUY", "BUY"), mode="quorum").action == "BUY"


def test_quorum_n_of_m():
    params = {"signal_quorum": 2}

    agreed = combine_signals(_three("BUY", "BUY", "HOLD"), mode="quorum", params=params)
    alone = combine_signals(_three("BUY", "HOLD", "HOLD"), mode="quorum", params=params)

    assert agreed.action == "BUY"
    assert agreed.metadata["agreed_strategies"] == ["ma", "ml"]
    assert "ma reason" in agreed.reason
    assert alone.action == "HOLD"
    assert "only 1 of 3" in alone.reason


def test_quorum_holds_on_any_disagreement():
    result = combine_signals(
        _three("BUY", "BUY", "SELL"), mode="quorum", params={"signal_quorum": 2}
    )

    assert result.action == "HOLD"
    assert result.metadata["conflict"] is True


def test_quorum_can_exit_on_fewer_votes_than_it_enters_on():
    params = {"signal_quorum": 3, "signal_exit_quorum": 1}

    assert combine_signals(_three("SELL", "HOLD", "HOLD"), mode="quorum", params=params).action == "SELL"
    assert combine_signals(_three("BUY", "HOLD", "HOLD"), mode="quorum", params=params).action == "HOLD"


def test_score_mode_blends_actions():
    buy = combine_signals(_three("BUY", "BUY", "HOLD"), mode="score")
    hold = combine_signals(_three("BUY", "HOLD", "HOLD"), mode="score")
    sell = combine_signals(_three("SELL", "SELL", "HOLD"), mode="score")

    assert buy.action == "BUY"
    assert buy.metadata["score"] == pytest.approx(2 / 3)
    assert buy.metadata["size_factor"] == pytest.approx(2 / 3)
    assert hold.action == "HOLD"
    assert sell.action == "SELL"


def test_score_mode_uses_graded_scores_weights_and_thresholds():
    results = {
        "trend": SignalResult("HOLD", "r", {"score": 0.9}),
        "ml": SignalResult("BUY", "r", {"score": 0.3}),
    }

    equal = combine_signals(results, mode="score")
    weighted = combine_signals(
        results, mode="score", params={"signal_weights": {"trend": 1, "ml": 3}}
    )
    low_bar = combine_signals(
        results, mode="score",
        params={"signal_weights": {"trend": 1, "ml": 3}, "signal_score_buy": 0.4,
                "signal_score_sizing": False},
    )

    assert equal.action == "BUY" and equal.metadata["score"] == pytest.approx(0.6)
    assert weighted.action == "HOLD" and weighted.metadata["score"] == pytest.approx(0.45)
    assert low_bar.action == "BUY" and "size_factor" not in low_bar.metadata


def test_default_mode_is_consensus():
    results = {"ma": _signal("BUY", "golden cross"), "ml": _signal("SELL", "ML predicts negative")}

    result = combine_signals(results)

    assert result.action == "HOLD"
