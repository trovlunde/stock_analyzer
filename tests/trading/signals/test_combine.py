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


def test_default_mode_is_consensus():
    results = {"ma": _signal("BUY", "golden cross"), "ml": _signal("SELL", "ML predicts negative")}

    result = combine_signals(results)

    assert result.action == "HOLD"
